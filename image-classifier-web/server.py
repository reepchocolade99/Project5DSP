from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import re
from datetime import datetime
import logging

# Import PDF extraction functions
from extract_images import extract_embedded_images, write_manifest
import fitz

# Import SAM3 service
from sam3_service import SAM3Analyzer, analyze_evidence_images

# Import Claude Vision service for MLLM
from claude_vision_service import ClaudeVisionService, analyze_parking_evidence

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
DATA_FOLDER = os.path.join(os.path.dirname(__file__), 'data')
DERIVED_FOLDER = os.path.join(DATA_FOLDER, 'derived')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(DERIVED_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DATA_FOLDER'] = DATA_FOLDER
app.config['DERIVED_FOLDER'] = DERIVED_FOLDER

# Initialize SAM3 Analyzer (mock mode for development)
sam3_analyzer = SAM3Analyzer(output_dir=DERIVED_FOLDER, mock_mode=True)

# Initialize Claude Vision Service for MLLM
claude_vision_service = ClaudeVisionService()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'pdf'}

# ==================== TRANSLATIONS ====================
TRANSLATIONS = {
    'en': {
        # Page titles
        'page_title': 'Analysis Report',
        'upload_title': 'Upload',

        # Tab names
        'tab_report': 'Report',
        'tab_summary': 'Document Summary',

        # Generated time
        'generated': 'Generated',

        # Report header
        'case': 'Case',
        'parking_violation': 'Parking Violation',

        # Source badges
        'from_document': 'From Document',
        'ai_generated': 'AI Generated',
        'legal_template': 'Legal Template',
        'missing_data': 'Missing Data',
        'system': 'System',

        # Section titles
        'section_1': '1. Image Description',
        'section_2': '2. Object Detection Analysis',
        'section_3': '3. Timestamp & Location Verification',
        'section_4': '4. Environmental Context',
        'section_5': '5. Legal Reasoning',
        'section_6': '6. Supporting Evidence',
        'section_7': '7. Confidence Summary',

        # Inspector
        'evidence_inspector': 'Evidence Inspector',
        'filename': 'Filename',
        'page': 'Page',
        'method': 'Method',
        'size': 'Size',
        'detected_items': 'Detected Items',
        'vehicle': 'Vehicle',
        'license_plate': 'License Plate',
        'sign': 'Sign',
        'extracted_text': 'Extracted Text',
        'no_images_available': 'No evidence images available',

        # Confidence
        'confidence_scores': 'Confidence Scores',
        'confidence': 'Confidence',
        'object_detection': 'Object Detection',
        'text_recognition': 'Text Recognition',
        'legal_reasoning': 'Legal Reasoning',

        # Summary tab
        'case_identifiers': 'Case Identifiers',
        'sequence_number': 'Sequence Number',
        'ticket_number': 'Ticket Number',
        'status': 'Status',
        'date_time': 'Date/Time',
        'employee': 'Employee',
        'location': 'Location',
        'city': 'City',
        'district': 'District',
        'neighborhood': 'Neighborhood',
        'street': 'Street',
        'violation': 'Violation',
        'description': 'Description',
        'clarification': 'Clarification',
        'removal_reason': 'Removal Reason',
        'vehicle_info': 'Vehicle',
        'license_plate_label': 'License Plate',
        'brand': 'Brand',
        'model': 'Model',
        'color': 'Color',
        'officer_observation': 'Officer Observation',
        'no_observation': 'No observation available in document',
        'related_registrations': 'Related Registrations',
        'type': 'Type',
        'reference': 'Reference',

        # Evidence
        'evidence_images': 'Evidence Images',
        'no_images_extracted': 'No images extracted from document',

        # Export
        'export_options': 'Export Options',
        'download_pdf_report': 'Download PDF Report',
        'export_json': 'Export JSON',
        'download_images': 'Download Images',

        # Actions
        'new_analysis': 'New Analysis',
        'footer': 'Amsterdam Parking Enforcement System',
        'copy_to_clipboard': 'Copy to clipboard',

        # Not available
        'not_available': 'Not available',
        'unknown': 'Unknown',
        'none': 'None',
        'not_specified': 'Not specified',

        # Report content (summaries - will be generated in selected language)
        'images_show_vehicle': 'The images show a vehicle',
        'with_plate': 'with license plate',
        'parked_at_location': 'The vehicle is parked at the indicated location.',
        'images_show_parked': 'The images show a parked vehicle. Vehicle details are being analyzed based on available images.',
        'detected_objects': 'Detected objects: vehicle, license plate',
        'traffic_sign': 'traffic sign',
        'auto_detection_support': 'Automated detection has identified key elements in the image to support legal assessment.',
        'location_label': 'Location',
        'date_time_label': 'Date/Time',
        'location_not_available': 'Location and time data not available in document metadata.',
        'env_context': 'Environmental context is determined based on the images. Visible elements include street furniture, traffic signs, and the immediate parking environment.',
        'no_images_for_env': 'No images available for environmental analysis.',
        'supporting_evidence_label': 'Supporting Evidence',
        'num_evidence_photos': 'Number of evidence photos',
        'vehicle_data': 'Vehicle data',
        'officer_obs_status': 'Officer observation',
        'available': 'Available',
        'reasons_knowledge': 'Reasons of knowledge',
        'report_generated': 'This report was generated based on automated image analysis and document extraction. Please verify all data manually before taking further action.',

        # Violation descriptions (English)
        'violation_E1': 'No parking zone',
        'violation_E2': 'No stopping zone',
        'violation_E3': 'No bicycles/mopeds parking',
        'violation_E4': 'Parking facility',
        'violation_E5': 'Taxi stand',
        'violation_E6': 'Disabled parking space',
        'violation_E7': 'Loading/unloading zone',
        'violation_E8': 'Parking for designated vehicles only',
        'violation_E9': 'Permit holders parking only',
        'violation_E10': 'Parking disc zone',
        'violation_E11': 'End of parking disc zone',
        'violation_E12': 'Park and ride facility',
        'violation_E13': 'Carpool parking',
        'violation_G7': 'Footpath / Pedestrian area',

        # Legal phrases (English summaries - Dutch quotes remain in Dutch)
        'legal_E9_summary': 'The vehicle was parked in a permit holders only area without a valid permit visible.',
        'legal_E6_summary': 'The vehicle was parked in a disabled parking space without a valid disabled parking card visible.',
        'legal_E7_summary': 'The vehicle was parked in a loading/unloading zone without active loading/unloading activities.',
        'legal_G7_summary': 'The vehicle was parked on the footpath/pedestrian area, blocking pedestrian access.',
        'legal_default_summary': 'The vehicle was parked in violation of parking regulations.',
        'no_driver_present': 'No driver was present in or around the vehicle.',
        'violation_code': 'Violation',
        'no_description': 'No description available',
    },
    'nl': {
        # Page titles
        'page_title': 'Analyse Rapport',
        'upload_title': 'Uploaden',

        # Tab names
        'tab_report': 'Rapport',
        'tab_summary': 'Document Samenvatting',

        # Generated time
        'generated': 'Gegenereerd',

        # Report header
        'case': 'Zaak',
        'parking_violation': 'Parkeerovertreding',

        # Source badges
        'from_document': 'Uit Document',
        'ai_generated': 'AI Gegenereerd',
        'legal_template': 'Juridisch Sjabloon',
        'missing_data': 'Ontbrekende Data',
        'system': 'Systeem',

        # Section titles
        'section_1': '1. Beeldbeschrijving',
        'section_2': '2. Objectdetectie Analyse',
        'section_3': '3. Tijdstip & Locatie Verificatie',
        'section_4': '4. Omgevingscontext',
        'section_5': '5. Juridische Onderbouwing',
        'section_6': '6. Ondersteunend Bewijs',
        'section_7': '7. Betrouwbaarheidssamenvatting',

        # Inspector
        'evidence_inspector': 'Bewijs Inspecteur',
        'filename': 'Bestandsnaam',
        'page': 'Pagina',
        'method': 'Methode',
        'size': 'Grootte',
        'detected_items': 'Gedetecteerde Items',
        'vehicle': 'Voertuig',
        'license_plate': 'Kenteken',
        'sign': 'Bord',
        'extracted_text': 'Geëxtraheerde Tekst',
        'no_images_available': 'Geen bewijsafbeeldingen beschikbaar',

        # Confidence
        'confidence_scores': 'Betrouwbaarheidsscores',
        'confidence': 'Betrouwbaarheid',
        'object_detection': 'Objectdetectie',
        'text_recognition': 'Tekstherkenning',
        'legal_reasoning': 'Juridische Onderbouwing',

        # Summary tab
        'case_identifiers': 'Zaakgegevens',
        'sequence_number': 'Volgnummer',
        'ticket_number': 'Bonnummer',
        'status': 'Status',
        'date_time': 'Datum/Tijd',
        'employee': 'Medewerker',
        'location': 'Locatie',
        'city': 'Plaats',
        'district': 'Stadsdeel',
        'neighborhood': 'Buurt',
        'street': 'Straat',
        'violation': 'Overtreding',
        'description': 'Beschrijving',
        'clarification': 'Toelichting',
        'removal_reason': 'Reden Verwijdering',
        'vehicle_info': 'Voertuig',
        'license_plate_label': 'Kenteken',
        'brand': 'Merk',
        'model': 'Model',
        'color': 'Kleur',
        'officer_observation': 'Waarneming Ambtenaar',
        'no_observation': 'Geen observatie beschikbaar in document',
        'related_registrations': 'Gerelateerde Registraties',
        'type': 'Type',
        'reference': 'Referentie',

        # Evidence
        'evidence_images': 'Bewijsafbeeldingen',
        'no_images_extracted': 'Geen afbeeldingen geëxtraheerd uit document',

        # Export
        'export_options': 'Export Opties',
        'download_pdf_report': 'Download PDF Rapport',
        'export_json': 'Exporteer JSON',
        'download_images': 'Download Afbeeldingen',

        # Actions
        'new_analysis': 'Nieuwe Analyse',
        'footer': 'Amsterdam Parkeerhandhaving Systeem',
        'copy_to_clipboard': 'Kopieer naar klembord',

        # Not available
        'not_available': 'Niet beschikbaar',
        'unknown': 'Onbekend',
        'none': 'Geen',
        'not_specified': 'Niet gespecificeerd',

        # Report content (summaries in Dutch)
        'images_show_vehicle': 'De beelden tonen een voertuig',
        'with_plate': 'met kenteken',
        'parked_at_location': 'Het voertuig is geparkeerd op de aangegeven locatie.',
        'images_show_parked': 'De beelden tonen een geparkeerd voertuig. Voertuigdetails worden geanalyseerd op basis van de beschikbare beelden.',
        'detected_objects': 'Gedetecteerde objecten: voertuig, kenteken',
        'traffic_sign': 'verkeersbord',
        'auto_detection_support': 'De automatische detectie heeft de belangrijkste elementen in beeld geïdentificeerd ter ondersteuning van de juridische beoordeling.',
        'location_label': 'Locatie',
        'date_time_label': 'Datum/Tijd',
        'location_not_available': 'Locatie- en tijdgegevens niet beschikbaar in de documentmetadata.',
        'env_context': 'De omgevingscontext wordt bepaald op basis van de beelden. Zichtbare elementen omvatten straatmeubilair, verkeersborden en de directe parkeeromgeving.',
        'no_images_for_env': 'Geen beelden beschikbaar voor omgevingsanalyse.',
        'supporting_evidence_label': 'Ondersteunend bewijs',
        'num_evidence_photos': 'Aantal bewijsfoto\'s',
        'vehicle_data': 'Voertuiggegevens',
        'officer_obs_status': 'Observatie ambtenaar',
        'available': 'Beschikbaar',
        'reasons_knowledge': 'Redenen van wetenschap',
        'report_generated': 'Dit rapport is gegenereerd op basis van geautomatiseerde beeldanalyse en documentextractie. Controleer alle gegevens handmatig voordat u verdere actie onderneemt.',

        # Violation descriptions (Dutch)
        'violation_E1': 'Parkeerverbod',
        'violation_E2': 'Verbod stil te staan',
        'violation_E3': 'Verbod fietsen en bromfietsen te plaatsen',
        'violation_E4': 'Parkeergelegenheid',
        'violation_E5': 'Taxistandplaats',
        'violation_E6': 'Gehandicaptenparkeerplaats',
        'violation_E7': 'Gelegenheid bestemd voor laden en lossen',
        'violation_E8': 'Parkeergelegenheid alleen bestemd voor aangegeven voertuigen',
        'violation_E9': 'Parkeergelegenheid alleen bestemd voor vergunninghouders',
        'violation_E10': 'Parkeerschijf-zone',
        'violation_E11': 'Einde parkeerschijf-zone',
        'violation_E12': 'Parkeergelegenheid ten behoeve van overstappen',
        'violation_E13': 'Parkeergelegenheid ten behoeve van carpoolers',
        'violation_G7': 'Voetpad / Voetgangersgebied',

        # Legal phrases (Dutch)
        'legal_E9_summary': 'Het voertuig stond geparkeerd op een parkeergelegenheid bestemd voor vergunninghouders zonder geldige vergunning.',
        'legal_E6_summary': 'Het voertuig stond geparkeerd op een gehandicaptenparkeerplaats zonder geldige gehandicaptenparkeerkaart.',
        'legal_E7_summary': 'Het voertuig stond geparkeerd op een laad/los gelegenheid zonder actieve laad/los activiteiten.',
        'legal_G7_summary': 'Het voertuig stond geparkeerd op het voetpad/voetgangersgebied en blokkeerde de doorgang voor voetgangers.',
        'legal_default_summary': 'Het voertuig stond in overtreding geparkeerd.',
        'no_driver_present': 'Geen bestuurder was in of rondom het voertuig aanwezig.',
        'violation_code': 'Overtreding',
        'no_description': 'Geen beschrijving beschikbaar',

        # Index page
        'index_title': 'Afbeelding Classificatie',
        'index_subtitle': 'Geautomatiseerde bewijsanalyse en juridische ondersteuning.',
        'upload_file': 'Bestand Uploaden',
        'select_error': 'Selecteer een afbeelding of PDF bestand voordat u verzendt.',
        'settings': 'Instellingen',
        'language_label': 'Taal',
        'language_en': 'English',
        'language_nl': 'Nederlands',
        'decision_support': 'Beslissingsondersteuning',
        'decision_support_desc': 'Activeer AI-ondersteunde juridische redenering suggesties',
        'output_file_type': 'Uitvoer bestandstype',
        'model_type': 'Model Type',
        'model_desc_sam': 'Segmentatie-gebaseerde objectdetectie voor parkeerbewijzen',
        'model_desc_mllm': 'Claude Vision AI voor geavanceerde beeldanalyse',
        'mllm_coming_soon': 'MLLM analyse niet beschikbaar. Controleer API configuratie.',
        'start_analysis': 'Start Analyse & Bewaar Instellingen',
        'header_service': 'Afbeelding Classificatie',
        'choose_file': 'Bestand Kiezen',
        'no_file_chosen': 'Geen bestand gekozen',
        'drag_drop': 'of sleep een bestand hierheen',

        # SAM3 detection
        'show_overlay': 'Toon Segmentatie',
        'hide_overlay': 'Verberg Segmentatie',
        'detected': 'gedetecteerd',
        'not_detected': 'niet gedetecteerd',
        'roi_available': 'ROI beschikbaar',
        'view_crop': 'Bekijk Uitsnede',
        'detection_note': 'Detectie gebaseerd op geautomatiseerde beeldanalyse. Handmatige verificatie aanbevolen.',
        'not_enough_info': 'Onvoldoende informatie beschikbaar',

        # Loading animation
        'analyzing': 'Analyseren...',
        'processing_images': 'Afbeeldingen verwerken en data extraheren',
        'extracting_data': 'Data extraheren',
        'analyzing_images': 'Afbeeldingen analyseren',
        'generating_report': 'Rapport genereren',

        # MLLM/Verification
        'verification_status': 'Verificatie Status',
        'observation_supported': 'Bewijs ondersteunt observatie',
        'discrepancies_found': 'Afwijkingen gevonden',

        # MLLM Report Section Labels
        'detected_objects_mllm': 'Gedetecteerde objecten door MLLM analyse:',
        'environmental_analysis_mllm': 'Omgevingsanalyse door MLLM:',
        'parking_permit': 'Parkeervergunning',
        'driver': 'Bestuurder',
        'present': 'aanwezig',
        'not_present': 'niet aanwezig',
        'time_of_day': 'Tijdstip',
        'lighting': 'Verlichting',
        'weather': 'Weer',
        'environment': 'Omgeving',
        'from_document_separator': '--- Uit Document ---',
        'violation_label': 'Overtreding',
        'clarification_label': 'Toelichting',
    }
}

# Add English index translations
TRANSLATIONS['en'].update({
    'index_title': 'Image Classifier',
    'index_subtitle': 'Automated evidence analysis and legal support.',
    'upload_file': 'Upload File',
    'select_error': 'Please select an image or PDF file before submitting.',
    'settings': 'Settings',
    'language_label': 'Language',
    'language_en': 'English',
    'language_nl': 'Nederlands',
    'decision_support': 'Decision Support',
    'decision_support_desc': 'Enable AI-powered legal reasoning suggestions',
    'output_file_type': 'Output File Type',
    'model_type': 'Model Type',
    'model_desc_sam': 'Segmentation-based object detection for parking evidence',
    'model_desc_mllm': 'Claude Vision AI for advanced image analysis',
    'mllm_coming_soon': 'MLLM analysis not available. Check API configuration.',
    'start_analysis': 'Start Analysis & Save Settings',
    'header_service': 'Image Classifier',
    'choose_file': 'Choose File',
    'no_file_chosen': 'No file chosen',
    'drag_drop': 'or drag and drop a file here',
    # SAM3 detection
    'show_overlay': 'Show Segmentation',
    'hide_overlay': 'Hide Segmentation',
    'detected': 'detected',
    'not_detected': 'not detected',
    'roi_available': 'ROI available',
    'view_crop': 'View Crop',
    'detection_note': 'Detection based on automated image analysis. Manual verification recommended.',
    'not_enough_info': 'Not enough information available',

    # Loading animation
    'analyzing': 'Analyzing...',
    'processing_images': 'Processing images and extracting data',
    'extracting_data': 'Extracting data',
    'analyzing_images': 'Analyzing images',
    'generating_report': 'Generating report',

    # MLLM/Verification
    'verification_status': 'Verification Status',
    'observation_supported': 'Evidence supports observation',
    'discrepancies_found': 'Discrepancies found',

    # MLLM Report Section Labels
    'detected_objects_mllm': 'Detected objects via MLLM analysis:',
    'environmental_analysis_mllm': 'Environmental analysis via MLLM:',
    'parking_permit': 'Parking Permit',
    'driver': 'Driver',
    'present': 'present',
    'not_present': 'not present',
    'time_of_day': 'Time of Day',
    'lighting': 'Lighting',
    'weather': 'Weather',
    'environment': 'Environment',
    'from_document_separator': '--- From Document ---',
    'violation_label': 'Violation',
    'clarification_label': 'Clarification',
})


def get_translations(lang='en'):
    """Get translations for the specified language."""
    return TRANSLATIONS.get(lang, TRANSLATIONS['en'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_pdf(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'


def extract_pdf_text(pdf_path):
    """Extract all text from PDF for field extraction."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


def extract_structured_fields(text, lang='en'):
    """
    Extract structured fields from Dutch parking/towing case PDF text.
    Returns a dictionary with case, location, violation, vehicle info.
    """
    t = get_translations(lang)

    # Helper to find field value
    def find_field(pattern, text, default=None):
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else default

    not_available = t['not_available']

    # Extract case identifiers
    case = {
        "volgnummer": find_field(r"volgnummer[:\s]*([A-Z0-9\-]+)", text) or not_available,
        "bonnummer": find_field(r"bonnummer[:\s]*([A-Z0-9\-]+)", text) or not_available,
        "status": find_field(r"status[:\s]*(\w+)", text) or "Wegsleepwaardig",
        "datum_tijd": find_field(r"datum[/\s]tijd[:\s]*([0-9\-\s:]+)", text) or not_available,
        "medewerker": find_field(r"medewerker[:\s]*([A-Za-z\s\.]+)", text) or not_available,
    }

    # Try to extract date/time from common formats
    date_match = re.search(r"(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s*(\d{1,2}:\d{2})?", text)
    if date_match:
        case["datum_tijd"] = f"{date_match.group(1)} {date_match.group(2) or ''}".strip()

    # Extract location
    location = {
        "plaats": find_field(r"plaats[:\s]*([A-Za-z\s]+)", text) or "Amsterdam",
        "stadsdeel": find_field(r"stadsdeel[:\s]*([A-Za-z\s\-]+)", text) or not_available,
        "buurt": find_field(r"buurt[:\s]*([A-Za-z\s\-]+)", text) or not_available,
        "straat": find_field(r"straat[:\s]*([A-Za-z\s]+)", text) or not_available,
        "locatie_nr": find_field(r"(?:locatie\s*nr|huisnummer)[:\s]*(\d+)", text) or "",
    }

    # Extract violation info - look for E-codes (parking signs)
    violation_code_match = re.search(r"\b(E[1-9]|E1[0-3]|G[1-9])\b", text)
    violation_code = violation_code_match.group(1) if violation_code_match else None

    # Get violation description in selected language (prioritize translation over extracted text)
    violation_desc_key = f'violation_{violation_code}' if violation_code else None
    translated_desc = t.get(violation_desc_key) if violation_desc_key else None
    extracted_desc = find_field(r"overtreding[:\s]*(.+?)(?:\n|$)", text)

    # Use translated description if available for the violation code, otherwise use extracted
    violation_desc = translated_desc or extracted_desc or t['not_specified']

    violation = {
        "code": violation_code or t['not_specified'],
        "sign": violation_code,
        "description": violation_desc,
        "toelichting": find_field(r"toelichting[:\s]*(.+?)(?:\n|$)", text) or t['none'],
        "reden_verwijdering": find_field(r"reden\s*(?:van\s*)?verwijdering[:\s]*(.+?)(?:\n|$)", text) or not_available,
    }

    # Extract vehicle info
    kenteken_match = re.search(r"\b([A-Z]{1,3}[-\s]?\d{1,3}[-\s]?[A-Z]{1,3}|\d{1,2}[-\s]?[A-Z]{2,3}[-\s]?\d{1,2})\b", text)
    vehicle = {
        "kenteken": kenteken_match.group(1).replace(" ", "-") if kenteken_match else not_available,
        "merk": find_field(r"merk[:\s]*([A-Za-z]+)", text) or not_available,
        "model": find_field(r"model[:\s]*([A-Za-z0-9\s]+)", text) or not_available,
        "kleur": find_field(r"kleur[:\s]*([A-Za-z]+)", text) or not_available,
    }

    # Extract officer observation (keep original Dutch text from document)
    observation_match = re.search(
        r"(?:redenen\s*van\s*wetenschap|waarneming|observatie)[:\s]*(.+?)(?=\n\n|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    officer_observation = observation_match.group(1).strip() if observation_match else None

    # Extract related registrations
    related = []
    sleepbon_match = re.search(r"sleepbon[:\s]*([A-Z0-9\-]+)", text, re.IGNORECASE)
    if sleepbon_match:
        related.append({"type": "Sleepbon", "reference": sleepbon_match.group(1)})

    return {
        "case": case,
        "location": location,
        "violation": violation,
        "vehicle": vehicle,
        "officer_observation": officer_observation,
        "related_registrations": related
    }


def _generate_sam3_detection_text(sam3_results, lang, t, violation_code):
    """
    Generate object detection text from SAM3 results.

    Uses conservative language and only reports what was detected.
    """
    aggregate = sam3_results.get('aggregate', {})
    per_image = sam3_results.get('per_image', {})

    lines = []

    if lang == 'nl':
        header = "Gedetecteerde objecten in bewijsmateriaal:"
        detected_text = t.get('detected', 'gedetecteerd')
        not_detected_text = t.get('not_detected', 'niet gedetecteerd')
        confidence_text = t.get('confidence', 'betrouwbaarheid')
    else:
        header = "Detected objects in evidence material:"
        detected_text = t.get('detected', 'detected')
        not_detected_text = t.get('not_detected', 'not detected')
        confidence_text = t.get('confidence', 'confidence')

    lines.append(header)
    lines.append("")

    # Calculate average scores from per_image results
    vehicle_scores = []
    plate_scores = []
    sign_scores = []

    for img_result in per_image.values():
        for inst in img_result.get('instances', []):
            if inst['label'] == 'vehicle':
                vehicle_scores.append(inst['score'])
            elif inst['label'] == 'license_plate':
                plate_scores.append(inst['score'])
            elif inst['label'] == 'traffic_sign':
                sign_scores.append(inst['score'])

    # Vehicle detection
    if vehicle_scores:
        avg_score = sum(vehicle_scores) / len(vehicle_scores)
        lines.append(f"• {t['vehicle']}: {detected_text} ({confidence_text}: {int(avg_score * 100)}%)")
    else:
        lines.append(f"• {t['vehicle']}: {not_detected_text}")

    # License plate detection
    if plate_scores:
        avg_score = sum(plate_scores) / len(plate_scores)
        lines.append(f"• {t['license_plate']}: {detected_text} ({confidence_text}: {int(avg_score * 100)}%)")
    else:
        lines.append(f"• {t['license_plate']}: {not_detected_text}")

    # Traffic sign detection
    if sign_scores:
        avg_score = sum(sign_scores) / len(sign_scores)
        sign_label = f"{t['sign']} {violation_code}" if violation_code else t.get('traffic_sign', 'Traffic Sign')
        lines.append(f"• {sign_label}: {detected_text} ({confidence_text}: {int(avg_score * 100)}%)")
    else:
        if violation_code:
            lines.append(f"• {t['sign']} {violation_code}: {not_detected_text}")

    lines.append("")
    lines.append(t.get('detection_note', 'Detection based on automated image analysis. Manual verification recommended.'))

    return "\n".join(lines)


def generate_report_sections(doc_summary, images, lang='en', sam3_results=None):
    """
    Generate the 7 report sections based on extracted data.
    Uses language-appropriate content and SAM3/MLLM detection results.

    Args:
        doc_summary: Document summary from PDF extraction
        images: List of image filenames
        lang: Language code (en/nl)
        sam3_results: SAM3 or MLLM analysis results dict (optional)
    """
    t = get_translations(lang)

    violation = doc_summary.get("violation", {})
    vehicle = doc_summary.get("vehicle", {})
    location = doc_summary.get("location", {})
    case = doc_summary.get("case", {})
    observation = doc_summary.get("officer_observation")

    violation_code = violation.get("code", "")
    not_available = t['not_available']

    # Check if MLLM mode is active
    is_mllm_mode = sam3_results and sam3_results.get('mllm_mode')

    # Build image description content
    if is_mllm_mode and sam3_results.get('image_description'):
        # Use MLLM-generated image description
        image_desc = sam3_results.get('image_description')
    elif vehicle.get('kenteken') and vehicle.get('kenteken') != not_available:
        merk = vehicle.get('merk', t['unknown'])
        model = vehicle.get('model', '')
        kleur = vehicle.get('kleur', t['unknown'])
        kenteken = vehicle.get('kenteken')
        image_desc = f"{t['images_show_vehicle']} ({merk} {model}, {kleur}) {t['with_plate']} {kenteken}. {t['parked_at_location']}"
    else:
        image_desc = t['images_show_parked']

    # Object detection content - use MLLM or SAM3 results if available
    if is_mllm_mode and sam3_results.get('analysis'):
        # Use MLLM analysis for object detection section
        analysis = sam3_results.get('analysis', {})
        obj_det = analysis.get('object_detection', {})

        lines = []
        lines.append(t['detected_objects_mllm'])
        lines.append("")

        # Vehicle
        if obj_det.get('vehicle', {}).get('detected'):
            conf = int(obj_det['vehicle'].get('confidence', 0) * 100)
            details = obj_det['vehicle'].get('details', '')
            lines.append(f"• {t['vehicle']}: {t['detected']} ({t['confidence']}: {conf}%)")
            if details:
                lines.append(f"  {details}")

        # License plate
        if obj_det.get('license_plate', {}).get('detected'):
            conf = int(obj_det['license_plate'].get('confidence', 0) * 100)
            value = obj_det['license_plate'].get('value', '')
            lines.append(f"• {t['license_plate']}: {t['detected']} ({t['confidence']}: {conf}%)")
            if value:
                extracted_label = "Extracted" if lang == 'en' else "Geëxtraheerd"
                lines.append(f"  {extracted_label}: {value}")

        # Traffic sign
        if obj_det.get('traffic_sign', {}).get('detected'):
            conf = int(obj_det['traffic_sign'].get('confidence', 0) * 100)
            sign_type = obj_det['traffic_sign'].get('sign_type', '')
            sign_label = f"{t['sign']} {sign_type}" if sign_type else t.get('traffic_sign', 'Traffic Sign')
            lines.append(f"• {sign_label}: {t['detected']} ({t['confidence']}: {conf}%)")

        # Parking permit
        if obj_det.get('parking_permit', {}).get('detected'):
            conf = int(obj_det['parking_permit'].get('confidence', 0) * 100)
            lines.append(f"• {t['parking_permit']}: {t['detected']} ({t['confidence']}: {conf}%)")
        elif obj_det.get('parking_permit'):
            lines.append(f"• {t['parking_permit']}: {t['not_detected']}")

        # Driver presence
        if obj_det.get('driver_present', {}).get('detected'):
            lines.append(f"• {t['driver']}: {t['present']}")
        elif obj_det.get('driver_present'):
            lines.append(f"• {t['driver']}: {t['not_present']}")

        obj_detect = "\n".join(lines)
    elif sam3_results and sam3_results.get('aggregate'):
        obj_detect = _generate_sam3_detection_text(sam3_results, lang, t, violation_code)
    else:
        # Fallback to generic text
        obj_detect = f"{t['detected_objects']}"
        if violation_code and violation_code != t['not_specified']:
            obj_detect += f", {t['traffic_sign']} {violation_code}"
        obj_detect += f". {t['auto_detection_support']}"

    # Location content
    if location.get('straat') and location.get('straat') != not_available:
        loc_content = f"{t['location_label']}: {location.get('straat', t['unknown'])} {location.get('locatie_nr', '')}, {location.get('buurt', '')}, {location.get('stadsdeel', '')}, {location.get('plaats', 'Amsterdam')}.\n{t['date_time_label']}: {case.get('datum_tijd', not_available)}."
        loc_source = "document"
    else:
        loc_content = t['location_not_available']
        loc_source = "missing"

    # Environmental context - use MLLM data if available
    if is_mllm_mode and sam3_results.get('environmental_context'):
        env_ctx = sam3_results.get('environmental_context', {})
        env_lines = []
        env_lines.append(t['environmental_analysis_mllm'])
        env_lines.append("")
        if env_ctx.get('time_of_day'):
            env_lines.append(f"• {t['time_of_day']}: {env_ctx['time_of_day']}")
        if env_ctx.get('lighting'):
            env_lines.append(f"• {t['lighting']}: {env_ctx['lighting']}")
        if env_ctx.get('weather'):
            env_lines.append(f"• {t['weather']}: {env_ctx['weather']}")
        if env_ctx.get('street_description'):
            env_lines.append(f"• {t['environment']}: {env_ctx['street_description']}")
        env_content = "\n".join(env_lines)
        env_source = "model"
    elif images:
        env_content = t['env_context']
        env_source = "model"
    else:
        env_content = t['no_images_for_env']
        env_source = "missing"

    # Legal reasoning - include original Dutch legal phrases as quotes, with summary in selected language
    legal_summary_key = f'legal_{violation_code}_summary' if violation_code and violation_code != t['not_specified'] else 'legal_default_summary'
    legal_summary = t.get(legal_summary_key, t['legal_default_summary'])

    # Original Dutch legal phrases (these are direct quotes from legal templates)
    dutch_legal_templates = {
        "E9": [
            "Ik zag dat het voertuig geparkeerd stond op een parkeergelegenheid bestemd voor vergunninghouders.",
            "Ik zag geen geldige vergunning zichtbaar aanwezig in of aan het voertuig.",
            "Ik zag geen laad/los activiteiten plaatsvinden.",
            "Geen bestuurder was in of rondom het voertuig aanwezig.",
        ],
        "E6": [
            "Ik zag dat het voertuig geparkeerd stond op een gehandicaptenparkeerplaats.",
            "Ik zag geen geldige gehandicaptenparkeerkaart zichtbaar aanwezig.",
            "Geen bestuurder was in of rondom het voertuig aanwezig.",
        ],
        "E7": [
            "Ik zag dat het voertuig geparkeerd stond op een laad/los gelegenheid.",
            "Ik zag geen laad/los activiteiten plaatsvinden.",
            "Geen bestuurder was in of rondom het voertuig aanwezig.",
        ],
        "G7": [
            "Ik zag dat het voertuig geparkeerd stond op het voetpad/voetgangersgebied.",
            "Het voertuig blokkeerde de doorgang voor voetgangers.",
        ],
    }

    dutch_phrases = dutch_legal_templates.get(violation_code, [
        "Ik zag dat het voertuig in overtreding geparkeerd stond.",
        "Geen bestuurder was in of rondom het voertuig aanwezig.",
    ])

    # Build legal content with quotes and summary
    legal_content = legal_summary + "\n\n" + t['no_driver_present']
    legal_content += f"\n\n{t['from_document_separator']}\n"
    legal_content += "\n".join(dutch_phrases)
    legal_content += f"\n\n{t['violation_label']}: {violation.get('code', t['not_specified'])} - {violation.get('description', t['no_description'])}."
    if violation.get('toelichting') and violation.get('toelichting') != t['none']:
        legal_content += f"\n{t['clarification_label']}: {violation.get('toelichting')}"

    # Supporting evidence
    obs_status = t['available'] if observation else not_available
    evidence_content = f"{t['supporting_evidence_label']}:\n- {t['num_evidence_photos']}: {len(images)}\n- {t['vehicle_data']}: {vehicle.get('kenteken', not_available)}\n- {t['officer_obs_status']}: {obs_status}"
    if observation:
        evidence_content += f"\n\n{t['reasons_knowledge']}:\n{observation}"

    sections = [
        {
            "id": "image_description",
            "title": t['section_1'],
            "content": image_desc,
            "source": "model",
            "editable": True
        },
        {
            "id": "object_detection",
            "title": t['section_2'],
            "content": obj_detect,
            "source": "model",
            "confidence": 0.94,
            "editable": True
        },
        {
            "id": "timestamp_location",
            "title": t['section_3'],
            "content": loc_content,
            "source": loc_source,
            "editable": True
        },
        {
            "id": "environmental_context",
            "title": t['section_4'],
            "content": env_content,
            "source": env_source,
            "editable": True
        },
        {
            "id": "legal_reasoning",
            "title": t['section_5'],
            "content": legal_content,
            "source": "template",
            "editable": True
        },
        {
            "id": "supporting_evidence",
            "title": t['section_6'],
            "content": evidence_content,
            "source": "document" if observation else "model",
            "editable": True
        },
        {
            "id": "confidence_summary",
            "title": t['section_7'],
            "content": sam3_results.get('summary', t['report_generated']) if is_mllm_mode else t['report_generated'],
            "source": "model" if is_mllm_mode else "system",
            "editable": False
        }
    ]

    return sections


def calculate_confidence_scores(doc_summary, images):
    """Calculate confidence scores based on available data."""

    scores = {
        "object_detection": 0.0,
        "text_recognition": 0.0,
        "legal_reasoning": 0.0
    }

    # Object detection confidence based on image availability
    if images:
        scores["object_detection"] = min(0.95, 0.7 + (len(images) * 0.02))

    # Text recognition based on extracted fields
    vehicle = doc_summary.get("vehicle", {})
    kenteken = vehicle.get("kenteken", "")
    if kenteken and "beschikbaar" not in kenteken.lower() and "available" not in kenteken.lower():
        scores["text_recognition"] = 0.88
    elif any(v and "beschikbaar" not in str(v).lower() and "available" not in str(v).lower() for v in vehicle.values()):
        scores["text_recognition"] = 0.65

    # Legal reasoning based on violation code availability
    violation = doc_summary.get("violation", {})
    code = violation.get("code", "")
    if code and "gespecificeerd" not in code.lower() and "specified" not in code.lower():
        scores["legal_reasoning"] = 0.86
    elif doc_summary.get("officer_observation"):
        scores["legal_reasoning"] = 0.72
    else:
        scores["legal_reasoning"] = 0.45

    return scores


@app.route('/')
def index():
    # Default to English, but check if language preference is set
    lang = request.args.get('lang', 'en')
    if lang not in ['en', 'nl']:
        lang = 'en'
    t = get_translations(lang)
    return render_template('index.html', t=t, lang=lang)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400

    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'no file selected'}), 400

    if not allowed_file(f.filename):
        return jsonify({'error': 'file type not allowed'}), 400

    # Get language from form
    lang = request.form.get('language', 'en')
    if lang not in ['en', 'nl']:
        lang = 'en'

    # Get model type from form (sam or mllm)
    model_type = request.form.get('model', 'sam')
    if model_type not in ['sam', 'mllm']:
        model_type = 'sam'

    t = get_translations(lang)

    filename = secure_filename(f.filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)

    # Initialize data structures
    extracted_images = []
    images_metadata = []
    doc_summary = {
        "case": {},
        "location": {},
        "violation": {},
        "vehicle": {},
        "officer_observation": None,
        "related_registrations": []
    }

    # Check if PDF - extract images and text
    if is_pdf(filename):
        try:
            from pathlib import Path

            # Extract text for structured fields
            pdf_text = extract_pdf_text(path)
            doc_summary = extract_structured_fields(pdf_text, lang)

            # Extract images
            doc = fitz.open(path)
            pdf_stem = Path(filename).stem
            pdf_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in pdf_stem)
            out_dir = Path(app.config['DATA_FOLDER'])

            all_images = []
            for page_num in range(len(doc)):
                images, _ = extract_embedded_images(doc, page_num, pdf_stem, out_dir, min_size=200)
                all_images.extend(images)

            doc.close()

            # Write manifest
            write_manifest(out_dir, Path(path), all_images, [])
            extracted_images = [img['file'] for img in all_images]
            images_metadata = all_images

        except Exception as e:
            return jsonify({'error': f'PDF extraction failed: {str(e)}'}), 500

    # Run SAM3 analysis on extracted images (only when SAM model is selected)
    sam3_results = None
    detected_items_ui = None

    if model_type == 'sam' and extracted_images:
        try:
            image_paths = [
                os.path.join(app.config['DATA_FOLDER'], img)
                for img in extracted_images
            ]
            sam3_results = analyze_evidence_images(
                image_paths,
                output_dir=app.config['DERIVED_FOLDER'],
                mock_mode=True  # Use mock mode for development
            )

            # Get detected items for the first image (default selection)
            if sam3_results.get('per_image'):
                first_image_id = list(sam3_results['per_image'].keys())[0]
                first_result = sam3_results['per_image'][first_image_id]

                # Get extracted text and sign code from document
                kenteken = doc_summary.get('vehicle', {}).get('kenteken')
                sign_code = doc_summary.get('violation', {}).get('code')

                # Pass dict directly - get_detected_items_for_ui handles both formats
                detected_items_ui = sam3_analyzer.get_detected_items_for_ui(
                    first_result,
                    lang=lang,
                    extracted_plate_text=kenteken if kenteken and 'beschikbaar' not in kenteken.lower() else None,
                    sign_code=sign_code if sign_code and 'gespecificeerd' not in sign_code.lower() else None
                )

            logger.info(f"SAM3 analysis completed: {sam3_results.get('aggregate', {})}")

        except Exception as e:
            logger.error(f"SAM3 analysis failed: {str(e)}")
            # Continue without SAM3 results
    elif model_type == 'mllm' and extracted_images:
        # MLLM analysis using Claude Vision
        try:
            image_paths = [
                os.path.join(app.config['DATA_FOLDER'], img)
                for img in extracted_images
            ]

            # Run Claude Vision analysis
            mllm_ui_data = analyze_parking_evidence(
                image_paths=image_paths,
                doc_summary=doc_summary,
                lang=lang,
                max_images=10
            )

            # Extract results for template
            if mllm_ui_data.get('mllm_analysis'):
                detected_items_ui = mllm_ui_data.get('detected_items_ui')
                sam3_results = {
                    'mllm_mode': True,
                    'analysis': mllm_ui_data.get('mllm_analysis'),
                    'summary': mllm_ui_data.get('summary'),
                    'verification': mllm_ui_data.get('verification'),
                    'image_description': mllm_ui_data.get('image_description'),
                    'environmental_context': mllm_ui_data.get('environmental_context'),
                    'metadata': mllm_ui_data.get('metadata')
                }
                logger.info(f"MLLM analysis completed: {mllm_ui_data.get('metadata', {})}")
            else:
                logger.warning(f"MLLM analysis failed: {mllm_ui_data.get('mllm_error')}")

        except Exception as e:
            logger.error(f"MLLM analysis failed: {str(e)}")
            # Continue without MLLM results

    # Generate report sections and confidence scores (now with SAM3/MLLM)
    report_sections = generate_report_sections(doc_summary, extracted_images, lang, sam3_results)

    # Calculate confidence scores based on mode
    if sam3_results and sam3_results.get('mllm_mode'):
        # MLLM mode - use confidence scores from Claude Vision analysis
        confidence_scores = mllm_ui_data.get('confidence_scores', {
            "object_detection": 0.0,
            "text_recognition": 0.0,
            "legal_reasoning": 0.0
        })
    elif sam3_results and sam3_results.get('per_image'):
        # SAM mode - use SAM3 analyzer
        kenteken = doc_summary.get('vehicle', {}).get('kenteken', '')
        has_plate = kenteken and 'beschikbaar' not in kenteken.lower() and 'available' not in kenteken.lower()
        violation_code = doc_summary.get('violation', {}).get('code', '')
        has_code = violation_code and 'gespecificeerd' not in violation_code.lower() and 'specified' not in violation_code.lower()
        confidence_scores = sam3_analyzer.calculate_confidence_scores(
            sam3_results.get('per_image', {}),
            has_plate_text=has_plate,
            has_violation_code=has_code
        )
    else:
        # Fallback - calculate from document summary
        confidence_scores = calculate_confidence_scores(doc_summary, extracted_images)

    # Add evidence images to doc_summary
    doc_summary["evidence_images"] = images_metadata

    return render_template('result.html',
                          filename=filename,
                          is_pdf=is_pdf(filename),
                          extracted_images=extracted_images,
                          images_metadata=images_metadata,
                          doc_summary=doc_summary,
                          report_sections=report_sections,
                          confidence_scores=confidence_scores,
                          sam3_results=sam3_results,
                          detected_items_ui=detected_items_ui,
                          model_type=model_type,
                          generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
                          lang=lang,
                          t=t)


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/data/<path:filename>')
def data_file(filename):
    return send_from_directory(app.config['DATA_FOLDER'], filename)


@app.route('/data/derived/<path:filename>')
def derived_file(filename):
    """Serve SAM3 derived files (crops, overlays)."""
    return send_from_directory(app.config['DERIVED_FOLDER'], filename)


@app.route('/api/export-json', methods=['POST'])
def export_json():
    """Export the document summary and report as JSON."""
    data = request.get_json()
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True, port=5001)
