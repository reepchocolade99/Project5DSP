"""
Claude Vision Service for MLLM-based parking violation image analysis.

This service:
1. Accepts images + police observation context
2. Calls Claude Vision API with structured prompts
3. Returns semantic metadata aligned with parking enforcement domain
"""

import base64
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import logging

# Make anthropic import optional
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)


class ClaudeVisionService:
    """
    Service for analyzing parking violation evidence images using Claude Vision.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Claude Vision service.

        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = None
        self.model = "claude-sonnet-4-20250514"

        if not ANTHROPIC_AVAILABLE:
            logger.warning("Anthropic package not installed - MLLM analysis will not be available")
            return

        if self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
                logger.info("ClaudeVisionService initialized with API key")
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
        else:
            logger.warning("No ANTHROPIC_API_KEY found - MLLM analysis will not be available")

    def is_available(self) -> bool:
        """Check if the service is available (API key configured)."""
        return self.client is not None

    def _encode_image(self, image_path: str) -> tuple[str, str]:
        """
        Encode image to base64 and determine media type.

        Returns:
            Tuple of (base64_data, media_type)
        """
        path = Path(image_path)
        suffix = path.suffix.lower()

        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }

        media_type = media_type_map.get(suffix, 'image/jpeg')

        with open(image_path, 'rb') as f:
            image_data = base64.standard_b64encode(f.read()).decode('utf-8')

        return image_data, media_type

    def _select_best_images(
        self,
        image_paths: List[str],
        max_images: int = 10
    ) -> List[str]:
        """
        Select the most representative images for analysis.

        Strategy:
        - Prefer larger images (more detail)
        - Ensure variety (different pages if from PDF)
        - Limit to max_images to control API costs

        Args:
            image_paths: List of all extracted image paths
            max_images: Maximum number of images to send to API

        Returns:
            List of selected image paths
        """
        if len(image_paths) <= max_images:
            return image_paths

        # Sort by file size (larger = more detail typically)
        sized_paths = []
        for p in image_paths:
            try:
                size = os.path.getsize(p)
                sized_paths.append((p, size))
            except:
                sized_paths.append((p, 0))

        sized_paths.sort(key=lambda x: x[1], reverse=True)

        # Take top max_images by size
        return [p[0] for p in sized_paths[:max_images]]

    def _build_analysis_prompt(
        self,
        officer_observation: str,
        violation_code: str,
        violation_description: str,
        vehicle_info: Dict[str, str],
        location_info: Dict[str, str],
        lang: str = 'nl'
    ) -> str:
        """
        Build the analysis prompt with all context.
        Language-aware: returns prompt that instructs Claude to respond in the selected language.
        """
        # Build context section (always includes Dutch source data)
        context_section = f"""
### Violation / Overtreding
- Code: {violation_code or 'Unknown'}
- Description: {violation_description or 'Not specified'}

### Vehicle / Voertuig
- License Plate: {vehicle_info.get('kenteken', 'Unknown')}
- Brand: {vehicle_info.get('merk', 'Unknown')}
- Model: {vehicle_info.get('model', 'Unknown')}
- Color: {vehicle_info.get('kleur', 'Unknown')}

### Location / Locatie
- Street: {location_info.get('straat', 'Unknown')}
- Neighborhood: {location_info.get('buurt', 'Unknown')}
- City: {location_info.get('plaats', 'Amsterdam')}

### Officer Observation (Dutch source document)
{officer_observation or 'No observation available'}
"""

        if lang == 'en':
            prompt = f"""You are a legal image analysis assistant for parking enforcement in Amsterdam.
Analyze the attached evidence photos of a parking violation.

IMPORTANT: All your text responses in the JSON output MUST be in ENGLISH.

## CONTEXT FROM THE OFFICIAL REPORT
{context_section}
---

## YOUR ANALYSIS TASKS

Analyze the photos and provide a structured JSON response with the following sections:

### 1. image_description
Objectively describe what is visible in the photos. Focus on:
- The vehicle (brand, model, color, condition)
- The parking location and surroundings
- Visible traffic signs
- The windshield (for permit/exemption verification)

### 2. object_detection
List of detected objects with confidence scores (0.0-1.0):
- vehicle: Is the vehicle visible?
- license_plate: Is the license plate readable? If so, what does it say?
- traffic_sign: Which traffic sign is visible? (E6, E7, E9, G7, etc.)
- parking_permit: Is there a parking permit/exemption visible behind the windshield?
- driver_present: Is there a driver present in or near the vehicle?

### 3. environmental_context
Describe the environmental factors:
- Time indication (day/night, lighting)
- Weather conditions (if visible)
- Street view and infrastructure

### 4. verification
Compare the images with the officer's observation:
- observation_supported: true/false - Do the images support the observation?
- matching_elements: List of elements that match
- discrepancies: List of discrepancies or uncertainties
- missing_evidence: What CANNOT be confirmed from the images?
- overall_confidence: 0.0-1.0 score for the total evidentiary strength

### 5. summary
A brief English summary (2-3 sentences) of the main findings.

---

## OUTPUT FORMAT

Respond ONLY with valid JSON in this exact format (no other text). ALL TEXT VALUES MUST BE IN ENGLISH:

{{
  "image_description": "string in English",
  "object_detection": {{
    "vehicle": {{"detected": true/false, "confidence": 0.0-1.0, "details": "string in English"}},
    "license_plate": {{"detected": true/false, "confidence": 0.0-1.0, "value": "string or null"}},
    "traffic_sign": {{"detected": true/false, "confidence": 0.0-1.0, "sign_type": "string or null"}},
    "parking_permit": {{"detected": true/false, "confidence": 0.0-1.0, "details": "string in English"}},
    "driver_present": {{"detected": true/false, "confidence": 0.0-1.0}}
  }},
  "environmental_context": {{
    "time_of_day": "string in English",
    "lighting": "string in English",
    "weather": "string in English",
    "street_description": "string in English"
  }},
  "verification": {{
    "observation_supported": true/false,
    "matching_elements": ["strings in English"],
    "discrepancies": ["strings in English"],
    "missing_evidence": ["strings in English"],
    "overall_confidence": 0.0-1.0
  }},
  "summary": "string in English"
}}"""
        else:
            # Dutch (nl) prompt
            prompt = f"""Je bent een juridische beeldanalyse-assistent voor parkeerhandhaving in Amsterdam.
Analyseer de bijgevoegde bewijsfoto's van een parkeerovertreding.

BELANGRIJK: Al je tekstuele antwoorden in de JSON output MOETEN in het NEDERLANDS zijn.

## CONTEXT UIT HET PROCES-VERBAAL
{context_section}
---

## JOUW ANALYSE TAKEN

Analyseer de foto's en geef een gestructureerd JSON antwoord met de volgende secties:

### 1. image_description
Beschrijf objectief wat zichtbaar is in de foto's. Focus op:
- Het voertuig (merk, model, kleur, staat)
- De parkeerlocatie en omgeving
- Zichtbare verkeersborden
- De voorruit (voor ontheffing/vergunning controle)

### 2. object_detection
Lijst van gedetecteerde objecten met confidence scores (0.0-1.0):
- vehicle: Is het voertuig zichtbaar?
- license_plate: Is het kenteken leesbaar? Zo ja, wat staat erop?
- traffic_sign: Welk verkeersbord is zichtbaar? (E6, E7, E9, G7, etc.)
- parking_permit: Is er een parkeervergunning/ontheffing zichtbaar achter de voorruit?
- driver_present: Is er een bestuurder aanwezig in of bij het voertuig?

### 3. environmental_context
Beschrijf de omgevingsfactoren:
- Tijdstip indicatie (dag/nacht, verlichting)
- Weersomstandigheden (indien zichtbaar)
- Straatbeeld en infrastructuur

### 4. verification
Vergelijk de beelden met de politie-observatie:
- observation_supported: true/false - Ondersteunen de beelden de observatie?
- matching_elements: Lijst van elementen die overeenkomen
- discrepancies: Lijst van afwijkingen of onduidelijkheden
- missing_evidence: Wat kan NIET worden bevestigd uit de beelden?
- overall_confidence: 0.0-1.0 score voor de totale bewijskracht

### 5. summary
Een korte Nederlandse samenvatting (2-3 zinnen) van de belangrijkste bevindingen.

---

## OUTPUT FORMAT

Antwoord ALLEEN met valid JSON in dit exacte format (geen andere tekst). ALLE TEKST WAARDEN MOETEN IN HET NEDERLANDS ZIJN:

{{
  "image_description": "string in Nederlands",
  "object_detection": {{
    "vehicle": {{"detected": true/false, "confidence": 0.0-1.0, "details": "string in Nederlands"}},
    "license_plate": {{"detected": true/false, "confidence": 0.0-1.0, "value": "string of null"}},
    "traffic_sign": {{"detected": true/false, "confidence": 0.0-1.0, "sign_type": "string of null"}},
    "parking_permit": {{"detected": true/false, "confidence": 0.0-1.0, "details": "string in Nederlands"}},
    "driver_present": {{"detected": true/false, "confidence": 0.0-1.0}}
  }},
  "environmental_context": {{
    "time_of_day": "string in Nederlands",
    "lighting": "string in Nederlands",
    "weather": "string in Nederlands",
    "street_description": "string in Nederlands"
  }},
  "verification": {{
    "observation_supported": true/false,
    "matching_elements": ["strings in Nederlands"],
    "discrepancies": ["strings in Nederlands"],
    "missing_evidence": ["strings in Nederlands"],
    "overall_confidence": 0.0-1.0
  }},
  "summary": "string in Nederlands"
}}"""
        return prompt

    def analyze_images(
        self,
        image_paths: List[str],
        officer_observation: str,
        violation_code: str,
        violation_description: str,
        vehicle_info: Dict[str, str],
        location_info: Dict[str, str],
        lang: str = 'nl',
        max_images: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze parking violation evidence images using Claude Vision.

        Args:
            image_paths: List of paths to evidence images
            officer_observation: "Redenen van wetenschap" text from PDF
            violation_code: e.g., "E9", "E6", "G7"
            violation_description: Full description of the violation
            vehicle_info: Dict with kenteken, merk, model, kleur
            location_info: Dict with straat, buurt, plaats
            lang: Language code for output
            max_images: Maximum images to analyze (controls API cost)

        Returns:
            Structured analysis results
        """
        if not self.client:
            return {
                "success": False,
                "analysis": None,
                "error": "Claude Vision service not available - no API key configured"
            }

        # Select best images
        selected_images = self._select_best_images(image_paths, max_images)
        logger.info(f"Selected {len(selected_images)} images for MLLM analysis")

        # Build message content with images
        content = []

        # Add images first
        for img_path in selected_images:
            try:
                img_data, media_type = self._encode_image(img_path)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": img_data
                    }
                })
                logger.debug(f"Encoded image: {img_path}")
            except Exception as e:
                logger.warning(f"Could not encode image {img_path}: {e}")
                continue

        if not content:
            return {
                "success": False,
                "analysis": None,
                "error": "No images could be encoded for analysis"
            }

        # Add the analysis prompt
        prompt = self._build_analysis_prompt(
            officer_observation=officer_observation,
            violation_code=violation_code,
            violation_description=violation_description,
            vehicle_info=vehicle_info,
            location_info=location_info,
            lang=lang
        )
        content.append({"type": "text", "text": prompt})

        # Call Claude Vision API
        try:
            logger.info("Calling Claude Vision API...")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": content}
                ]
            )

            # Extract JSON from response
            response_text = response.content[0].text
            logger.debug(f"Raw API response: {response_text[:500]}...")

            # Try to parse JSON (handle potential markdown code blocks)
            json_str = response_text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            analysis_result = json.loads(json_str.strip())

            # Add metadata
            analysis_result["_metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "images_analyzed": len(selected_images),
                "total_images_available": len(image_paths),
                "selected_image_paths": [os.path.basename(p) for p in selected_images]
            }

            logger.info("MLLM analysis completed successfully")

            return {
                "success": True,
                "analysis": analysis_result,
                "error": None
            }

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response as JSON: {e}")
            return {
                "success": False,
                "analysis": None,
                "error": f"Failed to parse API response as JSON: {e}",
                "raw_response": response_text if 'response_text' in locals() else None
            }
        except Exception as e:
            # Handle all API errors including anthropic.APIError
            error_type = type(e).__name__
            logger.error(f"{error_type} during MLLM analysis: {e}")
            return {
                "success": False,
                "analysis": None,
                "error": f"{error_type}: {str(e)}"
            }

    def format_for_ui(
        self,
        analysis_result: Dict[str, Any],
        lang: str = 'nl'
    ) -> Dict[str, Any]:
        """
        Format the analysis result for the UI template.

        Transforms the raw API response into the format expected by result.html
        """
        if not analysis_result.get("success"):
            return {
                "mllm_analysis": None,
                "mllm_error": analysis_result.get("error"),
                "detected_items_ui": {"items": []},
                "confidence_scores": {
                    "object_detection": 0.0,
                    "text_recognition": 0.0,
                    "legal_reasoning": 0.0
                }
            }

        analysis = analysis_result["analysis"]
        obj_det = analysis.get("object_detection", {})

        # Calculate confidence scores for UI
        vehicle_conf = obj_det.get("vehicle", {}).get("confidence", 0)
        sign_conf = obj_det.get("traffic_sign", {}).get("confidence", 0)
        plate_conf = obj_det.get("license_plate", {}).get("confidence", 0)
        verification_conf = analysis.get("verification", {}).get("overall_confidence", 0)

        confidence_scores = {
            "object_detection": (vehicle_conf + sign_conf) / 2 if (vehicle_conf or sign_conf) else 0,
            "text_recognition": plate_conf,
            "legal_reasoning": verification_conf
        }

        # Format detected items for UI sidebar
        detected_items = []

        # Labels based on language
        labels = {
            'nl': {
                'vehicle': 'Voertuig',
                'license_plate': 'Kenteken',
                'traffic_sign': 'Verkeersbord',
                'parking_permit': 'Parkeervergunning',
                'driver': 'Bestuurder'
            },
            'en': {
                'vehicle': 'Vehicle',
                'license_plate': 'License Plate',
                'traffic_sign': 'Traffic Sign',
                'parking_permit': 'Parking Permit',
                'driver': 'Driver'
            }
        }.get(lang, {})

        if obj_det.get("vehicle", {}).get("detected"):
            detected_items.append({
                "label": labels.get('vehicle', 'Vehicle'),
                "label_key": "vehicle",
                "confidence": int(obj_det["vehicle"]["confidence"] * 100),
                "detected": True,
                "details": obj_det["vehicle"].get("details", ""),
                "crop_available": False
            })

        if obj_det.get("license_plate", {}).get("detected"):
            plate_item = {
                "label": labels.get('license_plate', 'License Plate'),
                "label_key": "license_plate",
                "confidence": int(obj_det["license_plate"]["confidence"] * 100),
                "detected": True,
                "crop_available": False
            }
            if obj_det["license_plate"].get("value"):
                plate_item["extracted_text"] = obj_det["license_plate"]["value"]
            detected_items.append(plate_item)

        if obj_det.get("traffic_sign", {}).get("detected"):
            sign_item = {
                "label": labels.get('traffic_sign', 'Traffic Sign'),
                "label_key": "traffic_sign",
                "confidence": int(obj_det["traffic_sign"]["confidence"] * 100),
                "detected": True,
                "crop_available": False
            }
            if obj_det["traffic_sign"].get("sign_type"):
                sign_item["sign_code"] = obj_det["traffic_sign"]["sign_type"]
                sign_item["label"] = f"{labels.get('traffic_sign', 'Traffic Sign')} {obj_det['traffic_sign']['sign_type']}"
            detected_items.append(sign_item)

        # Add items that weren't detected
        if not obj_det.get("vehicle", {}).get("detected"):
            detected_items.append({
                "label": labels.get('vehicle', 'Vehicle'),
                "label_key": "vehicle",
                "confidence": 0,
                "detected": False,
                "crop_available": False
            })

        if not obj_det.get("license_plate", {}).get("detected"):
            detected_items.append({
                "label": labels.get('license_plate', 'License Plate'),
                "label_key": "license_plate",
                "confidence": 0,
                "detected": False,
                "crop_available": False
            })

        if not obj_det.get("traffic_sign", {}).get("detected"):
            detected_items.append({
                "label": labels.get('traffic_sign', 'Traffic Sign'),
                "label_key": "traffic_sign",
                "confidence": 0,
                "detected": False,
                "crop_available": False
            })

        return {
            "mllm_analysis": analysis,
            "mllm_error": None,
            "detected_items_ui": {"items": detected_items},
            "confidence_scores": confidence_scores,
            "image_description": analysis.get("image_description", ""),
            "environmental_context": analysis.get("environmental_context", {}),
            "verification": analysis.get("verification", {}),
            "summary": analysis.get("summary", ""),
            "metadata": analysis.get("_metadata", {})
        }


# Convenience function for direct use
def analyze_parking_evidence(
    image_paths: List[str],
    doc_summary: Dict[str, Any],
    lang: str = 'nl',
    max_images: int = 10
) -> Dict[str, Any]:
    """
    Convenience function to analyze parking violation evidence.

    Args:
        image_paths: List of image file paths
        doc_summary: Document summary from PDF extraction
        lang: Language code
        max_images: Maximum images to analyze

    Returns:
        Formatted results for UI
    """
    if not ANTHROPIC_AVAILABLE:
        return {
            "mllm_analysis": None,
            "mllm_error": "MLLM service not available - anthropic package not installed. Run: pip install anthropic",
            "detected_items_ui": {"items": []},
            "confidence_scores": {
                "object_detection": 0.0,
                "text_recognition": 0.0,
                "legal_reasoning": 0.0
            }
        }

    service = ClaudeVisionService()

    if not service.is_available():
        return {
            "mllm_analysis": None,
            "mllm_error": "MLLM service not available - ANTHROPIC_API_KEY not configured",
            "detected_items_ui": {"items": []},
            "confidence_scores": {
                "object_detection": 0.0,
                "text_recognition": 0.0,
                "legal_reasoning": 0.0
            }
        }

    # Extract data from doc_summary
    violation = doc_summary.get("violation", {})
    vehicle = doc_summary.get("vehicle", {})
    location = doc_summary.get("location", {})
    officer_observation = doc_summary.get("officer_observation", "")

    # Run analysis
    result = service.analyze_images(
        image_paths=image_paths,
        officer_observation=officer_observation,
        violation_code=violation.get("code", ""),
        violation_description=violation.get("description", ""),
        vehicle_info=vehicle,
        location_info=location,
        lang=lang,
        max_images=max_images
    )

    # Format for UI
    return service.format_for_ui(result, lang)
