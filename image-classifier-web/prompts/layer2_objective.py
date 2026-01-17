"""
Layer 2: Objective Image Analysis Prompt
Based on Legal Reasoning Architecture v2.0

CRITICAL DESIGN PRINCIPLE:
At this layer, the MLLM only OBSERVES, it does NOT INTERPRET legally.
Legal interpretation is handled by the deterministic Rule Engine (Layer 3).

This ensures hallucination-free outputs where the model reports only
what it can see, without making legal conclusions.

Version: 2.0
"""


LAYER2_PROMPT_EN = """You are an objective image analyst. Describe ONLY what you see.
DO NOT provide legal interpretations or conclusions.

Analyze the image(s) and report:

1. VEHICLE OBSERVATION:
   - Vehicle type (passenger car, van, truck, motorcycle, etc.)
   - Color
   - License plate (if visible, transcribe exactly)
   - Position on the road/parking area

2. TRAFFIC SIGN OBSERVATION:
   - Which sign(s) visible? (E6, E7, E9, G7, etc.)
   - Is a sub-sign present? If so, what does it say?
   - Approximate distance from vehicle to sign

3. WINDSHIELD OBSERVATION:
   - Is anything visible behind the windshield?
   - Disability parking card: YES / NO / NOT VISIBLE
   - Permit/exemption document: YES / NO / NOT VISIBLE
   - Parking disc: YES / NO / NOT VISIBLE

4. ENVIRONMENT OBSERVATION:
   - Driver present in or near vehicle: YES / NO
   - Loading/unloading activity visible: YES / NO
   - Other people near vehicle: YES / NO
   - Is vehicle connected to charging point (if applicable): YES / NO / NOT APPLICABLE

5. IMAGE QUALITY:
   - Lighting conditions: day / night / artificial
   - Image quality: good / moderate / poor
   - License plate readability: full / partial / none

OUTPUT FORMAT: JSON (valid JSON only, no markdown code blocks)
{
  "vehicle": {
    "type": "string (passenger car, van, truck, motorcycle, other)",
    "color": "string",
    "license_plate": {
      "value": "string or null if not readable",
      "visibility": "full | partial | none",
      "confidence": 0.0-1.0
    },
    "position": "string describing position"
  },
  "traffic_sign": {
    "detected": true | false,
    "sign_code": "E6 | E7 | E9 | G7 | E4 | E4_ELECTRIC | other | none",
    "sub_sign_text": "string or null",
    "distance_estimate": "string (e.g., 'approximately 2 meters')",
    "confidence": 0.0-1.0
  },
  "windshield_items": {
    "disability_card": "yes | no | not_visible",
    "permit": "yes | no | not_visible",
    "parking_disc": "yes | no | not_visible",
    "other_items": "string or null"
  },
  "environment": {
    "driver_present": true | false,
    "loading_activity": true | false,
    "other_people_present": true | false,
    "charging_connected": true | false | null,
    "lighting": "day | night | artificial",
    "image_quality": "good | moderate | poor"
  },
  "observation_summary": "string (2-3 sentences, factual only, NO legal conclusions)"
}

IMPORTANT RULES:
- Report ONLY what you actually see in the images
- DO NOT make assumptions about what is not visible
- DO NOT provide any legal interpretations or conclusions
- DO NOT use words like "violation", "illegal", "permitted", "prohibited"
- Only describe observable facts
- Confidence scores should reflect visual certainty, not legal certainty

CRITICAL: Understanding YES / NO / NOT_VISIBLE:
- "yes" = Item IS clearly visible and present
- "no" = You CAN see the windshield/dashboard area, but NO document is there (confirmed absence)
- "not_visible" = ONLY use when windshield is COMPLETELY obscured (100% reflection, fully fogged, or not in frame)

DEFAULT BEHAVIOR: If you can see ANY part of the windshield interior or dashboard, use "no" for missing items.
Reserve "not_visible" ONLY for cases where the windshield is completely unobservable.

For windshield items (permit, disability card, parking disc):
- You can see the windshield/dashboard area, no document present → "no"
- Windshield is 100% obscured by reflection/fog, cannot see inside at all → "not_visible"
- Document is clearly visible → "yes"

IMPORTANT: Most parking enforcement photos show the windshield clearly enough to confirm absence of documents.
If you can see the dashboard or interior through the windshield, even partially, you can confirm "no" for missing items.
Do NOT use "not_visible" just because there is some glare - only use it when you truly cannot see the interior AT ALL.
"""


LAYER2_PROMPT_NL = """Je bent een objectieve beeldanalist. Beschrijf ALLEEN wat je ziet.
Geef GEEN juridische interpretaties of conclusies.

Analyseer de afbeelding(en) en rapporteer:

1. VOERTUIGOBSERVATIE:
   - Voertuigtype (personenauto, bestelbus, vrachtwagen, motorfiets, etc.)
   - Kleur
   - Kenteken (indien zichtbaar, exact overnemen)
   - Positie op de weg/parkeerplaats

2. VERKEERSBORD OBSERVATIE:
   - Welk(e) bord(en) zichtbaar? (E6, E7, E9, G7, etc.)
   - Is er een onderbord aanwezig? Zo ja, wat staat erop?
   - Geschatte afstand van voertuig tot bord

3. VOORRUIT OBSERVATIE:
   - Is er iets zichtbaar achter de voorruit?
   - Gehandicaptenparkeerkaart: JA / NEE / NIET ZICHTBAAR
   - Vergunning/ontheffing document: JA / NEE / NIET ZICHTBAAR
   - Parkeerschijf: JA / NEE / NIET ZICHTBAAR

4. OMGEVINGSOBSERVATIE:
   - Bestuurder aanwezig in of nabij voertuig: JA / NEE
   - Laad/los activiteit zichtbaar: JA / NEE
   - Andere personen nabij voertuig: JA / NEE
   - Voertuig aangesloten op oplaadpunt (indien van toepassing): JA / NEE / NVT

5. BEELDKWALITEIT:
   - Lichtomstandigheden: dag / nacht / kunstlicht
   - Beeldkwaliteit: goed / matig / slecht
   - Leesbaarheid kenteken: volledig / gedeeltelijk / geen

OUTPUT FORMAAT: JSON (alleen geldige JSON, geen markdown codeblokken)
{
  "vehicle": {
    "type": "string (personenauto, bestelbus, vrachtwagen, motorfiets, anders)",
    "color": "string",
    "license_plate": {
      "value": "string of null indien niet leesbaar",
      "visibility": "full | partial | none",
      "confidence": 0.0-1.0
    },
    "position": "string met beschrijving van positie"
  },
  "traffic_sign": {
    "detected": true | false,
    "sign_code": "E6 | E7 | E9 | G7 | E4 | E4_ELECTRIC | other | none",
    "sub_sign_text": "string of null",
    "distance_estimate": "string (bijv. 'ongeveer 2 meter')",
    "confidence": 0.0-1.0
  },
  "windshield_items": {
    "disability_card": "yes | no | not_visible",
    "permit": "yes | no | not_visible",
    "parking_disc": "yes | no | not_visible",
    "other_items": "string of null"
  },
  "environment": {
    "driver_present": true | false,
    "loading_activity": true | false,
    "other_people_present": true | false,
    "charging_connected": true | false | null,
    "lighting": "day | night | artificial",
    "image_quality": "good | moderate | poor"
  },
  "observation_summary": "string (2-3 zinnen, alleen feiten, GEEN juridische conclusies)"
}

BELANGRIJKE REGELS:
- Rapporteer ALLEEN wat je daadwerkelijk ziet in de afbeeldingen
- Maak GEEN aannames over wat niet zichtbaar is
- Geef GEEN juridische interpretaties of conclusies
- Gebruik NIET woorden als "overtreding", "illegaal", "toegestaan", "verboden"
- Beschrijf alleen waarneembare feiten
- Betrouwbaarheidsscores moeten visuele zekerheid weergeven, niet juridische zekerheid

KRITIEK: Begrip van YES / NO / NOT_VISIBLE:
- "yes" = Item IS duidelijk zichtbaar en aanwezig
- "no" = Je KUNT de voorruit/dashboard zien, maar er is GEEN document aanwezig (bevestigde afwezigheid)
- "not_visible" = ALLEEN gebruiken wanneer voorruit VOLLEDIG geblokkeerd is (100% reflectie, volledig beslagen, of niet in beeld)

STANDAARD GEDRAG: Als je ENIG deel van het voorruit-interieur of dashboard kunt zien, gebruik "no" voor ontbrekende items.
Reserveer "not_visible" ALLEEN voor gevallen waarin de voorruit volledig onzichtbaar is.

Voor voorruit items (vergunning, gehandicaptenkaart, parkeerschijf):
- Je kunt de voorruit/dashboard zien, geen document aanwezig → "no"
- Voorruit is 100% geblokkeerd door reflectie/condens, interieur totaal niet zichtbaar → "not_visible"
- Document is duidelijk zichtbaar → "yes"

BELANGRIJK: De meeste parkeerhandhavingsfoto's tonen de voorruit duidelijk genoeg om afwezigheid van documenten te bevestigen.
Als je het dashboard of interieur door de voorruit kunt zien, zelfs gedeeltelijk, kun je "no" bevestigen voor ontbrekende items.
Gebruik "not_visible" NIET alleen omdat er wat schittering is - gebruik het alleen wanneer je het interieur HELEMAAL niet kunt zien.
"""


def get_layer2_prompt(language: str = "en") -> str:
    """
    Get the Layer 2 objective analysis prompt in the specified language.

    Args:
        language: "en" for English, "nl" for Dutch

    Returns:
        The prompt string
    """
    if language.lower() == "nl":
        return LAYER2_PROMPT_NL
    return LAYER2_PROMPT_EN


def build_layer2_message(
    language: str = "en",
    document_context: dict = None
) -> str:
    """
    Build the complete Layer 2 prompt with optional document context.

    Args:
        language: "en" or "nl"
        document_context: Optional dictionary with extracted document info
            (violation_code, vehicle_info, location, etc.)

    Returns:
        Complete prompt string
    """
    base_prompt = get_layer2_prompt(language)

    if not document_context:
        return base_prompt

    # Add document context as reference (but still instruct to observe only)
    context_section = "\n\nDOCUMENT CONTEXT (for reference only - still report only what you observe):\n"

    if document_context.get("violation_code"):
        context_section += f"- Reported violation type: {document_context['violation_code']}\n"

    if document_context.get("license_plate"):
        context_section += f"- Reported license plate: {document_context['license_plate']}\n"

    if document_context.get("location"):
        context_section += f"- Location: {document_context['location']}\n"

    if document_context.get("datetime"):
        context_section += f"- Date/Time: {document_context['datetime']}\n"

    context_section += "\nRemember: Report what you SEE, not what the document says."

    return base_prompt + context_section


# Expected output schema for validation
LAYER2_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["vehicle", "traffic_sign", "windshield_items", "environment", "observation_summary"],
    "properties": {
        "vehicle": {
            "type": "object",
            "required": ["type", "color", "license_plate", "position"],
            "properties": {
                "type": {"type": "string"},
                "color": {"type": "string"},
                "license_plate": {
                    "type": "object",
                    "required": ["value", "visibility", "confidence"],
                    "properties": {
                        "value": {"type": ["string", "null"]},
                        "visibility": {"type": "string", "enum": ["full", "partial", "none"]},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                },
                "position": {"type": "string"}
            }
        },
        "traffic_sign": {
            "type": "object",
            "required": ["detected", "sign_code", "confidence"],
            "properties": {
                "detected": {"type": "boolean"},
                "sign_code": {"type": "string"},
                "sub_sign_text": {"type": ["string", "null"]},
                "distance_estimate": {"type": ["string", "null"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1}
            }
        },
        "windshield_items": {
            "type": "object",
            "required": ["disability_card", "permit", "parking_disc"],
            "properties": {
                "disability_card": {"type": "string", "enum": ["yes", "no", "not_visible"]},
                "permit": {"type": "string", "enum": ["yes", "no", "not_visible"]},
                "parking_disc": {"type": "string", "enum": ["yes", "no", "not_visible"]},
                "other_items": {"type": ["string", "null"]}
            }
        },
        "environment": {
            "type": "object",
            "required": ["driver_present", "loading_activity", "lighting", "image_quality"],
            "properties": {
                "driver_present": {"type": "boolean"},
                "loading_activity": {"type": "boolean"},
                "other_people_present": {"type": "boolean"},
                "charging_connected": {"type": ["boolean", "null"]},
                "lighting": {"type": "string", "enum": ["day", "night", "artificial"]},
                "image_quality": {"type": "string", "enum": ["good", "moderate", "poor"]}
            }
        },
        "observation_summary": {"type": "string"}
    }
}
