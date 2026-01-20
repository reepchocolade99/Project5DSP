"""
Claude Vision Service for MLLM-based parking violation image analysis.

This service:
1. Accepts images + police observation context
2. Calls Claude Vision API with structured prompts
3. Returns semantic metadata aligned with parking enforcement domain
This service implements the Legal Reasoning Architecture v2.0:
- Layer 1: Document Parser (handled by server.py)
- Layer 2: Objective Image Analysis (MLLM observes, does NOT interpret legally)
- Layer 3: Rule Engine (deterministic legal matching)
- Layer 4: Officer Validation & Citation Generation

Version: 2.0 with backward compatibility
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

# Import Legal Reasoning v2 modules
try:
    from legal import (
        evaluate_legal_compliance,
        evaluate_with_auto_detection,
        generate_legal_statement,
        generate_full_legal_output,
        determine_action,
        get_supporting_articles,
        format_evidence_checklist,
        calculate_overall_confidence,
        get_confidence_label,
        format_action_for_ui,
        LEGAL_DECISION_TREES
    )
    from prompts import (
        get_layer2_prompt,
        build_layer2_message,
        build_layer4_prompt,
        parse_layer4_response,
        merge_verification_with_evaluation
    )
    LEGAL_V2_AVAILABLE = True
except ImportError as e:
    LEGAL_V2_AVAILABLE = False
    logging.warning(f"Legal Reasoning v2 modules not available: {e}")

logger = logging.getLogger(__name__)

# Feature flag for new Legal Pipeline v2
# Set to True to use the new 4-layer architecture
# Set to False to use the original prompt (backward compatibility)
USE_LEGAL_PIPELINE_V2 = True


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

    # =========================================================================
    # LEGAL REASONING ARCHITECTURE v2.0 - New Methods
    # =========================================================================

    def analyze_images_v2(
        self,
        image_paths: List[str],
        officer_observation: str,
        violation_code: str,
        vehicle_info: Dict[str, str],
        location_info: Dict[str, str],
        lang: str = 'nl',
        max_images: int = 10
    ) -> Dict[str, Any]:
        """
        Layer 2: Objective Image Analysis using the new hallucination-free prompt.

        CRITICAL: This method only OBSERVES, it does NOT INTERPRET legally.
        Legal interpretation is handled by the Rule Engine (Layer 3).

        Args:
            image_paths: List of paths to evidence images
            officer_observation: "Redenen van wetenschap" text from PDF
            violation_code: e.g., "E9", "E6", "G7"
            vehicle_info: Dict with kenteken, merk, model, kleur
            location_info: Dict with straat, buurt, plaats
            lang: Language code for output
            max_images: Maximum images to analyze

        Returns:
            Layer 2 structured observation results (no legal conclusions)
        """
        if not self.client:
            return {
                "success": False,
                "layer2_output": None,
                "error": "Claude Vision service not available - no API key configured"
            }

        if not LEGAL_V2_AVAILABLE:
            return {
                "success": False,
                "layer2_output": None,
                "error": "Legal Reasoning v2 modules not available"
            }

        # Select best images
        selected_images = self._select_best_images(image_paths, max_images)
        logger.info(f"[Layer 2] Selected {len(selected_images)} images for objective analysis")

        # Build message content with images
        content = []

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
            except Exception as e:
                logger.warning(f"Could not encode image {img_path}: {e}")
                continue

        if not content:
            return {
                "success": False,
                "layer2_output": None,
                "error": "No images could be encoded for analysis"
            }

        # Build Layer 2 objective analysis prompt
        document_context = {
            "violation_code": violation_code,
            "license_plate": vehicle_info.get("kenteken"),
            "location": f"{location_info.get('straat', '')}, {location_info.get('buurt', '')}",
        }
        prompt = build_layer2_message(lang, document_context)
        content.append({"type": "text", "text": prompt})

        # Call Claude Vision API
        try:
            logger.info("[Layer 2] Calling Claude Vision API for objective analysis...")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": content}]
            )

            response_text = response.content[0].text
            logger.debug(f"[Layer 2] Raw response: {response_text[:500]}...")

            # Parse JSON response
            json_str = response_text
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0]
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0]

            layer2_output = json.loads(json_str.strip())

            # Add metadata
            layer2_output["_metadata"] = {
                "layer": 2,
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "images_analyzed": len(selected_images),
                "pipeline_version": "2.0"
            }

            # Log key windshield items for debugging recommendation logic
            windshield = layer2_output.get("windshield_items", {})
            logger.info(f"[Layer 2] MLLM windshield_items: permit={windshield.get('permit')}, "
                       f"disability_card={windshield.get('disability_card')}, "
                       f"parking_disc={windshield.get('parking_disc')}")
            logger.info("[Layer 2] Objective analysis completed successfully")

            return {
                "success": True,
                "layer2_output": layer2_output,
                "error": None
            }

        except json.JSONDecodeError as e:
            logger.error(f"[Layer 2] Failed to parse response as JSON: {e}")
            return {
                "success": False,
                "layer2_output": None,
                "error": f"Failed to parse Layer 2 response: {e}",
                "raw_response": response_text if 'response_text' in locals() else None
            }
        except Exception as e:
            error_type = type(e).__name__
            logger.error(f"[Layer 2] {error_type}: {e}")
            return {
                "success": False,
                "layer2_output": None,
                "error": f"{error_type}: {str(e)}"
            }

    def run_full_legal_pipeline(
        self,
        image_paths: List[str],
        doc_summary: Dict[str, Any],
        lang: str = 'nl',
        max_images: int = 10
    ) -> Dict[str, Any]:
        """
        Run the complete 4-layer Legal Reasoning Pipeline v2.0.

        Layers:
        1. Document Parser (already done by server.py, data in doc_summary)
        2. Objective Image Analysis (MLLM observes only)
        3. Rule Engine (deterministic legal matching)
        4. Officer Validation & Citation Generation

        Args:
            image_paths: List of image file paths
            doc_summary: Document summary from PDF extraction
            lang: Language code
            max_images: Maximum images to analyze

        Returns:
            Complete legal assessment with article references
        """
        if not LEGAL_V2_AVAILABLE:
            logger.warning("Legal v2 not available, falling back to v1 pipeline")
            return self._fallback_to_v1(image_paths, doc_summary, lang, max_images)

        # Extract data from doc_summary
        violation = doc_summary.get("violation", {})
        vehicle = doc_summary.get("vehicle", {})
        location = doc_summary.get("location", {})
        officer_observation = doc_summary.get("officer_observation", "")
        violation_code = violation.get("code", "")

        logger.info(f"[Pipeline v2] Starting 4-layer analysis for violation: {violation_code}")

        # =====================================================================
        # LAYER 2: Objective Image Analysis
        # =====================================================================
        layer2_result = self.analyze_images_v2(
            image_paths=image_paths,
            officer_observation=officer_observation,
            violation_code=violation_code,
            vehicle_info=vehicle,
            location_info=location,
            lang=lang,
            max_images=max_images
        )

        if not layer2_result.get("success"):
            return self._format_pipeline_error(layer2_result, "Layer 2")

        layer2_output = layer2_result["layer2_output"]
        logger.info("[Layer 2] Complete - Objective observations recorded")

        # =====================================================================
        # LAYER 3: Rule Engine - Deterministic Legal Matching
        # =====================================================================
        logger.info("[Layer 3] Running Rule Engine evaluation...")

        # Auto-detect violation from sign if not provided
        if violation_code:
            rule_engine_result = evaluate_legal_compliance(layer2_output, violation_code)
        else:
            rule_engine_result = evaluate_with_auto_detection(layer2_output)

        logger.info(f"[Layer 3] Complete - Verification score: {rule_engine_result.get('verification_score', 0)}")

        # =====================================================================
        # LAYER 4: Officer Validation (optional MLLM call for complex cases)
        # =====================================================================
        # For now, we'll do a simplified verification without an additional API call
        # The full Layer 4 MLLM call can be enabled for complex cases
        verification_result = self._simple_verification(
            layer2_output, rule_engine_result, officer_observation
        )

        logger.info(f"[Layer 4] Complete - Observation match: {verification_result.get('observation_match_score', 0)}")

        # =====================================================================
        # COMBINE RESULTS & DETERMINE ACTION
        # =====================================================================
        merged_result = {
            **rule_engine_result,
            "observation_supported": verification_result.get("observation_supported", False),
            "observation_match_score": verification_result.get("observation_match_score", 0.5),
            "matching_elements": verification_result.get("matching_elements", []),
            "discrepancies": verification_result.get("discrepancies", []),
            "missing_from_image": verification_result.get("missing_from_image", []),
            "overall_confidence": calculate_overall_confidence(
                object_detection=self._get_avg_confidence(layer2_output),
                text_recognition=self._get_plate_confidence(layer2_output),
                legal_reasoning=rule_engine_result.get("verification_score", 0),
                observation_match=verification_result.get("observation_match_score", 0.5)
            )
        }

        # Determine recommended action
        action_result = determine_action(merged_result)

        # Generate legal statement
        statement_context = self._build_statement_context(layer2_output, doc_summary)
        legal_statement = generate_full_legal_output(
            violation_code=rule_engine_result.get("violation_code", violation_code),
            mllm_output=layer2_output,
            officer_observation=officer_observation,
            language=lang
        )

        # Get supporting articles
        articles = get_supporting_articles(rule_engine_result.get("violation_code", violation_code))

        # Format evidence checklist for UI
        evidence_checklist = format_evidence_checklist(rule_engine_result, lang)

        # Build final output
        final_output = {
            "success": True,
            "pipeline_version": "2.0",

            # Layer 2 output (observations)
            "layer2_observations": layer2_output,

            # Layer 3 output (legal evaluation)
            "legal_assessment": {
                "violation_code": rule_engine_result.get("violation_code"),
                "violation_name": rule_engine_result.get("violation_name"),
                "violation_name_en": rule_engine_result.get("violation_name_en"),
                "feit_code": rule_engine_result.get("legal_references", {}).get("feit_code"),
                "all_checks_passed": rule_engine_result.get("all_checks_passed"),
                "verification_score": rule_engine_result.get("verification_score"),
                "checks": rule_engine_result.get("checks", []),
                "legal_references": rule_engine_result.get("legal_references", {}),
                "supporting_articles": articles
            },

            # Layer 4 output (verification)
            "officer_verification": verification_result,

            # Combined results
            "recommendation": action_result,
            "overall_confidence": merged_result["overall_confidence"],

            # Generated content
            "legal_statement": legal_statement,
            "evidence_checklist": evidence_checklist,

            # Metadata
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "pipeline_version": "2.0",
                "images_analyzed": layer2_output.get("_metadata", {}).get("images_analyzed", 0),
                "language": lang
            }
        }

        logger.info(f"[Pipeline v2] Complete - Action: {action_result.get('action')}, Confidence: {merged_result['overall_confidence']}")

        return final_output

    def _simple_verification(
        self,
        layer2_output: Dict,
        rule_engine_result: Dict,
        officer_observation: str
    ) -> Dict[str, Any]:
        """
        Simple verification of Layer 2 output against officer observation.
        This is a deterministic check without additional MLLM calls.
        """
        matching_elements = []
        discrepancies = []
        missing_from_image = []

        # Check sign detection matches
        sign_code = layer2_output.get("traffic_sign", {}).get("sign_code", "none")
        if sign_code and sign_code != "none":
            matching_elements.append({
                "element": f"Traffic sign {sign_code} detected",
                "source": "image"
            })

        # Check permit/card status
        windshield = layer2_output.get("windshield_items", {})
        if windshield.get("permit") == "no":
            matching_elements.append({
                "element": "No permit visible in windshield",
                "source": "image"
            })
        elif windshield.get("permit") == "not_visible":
            missing_from_image.append("Permit visibility could not be confirmed")

        if windshield.get("disability_card") == "no":
            matching_elements.append({
                "element": "No disability card visible",
                "source": "image"
            })

        # Check driver presence
        env = layer2_output.get("environment", {})
        if not env.get("driver_present", True):
            matching_elements.append({
                "element": "No driver present",
                "source": "image"
            })

        # Calculate match score
        total_checks = len(rule_engine_result.get("checks", []))
        passed_checks = len(rule_engine_result.get("passed_checks", []))

        if total_checks > 0:
            base_score = passed_checks / total_checks
        else:
            base_score = 0.5

        # Adjust based on matches
        match_bonus = min(0.2, len(matching_elements) * 0.05)
        discrepancy_penalty = len(discrepancies) * 0.1

        match_score = min(1.0, max(0.0, base_score + match_bonus - discrepancy_penalty))

        return {
            "observation_supported": match_score >= 0.7,
            "observation_match_score": round(match_score, 2),
            "matching_elements": matching_elements,
            "discrepancies": discrepancies,
            "missing_from_image": missing_from_image
        }

    def _get_avg_confidence(self, layer2_output: Dict) -> float:
        """Get average detection confidence from Layer 2 output."""
        confidences = []

        if layer2_output.get("traffic_sign", {}).get("confidence"):
            confidences.append(layer2_output["traffic_sign"]["confidence"])

        plate_conf = layer2_output.get("vehicle", {}).get("license_plate", {}).get("confidence")
        if plate_conf:
            confidences.append(plate_conf)

        return sum(confidences) / len(confidences) if confidences else 0.5

    def _get_plate_confidence(self, layer2_output: Dict) -> float:
        """Get license plate recognition confidence."""
        return layer2_output.get("vehicle", {}).get("license_plate", {}).get("confidence", 0.5)

    def _build_statement_context(self, layer2_output: Dict, doc_summary: Dict) -> Dict:
        """Build context for legal statement generation."""
        return {
            "observation_time": "5",  # Default
            "sub_sign_text": layer2_output.get("traffic_sign", {}).get("sub_sign_text"),
            "vehicle_plate": doc_summary.get("vehicle", {}).get("kenteken", "[KENTEKEN]"),
        }

    def _format_pipeline_error(self, result: Dict, layer: str) -> Dict:
        """Format an error response from the pipeline."""
        return {
            "success": False,
            "pipeline_version": "2.0",
            "error": f"{layer} failed: {result.get('error', 'Unknown error')}",
            "failed_layer": layer,
            "layer2_observations": None,
            "legal_assessment": None,
            "recommendation": {
                "action": "manual_review",
                "reason": f"Pipeline error at {layer}",
                "requires_manual_review": True
            }
        }

    def _fallback_to_v1(
        self,
        image_paths: List[str],
        doc_summary: Dict[str, Any],
        lang: str,
        max_images: int
    ) -> Dict[str, Any]:
        """Fallback to v1 pipeline if v2 is not available."""
        violation = doc_summary.get("violation", {})
        vehicle = doc_summary.get("vehicle", {})
        location = doc_summary.get("location", {})
        officer_observation = doc_summary.get("officer_observation", "")

        result = self.analyze_images(
            image_paths=image_paths,
            officer_observation=officer_observation,
            violation_code=violation.get("code", ""),
            violation_description=violation.get("description", ""),
            vehicle_info=vehicle,
            location_info=location,
            lang=lang,
            max_images=max_images
        )

        return self.format_for_ui(result, lang)

    def format_v2_for_ui(
        self,
        pipeline_result: Dict[str, Any],
        lang: str = 'nl'
    ) -> Dict[str, Any]:
        """
        Format the v2 pipeline result for the existing UI template.
        Provides backward compatibility with the result.html template.
        """
        if not pipeline_result.get("success"):
            return {
                "mllm_analysis": None,
                "mllm_error": pipeline_result.get("error"),
                "detected_items_ui": {"items": []},
                "confidence_scores": {
                    "object_detection": 0.0,
                    "text_recognition": 0.0,
                    "legal_reasoning": 0.0
                },
                "pipeline_version": "2.0"
            }

        layer2 = pipeline_result.get("layer2_observations", {})
        legal = pipeline_result.get("legal_assessment", {})
        verification = pipeline_result.get("officer_verification", {})
        recommendation = pipeline_result.get("recommendation", {})

        # Calculate confidence scores for UI
        confidence_scores = {
            "object_detection": self._get_avg_confidence(layer2),
            "text_recognition": self._get_plate_confidence(layer2),
            "legal_reasoning": legal.get("verification_score", 0)
        }

        # Format detected items for UI sidebar
        detected_items = self._format_detected_items_v2(layer2, lang)

        # Convert Layer 2 observations to legacy format for UI compatibility
        legacy_analysis = self._convert_to_legacy_format(layer2, legal, verification)

        return {
            "mllm_analysis": legacy_analysis,
            "mllm_error": None,
            "detected_items_ui": {"items": detected_items},
            "confidence_scores": confidence_scores,
            "image_description": layer2.get("observation_summary", ""),
            "environmental_context": {
                "lighting": layer2.get("environment", {}).get("lighting", ""),
                "image_quality": layer2.get("environment", {}).get("image_quality", "")
            },
            "verification": {
                "observation_supported": verification.get("observation_supported", False),
                "matching_elements": [m.get("element", "") for m in verification.get("matching_elements", [])],
                "discrepancies": [d.get("item", "") for d in verification.get("discrepancies", [])],
                "missing_evidence": verification.get("missing_from_image", []),
                "overall_confidence": pipeline_result.get("overall_confidence", 0)
            },
            "summary": layer2.get("observation_summary", ""),
            "metadata": pipeline_result.get("metadata", {}),

            # New v2 specific fields
            "pipeline_version": "2.0",
            "legal_assessment": legal,
            "legal_statement": pipeline_result.get("legal_statement", {}),
            "evidence_checklist": pipeline_result.get("evidence_checklist", []),
            "recommendation": recommendation,
            "recommendation_ui": format_action_for_ui(recommendation, lang) if LEGAL_V2_AVAILABLE else None
        }

    def _format_detected_items_v2(self, layer2: Dict, lang: str) -> List[Dict]:
        """Format detected items from Layer 2 output for UI."""
        items = []

        labels = {
            'nl': {
                'vehicle': 'Voertuig',
                'license_plate': 'Kenteken',
                'traffic_sign': 'Verkeersbord',
                'parking_permit': 'Parkeervergunning',
                'disability_card': 'Gehandicaptenparkeerkaart'
            },
            'en': {
                'vehicle': 'Vehicle',
                'license_plate': 'License Plate',
                'traffic_sign': 'Traffic Sign',
                'parking_permit': 'Parking Permit',
                'disability_card': 'Disability Card'
            }
        }.get(lang, {})

        # Vehicle
        vehicle = layer2.get("vehicle", {})
        if vehicle.get("type"):
            items.append({
                "label": labels.get('vehicle', 'Vehicle'),
                "label_key": "vehicle",
                "confidence": int(layer2.get("traffic_sign", {}).get("confidence", 0.8) * 100),
                "detected": True,
                "details": f"{vehicle.get('color', '')} {vehicle.get('type', '')}".strip(),
                "crop_available": False
            })

        # License plate
        plate = vehicle.get("license_plate", {})
        if plate.get("visibility") in ["full", "partial"]:
            items.append({
                "label": labels.get('license_plate', 'License Plate'),
                "label_key": "license_plate",
                "confidence": int(plate.get("confidence", 0) * 100),
                "detected": True,
                "extracted_text": plate.get("value"),
                "crop_available": False
            })
        else:
            items.append({
                "label": labels.get('license_plate', 'License Plate'),
                "label_key": "license_plate",
                "confidence": 0,
                "detected": False,
                "crop_available": False
            })

        # Traffic sign
        sign = layer2.get("traffic_sign", {})
        if sign.get("detected"):
            sign_label = f"{labels.get('traffic_sign', 'Traffic Sign')} {sign.get('sign_code', '')}"
            items.append({
                "label": sign_label,
                "label_key": "traffic_sign",
                "confidence": int(sign.get("confidence", 0) * 100),
                "detected": True,
                "sign_code": sign.get("sign_code"),
                "sub_sign": sign.get("sub_sign_text"),
                "crop_available": False
            })
        else:
            items.append({
                "label": labels.get('traffic_sign', 'Traffic Sign'),
                "label_key": "traffic_sign",
                "confidence": 0,
                "detected": False,
                "crop_available": False
            })

        # Windshield items
        windshield = layer2.get("windshield_items", {})
        if windshield.get("permit") == "yes":
            items.append({
                "label": labels.get('parking_permit', 'Parking Permit'),
                "label_key": "permit",
                "confidence": 80,
                "detected": True,
                "crop_available": False
            })

        if windshield.get("disability_card") == "yes":
            items.append({
                "label": labels.get('disability_card', 'Disability Card'),
                "label_key": "disability_card",
                "confidence": 80,
                "detected": True,
                "crop_available": False
            })

        return items

    def _convert_to_legacy_format(
        self,
        layer2: Dict,
        legal: Dict,
        verification: Dict
    ) -> Dict:
        """Convert v2 output to legacy analysis format for UI compatibility."""
        vehicle = layer2.get("vehicle", {})
        sign = layer2.get("traffic_sign", {})
        plate = vehicle.get("license_plate", {})
        windshield = layer2.get("windshield_items", {})

        return {
            "image_description": layer2.get("observation_summary", ""),
            "object_detection": {
                "vehicle": {
                    "detected": bool(vehicle.get("type")),
                    "confidence": 0.85,
                    "details": f"{vehicle.get('color', '')} {vehicle.get('type', '')}".strip()
                },
                "license_plate": {
                    "detected": plate.get("visibility") in ["full", "partial"],
                    "confidence": plate.get("confidence", 0),
                    "value": plate.get("value")
                },
                "traffic_sign": {
                    "detected": sign.get("detected", False),
                    "confidence": sign.get("confidence", 0),
                    "sign_type": sign.get("sign_code")
                },
                "parking_permit": {
                    "detected": windshield.get("permit") == "yes",
                    "confidence": 0.8 if windshield.get("permit") == "yes" else 0,
                    "details": ""
                },
                "driver_present": {
                    "detected": layer2.get("environment", {}).get("driver_present", False),
                    "confidence": 0.9
                }
            },
            "environmental_context": {
                "lighting": layer2.get("environment", {}).get("lighting", ""),
                "image_quality": layer2.get("environment", {}).get("image_quality", "")
            },
            "verification": {
                "observation_supported": verification.get("observation_supported", False),
                "matching_elements": [m.get("element", "") for m in verification.get("matching_elements", [])],
                "discrepancies": [d.get("item", "") for d in verification.get("discrepancies", [])],
                "missing_evidence": verification.get("missing_from_image", []),
                "overall_confidence": verification.get("observation_match_score", 0.5)
            },
            "summary": layer2.get("observation_summary", ""),
            "_metadata": layer2.get("_metadata", {}),

            # Add legal assessment data
            "_legal_assessment": legal
        }


# Convenience function for direct use
def analyze_parking_evidence(
    image_paths: List[str],
    doc_summary: Dict[str, Any],
    lang: str = 'nl',
    max_images: int = 10,
    use_v2_pipeline: bool = None
) -> Dict[str, Any]:
    """
    Convenience function to analyze parking violation evidence.

    Args:
        image_paths: List of image file paths
        doc_summary: Document summary from PDF extraction
        lang: Language code
        max_images: Maximum images to analyze
        use_v2_pipeline: Whether to use Legal Reasoning v2 pipeline.
                        If None, uses USE_LEGAL_PIPELINE_V2 flag.

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

    # Determine which pipeline to use
    should_use_v2 = use_v2_pipeline if use_v2_pipeline is not None else USE_LEGAL_PIPELINE_V2

    # Use v2 pipeline if enabled and available
    if should_use_v2 and LEGAL_V2_AVAILABLE:
        logger.info("Using Legal Reasoning Pipeline v2.0")
        pipeline_result = service.run_full_legal_pipeline(
            image_paths=image_paths,
            doc_summary=doc_summary,
            lang=lang,
            max_images=max_images
        )
        return service.format_v2_for_ui(pipeline_result, lang)

    # Fallback to v1 pipeline
    logger.info("Using Legacy Pipeline v1.0")
    violation = doc_summary.get("violation", {})
    vehicle = doc_summary.get("vehicle", {})
    location = doc_summary.get("location", {})
    officer_observation = doc_summary.get("officer_observation", "")

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

    return service.format_for_ui(result, lang)


def analyze_parking_evidence_v2(
    image_paths: List[str],
    doc_summary: Dict[str, Any],
    lang: str = 'nl',
    max_images: int = 10
) -> Dict[str, Any]:
    """
    Convenience function to analyze parking violation evidence using v2 pipeline.

    This function explicitly uses the Legal Reasoning Architecture v2.0
    with the 4-layer pipeline.

    Args:
        image_paths: List of image file paths
        doc_summary: Document summary from PDF extraction
        lang: Language code
        max_images: Maximum images to analyze

    Returns:
        Full pipeline result (not formatted for legacy UI)
    """
    if not ANTHROPIC_AVAILABLE:
        return {
            "success": False,
            "error": "MLLM service not available - anthropic package not installed"
        }

    if not LEGAL_V2_AVAILABLE:
        return {
            "success": False,
            "error": "Legal Reasoning v2 modules not available"
        }

    service = ClaudeVisionService()

    if not service.is_available():
        return {
            "success": False,
            "error": "MLLM service not available - ANTHROPIC_API_KEY not configured"
        }

    return service.run_full_legal_pipeline(
        image_paths=image_paths,
        doc_summary=doc_summary,
        lang=lang,
        max_images=max_images
    )
