# SAM3 + OpenAI Vision API: Paralel Confidence Score Mimarisi

## 🎯 Amaç

OpenAI Vision API'nin semantik analizini SAM3'ün objektif segmentasyonuyla birleştirerek **doğrulanabilir, hallüsinasyona karşı dirençli** bir confidence score sistemi oluşturmak.

---

## 1. SAM3 Temel Özellikleri (HuggingFace'ten)

SAM3'ün Project 5 için kritik özellikleri:

| Özellik | Açıklama | Kullanım |
|---------|----------|----------|
| **Text Prompting** | "vehicle", "traffic sign" gibi text ile segment | Ana detection method |
| **Confidence Scores** | Her mask için `scores` döndürür | Objektif confidence |
| **Bounding Boxes** | Her detection için `boxes` (xyxy format) | Lokasyon doğrulama |
| **Multiple Instances** | Tüm eşleşen objeleri bulur | Birden fazla araç/işaret |
| **0.9B Parameters** | Hafif model | Hızlı inference |

---

## 2. Paralel Mimari Diyagramı

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PDF Upload                                       │
│                    (Proces_verbaal_wegslepen.pdf)                        │
└─────────────────────────────┬───────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      PDF Processing Layer                                │
│  ┌─────────────────────┐    ┌──────────────────────────────────────┐   │
│  │  Text Extraction    │    │  Image Extraction                     │   │
│  │  (PyMuPDF/fitz)     │    │  (extract_embedded_images)            │   │
│  │                     │    │                                        │   │
│  │  • volgnummer       │    │  • _p03_img01.jpg                     │   │
│  │  • kenteken         │    │  • _p03_img02.jpg                     │   │
│  │  • violation_code   │    │  • _p04_img01.jpg ...                 │   │
│  │  • reden_verwijdering│   │                                        │   │
│  └──────────┬──────────┘    └─────────────────┬────────────────────┘   │
└─────────────┼─────────────────────────────────┼─────────────────────────┘
              │                                 │
              │              ┌──────────────────┴──────────────────┐
              │              │                                     │
              │              ▼                                     ▼
              │   ┌─────────────────────────┐       ┌─────────────────────────┐
              │   │      SAM3 PIPELINE      │       │   OPENAI VISION PIPELINE │
              │   │    (Parallel Thread 1)  │       │    (Parallel Thread 2)   │
              │   │                         │       │                          │
              │   │  Text Prompts:          │       │  GPT-4o Vision:          │
              │   │  • "vehicle"            │       │  • Image description     │
              │   │  • "license plate"      │       │  • Context understanding │
              │   │  • "traffic sign"       │       │  • Legal interpretation  │
              │   │  • "parking permit"     │       │  • Officer observation   │
              │   │  • "person"             │       │    verification          │
              │   │  • "charging cable"     │       │                          │
              │   │                         │       │                          │
              │   │  Returns per prompt:    │       │  Returns:                │
              │   │  • masks[]              │       │  • image_description     │
              │   │  • boxes[] (xyxy)       │       │  • object_detection{}    │
              │   │  • scores[] (0.0-1.0)   │       │  • environmental_context │
              │   │                         │       │  • verification{}        │
              │   └───────────┬─────────────┘       └────────────┬────────────┘
              │               │                                  │
              │               │         AGGREGATION              │
              │               └──────────────┬───────────────────┘
              │                              │
              │                              ▼
              │               ┌─────────────────────────────────┐
              │               │      CONFIDENCE MERGER          │
              │               │                                 │
              │               │  Cross-Validation Logic:        │
              │               │  ┌───────────────────────────┐  │
              │               │  │ SAM3    │ OpenAI │ Result │  │
              │               │  ├─────────┼────────┼────────┤  │
              │               │  │ High    │ High   │ ✅ 100%│  │
              │               │  │ High    │ Low    │ ⚠️ SAM3│  │
              │               │  │ Low     │ High   │ 🚨 HAL │  │
              │               │  │ Low     │ Low    │ ❌ None│  │
              │               │  └───────────────────────────┘  │
              │               └──────────────┬──────────────────┘
              │                              │
              └──────────────────────────────┼─────────────────────────────
                                             │
                                             ▼
              ┌─────────────────────────────────────────────────────────────┐
              │                    REPORT GENERATION                        │
              │                                                             │
              │  ┌─────────────────────────────────────────────────────┐   │
              │  │ 1. Image Description          [AI GENERATED]        │   │
              │  │ 2. Object Detection Analysis  [SAM3 + OPENAI]       │   │
              │  │ 3. Timestamp & Location       [FROM DOCUMENT]       │   │
              │  │ 4. Environmental Context      [AI GENERATED]        │   │
              │  │ 5. Legal Reasoning            [LEGAL TEMPLATE]      │   │
              │  │ 6. Supporting Evidence         [FROM DOCUMENT]       │   │
              │  │ 7. Confidence Summary          [MERGED SCORES]       │   │
              │  └─────────────────────────────────────────────────────┘   │
              │                                                             │
              │  Confidence Scores (Sidebar):                               │
              │  ┌─────────────────────────────────────────────────────┐   │
              │  │ Object Detection:  ████████░░ 85% (SAM3: 88%)       │   │
              │  │ Text Recognition:  ████████░░ 82% (SAM3: 79%)       │   │
              │  │ Legal Reasoning:   ████████░░ 86% (Merged)          │   │
              │  └─────────────────────────────────────────────────────┘   │
              └─────────────────────────────────────────────────────────────┘
```

---

## 3. SAM3 Detection Service

```python
# sam3_detection_service.py

from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

@dataclass
class SAM3Detection:
    """Single detection result from SAM3."""
    prompt: str
    detected: bool
    confidence: float  # max score from SAM3
    num_instances: int  # how many instances found
    boxes: List[Tuple[int, int, int, int]]  # xyxy format
    masks: Optional[np.ndarray] = None
    all_scores: List[float] = field(default_factory=list)

@dataclass
class SAM3ImageResult:
    """All detections for a single image."""
    image_path: str
    detections: Dict[str, SAM3Detection]
    processing_time_ms: float

class SAM3DetectionService:
    """
    SAM3-based object detection for parking violation evidence.
    Uses text prompting to detect specific objects.
    """
    
    # Parking violation specific prompts (from DSP_data.docx + Layer 2 prompt patterns)
    DETECTION_PROMPTS = {
        # ═══════════════════════════════════════════════════════════════
        # VEHICLE TYPES
        # ═══════════════════════════════════════════════════════════════
        'vehicle': [
            "car",
            "vehicle", 
            "automobile",
            "personenauto",
            "passenger car"
        ],
        'van': [
            "van",
            "delivery van",
            "bestelwagen",
            "cargo van"
        ],
        'truck': [
            "truck",
            "lorry",
            "vrachtwagen"
        ],
        'motorcycle': [
            "motorcycle",
            "motorbike",
            "scooter",
            "motorfiets",
            "bromfiets"
        ],
        
        # ═══════════════════════════════════════════════════════════════
        # LICENSE PLATE
        # ═══════════════════════════════════════════════════════════════
        'license_plate': [
            "license plate",
            "kenteken",
            "number plate",
            "Dutch yellow license plate"
        ],
        
        # ═══════════════════════════════════════════════════════════════
        # TRAFFIC SIGNS (E6, E7, E9, G7, E4, E4_ELECTRIC)
        # ═══════════════════════════════════════════════════════════════
        'traffic_sign_e6': [
            # E6 = Disabled parking - MUST have WHITE WHEELCHAIR SYMBOL (♿)
            "handicapped parking sign",
            "wheelchair parking sign",
            "blue square sign with wheelchair symbol",
            "disabled parking sign with wheelchair icon",
            "gehandicaptenparkeerplaats bord"
        ],
        'traffic_sign_e7': [
            # E7 = Loading/unloading zone - truck/cargo symbol
            "loading zone sign",
            "loading unloading sign",
            "truck loading sign",
            "blue sign with truck symbol",
            "laden lossen bord"
        ],
        'traffic_sign_e9': [
            # E9 = Permit holders - Blue "P" with "vergunninghouders" text, NO wheelchair!
            "permit parking sign",
            "permit holders parking sign",
            "blue P sign with vergunninghouders text",
            "parking sign with permit text",
            "vergunninghouders bord"
        ],
        'traffic_sign_g7': [
            # G7 = Pedestrian area - pedestrian symbol
            "pedestrian zone sign",
            "pedestrian area sign",
            "walking person sign",
            "blue sign with pedestrian symbol",
            "voetgangersgebied bord"
        ],
        'traffic_sign_e4': [
            # E4 = Parking facility - Blue "P" sign WITHOUT permit text
            "parking sign",
            "blue P parking sign",
            "parking facility sign",
            "parkeerplaats bord"
        ],
        'traffic_sign_e4_electric': [
            # E4 Electric = EV charging parking
            "electric vehicle charging sign",
            "EV parking sign",
            "charging station sign",
            "electric car parking sign",
            "oplaadpunt bord",
            "elektrisch laden bord"
        ],
        
        # ═══════════════════════════════════════════════════════════════
        # SUB-SIGN (ONDERBORD)
        # ═══════════════════════════════════════════════════════════════
        'sub_sign': [
            "sub sign",
            "secondary sign",
            "sign below main sign",
            "onderbord",
            "additional sign plate"
        ],
        
        # ═══════════════════════════════════════════════════════════════
        # WINDSHIELD ITEMS
        # ═══════════════════════════════════════════════════════════════
        'parking_permit': [
            "parking permit card",
            "permit card behind windshield",
            "dashboard permit card",
            "parking permit on dashboard",
            "ontheffing",
            "parkeervergunning"
        ],
        'disability_card': [
            # Blue EU disability parking card
            "blue disability card",
            "wheelchair parking card",
            "disabled parking permit",
            "blue badge",
            "gehandicaptenparkeerkaart",
            "invalidenparkeerkaart"
        ],
        'parking_disc': [
            # Parkeerschijf - blue/white parking time disc
            "parking disc",
            "parking clock",
            "blue parking disc",
            "parkeerschijf",
            "parking time disc"
        ],
        
        # ═══════════════════════════════════════════════════════════════
        # CHARGING INFRASTRUCTURE
        # ═══════════════════════════════════════════════════════════════
        'charging_cable': [
            "electric charging cable",
            "EV charging cord",
            "charging cable connected to car",
            "oplaadkabel"
        ],
        'charging_station': [
            "electric vehicle charging station",
            "EV charger",
            "charging point",
            "oplaadpaal",
            "laadpaal"
        ],
        'charging_connected': [
            # Verify if vehicle is actually connected to charger
            "car connected to charging station",
            "charging cable plugged into vehicle",
            "vehicle charging"
        ],
        
        # ═══════════════════════════════════════════════════════════════
        # PEOPLE & ACTIVITY
        # ═══════════════════════════════════════════════════════════════
        'person': [
            "person",
            "human",
            "driver",
            "bestuurder",
            "person near vehicle"
        ],
        'driver_in_vehicle': [
            "driver inside car",
            "person sitting in vehicle",
            "driver behind steering wheel"
        ],
        'loading_activity': [
            # Laden/lossen activity detection
            "person loading cargo",
            "person unloading",
            "loading activity",
            "boxes being loaded",
            "cargo loading",
            "laden lossen activiteit"
        ],
        
        # ═══════════════════════════════════════════════════════════════
        # WINDSHIELD VISIBILITY CHECK
        # ═══════════════════════════════════════════════════════════════
        'windshield': [
            "car windshield",
            "front windshield",
            "voorruit",
            "windscreen"
        ]
    }
    
    def __init__(self, device: str = None, dtype: torch.dtype = torch.float16):
        """
        Initialize SAM3 model.
        
        Args:
            device: "cuda" or "cpu" (auto-detected if None)
            dtype: torch.float16 for speed, torch.float32 for accuracy
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        
        print(f"Loading SAM3 model on {self.device}...")
        self.model = Sam3Model.from_pretrained("facebook/sam3").to(self.device)
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        print("SAM3 model loaded successfully.")
    
    def detect_all(self, image_path: Path) -> SAM3ImageResult:
        """
        Run all parking-related detections on a single image.
        
        Returns comprehensive detection results with confidence scores.
        """
        import time
        start_time = time.time()
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size  # (width, height)
        
        detections = {}
        
        for category, prompts in self.DETECTION_PROMPTS.items():
            # Try each prompt variant, take best result
            best_detection = None
            best_confidence = 0.0
            
            for prompt in prompts:
                detection = self._detect_with_prompt(image, prompt, original_size)
                
                if detection.confidence > best_confidence:
                    best_confidence = detection.confidence
                    best_detection = detection
                    best_detection.prompt = f"{category} ('{prompt}')"
            
            if best_detection:
                detections[category] = best_detection
            else:
                # No detection for this category
                detections[category] = SAM3Detection(
                    prompt=category,
                    detected=False,
                    confidence=0.0,
                    num_instances=0,
                    boxes=[],
                    all_scores=[]
                )
        
        processing_time = (time.time() - start_time) * 1000
        
        return SAM3ImageResult(
            image_path=str(image_path),
            detections=detections,
            processing_time_ms=processing_time
        )
    
    def _detect_with_prompt(
        self, 
        image: Image.Image, 
        prompt: str,
        original_size: Tuple[int, int]
    ) -> SAM3Detection:
        """
        Run SAM3 detection with a specific text prompt.
        """
        # Prepare inputs
        inputs = self.processor(
            images=image, 
            text=prompt, 
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=[list(original_size)[::-1]]  # [height, width]
        )[0]
        
        # Extract detection info
        masks = results.get('masks', torch.tensor([]))
        boxes = results.get('boxes', torch.tensor([]))
        scores = results.get('scores', torch.tensor([]))
        
        if len(scores) == 0:
            return SAM3Detection(
                prompt=prompt,
                detected=False,
                confidence=0.0,
                num_instances=0,
                boxes=[],
                all_scores=[]
            )
        
        # Convert to Python types
        scores_list = scores.cpu().numpy().tolist()
        boxes_list = [tuple(box.cpu().numpy().astype(int).tolist()) for box in boxes]
        
        return SAM3Detection(
            prompt=prompt,
            detected=True,
            confidence=max(scores_list),  # Best confidence
            num_instances=len(scores_list),
            boxes=boxes_list,
            masks=masks.cpu().numpy() if len(masks) > 0 else None,
            all_scores=scores_list
        )
    
    def detect_specific(
        self, 
        image_path: Path, 
        categories: List[str]
    ) -> SAM3ImageResult:
        """
        Run detection only for specific categories.
        
        Args:
            image_path: Path to image
            categories: List like ['vehicle', 'license_plate', 'traffic_sign_e9']
        """
        import time
        start_time = time.time()
        
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        
        detections = {}
        
        for category in categories:
            if category not in self.DETECTION_PROMPTS:
                continue
                
            prompts = self.DETECTION_PROMPTS[category]
            best_detection = None
            best_confidence = 0.0
            
            for prompt in prompts:
                detection = self._detect_with_prompt(image, prompt, original_size)
                if detection.confidence > best_confidence:
                    best_confidence = detection.confidence
                    best_detection = detection
                    best_detection.prompt = f"{category} ('{prompt}')"
            
            detections[category] = best_detection or SAM3Detection(
                prompt=category,
                detected=False,
                confidence=0.0,
                num_instances=0,
                boxes=[],
                all_scores=[]
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        return SAM3ImageResult(
            image_path=str(image_path),
            detections=detections,
            processing_time_ms=processing_time
        )
    
    def aggregate_results(
        self, 
        results_list: List[SAM3ImageResult]
    ) -> Dict[str, float]:
        """
        Aggregate SAM3 detections across multiple images.
        Takes maximum confidence per category.
        
        Returns dict like {'vehicle': 0.92, 'license_plate': 0.85, ...}
        """
        aggregated = {}
        
        for result in results_list:
            for category, detection in result.detections.items():
                if category not in aggregated:
                    aggregated[category] = detection.confidence
                else:
                    aggregated[category] = max(
                        aggregated[category],
                        detection.confidence
                    )
        
        return aggregated
```

---

## 4. OpenAI Vision Service

```python
# openai_vision_service.py

import openai
import base64
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class OpenAIVisionResult:
    """Result from OpenAI Vision analysis."""
    image_description: str
    object_detection: Dict[str, Dict]
    environmental_context: Dict
    verification: Dict
    raw_response: str

class OpenAIVisionService:
    """
    OpenAI GPT-4o Vision API for semantic image analysis.
    Provides contextual understanding and legal interpretation.
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        """
        import os
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY")
        )
        self.model = "gpt-4o"  # GPT-4o with vision
    
    def analyze_images(
        self,
        image_paths: List[Path],
        officer_observation: str,
        violation_code: str,
        vehicle_info: Dict = None,
        location_info: Dict = None,
        lang: str = "en"
    ) -> OpenAIVisionResult:
        """
        Analyze parking violation images with GPT-4o Vision.
        
        Args:
            image_paths: List of image file paths
            officer_observation: "Redenen van wetenschap" text
            violation_code: E6, E7, E9, G7, etc.
            vehicle_info: Dict with kenteken, merk, model, kleur
            location_info: Dict with straat, stadsdeel, buurt
            lang: "en" or "nl" for output language
        
        Returns:
            OpenAIVisionResult with structured analysis
        """
        # Prepare images as base64
        image_contents = []
        for img_path in image_paths[:10]:  # Max 10 images
            with open(img_path, "rb") as f:
                base64_image = base64.b64encode(f.read()).decode('utf-8')
                image_contents.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                })
        
        # Build prompt
        prompt = self._build_analysis_prompt(
            officer_observation=officer_observation,
            violation_code=violation_code,
            vehicle_info=vehicle_info,
            location_info=location_info,
            lang=lang
        )
        
        # Add text prompt to content
        content = image_contents + [{"type": "text", "text": prompt}]
        
        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": self._get_system_prompt(lang)
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=2000,
            temperature=0.1  # Low temperature for consistency
        )
        
        # Parse response
        raw_response = response.choices[0].message.content
        parsed = self._parse_response(raw_response)
        
        return OpenAIVisionResult(
            image_description=parsed.get('image_description', ''),
            object_detection=parsed.get('object_detection', {}),
            environmental_context=parsed.get('environmental_context', {}),
            verification=parsed.get('verification', {}),
            raw_response=raw_response
        )
    
    def _get_system_prompt(self, lang: str) -> str:
        """System prompt for parking violation analysis."""
        return """You are an expert image analyst for parking enforcement evidence.
Your task is to objectively describe what is visible in parking violation photos.

CRITICAL RULES:
1. ONLY describe what you can actually see - never assume or infer
2. Do NOT make legal judgments - only report visual facts
3. Report confidence levels honestly
4. If something is unclear or not visible, say so explicitly
5. Focus on: vehicle, traffic signs, permits/cards, location features, people

You must respond in valid JSON format only."""
    
    def _build_analysis_prompt(
        self,
        officer_observation: str,
        violation_code: str,
        vehicle_info: Dict,
        location_info: Dict,
        lang: str
    ) -> str:
        """Build the analysis prompt with context."""
        
        vehicle_str = ""
        if vehicle_info:
            vehicle_str = f"""
Expected Vehicle Information (from document):
- License plate: {vehicle_info.get('kenteken', 'unknown')}
- Brand: {vehicle_info.get('merk', 'unknown')}
- Model: {vehicle_info.get('model', 'unknown')}
- Color: {vehicle_info.get('kleur', 'unknown')}
"""
        
        location_str = ""
        if location_info:
            location_str = f"""
Location Information:
- Street: {location_info.get('straat', 'unknown')}
- District: {location_info.get('stadsdeel', 'unknown')}
- Neighborhood: {location_info.get('buurt', 'unknown')}
"""
        
        return f"""Analyze these parking enforcement evidence photos.

Violation Code: {violation_code}

Officer Observation (Redenen van wetenschap):
"{officer_observation}"
{vehicle_str}
{location_str}

Respond with this exact JSON structure:
{{
    "image_description": "Detailed description of what is visible in the images",
    "object_detection": {{
        "vehicle": {{
            "detected": true/false,
            "confidence": 0.0-1.0,
            "details": "description of vehicle if visible"
        }},
        "license_plate": {{
            "detected": true/false,
            "confidence": 0.0-1.0,
            "value": "plate text if readable",
            "readable": true/false
        }},
        "traffic_sign": {{
            "detected": true/false,
            "confidence": 0.0-1.0,
            "sign_type": "E6/E7/E9/G7 or unknown",
            "details": "sign description"
        }},
        "parking_permit": {{
            "detected": true/false,
            "confidence": 0.0-1.0,
            "location": "behind windshield / dashboard / not visible"
        }},
        "driver_present": {{
            "detected": true/false,
            "confidence": 0.0-1.0
        }},
        "charging_cable": {{
            "detected": true/false,
            "confidence": 0.0-1.0,
            "connected": true/false/unknown
        }}
    }},
    "environmental_context": {{
        "time_of_day": "daytime/nighttime/twilight",
        "lighting": "natural/artificial/mixed",
        "weather_visible": "clear/rain/unknown",
        "street_features": "description of visible street elements"
    }},
    "verification": {{
        "observation_supported": true/false,
        "matching_elements": ["list of elements that match officer observation"],
        "discrepancies": ["list of any differences from officer observation"],
        "missing_evidence": ["list of things mentioned but not visible"],
        "overall_confidence": 0.0-1.0
    }}
}}

Output language: {"Dutch" if lang == "nl" else "English"}
Respond ONLY with valid JSON, no other text."""
    
    def _parse_response(self, raw_response: str) -> Dict:
        """Parse JSON response from OpenAI."""
        try:
            # Clean up response (remove markdown code blocks if present)
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            return json.loads(cleaned.strip())
        except json.JSONDecodeError as e:
            print(f"Failed to parse OpenAI response: {e}")
            return {
                "image_description": raw_response,
                "object_detection": {},
                "environmental_context": {},
                "verification": {"overall_confidence": 0.5}
            }
    
    def extract_confidences(self, result: OpenAIVisionResult) -> Dict[str, float]:
        """
        Extract confidence scores in format compatible with SAM3.
        
        Returns dict like {'vehicle': 0.85, 'license_plate': 0.78, ...}
        """
        confidences = {}
        
        obj_det = result.object_detection
        
        # Map OpenAI detection categories to SAM3 categories
        mapping = {
            'vehicle': 'vehicle',
            'license_plate': 'license_plate',
            'traffic_sign': 'traffic_sign',  # Will need to map to specific E6/E7/E9/G7
            'parking_permit': 'parking_permit',
            'driver_present': 'person',
            'charging_cable': 'charging_cable'
        }
        
        for openai_key, sam3_key in mapping.items():
            if openai_key in obj_det:
                det = obj_det[openai_key]
                if det.get('detected', False):
                    confidences[sam3_key] = det.get('confidence', 0.5)
                else:
                    confidences[sam3_key] = 0.0
        
        # Handle traffic sign specifically
        if 'traffic_sign' in obj_det:
            sign_det = obj_det['traffic_sign']
            if sign_det.get('detected', False):
                sign_type = sign_det.get('sign_type', '').upper()
                if 'E6' in sign_type:
                    confidences['traffic_sign_e6'] = sign_det.get('confidence', 0.5)
                elif 'E7' in sign_type:
                    confidences['traffic_sign_e7'] = sign_det.get('confidence', 0.5)
                elif 'E9' in sign_type:
                    confidences['traffic_sign_e9'] = sign_det.get('confidence', 0.5)
                elif 'G7' in sign_type:
                    confidences['traffic_sign_g7'] = sign_det.get('confidence', 0.5)
        
        return confidences
```

---

## 5. Confidence Merger (Cross-Validation)

```python
# confidence_merger.py

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum

class ConfidenceSource(Enum):
    SAM3 = "sam3"          # Objective segmentation
    OPENAI = "openai"      # Semantic analysis
    MERGED = "merged"      # Combined score
    HALLUCINATION = "hallucination_warning"

@dataclass
class MergedConfidence:
    """Result of merging SAM3 and OpenAI confidence scores."""
    category: str
    sam3_confidence: float
    openai_confidence: float
    merged_confidence: float
    agreement_score: float
    source_used: ConfidenceSource
    is_hallucination_risk: bool
    reasoning: str

class ConfidenceMerger:
    """
    Merge SAM3 and OpenAI confidence scores with cross-validation.
    
    KEY INSIGHT:
    - SAM3 = Objektif görsel kanıt (segmentasyon mask + score)
    - OpenAI = Semantik anlama (ne gördüğünü yorumlama)
    
    Cross-validation ile hallüsinasyon tespiti:
    - OpenAI bir şey gördüğünü söylüyor ama SAM3 bulamıyorsa = 🚨 Hallucination risk
    """
    
    # Category-specific weights
    # Higher SAM3 weight = more objective visual evidence matters
    # Higher OpenAI weight = more context/interpretation matters
    CATEGORY_WEIGHTS = {
        'vehicle':          {'sam3': 0.70, 'openai': 0.30},  # Objektif
        'license_plate':    {'sam3': 0.60, 'openai': 0.40},  # SAM3 + OCR context
        'traffic_sign_e6':  {'sam3': 0.65, 'openai': 0.35},  # Visual + interpretation
        'traffic_sign_e7':  {'sam3': 0.65, 'openai': 0.35},
        'traffic_sign_e9':  {'sam3': 0.65, 'openai': 0.35},
        'traffic_sign_g7':  {'sam3': 0.65, 'openai': 0.35},
        'parking_permit':   {'sam3': 0.50, 'openai': 0.50},  # Context important
        'disability_card':  {'sam3': 0.55, 'openai': 0.45},
        'charging_cable':   {'sam3': 0.70, 'openai': 0.30},  # Objektif
        'person':           {'sam3': 0.75, 'openai': 0.25},  # Very objektif
    }
    
    # Thresholds
    HIGH_CONFIDENCE = 0.70
    LOW_CONFIDENCE = 0.35
    HALLUCINATION_THRESHOLD = 0.40  # If diff > this, flag as risk
    
    def merge(
        self,
        sam3_confidences: Dict[str, float],
        openai_confidences: Dict[str, float]
    ) -> Dict[str, MergedConfidence]:
        """
        Merge confidence scores from both sources.
        
        Args:
            sam3_confidences: {'vehicle': 0.92, 'license_plate': 0.85, ...}
            openai_confidences: {'vehicle': 0.88, 'license_plate': 0.82, ...}
        
        Returns:
            Dict of MergedConfidence objects per category
        """
        all_categories = set(sam3_confidences.keys()) | set(openai_confidences.keys())
        merged_results = {}
        
        for category in all_categories:
            sam3_conf = sam3_confidences.get(category, 0.0)
            openai_conf = openai_confidences.get(category, 0.0)
            
            merged_results[category] = self._merge_single(
                category, sam3_conf, openai_conf
            )
        
        return merged_results
    
    def _merge_single(
        self,
        category: str,
        sam3_conf: float,
        openai_conf: float
    ) -> MergedConfidence:
        """Apply merge strategy for a single category."""
        
        weights = self.CATEGORY_WEIGHTS.get(
            category,
            {'sam3': 0.60, 'openai': 0.40}  # Default
        )
        
        # Calculate agreement (1.0 = perfect agreement)
        agreement = 1.0 - abs(sam3_conf - openai_conf)
        
        # Determine scenario and apply strategy
        
        # SCENARIO 1: Both HIGH → Strong agreement, use weighted average
        if sam3_conf >= self.HIGH_CONFIDENCE and openai_conf >= self.HIGH_CONFIDENCE:
            merged = sam3_conf * weights['sam3'] + openai_conf * weights['openai']
            return MergedConfidence(
                category=category,
                sam3_confidence=sam3_conf,
                openai_confidence=openai_conf,
                merged_confidence=merged,
                agreement_score=agreement,
                source_used=ConfidenceSource.MERGED,
                is_hallucination_risk=False,
                reasoning="✅ Both SAM3 and OpenAI confirm detection with high confidence"
            )
        
        # SCENARIO 2: SAM3 HIGH, OpenAI LOW → Trust SAM3 (OpenAI missed it)
        elif sam3_conf >= self.HIGH_CONFIDENCE and openai_conf < self.LOW_CONFIDENCE:
            return MergedConfidence(
                category=category,
                sam3_confidence=sam3_conf,
                openai_confidence=openai_conf,
                merged_confidence=sam3_conf * 0.90,  # Slight penalty
                agreement_score=agreement,
                source_used=ConfidenceSource.SAM3,
                is_hallucination_risk=False,
                reasoning="⚠️ SAM3 detects object that OpenAI did not mention - trusting visual segmentation"
            )
        
        # SCENARIO 3: SAM3 LOW, OpenAI HIGH → 🚨 HALLUCINATION RISK
        elif sam3_conf < self.LOW_CONFIDENCE and openai_conf >= self.HIGH_CONFIDENCE:
            return MergedConfidence(
                category=category,
                sam3_confidence=sam3_conf,
                openai_confidence=openai_conf,
                merged_confidence=openai_conf * 0.40,  # Heavy penalty
                agreement_score=agreement,
                source_used=ConfidenceSource.HALLUCINATION,
                is_hallucination_risk=True,
                reasoning="🚨 HALLUCINATION RISK: OpenAI claims detection but SAM3 cannot segment - manual verification required"
            )
        
        # SCENARIO 4: Both LOW → Consistent non-detection
        elif sam3_conf < self.LOW_CONFIDENCE and openai_conf < self.LOW_CONFIDENCE:
            merged = max(sam3_conf, openai_conf)
            return MergedConfidence(
                category=category,
                sam3_confidence=sam3_conf,
                openai_confidence=openai_conf,
                merged_confidence=merged,
                agreement_score=agreement,
                source_used=ConfidenceSource.MERGED,
                is_hallucination_risk=False,
                reasoning="❌ Neither source confidently detects this object"
            )
        
        # SCENARIO 5: Medium confidence range → Weighted average
        else:
            merged = sam3_conf * weights['sam3'] + openai_conf * weights['openai']
            is_risk = (openai_conf - sam3_conf) > self.HALLUCINATION_THRESHOLD
            
            return MergedConfidence(
                category=category,
                sam3_confidence=sam3_conf,
                openai_confidence=openai_conf,
                merged_confidence=merged,
                agreement_score=agreement,
                source_used=ConfidenceSource.MERGED,
                is_hallucination_risk=is_risk,
                reasoning=f"Mixed confidence - weighted merge applied (SAM3: {weights['sam3']:.0%}, OpenAI: {weights['openai']:.0%})"
            )
    
    def calculate_final_scores(
        self,
        merged_results: Dict[str, MergedConfidence]
    ) -> Dict[str, float]:
        """
        Calculate final UI confidence scores.
        
        Returns:
            {
                'object_detection': 0.85,
                'text_recognition': 0.82,
                'legal_reasoning': 0.86
            }
        """
        # Object Detection Score
        obj_categories = [
            'vehicle', 'traffic_sign_e6', 'traffic_sign_e7', 
            'traffic_sign_e9', 'traffic_sign_g7', 'parking_permit',
            'disability_card', 'charging_cable', 'person'
        ]
        obj_scores = [
            merged_results[cat].merged_confidence
            for cat in obj_categories
            if cat in merged_results and merged_results[cat].merged_confidence > 0
        ]
        object_detection = sum(obj_scores) / len(obj_scores) if obj_scores else 0.0
        
        # Text Recognition Score (primarily license plate)
        text_score = merged_results.get('license_plate', MergedConfidence(
            category='license_plate', sam3_confidence=0, openai_confidence=0,
            merged_confidence=0, agreement_score=0, source_used=ConfidenceSource.MERGED,
            is_hallucination_risk=False, reasoning=''
        )).merged_confidence
        
        # Legal Reasoning Score (based on evidence completeness)
        legal_score = self._calculate_legal_score(merged_results)
        
        return {
            'object_detection': round(object_detection * 100),  # As percentage
            'text_recognition': round(text_score * 100),
            'legal_reasoning': round(legal_score * 100)
        }
    
    def _calculate_legal_score(
        self,
        merged_results: Dict[str, MergedConfidence]
    ) -> float:
        """
        Calculate legal reasoning score based on evidence completeness.
        
        For a valid parking violation case, we need:
        1. Vehicle confirmed (required)
        2. Location/sign confirmed (required)
        3. Absence of permit confirmed (if applicable)
        4. No driver present (supports violation)
        """
        vehicle_conf = merged_results.get('vehicle', MergedConfidence(
            category='vehicle', sam3_confidence=0, openai_confidence=0,
            merged_confidence=0, agreement_score=0, source_used=ConfidenceSource.MERGED,
            is_hallucination_risk=False, reasoning=''
        )).merged_confidence
        
        # Get best traffic sign score
        sign_scores = [
            merged_results.get(f'traffic_sign_{code}', MergedConfidence(
                category=f'traffic_sign_{code}', sam3_confidence=0, openai_confidence=0,
                merged_confidence=0, agreement_score=0, source_used=ConfidenceSource.MERGED,
                is_hallucination_risk=False, reasoning=''
            )).merged_confidence
            for code in ['e6', 'e7', 'e9', 'g7']
        ]
        sign_conf = max(sign_scores) if sign_scores else 0.0
        
        # Permit not found is good for violation (inverted logic)
        permit_conf = merged_results.get('parking_permit', MergedConfidence(
            category='parking_permit', sam3_confidence=0, openai_confidence=0,
            merged_confidence=0, agreement_score=0, source_used=ConfidenceSource.MERGED,
            is_hallucination_risk=False, reasoning=''
        )).merged_confidence
        no_permit_score = 1.0 - permit_conf  # No permit found = good for case
        
        # Calculate weighted score
        legal_score = (
            vehicle_conf * 0.35 +      # Vehicle must be present
            sign_conf * 0.35 +          # Sign must be visible
            no_permit_score * 0.20 +    # No permit supports violation
            0.10                        # Base score for documentation
        )
        
        # Cap at 1.0
        return min(legal_score, 1.0)
    
    def get_hallucination_warnings(
        self,
        merged_results: Dict[str, MergedConfidence]
    ) -> List[str]:
        """Get list of hallucination warnings for UI display."""
        warnings = []
        for category, result in merged_results.items():
            if result.is_hallucination_risk:
                warnings.append(
                    f"⚠️ {category}: {result.reasoning}"
                )
        return warnings
```

---

## 6. Entegre Server Endpoint

```python
# server.py - Updated /predict endpoint

from flask import Flask, request, render_template
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

app = Flask(__name__)

# Initialize services
sam3_service = None  # Lazy load
openai_service = None  # Lazy load

def get_sam3_service():
    global sam3_service
    if sam3_service is None:
        sam3_service = SAM3DetectionService()
    return sam3_service

def get_openai_service():
    global openai_service
    if openai_service is None:
        openai_service = OpenAIVisionService()
    return openai_service

@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint with parallel SAM3 + OpenAI processing.
    """
    start_time = time.time()
    
    # Get form data
    file = request.files.get('file')
    lang = request.form.get('language', 'en')
    model_type = request.form.get('model_type', 'mllm')  # 'mllm', 'openai', 'mock'
    
    # Save and process file
    # ... (existing file handling code) ...
    
    # Extract PDF data
    pdf_text = extract_pdf_text(pdf_path)
    doc_summary = extract_structured_fields(pdf_text)
    images = extract_embedded_images(pdf_path, output_dir)
    
    # Get context for analysis
    officer_observation = doc_summary.get('reden_verwijdering', '')
    violation_code = doc_summary.get('code', 'unknown')
    vehicle_info = {
        'kenteken': doc_summary.get('kenteken'),
        'merk': doc_summary.get('merk'),
        'model': doc_summary.get('model'),
        'kleur': doc_summary.get('kleur')
    }
    location_info = {
        'straat': doc_summary.get('straat'),
        'stadsdeel': doc_summary.get('stadsdeel'),
        'buurt': doc_summary.get('buurt')
    }
    
    # ═══════════════════════════════════════════════════════════════
    # PARALLEL PROCESSING: SAM3 + OpenAI
    # ═══════════════════════════════════════════════════════════════
    
    sam3_results = {}
    openai_results = {}
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {}
        
        # Submit SAM3 task
        futures['sam3'] = executor.submit(
            run_sam3_analysis,
            images
        )
        
        # Submit OpenAI task (if enabled)
        if model_type in ['mllm', 'openai']:
            futures['openai'] = executor.submit(
                run_openai_analysis,
                images,
                officer_observation,
                violation_code,
                vehicle_info,
                location_info,
                lang
            )
        
        # Collect results
        for name, future in futures.items():
            try:
                if name == 'sam3':
                    sam3_results = future.result(timeout=60)
                elif name == 'openai':
                    openai_results = future.result(timeout=30)
            except Exception as e:
                print(f"Error in {name} pipeline: {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # MERGE CONFIDENCE SCORES
    # ═══════════════════════════════════════════════════════════════
    
    merger = ConfidenceMerger()
    merged_confidences = merger.merge(sam3_results, openai_results)
    final_scores = merger.calculate_final_scores(merged_confidences)
    hallucination_warnings = merger.get_hallucination_warnings(merged_confidences)
    
    # ═══════════════════════════════════════════════════════════════
    # GENERATE REPORT
    # ═══════════════════════════════════════════════════════════════
    
    report_sections = generate_report_sections(
        doc_summary=doc_summary,
        openai_results=openai_results,
        merged_confidences=merged_confidences,
        lang=lang
    )
    
    # Format detected items for UI
    detected_items_ui = format_detected_items_for_ui(merged_confidences)
    
    processing_time = time.time() - start_time
    
    return render_template('result.html',
        filename=filename,
        extracted_images=images,
        images_metadata=get_images_metadata(images),
        doc_summary=doc_summary,
        report_sections=report_sections,
        confidence_scores=final_scores,
        sam3_results=sam3_results,
        detected_items_ui=detected_items_ui,
        hallucination_warnings=hallucination_warnings,
        model_type=model_type,
        lang=lang,
        processing_time=f"{processing_time:.2f}s"
    )


def run_sam3_analysis(images: List[Path]) -> Dict[str, float]:
    """Run SAM3 analysis on all images."""
    service = get_sam3_service()
    
    all_results = []
    for img_path in images:
        result = service.detect_all(img_path)
        all_results.append(result)
    
    # Aggregate across all images
    return service.aggregate_results(all_results)


def run_openai_analysis(
    images: List[Path],
    officer_observation: str,
    violation_code: str,
    vehicle_info: Dict,
    location_info: Dict,
    lang: str
) -> Dict[str, float]:
    """Run OpenAI Vision analysis."""
    service = get_openai_service()
    
    result = service.analyze_images(
        image_paths=images,
        officer_observation=officer_observation,
        violation_code=violation_code,
        vehicle_info=vehicle_info,
        location_info=location_info,
        lang=lang
    )
    
    return service.extract_confidences(result)


def format_detected_items_for_ui(
    merged: Dict[str, MergedConfidence]
) -> List[Dict]:
    """Format merged results for UI sidebar display."""
    
    display_names = {
        'vehicle': 'Vehicle',
        'license_plate': 'License Plate',
        'traffic_sign_e6': 'Sign E6 (Handicapped)',
        'traffic_sign_e7': 'Sign E7 (Loading)',
        'traffic_sign_e9': 'Sign E9 (Permit)',
        'traffic_sign_g7': 'Sign G7 (Pedestrian)',
        'parking_permit': 'Parking Permit',
        'disability_card': 'Disability Card',
        'charging_cable': 'Charging Cable',
        'person': 'Driver/Person'
    }
    
    items = []
    for category, data in merged.items():
        items.append({
            'label': display_names.get(category, category.replace('_', ' ').title()),
            'category': category,
            'detected': data.merged_confidence >= 0.5,
            'confidence': int(data.merged_confidence * 100),
            'sam3_confidence': int(data.sam3_confidence * 100),
            'openai_confidence': int(data.openai_confidence * 100),
            'agreement': int(data.agreement_score * 100),
            'source': data.source_used.value,
            'is_hallucination_risk': data.is_hallucination_risk,
            'reasoning': data.reasoning
        })
    
    # Sort by confidence descending
    items.sort(key=lambda x: x['confidence'], reverse=True)
    
    return items
```

---

## 7. UI Updates (result.html)

```html
<!-- Enhanced Detected Items with SAM3 + OpenAI dual bars -->
<div class="detected-items-panel">
    <h4>DETECTED ITEMS</h4>
    <p class="subtitle">SAM3 Segmentation + OpenAI Vision</p>
    
    {% for item in detected_items_ui %}
    <div class="detection-card {{ 'hallucination-risk' if item.is_hallucination_risk else '' }}">
        <div class="detection-header">
            <span class="label">{{ item.label }}</span>
            <span class="status-badge {{ 'detected' if item.detected else 'not-detected' }}">
                {{ 'Detected' if item.detected else 'Not Found' }}
            </span>
        </div>
        
        <div class="dual-confidence">
            <!-- SAM3 Bar -->
            <div class="conf-row">
                <span class="source-label" title="Objektif segmentasyon">SAM3</span>
                <div class="bar-track">
                    <div class="bar-fill sam3" style="width: {{ item.sam3_confidence }}%"></div>
                </div>
                <span class="percentage">{{ item.sam3_confidence }}%</span>
            </div>
            
            <!-- OpenAI Bar -->
            <div class="conf-row">
                <span class="source-label" title="Semantik analiz">OpenAI</span>
                <div class="bar-track">
                    <div class="bar-fill openai" style="width: {{ item.openai_confidence }}%"></div>
                </div>
                <span class="percentage">{{ item.openai_confidence }}%</span>
            </div>
            
            <!-- Merged/Final Bar -->
            <div class="conf-row final">
                <span class="source-label">FINAL</span>
                <div class="bar-track">
                    <div class="bar-fill {{ item.source }}" style="width: {{ item.confidence }}%"></div>
                </div>
                <span class="percentage {{ 'warning' if item.is_hallucination_risk else '' }}">
                    {{ item.confidence }}%
                    {% if item.is_hallucination_risk %}⚠️{% endif %}
                </span>
            </div>
        </div>
        
        {% if item.is_hallucination_risk %}
        <div class="hallucination-warning">
            🚨 {{ item.reasoning }}
        </div>
        {% endif %}
    </div>
    {% endfor %}
    
    <!-- Hallucination Warnings Summary -->
    {% if hallucination_warnings %}
    <div class="warnings-summary">
        <h5>⚠️ Verification Warnings</h5>
        <ul>
        {% for warning in hallucination_warnings %}
            <li>{{ warning }}</li>
        {% endfor %}
        </ul>
        <p class="action-required">Manual review recommended before case approval.</p>
    </div>
    {% endif %}
</div>

<style>
.detected-items-panel {
    background: #fff;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.detection-card {
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 12px;
}

.detection-card.hallucination-risk {
    border-color: #f59e0b;
    background: #fffbeb;
}

.detection-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.status-badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 12px;
    font-weight: 500;
}

.status-badge.detected {
    background: #d1fae5;
    color: #065f46;
}

.status-badge.not-detected {
    background: #fee2e2;
    color: #991b1b;
}

.dual-confidence {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.conf-row {
    display: flex;
    align-items: center;
    gap: 8px;
}

.conf-row.final {
    margin-top: 4px;
    padding-top: 6px;
    border-top: 1px dashed #e5e7eb;
}

.source-label {
    width: 50px;
    font-size: 10px;
    color: #6b7280;
    text-transform: uppercase;
}

.bar-track {
    flex: 1;
    height: 6px;
    background: #e5e7eb;
    border-radius: 3px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.3s ease;
}

.bar-fill.sam3 { background: #3b82f6; }       /* Blue */
.bar-fill.openai { background: #8b5cf6; }     /* Purple */
.bar-fill.merged { background: #10b981; }     /* Green */
.bar-fill.hallucination { background: #f59e0b; } /* Orange/Warning */

.percentage {
    width: 45px;
    text-align: right;
    font-size: 12px;
    font-weight: 500;
}

.percentage.warning {
    color: #d97706;
}

.hallucination-warning {
    margin-top: 8px;
    padding: 8px;
    background: #fef3c7;
    border-radius: 4px;
    font-size: 11px;
    color: #92400e;
}

.warnings-summary {
    margin-top: 16px;
    padding: 12px;
    background: #fef3c7;
    border: 1px solid #f59e0b;
    border-radius: 6px;
}

.warnings-summary h5 {
    margin: 0 0 8px 0;
    color: #92400e;
}

.warnings-summary ul {
    margin: 0;
    padding-left: 20px;
    font-size: 12px;
}

.action-required {
    margin: 8px 0 0 0;
    font-weight: 600;
    color: #b45309;
}
</style>
```

---

## 8. Akış Özeti

```
┌──────────────────────────────────────────────────────────────────────┐
│                        COMPLETE FLOW                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. USER UPLOAD                                                       │
│     └─→ Proces_verbaal_wegslepen.pdf                                 │
│                                                                       │
│  2. PDF PROCESSING                                                    │
│     ├─→ Text extraction (PyMuPDF)                                    │
│     │   • volgnummer, kenteken, code, reden_verwijdering             │
│     └─→ Image extraction                                             │
│         • 12 evidence photos                                         │
│                                                                       │
│  3. PARALLEL ANALYSIS (ThreadPoolExecutor)                           │
│     │                                                                │
│     ├─→ THREAD 1: SAM3 Detection                                     │
│     │   ┌─────────────────────────────────────────────┐              │
│     │   │ Text Prompts:                               │              │
│     │   │ • "vehicle" → masks, boxes, scores          │              │
│     │   │ • "license plate" → masks, boxes, scores    │              │
│     │   │ • "traffic sign E9" → masks, boxes, scores  │              │
│     │   │ • "parking permit" → masks, boxes, scores   │              │
│     │   │ • "person" → masks, boxes, scores           │              │
│     │   └─────────────────────────────────────────────┘              │
│     │   Output: {vehicle: 0.92, license_plate: 0.85, ...}           │
│     │                                                                │
│     └─→ THREAD 2: OpenAI GPT-4o Vision                              │
│         ┌─────────────────────────────────────────────┐              │
│         │ Input:                                      │              │
│         │ • 10 images (base64)                        │              │
│         │ • Officer observation text                  │              │
│         │ • Violation code (E9)                       │              │
│         │ • Vehicle info (kenteken, merk, kleur)      │              │
│         │                                             │              │
│         │ Output: JSON                                │              │
│         │ • image_description                         │              │
│         │ • object_detection{} with confidences       │              │
│         │ • environmental_context{}                   │              │
│         │ • verification{}                            │              │
│         └─────────────────────────────────────────────┘              │
│         Output: {vehicle: 0.88, license_plate: 0.82, ...}           │
│                                                                       │
│  4. CONFIDENCE MERGER (Cross-Validation)                             │
│     ┌─────────────────────────────────────────────────────────────┐  │
│     │ For each category:                                          │  │
│     │                                                             │  │
│     │ vehicle:       SAM3=0.92, OpenAI=0.88 → ✅ Merged=0.90     │  │
│     │ license_plate: SAM3=0.85, OpenAI=0.82 → ✅ Merged=0.84     │  │
│     │ traffic_sign:  SAM3=0.78, OpenAI=0.85 → ✅ Merged=0.81     │  │
│     │ parking_permit: SAM3=0.12, OpenAI=0.75 → 🚨 HALLUCINATION  │  │
│     │ person:        SAM3=0.05, OpenAI=0.10 → ❌ Not detected    │  │
│     └─────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  5. FINAL SCORES                                                     │
│     ├─→ Object Detection: 85%                                        │
│     ├─→ Text Recognition: 84%                                        │
│     └─→ Legal Reasoning: 86%                                         │
│                                                                       │
│  6. REPORT GENERATION                                                │
│     └─→ result.html with:                                            │
│         • 7 report sections                                          │
│         • Dual confidence bars (SAM3 + OpenAI)                       │
│         • Hallucination warnings                                     │
│         • Evidence images                                            │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 9. Avantajlar Tablosu

| Metrik | Sadece OpenAI | SAM3 + OpenAI (Paralel) |
|--------|--------------|-------------------------|
| **Objektivite** | LLM yorumu | ✅ SAM3 segmentasyon |
| **Hallüsinasyon Tespiti** | ❌ Yok | ✅ Cross-validation |
| **Görsel Kanıt** | ❌ Yok | ✅ Masks + Bboxes |
| **İşlem Süresi** | ~3s | ~4s (paralel) |
| **Audit Trail** | Sadece text | ✅ Segmentasyon data |
| **Güvenilirlik** | Değişken | ✅ Tutarlı |
| **Manuel Review İhtiyacı** | Belirsiz | ✅ Otomatik flag |

---

## 10. Dosya Yapısı

```
project/
├── server.py                      # Updated Flask app
├── sam3_detection_service.py      # SAM3 wrapper
├── openai_vision_service.py       # OpenAI Vision wrapper  
├── confidence_merger.py           # Cross-validation logic
├── extract_images.py              # PDF image extraction
├── templates/
│   ├── index.html                 # Upload form
│   └── result.html                # Results with dual bars
├── static/
│   └── css/
│       └── styles.css             # Updated styles
└── data/
    └── manifest.json              # Extraction metadata
```
