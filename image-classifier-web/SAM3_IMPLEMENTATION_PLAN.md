# SAM3 Integration Implementation Plan

## Overview

This document outlines the implementation plan for integrating SAM3 (Segment Anything Model 3) into the Parking Violation Report Tool for extracting useful, non-hallucinated insights from evidence images.

---

## A) UI Mapping Specification

### Current UI Structure

#### Evidence Inspector (Right Panel - Report Tab)
| Component | Data Source | Behavior |
|-----------|-------------|----------|
| Main Image Preview | `extracted_images[selectedIndex]` | Updates on thumbnail click |
| Filename | `images_metadata[selectedIndex].file` | Updates on selection |
| Page | `images_metadata[selectedIndex].page` | Updates on selection |
| Method | `images_metadata[selectedIndex].method` | Updates on selection |
| Size | `images_metadata[selectedIndex].width x height` | Updates on selection |
| Detected Items | Currently hardcoded from confidence_scores | **WILL USE SAM3 results** |
| Extracted Text | `doc_summary.vehicle.kenteken` | From document OCR |
| Thumbnail Strip | `extracted_images[]` | Click selects image |

#### Report Sections (Left Panel - Report Tab)
| Section | ID | Source Badge | SAM3 Influence |
|---------|-----|--------------|----------------|
| 1. Image Description | `image_description` | AI GENERATED | Uses SAM3 detections |
| 2. Object Detection Analysis | `object_detection` | AI GENERATED | **Primary SAM3 output** |
| 3. Timestamp & Location | `timestamp_location` | FROM DOCUMENT | No change |
| 4. Environmental Context | `environmental_context` | AI GENERATED | SAM3 sign detection |
| 5. Legal Reasoning | `legal_reasoning` | LEGAL TEMPLATE | No change (template only) |
| 6. Supporting Evidence | `supporting_evidence` | FROM DOCUMENT | SAM3 ROI references |
| 7. Confidence Summary | `confidence_summary` | SYSTEM | SAM3 confidence scores |

### SAM3 Data Flow
```
PDF Upload → Image Extraction → SAM3 Analysis → UI Display
                                     ↓
                              Per-image results:
                              - Segmentation masks
                              - Bounding boxes
                              - Confidence scores
                              - Cropped ROIs
```

---

## B) Data Contracts (JSON Schemas)

### 1. Evidence Image Schema
```json
{
  "id": "string (unique identifier)",
  "filename": "string",
  "page": "number",
  "method": "string (embedded|rendered)",
  "width": "number",
  "height": "number",
  "url": "string (/data/filename.jpg)"
}
```

### 2. SAM3 Analysis Result Schema (per image)
```json
{
  "image_id": "string",
  "analysis_timestamp": "ISO8601 string",
  "prompts_used": ["vehicle", "license_plate", "traffic_sign", "windshield"],
  "instances": [
    {
      "label": "string (vehicle|license_plate|traffic_sign|windshield|ground_marking)",
      "score": "number (0.0-1.0)",
      "box_xyxy": [x1, y1, x2, y2],
      "area_ratio": "number (0.0-1.0, portion of image)",
      "crop_url": "string|null (/data/derived/image_id_label_0.jpg)",
      "mask_url": "string|null (optional PNG mask)"
    }
  ],
  "derived_rois": {
    "vehicle_crop_url": "string|null",
    "plate_crop_url": "string|null",
    "sign_crop_url": "string|null",
    "windshield_crop_url": "string|null"
  },
  "overlay_url": "string|null (visualization with masks)",
  "warnings": ["string (any issues during analysis)"]
}
```

### 3. Detected Items for UI (derived from SAM3)
```json
{
  "items": [
    {
      "label": "Vehicle",
      "label_key": "vehicle",
      "confidence": 94,
      "detected": true,
      "crop_available": true
    },
    {
      "label": "License Plate",
      "label_key": "license_plate",
      "confidence": 88,
      "detected": true,
      "crop_available": true,
      "extracted_text": "G904XR"
    },
    {
      "label": "Sign E9",
      "label_key": "traffic_sign",
      "confidence": 86,
      "detected": true,
      "sign_code": "E9"
    }
  ],
  "missing": ["windshield"]
}
```

### 4. Confidence Scores Schema
```json
{
  "object_detection": {
    "score": 0.94,
    "source": "sam3",
    "details": "Based on 3 detected instances across 12 images"
  },
  "text_recognition": {
    "score": 0.88,
    "source": "document",
    "details": "License plate extracted from PDF metadata"
  },
  "legal_reasoning": {
    "score": 0.86,
    "source": "template",
    "details": "E9 violation code matched with legal template"
  }
}
```

### 5. Complete Report Payload
```json
{
  "document_summary": { /* existing doc_summary */ },
  "evidence_images": [ /* array of evidence image objects */ ],
  "sam3_analysis": {
    "per_image": { /* image_id -> SAM3 result */ },
    "aggregate": {
      "total_images_analyzed": 12,
      "vehicle_detections": 10,
      "plate_detections": 8,
      "sign_detections": 4,
      "best_vehicle_image": "image_id",
      "best_plate_image": "image_id",
      "best_sign_image": "image_id"
    }
  },
  "detected_items_summary": { /* for UI */ },
  "report_sections": [ /* 7 sections */ ],
  "confidence_scores": { /* scores with provenance */ },
  "generated_at": "ISO8601"
}
```

---

## C) SAM3 Service Implementation

### File: `sam3_service.py`

#### Core Components:
1. **SAM3Analyzer class** - Main inference engine
2. **ROI extraction** - Crop generation from masks
3. **Heuristic validation** - Conservative filtering
4. **Result serialization** - JSON-safe output

#### Detection Strategy:
```
1. Load image
2. Run prompted segmentation for:
   - "car" / "vehicle" → vehicle detection
   - "license plate" / "number plate" → plate detection
   - "traffic sign" / "parking sign" → sign detection
   - "windshield" / "front window" → windshield (optional)
3. If prompt fails (no masks > 0.5 confidence):
   - Run automatic mask generation
   - Apply heuristics to select candidates
4. Validate detections:
   - Vehicle: largest object, aspect ratio ~1.2-2.5
   - Plate: small rectangle on vehicle, aspect ~2-5
   - Sign: upper portion of image, rectangular
5. Generate crops and overlays
6. Return structured results
```

#### Heuristics (Conservative):
- **License Plate**:
  - Aspect ratio 2.0-5.0
  - Area < 5% of image
  - Located in lower half of detected vehicle bbox
  - High contrast region
- **Traffic Sign**:
  - Upper 60% of image
  - Roughly rectangular or circular
  - Distinct from background
- **Vehicle**:
  - Largest detected object
  - Area > 10% of image
  - Aspect ratio 1.0-3.0

---

## D) Insight Extraction Rules

### Allowed Insights (grounded in SAM3 + document):
| Insight | Condition | Example Output |
|---------|-----------|----------------|
| Vehicle detected | SAM3 vehicle mask > 0.7 | "Voertuig gedetecteerd (94%)" |
| License plate detected | SAM3 plate mask > 0.6 | "Kenteken regio gedetecteerd" |
| Plate text | From document OCR | "Geëxtraheerde tekst: G904XR" |
| Traffic sign detected | SAM3 sign mask > 0.6 | "Verkeersbord gedetecteerd" |
| Sign code | From document | "Bord E9 (vergunninghouders)" |

### NOT Allowed (would be hallucination):
| Claim | Reason |
|-------|--------|
| "No driver present" | Cannot verify from static image |
| "No permit visible" | Requires windshield inspection + OCR |
| "No loading activity" | Requires temporal observation |
| "Vehicle blocking access" | Requires spatial context |

### Default Fallback:
When detection confidence < 0.5 or heuristics fail:
```
"Onvoldoende informatie beschikbaar voor [aspect]"
"Not enough information available for [aspect]"
```

---

## E) UI Integration

### Evidence Inspector Updates:
1. **Detected Items List**: Render from `sam3_analysis.detected_items`
2. **Segmentation Toggle**: Add "Show Overlay" button
3. **Missing Items**: Show grey text for undetected items
4. **Crop Links**: Add "View Crop" links for available ROIs

### Report Section Updates:
1. **Section 2 (Object Detection)**:
   - Generate bullet list from SAM3 instances
   - Include confidence percentages
   - Add "Detection Confidence" bar

2. **Section 6 (Supporting Evidence)**:
   - List available ROI crops
   - Reference SAM3 analysis summary
   - Link to overlay visualizations

### State Management:
- Selected image index persists across tab switches
- SAM3 results cached per session
- Overlay toggle state preserved

---

## F) File Structure

```
image-classifier-web/
├── server.py                 # Main Flask app (updated)
├── sam3_service.py           # NEW: SAM3 inference service
├── extract_images.py         # Existing PDF extraction
├── requirements.txt          # Updated with SAM3 deps
├── data/
│   ├── *.jpg                 # Extracted images
│   └── derived/              # NEW: SAM3 crops & overlays
│       ├── {image_id}_vehicle_0.jpg
│       ├── {image_id}_plate_0.jpg
│       ├── {image_id}_sign_0.jpg
│       └── {image_id}_overlay.png
├── templates/
│   ├── result.html           # Updated for SAM3 display
│   └── ...
└── static/
    └── styles.css            # Updated for new components
```

---

## G) Implementation Steps

### Phase 1: Backend (sam3_service.py)
1. Create SAM3Analyzer class with model loading
2. Implement prompted segmentation
3. Implement automatic mask fallback
4. Add heuristic validation
5. Create crop extraction
6. Generate overlay visualization
7. Serialize results to JSON

### Phase 2: Server Integration
1. Import SAM3 service in server.py
2. Call SAM3 analysis after image extraction
3. Store results in derived/ folder
4. Add new API endpoint for analysis
5. Update predict() to include SAM3 data

### Phase 3: Frontend
1. Update result.html detected items section
2. Add overlay toggle functionality
3. Update Section 2 content generation
4. Update Section 6 with ROI references
5. Add translations for new UI elements

### Phase 4: Testing & Validation
1. Test with sample parking PDFs
2. Verify conservative detection
3. Check "Not enough information" fallbacks
4. Validate JSON output structure
5. Test UI responsiveness

---

## H) Dependencies

```
# requirements.txt additions
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
segment-anything-2  # or sam2 depending on package
Pillow>=9.0.0
numpy>=1.24.0
```

---

## I) Mocked Example Output

See `mock_sam3_output.json` for a complete example matching an E9 permit-holder parking case with volgnummer 3883491 and kenteken G904XR.
