# UX Specification: Parking Violation Report Tool

## A) UX Spec

### Page Layout
```
+----------------------------------------------------------+
|  [Amsterdam Header - Logo + "Image Classifier"]          |
+----------------------------------------------------------+
|  [ Report ]  [ Document Summary ]     <- Tab Switcher    |
+----------------------------------------------------------+
|                                                          |
|  +------------------+  +-----------------------------+   |
|  |                  |  |                             |   |
|  |  Evidence Grid   |  |  Detail Panel               |   |
|  |  (Left 60%)      |  |  (Right 40%)                |   |
|  |                  |  |                             |   |
|  +------------------+  +-----------------------------+   |
|                                                          |
+----------------------------------------------------------+
|  [Download Actions Bar]                                  |
+----------------------------------------------------------+
```

### Navigation
- **Tab Switcher**: Two tabs at top - "Report" (default) and "Document Summary"
- **State Preservation**: Selected image persists across tab switches
- **No nested routing**: Single page with tab-based content switching

### Primary Actions
1. **Download PDF** - Full report as PDF
2. **Download Images** - All evidence images as ZIP
3. **Export JSON** - Structured data export
4. **Copy Section** - Per-section clipboard copy

### Editing Interactions
- Narrative sections: Click to edit inline
- Save indicator: Auto-save with "Saved" badge
- Copy button: Per-section copy-to-clipboard

### Error/Empty States
- "No images extracted" - Show placeholder with retry option
- "Missing field" - Grey italic text: "Not available"
- "Low confidence" - Orange warning badge
- "Model inference" - Blue badge indicating AI-generated

### Accessibility
- Tab order: Header → Tabs → Main content → Actions
- ARIA labels on all interactive elements
- Focus visible states
- Color contrast ≥ 4.5:1

---

## B) UI Content Plan

### Tab 1: Document Summary

**Left Column (60%)**
```
┌─ Case Identifiers ─────────────────┐
│ Volgnummer: [value]                │
│ Bonnummer: [value]                 │
│ Status: [badge]                    │
│ Datum/Tijd: [value]                │
│ Medewerker: [value]                │
└────────────────────────────────────┘

┌─ Location ─────────────────────────┐
│ Stadsdeel: [value]                 │
│ Buurt: [value]                     │
│ Straat: [value]                    │
│ Locatie Nr: [value]                │
└────────────────────────────────────┘

┌─ Violation ────────────────────────┐
│ Code: [E9] [Sign Icon]             │
│ Description: [value]               │
│ Toelichting: [value]               │
└────────────────────────────────────┘

┌─ Vehicle ──────────────────────────┐
│ Kenteken: [value]                  │
│ Merk: [value]                      │
│ Model: [value]                     │
│ Kleur: [value]                     │
└────────────────────────────────────┘

┌─ Officer Observation ──────────────┐
│ [Redenen van wetenschap text]      │
│ Source: [From Document] badge      │
└────────────────────────────────────┘
```

**Right Column (40%)**
```
┌─ Evidence Images ──────────────────┐
│ ┌────┐ ┌────┐ ┌────┐ ┌────┐       │
│ │img1│ │img2│ │img3│ │img4│       │
│ └────┘ └────┘ └────┘ └────┘       │
│ ┌────┐ ┌────┐ ...                 │
│ │img5│ │img6│                     │
│ └────┘ └────┘                     │
│                                   │
│ Total: 12 images                  │
└───────────────────────────────────┘

┌─ Related Registrations ───────────┐
│ • Sleepbon: [reference]           │
│ • [other registrations]           │
└───────────────────────────────────┘

┌─ Export Actions ──────────────────┐
│ [Export JSON] [Download Images]   │
└───────────────────────────────────┘
```

### Tab 2: Report

**Left Column (60%) - Narrative Sections**
```
┌─ 1. Image Description ─────────────┐
│ [Editable textarea]                │
│ Source: [From Image] [Model]       │
└────────────────────────────────────┘

┌─ 2. Object Detection Analysis ─────┐
│ [Editable textarea]                │
│ Confidence: [■■■■■■■░░░] 78%      │
└────────────────────────────────────┘

┌─ 3. Timestamp & Location ──────────┐
│ [Editable textarea]                │
│ Source: [From Document]            │
└────────────────────────────────────┘

┌─ 4. Environmental Context ─────────┐
│ [Editable textarea]                │
│ Source: [From Image]               │
└────────────────────────────────────┘

┌─ 5. Legal Reasoning ───────────────┐
│ [Editable textarea - Dutch phrases]│
│ Source: [From Document] [Template] │
└────────────────────────────────────┘

┌─ 6. Supporting Evidence ───────────┐
│ [Editable textarea]                │
│ Source: [From Image]               │
└────────────────────────────────────┘

┌─ 7. Confidence Summary ────────────┐
│ [Read-only confidence metrics]     │
└────────────────────────────────────┘
```

**Right Column (40%) - Image Inspector**
```
┌─ Selected Evidence ────────────────┐
│ ┌──────────────────────────────┐  │
│ │                              │  │
│ │      [Large Image View]      │  │
│ │                              │  │
│ └──────────────────────────────┘  │
│                                   │
│ Filename: [name.jpg]              │
│ Page: 3 | Method: embedded        │
│ Size: 1023 × 461                  │
│                                   │
│ ┌─ Detected Items ─────────────┐  │
│ │ • Vehicle (94%)              │  │
│ │ • License Plate (88%)        │  │
│ │ • Parking Sign E9 (91%)      │  │
│ └──────────────────────────────┘  │
│                                   │
│ ┌─ Extracted Text ─────────────┐  │
│ │ Plate: XX-123-YY             │  │
│ └──────────────────────────────┘  │
└───────────────────────────────────┘

┌─ Image Thumbnails ─────────────────┐
│ [○] [○] [●] [○] [○] [○]           │
│ Click to select primary evidence   │
└───────────────────────────────────┘
```

---

## C) Component Architecture (Vanilla JS + Jinja)

### State Model
```javascript
const appState = {
  activeTab: 'report',        // 'report' | 'summary'
  selectedImageIndex: 0,      // Currently selected evidence image
  documentSummary: {...},     // Extracted PDF fields
  reportSections: [...],      // 7 narrative sections
  confidenceScores: {...},    // Detection confidence values
  editingSection: null,       // Currently editing section ID
  isDirty: false              // Unsaved changes flag
};
```

### Data Contracts

**Document Summary (from backend):**
```json
{
  "case": {
    "volgnummer": "2024-123456",
    "bonnummer": "BON-789",
    "status": "Wegsleepwaardig",
    "datum_tijd": "2024-01-15 14:30",
    "medewerker": "A. Jansen"
  },
  "location": {
    "stadsdeel": "Centrum",
    "buurt": "Grachtengordel-West",
    "straat": "Prinsengracht",
    "locatie_nr": "123"
  },
  "violation": {
    "code": "E9",
    "sign": "E9",
    "description": "Parkeren in strijd met bord E9",
    "toelichting": "Geen deelauto"
  },
  "vehicle": {
    "kenteken": "XX-123-YY",
    "merk": "Volkswagen",
    "model": "Golf",
    "kleur": "Grijs"
  },
  "officer_observation": "Ik zag dat het voertuig...",
  "related_registrations": [
    {"type": "Sleepbon", "reference": "SLP-2024-001"}
  ],
  "evidence_images": [
    {"file": "img_p03_01.jpg", "page": 3, "method": "embedded", "width": 1023, "height": 461}
  ]
}
```

**Report Sections (from backend):**
```json
{
  "sections": [
    {
      "id": "image_description",
      "title": "Image Description",
      "content": "...",
      "source": "model",
      "editable": true
    },
    ...
  ],
  "confidence": {
    "object_detection": 0.94,
    "text_recognition": 0.88,
    "legal_reasoning": 0.86
  }
}
```

---

## D) Implementation

See code files below.

---

## E) QA Checklist

- [ ] Switching tabs preserves selected image
- [ ] Document Summary fields match backend JSON
- [ ] Report sections appear in correct order (1-7)
- [ ] Missing data shows "Not available" (no hallucination)
- [ ] Download buttons trigger correct actions
- [ ] Image click selects as primary evidence
- [ ] Source badges display correctly (From Document / Model / Missing)
- [ ] Confidence bars render with correct percentages
- [ ] Edit mode works for narrative sections
- [ ] Copy-to-clipboard works per section
- [ ] No non-parking scenarios mentioned
- [ ] Dutch legal phrases from dataset used
- [ ] Responsive layout on smaller screens
- [ ] Keyboard navigation works
- [ ] Focus states visible
