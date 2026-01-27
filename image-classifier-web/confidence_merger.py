# confidence_merger.py
"""
Confidence Merger for SAM3 + OpenAI Vision Parallel Architecture.

This module implements cross-validation logic to merge confidence scores
from SAM3 (objective segmentation) and OpenAI Vision (semantic analysis).

KEY INSIGHT:
- SAM3 = Objective visual evidence (segmentation mask + score)
- OpenAI = Semantic understanding (interpretation of what it sees)

PARKING VIOLATION LOGIC:
For parking violations, the evidence logic is INVERTED for certain categories:
- Finding NO driver = GOOD (violation confirmed) = GREEN checkmark
- Finding NO permit = GOOD (violation confirmed) = GREEN checkmark
- Finding NO disability card = GOOD (E6 violation confirmed) = GREEN checkmark
- Finding the SIGN = GOOD (violation location confirmed) = GREEN checkmark

Cross-validation detects hallucinations:
- OpenAI says it sees something but SAM3 can't find it = HALLUCINATION RISK
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════════════
# CATEGORY CLASSIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

# ABSENCE-BASED: NOT finding these = GOOD for parking violation case
# For these categories, we INVERT the confidence for display
ABSENCE_BASED_CATEGORIES = {
    'person',
    'driver',
    'driver_present',
    'driver_in_vehicle',
    'parking_permit',
    'permit',
    'disability_card',
    'loading_activity',
}

# PRESENCE-BASED: Finding these = GOOD for parking violation case
# Standard detection logic applies
PRESENCE_BASED_CATEGORIES = {
    'vehicle',
    'van',
    'truck',
    'motorcycle',
    'license_plate',
    'traffic_sign',
    'traffic_sign_e1',
    'traffic_sign_e2',
    'traffic_sign_e4',
    'traffic_sign_e4_electric',
    'traffic_sign_e5',
    'traffic_sign_e6',
    'traffic_sign_e7',
    'traffic_sign_e8',
    'traffic_sign_e9',
    'traffic_sign_g7',
    'yellow_line',
    'windshield',
    'charging_cable',
    'charging_station',
    'charging_connected',
    'parking_disc',
}

# Display labels for categories
DISPLAY_LABELS = {
    # Presence-based (finding = good)
    'vehicle': 'Vehicle',
    'van': 'Van',
    'truck': 'Truck',
    'motorcycle': 'Motorcycle',
    'license_plate': 'License Plate',
    'traffic_sign': 'Traffic Sign',
    'traffic_sign_e1': 'Sign E1 (No Parking)',
    'traffic_sign_e2': 'Sign E2 (No Stopping)',
    'traffic_sign_e4': 'Sign E4 (Parking)',
    'traffic_sign_e4_electric': 'Sign E4 (Electric)',
    'traffic_sign_e5': 'Sign E5 (Taxi)',
    'traffic_sign_e6': 'Sign E6 (Disabled)',
    'traffic_sign_e7': 'Sign E7 (Loading)',
    'traffic_sign_e8': 'Sign E8 (Carpool)',
    'traffic_sign_e9': 'Sign E9 (Permit)',
    'traffic_sign_g7': 'Sign G7 (Pedestrian)',
    'yellow_line': 'Yellow Line Marking',
    'windshield': 'Windshield',
    'charging_cable': 'Charging Cable',
    'charging_station': 'Charging Station',
    'charging_connected': 'Charging Connected',
    'parking_disc': 'Parking Disc',
    # Absence-based (not finding = good) - standard labels
    'person': 'Driver/Person',
    'driver': 'Driver',
    'driver_present': 'Driver Present',
    'driver_in_vehicle': 'Driver in Vehicle',
    'parking_permit': 'Parking Permit',
    'permit': 'Permit',
    'disability_card': 'Disability Card',
    'loading_activity': 'Loading Activity',
}

# Labels when ABSENCE is confirmed (for violation cases)
ABSENCE_LABELS = {
    'person': 'No Driver Present',
    'driver': 'No Driver',
    'driver_present': 'No Driver Present',
    'driver_in_vehicle': 'No Driver in Vehicle',
    'parking_permit': 'No Valid Permit',
    'permit': 'No Valid Permit',
    'disability_card': 'No Disability Card',
    'loading_activity': 'No Loading Activity',
}

# Evidence Checklist configurations per violation type
VIOLATION_CHECKS = {
    'E1': [  # No Parking Zone
        {'label': 'Sign E1 visible', 'label_nl': 'Bord E1 zichtbaar', 'category': 'traffic_sign_e1', 'absence': False, 'ref': 'RVV 1990 Bijlage 1, Bord E1'},
        {'label': 'Vehicle identified', 'label_nl': 'Voertuig geïdentificeerd', 'category': 'vehicle', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'License plate visible', 'label_nl': 'Kenteken zichtbaar', 'category': 'license_plate', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'No valid exemption visible', 'label_nl': 'Geen geldige ontheffing zichtbaar', 'category': 'parking_permit', 'absence': True, 'ref': 'RVV 1990 Art. 87'},
        {'label': 'No driver present', 'label_nl': 'Geen bestuurder aanwezig', 'category': 'person', 'absence': True, 'ref': 'RVV 1990 Art. 1 (definitie parkeren)'},
    ],
    'E2': [  # No Stopping Zone
        {'label': 'Sign E2 visible', 'label_nl': 'Bord E2 zichtbaar', 'category': 'traffic_sign_e2', 'absence': False, 'ref': 'RVV 1990 Bijlage 1, Bord E2'},
        {'label': 'Vehicle identified', 'label_nl': 'Voertuig geïdentificeerd', 'category': 'vehicle', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'License plate visible', 'label_nl': 'Kenteken zichtbaar', 'category': 'license_plate', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'No valid exemption visible', 'label_nl': 'Geen geldige ontheffing zichtbaar', 'category': 'parking_permit', 'absence': True, 'ref': 'RVV 1990 Art. 87'},
    ],
    'E4': [  # Parking Facility (with conditions)
        {'label': 'Sign E4 visible', 'label_nl': 'Bord E4 zichtbaar', 'category': 'traffic_sign_e4', 'absence': False, 'ref': 'RVV 1990 Bijlage 1, Bord E4'},
        {'label': 'Vehicle identified', 'label_nl': 'Voertuig geïdentificeerd', 'category': 'vehicle', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'License plate visible', 'label_nl': 'Kenteken zichtbaar', 'category': 'license_plate', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'No driver present', 'label_nl': 'Geen bestuurder aanwezig', 'category': 'person', 'absence': True, 'ref': 'RVV 1990 Art. 1 (definitie parkeren)'},
    ],
    'E5': [  # Taxi Stand
        {'label': 'Sign E5 visible', 'label_nl': 'Bord E5 zichtbaar', 'category': 'traffic_sign_e5', 'absence': False, 'ref': 'RVV 1990 Bijlage 1, Bord E5'},
        {'label': 'Vehicle identified', 'label_nl': 'Voertuig geïdentificeerd', 'category': 'vehicle', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'License plate visible', 'label_nl': 'Kenteken zichtbaar', 'category': 'license_plate', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'No driver present', 'label_nl': 'Geen bestuurder aanwezig', 'category': 'person', 'absence': True, 'ref': 'RVV 1990 Art. 1 (definitie parkeren)'},
    ],
    'E9': [  # Permit holders parking
        {'label': 'Sign E9 visible', 'label_nl': 'Bord E9 zichtbaar', 'category': 'traffic_sign_e9', 'absence': False, 'ref': 'RVV 1990 Bijlage 1, Bord E9'},
        {'label': 'Vehicle identified', 'label_nl': 'Voertuig geïdentificeerd', 'category': 'vehicle', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'License plate visible', 'label_nl': 'Kenteken zichtbaar', 'category': 'license_plate', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'No valid permit visible', 'label_nl': 'Geen geldige vergunning zichtbaar', 'category': 'parking_permit', 'absence': True, 'ref': 'RVV 1990 Art. 24 lid 1g'},
        {'label': 'No driver present', 'label_nl': 'Geen bestuurder aanwezig', 'category': 'person', 'absence': True, 'ref': 'RVV 1990 Art. 1 (definitie parkeren)'},
    ],
    'E6': [  # Disabled parking
        {'label': 'Sign E6 visible', 'label_nl': 'Bord E6 zichtbaar', 'category': 'traffic_sign_e6', 'absence': False, 'ref': 'RVV 1990 Bijlage 1, Bord E6'},
        {'label': 'Vehicle identified', 'label_nl': 'Voertuig geïdentificeerd', 'category': 'vehicle', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'License plate visible', 'label_nl': 'Kenteken zichtbaar', 'category': 'license_plate', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'No disability card visible', 'label_nl': 'Geen gehandicaptenkaart zichtbaar', 'category': 'disability_card', 'absence': True, 'ref': 'RVV 1990 Art. 26'},
        {'label': 'No driver present', 'label_nl': 'Geen bestuurder aanwezig', 'category': 'person', 'absence': True, 'ref': 'RVV 1990 Art. 1 (definitie parkeren)'},
    ],
    'E7': [  # Loading zone
        {'label': 'Sign E7 visible', 'label_nl': 'Bord E7 zichtbaar', 'category': 'traffic_sign_e7', 'absence': False, 'ref': 'RVV 1990 Bijlage 1, Bord E7'},
        {'label': 'Vehicle identified', 'label_nl': 'Voertuig geïdentificeerd', 'category': 'vehicle', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'License plate visible', 'label_nl': 'Kenteken zichtbaar', 'category': 'license_plate', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'No loading/unloading activity', 'label_nl': 'Geen laad/los activiteit', 'category': 'loading_activity', 'absence': True, 'ref': 'RVV 1990 Art. 24'},
        {'label': 'No valid exemption visible', 'label_nl': 'Geen geldige ontheffing zichtbaar', 'category': 'parking_permit', 'absence': True, 'ref': 'RVV 1990 Art. 24 lid 1c'},
        {'label': 'No driver present', 'label_nl': 'Geen bestuurder aanwezig', 'category': 'person', 'absence': True, 'ref': 'RVV 1990 Art. 1'},
    ],
    'G7': [  # Pedestrian zone
        {'label': 'Sign G7 visible', 'label_nl': 'Bord G7 zichtbaar', 'category': 'traffic_sign_g7', 'absence': False, 'ref': 'RVV 1990 Bijlage 1, Bord G7'},
        {'label': 'Vehicle identified', 'label_nl': 'Voertuig geïdentificeerd', 'category': 'vehicle', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'License plate visible', 'label_nl': 'Kenteken zichtbaar', 'category': 'license_plate', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'No valid exemption visible', 'label_nl': 'Geen geldige ontheffing zichtbaar', 'category': 'parking_permit', 'absence': True, 'ref': 'RVV 1990 Art. 24'},
    ],
    'E8': [  # Carpool parking
        {'label': 'Sign E8 visible', 'label_nl': 'Bord E8 zichtbaar', 'category': 'traffic_sign_e8', 'absence': False, 'ref': 'RVV 1990 Bijlage 1, Bord E8'},
        {'label': 'Vehicle identified', 'label_nl': 'Voertuig geïdentificeerd', 'category': 'vehicle', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'License plate visible', 'label_nl': 'Kenteken zichtbaar', 'category': 'license_plate', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'No driver present', 'label_nl': 'Geen bestuurder aanwezig', 'category': 'person', 'absence': True, 'ref': 'RVV 1990 Art. 1 (definitie parkeren)'},
    ],
    'YELLOW_LINE': [  # Yellow continuous line (gele doorgetrokken streep)
        {'label': 'Yellow line visible', 'label_nl': 'Gele streep zichtbaar', 'category': 'yellow_line', 'absence': False, 'ref': 'RVV 1990 Art. 24 lid 1 sub e'},
        {'label': 'Vehicle identified', 'label_nl': 'Voertuig geïdentificeerd', 'category': 'vehicle', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'License plate visible', 'label_nl': 'Kenteken zichtbaar', 'category': 'license_plate', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'No driver present', 'label_nl': 'Geen bestuurder aanwezig', 'category': 'person', 'absence': True, 'ref': 'RVV 1990 Art. 1 (definitie parkeren)'},
    ],
    'R396I': [  # R396i code alias for yellow line
        {'label': 'Yellow line visible', 'label_nl': 'Gele streep zichtbaar', 'category': 'yellow_line', 'absence': False, 'ref': 'RVV 1990 Art. 24 lid 1 sub e'},
        {'label': 'Vehicle identified', 'label_nl': 'Voertuig geïdentificeerd', 'category': 'vehicle', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'License plate visible', 'label_nl': 'Kenteken zichtbaar', 'category': 'license_plate', 'absence': False, 'ref': 'Art. 5 Wahv'},
        {'label': 'No driver present', 'label_nl': 'Geen bestuurder aanwezig', 'category': 'person', 'absence': True, 'ref': 'RVV 1990 Art. 1 (definitie parkeren)'},
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

def is_absence_based(category: str) -> bool:
    """
    Check if category should use inverted logic (absence = good).

    For parking violations:
    - Not finding a driver = GOOD (confirms parking, not stopping)
    - Not finding a permit = GOOD (confirms violation)
    - Not finding a disability card = GOOD (confirms E6 violation)
    """
    return category.lower() in ABSENCE_BASED_CATEGORIES


def get_display_label(category: str, show_absence: bool = False) -> str:
    """
    Get display label for a category.

    Args:
        category: The detection category key
        show_absence: If True and category is absence-based, return "No X" label

    Returns:
        Human-readable label for display
    """
    category_lower = category.lower()

    if show_absence and category_lower in ABSENCE_LABELS:
        return ABSENCE_LABELS[category_lower]

    return DISPLAY_LABELS.get(category_lower, category.replace('_', ' ').title())


def invert_confidence(confidence: float) -> float:
    """
    Invert confidence for absence-based items.

    Example: 0% detection → 100% absence confidence
             90% detection → 10% absence confidence
    """
    return 1.0 - confidence


class ConfidenceSource(Enum):
    """Source of the final confidence score."""
    SAM3 = "sam3"                      # Objective segmentation
    OPENAI = "openai"                  # Semantic analysis
    MERGED = "merged"                  # Combined weighted score
    HALLUCINATION = "hallucination"    # Hallucination warning
    ABSENCE = "absence"                # Absence confirmed (for violation)


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

    Implements the 5 scenario merge strategy:
    1. Both HIGH → Weighted merge
    2. SAM3 HIGH, OpenAI LOW → Trust SAM3
    3. SAM3 LOW, OpenAI HIGH → HALLUCINATION RISK (unless absence-based)
    4. Both LOW → Not detected / Absence confirmed
    5. Medium range → Weighted merge with potential risk flag

    PARKING VIOLATION LOGIC:
    For absence-based categories (driver, permit, disability card):
    - SAM3 LOW = GOOD (absence confirmed, supports violation)
    - Display confidence is INVERTED (0% detection → 100% absence)
    """

    # Category-specific weights
    CATEGORY_WEIGHTS = {
        'vehicle':          {'sam3': 0.70, 'openai': 0.30},
        'van':              {'sam3': 0.70, 'openai': 0.30},
        'truck':            {'sam3': 0.70, 'openai': 0.30},
        'motorcycle':       {'sam3': 0.70, 'openai': 0.30},
        'license_plate':    {'sam3': 0.60, 'openai': 0.40},
        'traffic_sign':     {'sam3': 0.65, 'openai': 0.35},
        'traffic_sign_e1':  {'sam3': 0.65, 'openai': 0.35},
        'traffic_sign_e2':  {'sam3': 0.65, 'openai': 0.35},
        'traffic_sign_e4':  {'sam3': 0.65, 'openai': 0.35},
        'traffic_sign_e4_electric': {'sam3': 0.65, 'openai': 0.35},
        'traffic_sign_e5':  {'sam3': 0.65, 'openai': 0.35},
        'traffic_sign_e6':  {'sam3': 0.65, 'openai': 0.35},
        'traffic_sign_e7':  {'sam3': 0.65, 'openai': 0.35},
        'traffic_sign_e8':  {'sam3': 0.65, 'openai': 0.35},
        'traffic_sign_e9':  {'sam3': 0.65, 'openai': 0.35},
        'traffic_sign_g7':  {'sam3': 0.65, 'openai': 0.35},
        'yellow_line':      {'sam3': 0.70, 'openai': 0.30},
        'parking_permit':   {'sam3': 0.50, 'openai': 0.50},
        'disability_card':  {'sam3': 0.55, 'openai': 0.45},
        'parking_disc':     {'sam3': 0.55, 'openai': 0.45},
        'charging_cable':   {'sam3': 0.70, 'openai': 0.30},
        'charging_station': {'sam3': 0.70, 'openai': 0.30},
        'charging_connected': {'sam3': 0.60, 'openai': 0.40},
        'person':           {'sam3': 0.75, 'openai': 0.25},
        'driver_in_vehicle': {'sam3': 0.60, 'openai': 0.40},
        'driver_present':   {'sam3': 0.75, 'openai': 0.25},
        'loading_activity': {'sam3': 0.50, 'openai': 0.50},
        'windshield':       {'sam3': 0.80, 'openai': 0.20},
    }

    # Thresholds
    HIGH_CONFIDENCE = 0.70
    LOW_CONFIDENCE = 0.35
    HALLUCINATION_THRESHOLD = 0.40

    def merge(
        self,
        sam3_confidences: Dict[str, float],
        openai_confidences: Dict[str, float]
    ) -> Dict[str, MergedConfidence]:
        """
        Merge confidence scores from both sources.
        """
        all_categories = set(sam3_confidences.keys()) | set(openai_confidences.keys())
        merged_results = {}

        for category in all_categories:
            sam3_conf = sam3_confidences.get(category, 0.0)
            openai_conf = openai_confidences.get(category, 0.0)
            merged_results[category] = self._merge_single(category, sam3_conf, openai_conf)

        return merged_results

    def _merge_single(
        self,
        category: str,
        sam3_conf: float,
        openai_conf: float
    ) -> MergedConfidence:
        """
        Apply simplified merge strategy for a single category.

        NEW LOGIC:
        - If SAM3 > OpenAI → FINAL = SAM3 (trust objective segmentation)
        - If SAM3 <= OpenAI → FINAL = average of both

        Exception for absence-based categories (driver, permit, disability card):
        - These are inverted for display (low detection = high absence confidence)
        - Hallucination detection still applies when OpenAI claims something SAM3 can't find
        """
        agreement = 1.0 - abs(sam3_conf - openai_conf)
        is_absence = is_absence_based(category)

        # Calculate merged confidence using new simplified logic
        if sam3_conf > openai_conf:
            # SAM3 > OpenAI → Trust SAM3 directly
            merged = sam3_conf
            source = ConfidenceSource.SAM3
            reasoning = "SAM3 confidence higher - using SAM3 value directly"
        else:
            # SAM3 <= OpenAI → Use average
            merged = (sam3_conf + openai_conf) / 2.0
            source = ConfidenceSource.MERGED
            reasoning = "SAM3 <= OpenAI - using average of both"

        # Check for hallucination risk (OpenAI sees something SAM3 doesn't)
        is_hallucination = False
        if not is_absence and sam3_conf < self.LOW_CONFIDENCE and openai_conf >= self.HIGH_CONFIDENCE:
            is_hallucination = True
            reasoning = "HALLUCINATION RISK: OpenAI claims detection but SAM3 cannot segment"
            source = ConfidenceSource.HALLUCINATION

        # Handle absence-based categories (for parking violations)
        if is_absence:
            if sam3_conf < self.LOW_CONFIDENCE and openai_conf < self.LOW_CONFIDENCE:
                # Both low = absence confirmed
                source = ConfidenceSource.ABSENCE
                reasoning = "ABSENCE CONFIRMED: Both sources agree item is not present"
            elif sam3_conf < self.LOW_CONFIDENCE:
                # SAM3 low (absence) but OpenAI high - trust SAM3 for absence
                source = ConfidenceSource.ABSENCE
                reasoning = "ABSENCE CONFIRMED: SAM3 confirms absence (supports violation)"
                is_hallucination = False  # Not a hallucination for absence items

        return MergedConfidence(
            category=category,
            sam3_confidence=sam3_conf,
            openai_confidence=openai_conf,
            merged_confidence=merged,
            agreement_score=agreement,
            source_used=source,
            is_hallucination_risk=is_hallucination,
            reasoning=reasoning
        )

    def calculate_final_scores(
        self,
        merged_results: Dict[str, MergedConfidence]
    ) -> Dict[str, float]:
        """Calculate final UI confidence scores."""

        obj_categories = [
            'vehicle', 'van', 'truck', 'motorcycle',
            'traffic_sign', 'traffic_sign_e1', 'traffic_sign_e2',
            'traffic_sign_e4', 'traffic_sign_e5', 'traffic_sign_e6',
            'traffic_sign_e7', 'traffic_sign_e8', 'traffic_sign_e9',
            'traffic_sign_g7', 'yellow_line',
            'parking_permit', 'disability_card', 'parking_disc',
            'charging_cable', 'charging_station', 'person', 'windshield'
        ]
        obj_scores = [
            merged_results[cat].merged_confidence
            for cat in obj_categories
            if cat in merged_results and merged_results[cat].merged_confidence > 0
        ]
        object_detection = sum(obj_scores) / len(obj_scores) if obj_scores else 0.0

        text_score = merged_results.get('license_plate', MergedConfidence(
            category='license_plate', sam3_confidence=0, openai_confidence=0,
            merged_confidence=0, agreement_score=0, source_used=ConfidenceSource.MERGED,
            is_hallucination_risk=False, reasoning=''
        )).merged_confidence

        legal_score = self._calculate_legal_score(merged_results)

        return {
            'object_detection': object_detection,
            'text_recognition': text_score,
            'legal_reasoning': legal_score
        }

    def _calculate_legal_score(
        self,
        merged_results: Dict[str, MergedConfidence]
    ) -> float:
        """Calculate legal reasoning score with proper inversion logic."""

        # Vehicle confidence (any type)
        vehicle_confs = [
            merged_results.get(vtype, MergedConfidence(
                category=vtype, sam3_confidence=0, openai_confidence=0,
                merged_confidence=0, agreement_score=0, source_used=ConfidenceSource.MERGED,
                is_hallucination_risk=False, reasoning=''
            )).merged_confidence
            for vtype in ['vehicle', 'van', 'truck', 'motorcycle']
        ]
        vehicle_conf = max(vehicle_confs) if vehicle_confs else 0.0

        # Traffic sign score (includes all E-codes and yellow line)
        sign_scores = [
            merged_results.get(f'traffic_sign_{code}', MergedConfidence(
                category=f'traffic_sign_{code}', sam3_confidence=0, openai_confidence=0,
                merged_confidence=0, agreement_score=0, source_used=ConfidenceSource.MERGED,
                is_hallucination_risk=False, reasoning=''
            )).merged_confidence
            for code in ['e1', 'e2', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'g7']
        ]
        generic_sign = merged_results.get('traffic_sign', MergedConfidence(
            category='traffic_sign', sam3_confidence=0, openai_confidence=0,
            merged_confidence=0, agreement_score=0, source_used=ConfidenceSource.MERGED,
            is_hallucination_risk=False, reasoning=''
        )).merged_confidence
        sign_scores.append(generic_sign)
        # Also check yellow line marking
        yellow_line = merged_results.get('yellow_line', MergedConfidence(
            category='yellow_line', sam3_confidence=0, openai_confidence=0,
            merged_confidence=0, agreement_score=0, source_used=ConfidenceSource.MERGED,
            is_hallucination_risk=False, reasoning=''
        )).merged_confidence
        sign_scores.append(yellow_line)
        sign_conf = max(sign_scores) if sign_scores else 0.0

        # INVERTED: Permit not found is GOOD
        permit_conf = merged_results.get('parking_permit', MergedConfidence(
            category='parking_permit', sam3_confidence=0, openai_confidence=0,
            merged_confidence=0, agreement_score=0, source_used=ConfidenceSource.MERGED,
            is_hallucination_risk=False, reasoning=''
        )).merged_confidence
        no_permit_score = 1.0 - permit_conf

        # INVERTED: Driver not present is GOOD
        person_conf = merged_results.get('person', MergedConfidence(
            category='person', sam3_confidence=0, openai_confidence=0,
            merged_confidence=0, agreement_score=0, source_used=ConfidenceSource.MERGED,
            is_hallucination_risk=False, reasoning=''
        )).merged_confidence
        no_driver_score = 1.0 - person_conf

        legal_score = (
            vehicle_conf * 0.35 +
            sign_conf * 0.30 +
            no_permit_score * 0.20 +
            no_driver_score * 0.05 +
            0.10
        )

        return min(legal_score, 1.0)

    def get_hallucination_warnings(
        self,
        merged_results: Dict[str, MergedConfidence]
    ) -> List[str]:
        """Get list of hallucination warnings for UI display."""
        warnings = []
        for category, result in merged_results.items():
            if result.is_hallucination_risk:
                display_name = get_display_label(category)
                warnings.append(f"{display_name}: {result.reasoning}")
        return warnings

    def format_for_ui(
        self,
        merged_results: Dict[str, MergedConfidence]
    ) -> List[Dict]:
        """
        Format merged results for UI display with INVERTED LOGIC for absence-based items.

        PARKING VIOLATION LOGIC:
        - For presence-based items (vehicle, sign): Show raw confidence
        - For absence-based items (driver, permit): INVERT confidence
          - 0% detection → 100% absence confidence → GREEN checkmark
          - 90% detection → 10% absence confidence → RED (something found)

        Returns list of items for the detected items panel.
        """
        items = []

        # Track processed categories to avoid duplicates
        # (e.g., 'person' and 'driver_present' both mean "driver")
        processed_concepts = set()
        DUPLICATE_GROUPS = {
            'person': 'driver',
            'driver_present': 'driver',
            'driver_in_vehicle': 'driver',
            'driver': 'driver',
            'parking_permit': 'permit',
            'permit': 'permit',
        }

        for category, data in merged_results.items():
            # Check for duplicate concepts
            concept = DUPLICATE_GROUPS.get(category)
            if concept:
                if concept in processed_concepts:
                    continue  # Skip duplicate
                processed_concepts.add(concept)

            is_absence = is_absence_based(category)

            # Original raw values (0.0 - 1.0)
            raw_sam3 = data.sam3_confidence
            raw_openai = data.openai_confidence
            raw_merged = data.merged_confidence

            if is_absence:
                # ═══════════════════════════════════════════════════════════════
                # ABSENCE-BASED: Invert all confidences
                # 0% detection → 100% absence confidence (GOOD for violation)
                # ═══════════════════════════════════════════════════════════════
                display_sam3 = int(invert_confidence(raw_sam3) * 100)
                display_openai = int(invert_confidence(raw_openai) * 100)
                display_final = int(invert_confidence(raw_merged) * 100)

                # For absence: HIGH inverted confidence = item is ABSENT = GOOD
                is_detected = display_final >= 70

                # Label shows "No X" when absence is confirmed
                label = get_display_label(category, show_absence=is_detected)

                # Reasoning for absence
                if is_detected:
                    reasoning = f"No {get_display_label(category)} detected - supports violation case"
                else:
                    reasoning = f"Possible {get_display_label(category)} present - manual verification needed"

                items.append({
                    'category': category,
                    'label': label,
                    'detected': is_detected,
                    'confidence': display_final,
                    'sam3_confidence': display_sam3,
                    'openai_confidence': display_openai,
                    'agreement': int(data.agreement_score * 100),
                    'source': data.source_used.value,
                    'is_hallucination_risk': False,  # No hallucination for absence items
                    'is_absence_based': True,
                    'reasoning': reasoning,
                    # Keep original values for debugging/checklist
                    'original_sam3': int(raw_sam3 * 100),
                    'original_openai': int(raw_openai * 100),
                    'original_merged': int(raw_merged * 100),
                })
            else:
                # ═══════════════════════════════════════════════════════════════
                # PRESENCE-BASED: Keep standard logic
                # HIGH confidence = item IS present = GOOD
                # ═══════════════════════════════════════════════════════════════
                display_sam3 = int(raw_sam3 * 100)
                display_openai = int(raw_openai * 100)
                display_final = int(raw_merged * 100)

                is_detected = display_final >= 50
                label = get_display_label(category)

                items.append({
                    'category': category,
                    'label': label,
                    'detected': is_detected,
                    'confidence': display_final,
                    'sam3_confidence': display_sam3,
                    'openai_confidence': display_openai,
                    'agreement': int(data.agreement_score * 100),
                    'source': data.source_used.value,
                    'is_hallucination_risk': data.is_hallucination_risk,
                    'is_absence_based': False,
                    'reasoning': data.reasoning,
                    'original_sam3': display_sam3,
                    'original_openai': display_openai,
                    'original_merged': display_final,
                })

        # Sort: detected items first, then by confidence descending
        items.sort(key=lambda x: (x['detected'], x['confidence']), reverse=True)

        return items


# ═══════════════════════════════════════════════════════════════════════════════════════
# EVIDENCE CHECKLIST GENERATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def determine_checklist_status(
    detection_result: Optional[Dict],
    is_absence_check: bool
) -> str:
    """
    Determine Evidence Checklist item status from detection result.

    Args:
        detection_result: The UI item dict from format_for_ui()
        is_absence_check: Whether this check expects absence (True) or presence (False)

    Returns:
        'passed', 'unverifiable', or 'failed' (matching template expectations)
    """
    if detection_result is None:
        return 'unverifiable'

    confidence = detection_result.get('confidence', 0)
    is_hallucination = detection_result.get('is_hallucination_risk', False)
    is_detected = detection_result.get('detected', False)

    # For both presence and absence checks, 'detected' means the check passed
    # (for absence items, confidence is already inverted, so detected=True means absence confirmed)

    if is_detected and confidence >= 70 and not is_hallucination:
        return 'passed'  # Template expects 'passed' not 'confirmed'
    elif confidence >= 40 or is_hallucination:
        return 'unverifiable'
    else:
        return 'failed'


def generate_evidence_checklist(
    detected_items_ui: List[Dict],
    violation_type: str,
    lang: str = 'en'
) -> Dict:
    """
    Generate Evidence Checklist based on detection results.

    This connects the Detected Items panel to the Evidence Checklist,
    ensuring visual consistency between them.

    Args:
        detected_items_ui: List of UI items from format_for_ui()
        violation_type: 'E9', 'E6', 'E7', 'G7', etc.
        lang: 'en' or 'nl'

    Returns:
        {
            'items': [...],
            'verified_percentage': 80,
            'confirmed_count': 4,
            'total_count': 5,
        }
    """
    checks = VIOLATION_CHECKS.get(violation_type.upper(), VIOLATION_CHECKS['E9'])
    checklist_items = []

    # Create lookup dict for detected items by category
    detection_lookup = {item['category']: item for item in detected_items_ui}

    for check in checks:
        category = check['category']
        is_absence = check['absence']

        # Find the detection result for this category
        detection = detection_lookup.get(category)

        # Also try alternative category names
        if detection is None:
            alt_categories = {
                'person': ['driver_present', 'driver_in_vehicle', 'driver'],
                'driver_present': ['person', 'driver_in_vehicle', 'driver'],
                'parking_permit': ['permit'],
            }
            for alt in alt_categories.get(category, []):
                if alt in detection_lookup:
                    detection = detection_lookup[alt]
                    break

        # Determine status
        status = determine_checklist_status(detection, is_absence)

        # Get confidence (already inverted for absence items in UI)
        # Convert to 0.0-1.0 range for template (template multiplies by 100)
        confidence = (detection.get('confidence', 0) / 100.0) if detection else 0.0

        # Get label based on language
        description = check.get(f'label_{lang}', check['label']) if lang != 'en' else check['label']

        # Template expects these field names:
        # - description (not label)
        # - legal_reference (not reference)
        # - confidence as 0.0-1.0 (not 0-100)
        checklist_items.append({
            'description': description,
            'status': status,
            'legal_reference': check['ref'],
            'confidence': confidence,
            'category': category,
            'is_absence_based': is_absence,
        })

    # Calculate overall verification
    confirmed_count = sum(1 for item in checklist_items if item['status'] == 'passed')
    total_count = len(checklist_items)
    verified_percentage = round((confirmed_count / total_count) * 100) if total_count > 0 else 0

    return {
        'items': checklist_items,
        'verified_percentage': verified_percentage,
        'confirmed_count': confirmed_count,
        'total_count': total_count,
    }


# ═══════════════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════════════

def merge_confidences(
    sam3_confidences: Dict[str, float],
    openai_confidences: Dict[str, float]
) -> Tuple[Dict[str, MergedConfidence], Dict[str, float], List[str], List[Dict]]:
    """
    Convenience function to merge confidences and return all needed data.

    Returns:
        (merged_results, final_scores, hallucination_warnings, ui_items)
    """
    merger = ConfidenceMerger()

    merged_results = merger.merge(sam3_confidences, openai_confidences)
    final_scores = merger.calculate_final_scores(merged_results)
    hallucination_warnings = merger.get_hallucination_warnings(merged_results)
    ui_items = merger.format_for_ui(merged_results)

    return merged_results, final_scores, hallucination_warnings, ui_items


def prepare_detected_items_for_display(
    ui_items: List[Dict],
    include_zero_detection: bool = False
) -> Tuple[List[Dict], List[str]]:
    """
    Filter and prepare detected items for display.

    Args:
        ui_items: List from format_for_ui()
        include_zero_detection: Whether to include items with 0% on both sources

    Returns:
        (filtered_items, not_detected_labels)
    """
    if include_zero_detection:
        return ui_items, []

    shown_items = []
    not_detected_labels = []

    for item in ui_items:
        # Check original values (before inversion)
        orig_sam3 = item.get('original_sam3', 0)
        orig_openai = item.get('original_openai', 0)

        if orig_sam3 > 0 or orig_openai > 0:
            shown_items.append(item)
        else:
            not_detected_labels.append(item['label'])

    return shown_items, not_detected_labels
