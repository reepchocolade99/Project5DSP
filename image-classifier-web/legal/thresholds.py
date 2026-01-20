"""
Validation Thresholds for Legal Compliance Evaluation.
Based on Legal Reasoning Architecture v2.0

This module defines the thresholds for automatic approval, manual review,
and rejection of parking violation cases.

Version: 2.0
"""

from typing import List, Optional


VALIDATION_THRESHOLDS = {
    "auto_approve": {
        "overall_confidence": 0.85,
        "observation_match_score": 0.90,
        "max_unverifiable_checks": 0,
        "max_minor_discrepancies": 1,
        "max_major_discrepancies": 0,
        "min_verification_score": 0.85
    },
    "manual_review": {
        "overall_confidence": 0.70,
        "observation_match_score": 0.75,
        "max_unverifiable_checks": 2,
        "max_minor_discrepancies": 2,
        "max_major_discrepancies": 1,
        "min_verification_score": 0.60
    },
    "auto_reject": {
        "overall_confidence_below": 0.50,
        "observation_match_score_below": 0.50,
        "min_verification_score_below": 0.40
    }
}


# Confidence score weights for calculating overall confidence
CONFIDENCE_WEIGHTS = {
    "object_detection": 0.30,
    "text_recognition": 0.25,
    "legal_reasoning": 0.25,
    "observation_match": 0.20
}


def determine_action(results: dict) -> dict:
    """
    Determine the recommended action based on validation results.

    This function evaluates the combined results from MLLM analysis,
    rule engine evaluation, and officer observation comparison to
    recommend an action: approve, manual_review, or reject.

    Args:
        results: Combined validation results dictionary containing:
            - overall_confidence: float (0.0-1.0)
            - observation_match_score: float (0.0-1.0)
            - verification_score: float (0.0-1.0) from rule engine
            - unverifiable_checks: list of check IDs
            - discrepancies: list of discrepancy dicts with 'severity' key
            - all_checks_passed: bool

    Returns:
        Action recommendation dictionary:
            - action: "approve" | "manual_review" | "reject"
            - reason: Human-readable reason
            - requires_manual_review: bool
            - review_points: list of specific items needing review
            - confidence_level: "high" | "medium" | "low"

    Example:
        >>> results = {
        ...     "overall_confidence": 0.92,
        ...     "observation_match_score": 0.95,
        ...     "verification_score": 0.88,
        ...     "unverifiable_checks": [],
        ...     "discrepancies": [],
        ...     "all_checks_passed": True
        ... }
        >>> action = determine_action(results)
        >>> action["action"]
        'approve'
    """
    # Extract values with defaults
    confidence = results.get("overall_confidence", 0.0)
    match_score = results.get("observation_match_score", 0.0)
    verification_score = results.get("verification_score", 0.0)
    unverifiable = results.get("unverifiable_checks", [])
    discrepancies = results.get("discrepancies", [])
    all_checks_passed = results.get("all_checks_passed", False)

    # Count discrepancies by severity
    minor_discrepancies = sum(
        1 for d in discrepancies
        if d.get("severity") == "minor"
    )
    major_discrepancies = sum(
        1 for d in discrepancies
        if d.get("severity") == "major"
    )
    unverifiable_count = len(unverifiable) if isinstance(unverifiable, list) else 0

    thresholds = VALIDATION_THRESHOLDS

    # Check auto-reject first (lowest priority threshold)
    reject_thresholds = thresholds["auto_reject"]
    if (confidence < reject_thresholds["overall_confidence_below"] or
        match_score < reject_thresholds["observation_match_score_below"] or
        verification_score < reject_thresholds["min_verification_score_below"]):

        reject_reasons = []
        if confidence < reject_thresholds["overall_confidence_below"]:
            reject_reasons.append(f"Overall confidence ({confidence:.0%}) below minimum ({reject_thresholds['overall_confidence_below']:.0%})")
        if match_score < reject_thresholds["observation_match_score_below"]:
            reject_reasons.append(f"Officer match score ({match_score:.0%}) below minimum ({reject_thresholds['observation_match_score_below']:.0%})")
        if verification_score < reject_thresholds["min_verification_score_below"]:
            reject_reasons.append(f"Verification score ({verification_score:.0%}) below minimum ({reject_thresholds['min_verification_score_below']:.0%})")

        return {
            "action": "reject",
            "reason": "Confidence scores below minimum thresholds",
            "requires_manual_review": True,
            "review_points": reject_reasons,
            "confidence_level": "low",
            "scores": {
                "overall_confidence": confidence,
                "observation_match_score": match_score,
                "verification_score": verification_score
            }
        }

    # Check auto-approve (highest priority threshold)
    approve_thresholds = thresholds["auto_approve"]
    can_auto_approve = (
        confidence >= approve_thresholds["overall_confidence"] and
        match_score >= approve_thresholds["observation_match_score"] and
        verification_score >= approve_thresholds["min_verification_score"] and
        unverifiable_count <= approve_thresholds["max_unverifiable_checks"] and
        minor_discrepancies <= approve_thresholds["max_minor_discrepancies"] and
        major_discrepancies <= approve_thresholds["max_major_discrepancies"] and
        all_checks_passed
    )

    if can_auto_approve:
        return {
            "action": "approve",
            "reason": "All validation criteria met",
            "requires_manual_review": False,
            "review_points": [],
            "confidence_level": "high",
            "scores": {
                "overall_confidence": confidence,
                "observation_match_score": match_score,
                "verification_score": verification_score
            }
        }

    # Default to manual review
    review_thresholds = thresholds["manual_review"]
    review_points = []

    # Identify specific review points
    if unverifiable_count > 0:
        review_points.append(
            f"{unverifiable_count} check(s) could not be verified from images: {', '.join(unverifiable)}"
        )

    if major_discrepancies > 0:
        review_points.append(
            f"{major_discrepancies} major discrepancy(ies) between image and officer observation"
        )

    if minor_discrepancies > review_thresholds["max_minor_discrepancies"]:
        review_points.append(
            f"{minor_discrepancies} minor discrepancy(ies) noted (threshold: {review_thresholds['max_minor_discrepancies']})"
        )

    if not all_checks_passed:
        review_points.append("Not all required legal checks passed")

    if confidence < approve_thresholds["overall_confidence"]:
        review_points.append(
            f"Overall confidence ({confidence:.0%}) below auto-approve threshold ({approve_thresholds['overall_confidence']:.0%})"
        )

    if match_score < approve_thresholds["observation_match_score"]:
        review_points.append(
            f"Officer observation match ({match_score:.0%}) below auto-approve threshold ({approve_thresholds['observation_match_score']:.0%})"
        )

    # Determine confidence level
    if confidence >= 0.80 and match_score >= 0.80:
        confidence_level = "medium-high"
    elif confidence >= 0.70 and match_score >= 0.70:
        confidence_level = "medium"
    else:
        confidence_level = "medium-low"

    return {
        "action": "manual_review",
        "reason": "Results require human verification",
        "requires_manual_review": True,
        "review_points": review_points,
        "confidence_level": confidence_level,
        "scores": {
            "overall_confidence": confidence,
            "observation_match_score": match_score,
            "verification_score": verification_score
        }
    }


def calculate_overall_confidence(
    object_detection: float = 0.0,
    text_recognition: float = 0.0,
    legal_reasoning: float = 0.0,
    observation_match: float = 0.0
) -> float:
    """
    Calculate weighted overall confidence score.

    Args:
        object_detection: Confidence in object detection (0.0-1.0)
        text_recognition: Confidence in text/plate recognition (0.0-1.0)
        legal_reasoning: Confidence in legal rule matching (0.0-1.0)
        observation_match: Confidence in officer observation match (0.0-1.0)

    Returns:
        Weighted overall confidence score (0.0-1.0)
    """
    weights = CONFIDENCE_WEIGHTS

    weighted_sum = (
        object_detection * weights["object_detection"] +
        text_recognition * weights["text_recognition"] +
        legal_reasoning * weights["legal_reasoning"] +
        observation_match * weights["observation_match"]
    )

    return round(weighted_sum, 2)


def get_confidence_label(confidence: float) -> str:
    """
    Get a human-readable label for a confidence score.

    Args:
        confidence: Confidence score (0.0-1.0)

    Returns:
        Label string: "Very High", "High", "Medium", "Low", "Very Low"
    """
    if confidence >= 0.90:
        return "Very High"
    elif confidence >= 0.80:
        return "High"
    elif confidence >= 0.70:
        return "Medium"
    elif confidence >= 0.50:
        return "Low"
    else:
        return "Very Low"


def get_confidence_color(confidence: float) -> str:
    """
    Get a CSS color class for a confidence score.

    Args:
        confidence: Confidence score (0.0-1.0)

    Returns:
        CSS color class string
    """
    if confidence >= 0.85:
        return "success"  # Green
    elif confidence >= 0.70:
        return "warning"  # Yellow/Orange
    else:
        return "danger"   # Red


def validate_scores(scores: dict) -> dict:
    """
    Validate and normalize confidence scores.

    Ensures all scores are within 0.0-1.0 range and
    provides defaults for missing values.

    Args:
        scores: Dictionary of score names to values

    Returns:
        Validated scores dictionary
    """
    validated = {}

    for key, value in scores.items():
        if value is None:
            validated[key] = 0.0
        elif isinstance(value, (int, float)):
            validated[key] = max(0.0, min(1.0, float(value)))
        else:
            validated[key] = 0.0

    return validated


def format_action_for_ui(action_result: dict, language: str = "en") -> dict:
    """
    Format the action result for UI display.

    Args:
        action_result: Result from determine_action()
        language: "en" or "nl"

    Returns:
        UI-formatted action dictionary
    """
    action = action_result.get("action", "manual_review")

    # Action labels and styling
    action_config = {
        "approve": {
            "label_en": "Approved",
            "label_nl": "Goedgekeurd",
            "icon": "check-circle",
            "color": "success",
            "description_en": "Case meets all validation criteria",
            "description_nl": "Zaak voldoet aan alle validatiecriteria"
        },
        "manual_review": {
            "label_en": "Manual Review Required",
            "label_nl": "Handmatige beoordeling vereist",
            "icon": "eye",
            "color": "warning",
            "description_en": "Case requires human verification",
            "description_nl": "Zaak vereist menselijke verificatie"
        },
        "reject": {
            "label_en": "Rejected",
            "label_nl": "Afgewezen",
            "icon": "x-circle",
            "color": "danger",
            "description_en": "Case does not meet minimum criteria",
            "description_nl": "Zaak voldoet niet aan minimale criteria"
        }
    }

    config = action_config.get(action, action_config["manual_review"])
    lang_suffix = "_nl" if language == "nl" else "_en"

    return {
        "action": action,
        "label": config.get(f"label{lang_suffix}"),
        "icon": config["icon"],
        "color": config["color"],
        "description": config.get(f"description{lang_suffix}"),
        "reason": action_result.get("reason"),
        "review_points": action_result.get("review_points", []),
        "confidence_level": action_result.get("confidence_level"),
        "scores": action_result.get("scores", {})
    }


def get_threshold_info() -> dict:
    """
    Get information about current thresholds for documentation/UI.

    Returns:
        Dictionary describing all thresholds
    """
    return {
        "auto_approve": {
            "description": "Cases meeting all these criteria are automatically approved",
            "criteria": VALIDATION_THRESHOLDS["auto_approve"]
        },
        "manual_review": {
            "description": "Cases meeting these minimum criteria go to manual review",
            "criteria": VALIDATION_THRESHOLDS["manual_review"]
        },
        "auto_reject": {
            "description": "Cases below these thresholds are automatically flagged for rejection",
            "criteria": VALIDATION_THRESHOLDS["auto_reject"]
        },
        "weights": {
            "description": "Weights used to calculate overall confidence",
            "values": CONFIDENCE_WEIGHTS
        }
    }
