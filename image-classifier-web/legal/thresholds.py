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
    Determine the recommended action based on Evidence Checklist results.

    Simple, clear logic based on Evidence Checklist item statuses:
    - APPROVED: All items are green checkmarks (passed)
    - MANUAL REVIEW: At least one question mark (unverifiable/hallucination risk)
    - REJECTED: At least one X mark (failed - neither LLM could verify)

    Args:
        results: Combined validation results dictionary containing:
            - evidence_checklist: dict with 'items' list, each item has 'status'
            - overall_confidence: float (0.0-1.0) for display
            - verification_score: float (0.0-1.0) for display

    Returns:
        Action recommendation dictionary:
            - action: "approve" | "manual_review" | "reject"
            - reason: Human-readable reason
            - requires_manual_review: bool
            - review_points: list of specific items needing review
            - confidence_level: "high" | "medium" | "low"
    """
    # Extract Evidence Checklist
    evidence_checklist = results.get("evidence_checklist", {})
    checklist_items = evidence_checklist.get("items", [])

    # Extract scores for display purposes
    confidence = results.get("overall_confidence", 0.0)
    verification_score = results.get("verification_score", 0.0)

    # Count statuses from Evidence Checklist
    passed_count = 0
    unverifiable_count = 0
    failed_count = 0

    failed_items = []
    unverifiable_items = []

    for item in checklist_items:
        status = item.get("status", "unverifiable")
        description = item.get("description", "Unknown check")

        if status == "passed":
            passed_count += 1
        elif status == "unverifiable":
            unverifiable_count += 1
            unverifiable_items.append(description)
        elif status == "failed":
            failed_count += 1
            failed_items.append(description)

    total_count = len(checklist_items)

    # ═══════════════════════════════════════════════════════════════════════════
    # DECISION LOGIC (Priority Order)
    # ═══════════════════════════════════════════════════════════════════════════

    # 1. REJECTED - At least one X mark (failed)
    #    Both LLMs couldn't find required evidence
    if failed_count > 0:
        return {
            "action": "reject",
            "reason": f"{failed_count} required evidence item(s) could not be verified",
            "requires_manual_review": True,
            "review_points": [f"Not detected: {item}" for item in failed_items],
            "confidence_level": "low",
            "scores": {
                "overall_confidence": confidence,
                "verification_score": verification_score,
                "passed": passed_count,
                "unverifiable": unverifiable_count,
                "failed": failed_count,
                "total": total_count
            }
        }

    # 2. MANUAL REVIEW - At least one question mark (unverifiable/hallucination risk)
    #    One LLM detected but the other didn't confirm
    if unverifiable_count > 0:
        return {
            "action": "manual_review",
            "reason": f"{unverifiable_count} item(s) require manual verification",
            "requires_manual_review": True,
            "review_points": [f"Needs verification: {item}" for item in unverifiable_items],
            "confidence_level": "medium",
            "scores": {
                "overall_confidence": confidence,
                "verification_score": verification_score,
                "passed": passed_count,
                "unverifiable": unverifiable_count,
                "failed": failed_count,
                "total": total_count
            }
        }

    # 3. APPROVED - All items are green checkmarks (passed)
    #    Both LLMs confirmed all required evidence
    if passed_count == total_count and total_count > 0:
        return {
            "action": "approve",
            "reason": "All evidence items verified by both detection systems",
            "requires_manual_review": False,
            "review_points": [],
            "confidence_level": "high",
            "scores": {
                "overall_confidence": confidence,
                "verification_score": verification_score,
                "passed": passed_count,
                "unverifiable": unverifiable_count,
                "failed": failed_count,
                "total": total_count
            }
        }

    # 4. Fallback - No checklist items (shouldn't happen, but handle gracefully)
    return {
        "action": "manual_review",
        "reason": "No evidence checklist available for evaluation",
        "requires_manual_review": True,
        "review_points": ["Evidence checklist is empty or missing"],
        "confidence_level": "low",
        "scores": {
            "overall_confidence": confidence,
            "verification_score": verification_score,
            "passed": 0,
            "unverifiable": 0,
            "failed": 0,
            "total": 0
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
            "label_en": "Evidence Verified",
            "label_nl": "Bewijs Geverifieerd",
            "icon": "check-circle",
            "color": "success",
            "description_en": "Evidence meets validation criteria - ready for officer review",
            "description_nl": "Bewijs voldoet aan validatiecriteria - klaar voor beoordeling"
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
