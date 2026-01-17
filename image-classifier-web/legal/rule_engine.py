"""
Rule Engine for Legal Compliance Evaluation.
Based on Legal Reasoning Architecture v2.0

This module provides deterministic evaluation of MLLM observations
against legal requirements defined in decision trees.

Layer 3 of the 4-Layer Legal Validation Pipeline:
- Compares evidence against legal article requirements
- Uses pre-defined decision trees
- Produces structured legal assessment with article references

Version: 2.0
"""

from typing import Any, Optional
from .decision_trees import LEGAL_DECISION_TREES, get_decision_tree, get_violation_from_sign


def get_nested_value(data: dict, path: str) -> Any:
    """
    Get a value from a nested dictionary using dot notation.

    Args:
        data: The dictionary to search
        path: Dot-separated path (e.g., "vehicle.license_plate.value")

    Returns:
        The value at the path, or None if not found

    Example:
        >>> data = {"vehicle": {"license_plate": {"value": "AB-123-CD"}}}
        >>> get_nested_value(data, "vehicle.license_plate.value")
        'AB-123-CD'
    """
    if not data or not path:
        return None

    keys = path.split(".")
    value = data

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None

    return value


def normalize_value(value: Any) -> Any:
    """
    Normalize a value for comparison.
    Handles string/boolean conversions and case normalization.

    Args:
        value: The value to normalize

    Returns:
        Normalized value
    """
    if value is None:
        return None

    if isinstance(value, str):
        lower = value.lower().strip()
        # Convert string booleans
        if lower in ("true", "yes"):
            return True
        if lower in ("false", "no"):
            return False
        return lower

    return value


def evaluate_check(check: dict, mllm_output: dict) -> dict:
    """
    Evaluate a single check against MLLM output.

    Args:
        check: Check definition from decision tree
        mllm_output: MLLM analysis output (Layer 2 format)

    Returns:
        Check result dictionary with status and details
    """
    check_result = {
        "check_id": check["check_id"],
        "description": check["description"],
        "description_nl": check.get("description_nl", check["description"]),
        "legal_reference": check["legal_reference"],
        "legal_url": check.get("legal_url"),
        "status": "unknown",
        "actual_value": None,
        "expected_value": check.get("expected_value"),
        "reason": None
    }

    # Get actual value from MLLM output
    source_field = check.get("source_field")
    actual_value = get_nested_value(mllm_output, source_field) if source_field else None
    check_result["actual_value"] = actual_value

    # Handle not visible / unknown cases
    normalized_actual = normalize_value(actual_value)

    if actual_value is None or normalized_actual in ("not_visible", "unknown", "not visible"):
        check_result["status"] = "unverifiable"
        check_result["reason"] = "Not visible in image material"
        return check_result

    # Handle comparison checks (comparing two fields)
    if "compare_with" in check:
        compare_value = get_nested_value(mllm_output, check["compare_with"])
        check_result["compare_value"] = compare_value
        expected_result = check.get("expected_result", "match")

        if compare_value is None:
            check_result["status"] = "unverifiable"
            check_result["reason"] = "Comparison value not available"
        elif expected_result == "mismatch":
            # We expect the values to NOT match
            if normalize_value(actual_value) != normalize_value(compare_value):
                check_result["status"] = "passed"
                check_result["reason"] = f"Values differ as expected: '{actual_value}' vs '{compare_value}'"
            else:
                check_result["status"] = "failed"
                check_result["reason"] = f"Values match but should differ: '{actual_value}'"
        else:
            # We expect the values to match
            if normalize_value(actual_value) == normalize_value(compare_value):
                check_result["status"] = "passed"
                check_result["reason"] = f"Values match: '{actual_value}'"
            else:
                check_result["status"] = "failed"
                check_result["reason"] = f"Values differ: '{actual_value}' vs '{compare_value}'"

        return check_result

    # Handle direct comparison checks
    expected_value = check.get("expected_value")
    if normalize_value(actual_value) == normalize_value(expected_value):
        check_result["status"] = "passed"
        check_result["reason"] = f"Value matches expected: {actual_value}"
    else:
        check_result["status"] = "failed"
        check_result["reason"] = f"Expected '{expected_value}', got '{actual_value}'"

    return check_result


def evaluate_legal_compliance(mllm_output: dict, violation_code: str) -> dict:
    """
    Deterministic evaluation of MLLM output against legal requirements.
    Returns structured legal assessment with article references.

    This is the main function for Layer 3 of the pipeline.

    Args:
        mllm_output: MLLM analysis output (Layer 2 format)
        violation_code: The violation type (E6, E7, E9, G7, ELECTRIC_CHARGING)

    Returns:
        Structured legal assessment dictionary

    Example:
        >>> mllm_output = {
        ...     "traffic_sign": {"sign_code": "E9", "detected": True},
        ...     "windshield_items": {"permit": "no"},
        ...     "environment": {"driver_present": False}
        ... }
        >>> result = evaluate_legal_compliance(mllm_output, "E9")
        >>> result["all_checks_passed"]
        True
    """
    tree = get_decision_tree(violation_code)

    if not tree:
        return {
            "error": f"Unknown violation code: {violation_code}",
            "violation_code": violation_code,
            "checks": [],
            "all_checks_passed": False,
            "verification_score": 0.0,
            "requires_manual_review": True
        }

    results = {
        "violation_code": violation_code,
        "violation_name": tree["name"],
        "violation_name_en": tree.get("name_en", tree["name"]),
        "checks": [],
        "all_checks_passed": True,
        "passed_checks": [],
        "failed_checks": [],
        "unverifiable_checks": [],
        "legal_references": {
            "violation_article": tree["violation_article"],
            "violation_article_url": tree.get("violation_article_url"),
            "violation_text_nl": tree.get("violation_text_nl"),
            "violation_text_en": tree.get("violation_text_en"),
            "wegslepen_basis": tree["wegslepen_basis"],
            "wegslepen_url": tree.get("wegslepen_url"),
            "feit_code": tree["feit_code"]
        }
    }

    # Evaluate each required check
    for check in tree["required_checks"]:
        check_result = evaluate_check(check, mllm_output)
        results["checks"].append(check_result)

        if check_result["status"] == "passed":
            results["passed_checks"].append(check_result["check_id"])
        elif check_result["status"] == "failed":
            results["failed_checks"].append(check_result["check_id"])
            results["all_checks_passed"] = False
        elif check_result["status"] == "unverifiable":
            results["unverifiable_checks"].append(check_result["check_id"])

    # Calculate verification score
    total_checks = len(results["checks"])
    passed_count = len(results["passed_checks"])
    unverifiable_count = len(results["unverifiable_checks"])

    if total_checks > 0:
        # Passed checks contribute fully, unverifiable contribute 0.5
        score = (passed_count + (unverifiable_count * 0.5)) / total_checks
        results["verification_score"] = round(score, 2)
    else:
        results["verification_score"] = 0.0

    # Determine if manual review is required
    results["requires_manual_review"] = (
        unverifiable_count > 0 or
        not results["all_checks_passed"]
    )

    return results


def auto_detect_violation(mllm_output: dict) -> Optional[str]:
    """
    Automatically detect the violation type from MLLM output based on detected sign.

    Args:
        mllm_output: MLLM analysis output (Layer 2 format)

    Returns:
        Detected violation code or None if not determinable
    """
    sign_code = get_nested_value(mllm_output, "traffic_sign.sign_code")

    if not sign_code:
        return None

    # Handle case variations
    sign_code_upper = str(sign_code).upper().strip()

    return get_violation_from_sign(sign_code_upper)


def evaluate_with_auto_detection(mllm_output: dict, fallback_code: str = None) -> dict:
    """
    Evaluate legal compliance with automatic violation type detection.

    First attempts to detect violation type from the traffic sign in MLLM output.
    Falls back to provided code if detection fails.

    Args:
        mllm_output: MLLM analysis output (Layer 2 format)
        fallback_code: Fallback violation code if auto-detection fails

    Returns:
        Structured legal assessment dictionary
    """
    detected_code = auto_detect_violation(mllm_output)

    if detected_code:
        result = evaluate_legal_compliance(mllm_output, detected_code)
        result["violation_auto_detected"] = True
        result["detected_from_sign"] = get_nested_value(mllm_output, "traffic_sign.sign_code")
        return result

    if fallback_code:
        result = evaluate_legal_compliance(mllm_output, fallback_code)
        result["violation_auto_detected"] = False
        result["fallback_used"] = True
        return result

    return {
        "error": "Could not determine violation type",
        "violation_code": None,
        "checks": [],
        "all_checks_passed": False,
        "verification_score": 0.0,
        "requires_manual_review": True,
        "violation_auto_detected": False,
        "reason": "No traffic sign detected and no fallback code provided"
    }


def get_supporting_articles(violation_code: str) -> list:
    """
    Get all supporting legal articles for a violation type.

    Args:
        violation_code: The violation type

    Returns:
        List of article reference dictionaries
    """
    tree = get_decision_tree(violation_code)

    if not tree:
        return []

    articles = []

    # Add main violation article
    articles.append({
        "type": "primary",
        "article": tree["violation_article"],
        "url": tree.get("violation_article_url"),
        "text_nl": tree.get("violation_text_nl"),
        "text_en": tree.get("violation_text_en")
    })

    # Add wegslepen basis
    articles.append({
        "type": "wegslepen",
        "article": tree["wegslepen_basis"],
        "url": tree.get("wegslepen_url")
    })

    # Add articles from individual checks
    for check in tree.get("required_checks", []):
        if check.get("legal_reference"):
            articles.append({
                "type": "supporting",
                "article": check["legal_reference"],
                "url": check.get("legal_url"),
                "check_id": check["check_id"]
            })

    return articles


def format_evidence_checklist(evaluation_result: dict, language: str = "en") -> list:
    """
    Format the evaluation checks as an evidence checklist for UI display.

    Args:
        evaluation_result: Result from evaluate_legal_compliance()
        language: "en" or "nl"

    Returns:
        List of checklist items for UI
    """
    checklist = []

    for check in evaluation_result.get("checks", []):
        desc_key = "description_nl" if language == "nl" else "description"

        item = {
            "id": check["check_id"],
            "description": check.get(desc_key, check["description"]),
            "status": check["status"],
            "legal_reference": check["legal_reference"],
            "legal_url": check.get("legal_url"),
            "confidence": None,  # Can be filled from MLLM confidence scores
            "source": "image"
        }

        # Determine icon/styling based on status
        if check["status"] == "passed":
            item["icon"] = "check"
            item["style"] = "success"
        elif check["status"] == "failed":
            item["icon"] = "x"
            item["style"] = "error"
        else:  # unverifiable
            item["icon"] = "question"
            item["style"] = "warning"

        checklist.append(item)

    return checklist
