"""
Legal Reasoning Module for Amsterdam Parking Enforcement System.
Based on Legal Reasoning Architecture v2.0

This module provides hallucination-free, article-referenced legal validation
for parking violation cases.

Main Components:
- Decision Trees: Deterministic legal requirement definitions
- Rule Engine: Evaluates evidence against legal requirements
- Templates: Pre-defined legal statement generators
- Thresholds: Validation and action determination

Usage:
    from legal import (
        evaluate_legal_compliance,
        generate_legal_statement,
        determine_action,
        LEGAL_DECISION_TREES
    )

    # Evaluate MLLM output against legal requirements
    result = evaluate_legal_compliance(mllm_output, "E9")

    # Generate legal statement
    statement = generate_legal_statement("E9", context, "nl")

    # Determine recommended action
    action = determine_action(result)

Version: 2.0
"""

# Decision Trees
from .decision_trees import (
    LEGAL_DECISION_TREES,
    SIGN_CODE_TO_VIOLATION,
    LEGAL_SOURCES,
    get_decision_tree,
    get_violation_from_sign,
    get_all_violation_codes
)

# Rule Engine
from .rule_engine import (
    evaluate_legal_compliance,
    evaluate_with_auto_detection,
    auto_detect_violation,
    get_nested_value,
    get_supporting_articles,
    format_evidence_checklist
)

# Legal Statement Templates
from .templates import (
    LEGAL_TEMPLATES,
    LEGAL_CONCLUSION_TEMPLATES,
    generate_legal_statement,
    get_legal_conclusion,
    generate_full_legal_output,
    get_available_templates
)

# Validation Thresholds
from .thresholds import (
    VALIDATION_THRESHOLDS,
    CONFIDENCE_WEIGHTS,
    determine_action,
    calculate_overall_confidence,
    get_confidence_label,
    get_confidence_color,
    validate_scores,
    format_action_for_ui,
    get_threshold_info
)


__version__ = "2.0.0"
__author__ = "Amsterdam Parking Enforcement AI Team"

__all__ = [
    # Decision Trees
    "LEGAL_DECISION_TREES",
    "SIGN_CODE_TO_VIOLATION",
    "LEGAL_SOURCES",
    "get_decision_tree",
    "get_violation_from_sign",
    "get_all_violation_codes",

    # Rule Engine
    "evaluate_legal_compliance",
    "evaluate_with_auto_detection",
    "auto_detect_violation",
    "get_nested_value",
    "get_supporting_articles",
    "format_evidence_checklist",

    # Templates
    "LEGAL_TEMPLATES",
    "LEGAL_CONCLUSION_TEMPLATES",
    "generate_legal_statement",
    "get_legal_conclusion",
    "generate_full_legal_output",
    "get_available_templates",

    # Thresholds
    "VALIDATION_THRESHOLDS",
    "CONFIDENCE_WEIGHTS",
    "determine_action",
    "calculate_overall_confidence",
    "get_confidence_label",
    "get_confidence_color",
    "validate_scores",
    "format_action_for_ui",
    "get_threshold_info",
]
