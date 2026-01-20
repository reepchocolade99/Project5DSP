"""
Prompts Module for Legal Reasoning Architecture v2.0

This module contains the MLLM prompts for the 4-layer legal validation pipeline:

- Layer 2: Objective Image Analysis (observation only, no legal interpretation)
- Layer 4: Officer Validation & Citation Generation (comparison and verification)

Layers 1 (Document Parser) and 3 (Rule Engine) do not require MLLM prompts.

Version: 2.0
"""

# Layer 2: Objective Analysis
from .layer2_objective import (
    LAYER2_PROMPT_EN,
    LAYER2_PROMPT_NL,
    LAYER2_OUTPUT_SCHEMA,
    get_layer2_prompt,
    build_layer2_message
)

# Layer 4: Verification
from .layer4_verification import (
    LAYER4_PROMPT_EN,
    LAYER4_PROMPT_NL,
    LAYER4_OUTPUT_SCHEMA,
    get_layer4_prompt,
    build_layer4_prompt,
    parse_layer4_response,
    calculate_observation_match_score,
    merge_verification_with_evaluation
)


__version__ = "2.0.0"

__all__ = [
    # Layer 2
    "LAYER2_PROMPT_EN",
    "LAYER2_PROMPT_NL",
    "LAYER2_OUTPUT_SCHEMA",
    "get_layer2_prompt",
    "build_layer2_message",

    # Layer 4
    "LAYER4_PROMPT_EN",
    "LAYER4_PROMPT_NL",
    "LAYER4_OUTPUT_SCHEMA",
    "get_layer4_prompt",
    "build_layer4_prompt",
    "parse_layer4_response",
    "calculate_observation_match_score",
    "merge_verification_with_evaluation",
]
