"""
Layer 4: Officer Validation & Citation Generation Prompt
Based on Legal Reasoning Architecture v2.0

This layer compares the MLLM image analysis with the police officer's
observation and produces the final verification output.

The police observation is treated as the GOLD STANDARD - in case of
discrepancies, the officer's observation takes precedence.

Version: 2.0
"""

import json
from typing import Optional


LAYER4_PROMPT_EN = """You are a legal verification assistant. Compare the image analysis with the police observation and identify matches and discrepancies.

IMAGE ANALYSIS RESULTS:
{mllm_analysis}

RULE ENGINE RESULTS:
{rule_engine_results}

POLICE OBSERVATION (Redenen van wetenschap):
{officer_observation}

TASKS:
1. Compare the image analysis with the police observation
2. Identify matches and discrepancies
3. Assess the overall consistency between sources

OUTPUT FORMAT: JSON (valid JSON only, no markdown code blocks)
{{
  "verification": {{
    "observation_supported": true | false,
    "matching_elements": [
      {{"element": "string describing what matches", "source": "image | officer | both"}}
    ],
    "discrepancies": [
      {{
        "item": "string describing the discrepancy",
        "image_says": "what the image analysis found",
        "officer_says": "what the officer reported",
        "severity": "minor | major"
      }}
    ],
    "missing_from_image": ["items mentioned by officer but not visible in images"],
    "overall_confidence": 0.0-1.0
  }},
  "recommendation": {{
    "action": "approve | manual_review | reject",
    "reason": "string explaining the recommendation",
    "manual_review_points": ["specific items needing human review"]
  }}
}}

IMPORTANT RULES:
- The police observation is the GOLD STANDARD
- When in doubt: recommendation = "manual_review"
- "major" discrepancies: contradictions that affect legal validity
- "minor" discrepancies: small differences that don't affect the case
- If officer says something the image cannot confirm: mark as "missing_from_image", NOT as discrepancy
- Only mark as "approve" if image analysis SUPPORTS the officer observation
"""


LAYER4_PROMPT_NL = """Je bent een juridische verificatie-assistent. Vergelijk de beeldanalyse met de politie-observatie en identificeer overeenkomsten en discrepanties.

BEELDANALYSE RESULTATEN:
{mllm_analysis}

REGEL ENGINE RESULTATEN:
{rule_engine_results}

POLITIE OBSERVATIE (Redenen van wetenschap):
{officer_observation}

TAKEN:
1. Vergelijk de beeldanalyse met de politie-observatie
2. Identificeer overeenkomsten en discrepanties
3. Beoordeel de algehele consistentie tussen bronnen

OUTPUT FORMAAT: JSON (alleen geldige JSON, geen markdown codeblokken)
{{
  "verification": {{
    "observation_supported": true | false,
    "matching_elements": [
      {{"element": "beschrijving van wat overeenkomt", "source": "image | officer | both"}}
    ],
    "discrepancies": [
      {{
        "item": "beschrijving van de discrepantie",
        "image_says": "wat de beeldanalyse vond",
        "officer_says": "wat de agent rapporteerde",
        "severity": "minor | major"
      }}
    ],
    "missing_from_image": ["items genoemd door agent maar niet zichtbaar in afbeeldingen"],
    "overall_confidence": 0.0-1.0
  }},
  "recommendation": {{
    "action": "approve | manual_review | reject",
    "reason": "uitleg van de aanbeveling",
    "manual_review_points": ["specifieke punten die menselijke beoordeling nodig hebben"]
  }}
}}

BELANGRIJKE REGELS:
- De politie-observatie is de GOUDEN STANDAARD
- Bij twijfel: recommendation = "manual_review"
- "major" discrepanties: tegenstrijdigheden die juridische geldigheid beïnvloeden
- "minor" discrepanties: kleine verschillen die de zaak niet beïnvloeden
- Als agent iets zegt dat beeld niet kan bevestigen: markeer als "missing_from_image", NIET als discrepantie
- Markeer alleen als "approve" als beeldanalyse de agent-observatie ONDERSTEUNT
"""


def get_layer4_prompt(language: str = "en") -> str:
    """
    Get the Layer 4 verification prompt in the specified language.

    Args:
        language: "en" for English, "nl" for Dutch

    Returns:
        The prompt template string
    """
    if language.lower() == "nl":
        return LAYER4_PROMPT_NL
    return LAYER4_PROMPT_EN


def build_layer4_prompt(
    mllm_analysis: dict,
    rule_engine_results: dict,
    officer_observation: str,
    language: str = "en"
) -> str:
    """
    Build the complete Layer 4 verification prompt with all inputs.

    Args:
        mllm_analysis: Output from Layer 2 (MLLM objective analysis)
        rule_engine_results: Output from Layer 3 (Rule Engine evaluation)
        officer_observation: Original officer observation text (Redenen van wetenschap)
        language: "en" or "nl"

    Returns:
        Complete prompt string ready for MLLM
    """
    template = get_layer4_prompt(language)

    # Format the inputs as JSON strings for the prompt
    mllm_json = json.dumps(mllm_analysis, indent=2, ensure_ascii=False)
    rule_json = json.dumps(rule_engine_results, indent=2, ensure_ascii=False)

    return template.format(
        mllm_analysis=mllm_json,
        rule_engine_results=rule_json,
        officer_observation=officer_observation or "[No officer observation provided]"
    )


def parse_layer4_response(response_text: str) -> dict:
    """
    Parse the Layer 4 MLLM response into a structured dictionary.

    Args:
        response_text: Raw text response from MLLM

    Returns:
        Parsed verification result dictionary
    """
    # Try to extract JSON from the response
    try:
        # Handle markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        else:
            json_str = response_text.strip()

        return json.loads(json_str)

    except json.JSONDecodeError:
        # Return a safe default structure
        return {
            "verification": {
                "observation_supported": False,
                "matching_elements": [],
                "discrepancies": [],
                "missing_from_image": [],
                "overall_confidence": 0.0
            },
            "recommendation": {
                "action": "manual_review",
                "reason": "Failed to parse MLLM verification response",
                "manual_review_points": ["Response parsing error - manual review required"]
            },
            "parse_error": True,
            "raw_response": response_text[:500]  # Truncate for logging
        }


def calculate_observation_match_score(verification_result: dict) -> float:
    """
    Calculate the observation match score from verification results.

    Args:
        verification_result: Parsed Layer 4 verification output

    Returns:
        Match score (0.0-1.0)
    """
    verification = verification_result.get("verification", {})

    # Base score from overall confidence
    base_score = verification.get("overall_confidence", 0.5)

    # Adjust based on discrepancies
    discrepancies = verification.get("discrepancies", [])
    major_count = sum(1 for d in discrepancies if d.get("severity") == "major")
    minor_count = sum(1 for d in discrepancies if d.get("severity") == "minor")

    # Deduct for discrepancies
    penalty = (major_count * 0.15) + (minor_count * 0.05)
    adjusted_score = max(0.0, base_score - penalty)

    # Boost if observation is supported
    if verification.get("observation_supported", False):
        adjusted_score = min(1.0, adjusted_score + 0.1)

    return round(adjusted_score, 2)


def merge_verification_with_evaluation(
    rule_engine_result: dict,
    verification_result: dict
) -> dict:
    """
    Merge Rule Engine evaluation with Layer 4 verification results.

    Creates the combined result structure used for action determination.

    Args:
        rule_engine_result: Output from evaluate_legal_compliance()
        verification_result: Parsed Layer 4 verification output

    Returns:
        Merged result dictionary
    """
    verification = verification_result.get("verification", {})
    recommendation = verification_result.get("recommendation", {})

    # Calculate observation match score
    match_score = calculate_observation_match_score(verification_result)

    # Merge the results
    merged = {
        # From Rule Engine
        "violation_code": rule_engine_result.get("violation_code"),
        "violation_name": rule_engine_result.get("violation_name"),
        "all_checks_passed": rule_engine_result.get("all_checks_passed", False),
        "verification_score": rule_engine_result.get("verification_score", 0.0),
        "unverifiable_checks": rule_engine_result.get("unverifiable_checks", []),
        "legal_references": rule_engine_result.get("legal_references", {}),
        "checks": rule_engine_result.get("checks", []),

        # From Layer 4 Verification
        "observation_supported": verification.get("observation_supported", False),
        "observation_match_score": match_score,
        "matching_elements": verification.get("matching_elements", []),
        "discrepancies": verification.get("discrepancies", []),
        "missing_from_image": verification.get("missing_from_image", []),

        # Combined confidence
        "overall_confidence": (
            rule_engine_result.get("verification_score", 0.0) * 0.5 +
            match_score * 0.5
        ),

        # Recommendation from Layer 4
        "layer4_recommendation": recommendation
    }

    return merged


# Expected output schema for validation
LAYER4_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["verification", "recommendation"],
    "properties": {
        "verification": {
            "type": "object",
            "required": ["observation_supported", "matching_elements", "discrepancies", "overall_confidence"],
            "properties": {
                "observation_supported": {"type": "boolean"},
                "matching_elements": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["element", "source"],
                        "properties": {
                            "element": {"type": "string"},
                            "source": {"type": "string", "enum": ["image", "officer", "both"]}
                        }
                    }
                },
                "discrepancies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["item", "image_says", "officer_says", "severity"],
                        "properties": {
                            "item": {"type": "string"},
                            "image_says": {"type": "string"},
                            "officer_says": {"type": "string"},
                            "severity": {"type": "string", "enum": ["minor", "major"]}
                        }
                    }
                },
                "missing_from_image": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "overall_confidence": {"type": "number", "minimum": 0, "maximum": 1}
            }
        },
        "recommendation": {
            "type": "object",
            "required": ["action", "reason"],
            "properties": {
                "action": {"type": "string", "enum": ["approve", "manual_review", "reject"]},
                "reason": {"type": "string"},
                "manual_review_points": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    }
}
