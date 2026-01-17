"""
Legal Statement Templates for Amsterdam Parking Enforcement.
Based on Legal Reasoning Architecture v2.0

This module provides pre-defined templates for generating legally
substantiated statements in Dutch and English.

Version: 2.0
"""

from typing import Optional


LEGAL_TEMPLATES = {
    "E6": {
        "template_nl": """Ik zag dat het voertuig geparkeerd stond op een door bord E6 RVV 1990 aangeduide gehandicaptenparkeerplaats. {sub_sign_clause} Ik heb geen geldige gehandicaptenparkeerkaart waargenomen achter de voorruit van het voertuig{card_reason}. Bij het constateren van het feit werd vastgesteld dat er gedurende een tijd van ongeveer {observation_time} minuten geen activiteit met betrekking tot het voertuig plaats vond, zodat er geen sprake was van onmiddellijk laden of lossen van goederen, dan wel het in of uit laten stappen van personen.""",

        "template_en": """I observed that the vehicle was parked in a disabled parking space indicated by sign E6 RVV 1990. {sub_sign_clause} No valid disability parking card was observed behind the windshield of the vehicle{card_reason}. When establishing the violation, it was determined that for a period of approximately {observation_time} minutes no activity related to the vehicle took place, meaning there was no immediate loading or unloading of goods, nor persons getting in or out.""",

        "sub_sign_clauses": {
            "general_nl": "Het bord was duidelijk zichtbaar aanwezig.",
            "general_en": "The sign was clearly visible.",
            "reserved_nl": "Blijkens het onderbord is het gebruik van deze gehandicaptenparkeerplaats voorbehouden aan het voertuig met kenteken {reserved_plate}.",
            "reserved_en": "According to the sub-sign, the use of this disabled parking space is reserved for the vehicle with license plate {reserved_plate}."
        },

        "card_reason_clauses": {
            "no_card_nl": ", want er was geen kaart aanwezig",
            "no_card_en": " because no card was present",
            "invalid_card_nl": ", want de aanwezige kaart was verlopen/ongeldig",
            "invalid_card_en": " because the card present was expired/invalid",
            "wrong_vehicle_nl": ", want de kaart behoorde niet bij dit voertuig",
            "wrong_vehicle_en": " because the card did not belong to this vehicle",
            "not_visible_nl": "",
            "not_visible_en": ""
        },

        "default_context": {
            "observation_time": "5",
            "sub_sign_clause": "",
            "card_reason": ""
        }
    },

    "E6_RESERVED": {
        "template_nl": """Ik zag dat het voertuig met kenteken {vehicle_plate} geparkeerd stond op een door bord E6 RVV 1990 aangeduide gereserveerde gehandicaptenparkeerplaats. Blijkens het onderbord is het gebruik van deze parkeerplaats voorbehouden aan het voertuig met kenteken {reserved_plate}. Het kenteken van het geparkeerde voertuig komt niet overeen met het kenteken op het onderbord. Bij het constateren van het feit werd vastgesteld dat er gedurende een tijd van ongeveer {observation_time} minuten geen activiteit met betrekking tot het voertuig plaats vond.""",

        "template_en": """I observed that the vehicle with license plate {vehicle_plate} was parked in a reserved disabled parking space indicated by sign E6 RVV 1990. According to the sub-sign, the use of this parking space is reserved for the vehicle with license plate {reserved_plate}. The license plate of the parked vehicle does not match the license plate on the sub-sign. When establishing the violation, it was determined that for a period of approximately {observation_time} minutes no activity related to the vehicle took place.""",

        "default_context": {
            "observation_time": "5",
            "vehicle_plate": "[KENTEKEN]",
            "reserved_plate": "[ONDERBORD KENTEKEN]"
        }
    },

    "E7": {
        "template_nl": """Ik zag dat het voertuig geparkeerd stond in strijd met bord E7 RVV 1990 (gelegenheid bestemd voor het onmiddellijk laden en lossen van goederen). Ik zag geen geldige ontheffing zichtbaar aanwezig in het voertuig. Ik zag geen laad/los activiteiten rondom het voertuig. Ik zag geen bestuurder in of rondom het voertuig. {time_restriction_clause}""",

        "template_en": """I observed that the vehicle was parked in violation of sign E7 RVV 1990 (area designated for immediate loading and unloading of goods). No valid exemption was visible in the vehicle. No loading/unloading activities were observed around the vehicle. No driver was present in or around the vehicle. {time_restriction_clause}""",

        "time_restriction_clauses": {
            "with_times_nl": "Waarnemingstijd {observation_time} minuten. Het bord was voorzien van onderbord met tijdvenster {time_window}.",
            "with_times_en": "Observation time {observation_time} minutes. The sign included a sub-sign with time window {time_window}.",
            "no_times_nl": "Waarnemingstijd {observation_time} minuten.",
            "no_times_en": "Observation time {observation_time} minutes."
        },

        "default_context": {
            "observation_time": "5",
            "time_restriction_clause": ""
        }
    },

    "E9": {
        "template_nl": """Ik zag dat het voertuig geparkeerd stond op een parkeergelegenheid bestemd voor vergunninghouders, aangeduid door bord E9 RVV 1990{sub_sign_clause}. Ik zag geen geldige vergunning zichtbaar aanwezig in of aan het voertuig. Ik zag geen laad/los activiteiten rondom het voertuig. Tevens zag ik geen bestuurder in of rondom het voertuig.""",

        "template_en": """I observed that the vehicle was parked in a parking area designated for permit holders, indicated by sign E9 RVV 1990{sub_sign_clause}. No valid permit was visible in or on the vehicle. No loading/unloading activities were observed around the vehicle. Additionally, no driver was present in or around the vehicle.""",

        "sub_sign_clauses": {
            "with_subsign_nl": " met onderbord \"{sub_sign_text}\"",
            "with_subsign_en": " with sub-sign \"{sub_sign_text}\"",
            "no_subsign_nl": "",
            "no_subsign_en": ""
        },

        "default_context": {
            "sub_sign_clause": "",
            "sub_sign_text": ""
        }
    },

    "G7": {
        "template_nl": """Het voertuig stond geparkeerd in een door bord G7 RVV 1990 aangeduid voetgangersgebied. Ik heb geen voor dat gebied geldige ontheffing waargenomen. {time_restriction_clause}""",

        "template_en": """The vehicle was parked in a pedestrian area indicated by sign G7 RVV 1990. No valid exemption for that area was observed. {time_restriction_clause}""",

        "time_restriction_clauses": {
            "with_times_nl": "Het verbod geldt {time_window}.",
            "with_times_en": "The prohibition applies {time_window}.",
            "no_times_nl": "",
            "no_times_en": ""
        },

        "default_context": {
            "time_restriction_clause": ""
        }
    },

    "ELECTRIC_CHARGING": {
        "template_nl": """Ik zag dat het voertuig geparkeerd stond op een oplaadpunt voor elektrische voertuigen, aangeduid door bord E4 met oplaadsymbool. Het voertuig was niet aangesloten op het oplaadpunt. Derhalve is sprake van parkeren op een oplaadpunt zonder daarvan gebruik te maken.""",

        "template_en": """I observed that the vehicle was parked at an electric vehicle charging point, indicated by sign E4 with charging symbol. The vehicle was not connected to the charging point. Therefore, this constitutes parking at a charging point without using it.""",

        "default_context": {}
    }
}


# Legal conclusion templates (formal legal language)
LEGAL_CONCLUSION_TEMPLATES = {
    "E6": {
        "nl": "Derhalve is sprake van een overtreding van artikel 26 van het RVV 1990. Op grond van artikel 2, onder e, van het Besluit wegslepen van voertuigen is verwijdering van het voertuig gerechtvaardigd.",
        "en": "Therefore, this constitutes a violation of Article 26 of RVV 1990. Under Article 2, under e, of the Vehicle Towing Decree, removal of the vehicle is justified."
    },
    "E6_RESERVED": {
        "nl": "Derhalve is sprake van een overtreding van artikel 26, eerste lid, onder c, van het RVV 1990. Op grond van artikel 2, onder e, van het Besluit wegslepen van voertuigen is verwijdering van het voertuig gerechtvaardigd.",
        "en": "Therefore, this constitutes a violation of Article 26, paragraph 1, under c, of RVV 1990. Under Article 2, under e, of the Vehicle Towing Decree, removal of the vehicle is justified."
    },
    "E7": {
        "nl": "Derhalve is sprake van een overtreding van artikel 24, eerste lid, onder f, van het RVV 1990. Op grond van artikel 2, onder f, van het Besluit wegslepen van voertuigen is verwijdering van het voertuig gerechtvaardigd.",
        "en": "Therefore, this constitutes a violation of Article 24, paragraph 1, under f, of RVV 1990. Under Article 2, under f, of the Vehicle Towing Decree, removal of the vehicle is justified."
    },
    "E9": {
        "nl": "Derhalve is sprake van een overtreding van artikel 24, eerste lid, onder g, van het RVV 1990. Op grond van artikel 2, onder h, van het Besluit wegslepen van voertuigen is verwijdering van het voertuig noodzakelijk in verband met het vrijhouden van aangewezen weggedeelten.",
        "en": "Therefore, this constitutes a violation of Article 24, paragraph 1, under g, of RVV 1990. Under Article 2, under h, of the Vehicle Towing Decree, removal of the vehicle is necessary to keep designated road sections clear."
    },
    "G7": {
        "nl": "Derhalve is sprake van een overtreding van artikel 87 van het RVV 1990. Op grond van artikel 2, onder i, van het Besluit wegslepen van voertuigen is verwijdering van het voertuig gerechtvaardigd.",
        "en": "Therefore, this constitutes a violation of Article 87 of RVV 1990. Under Article 2, under i, of the Vehicle Towing Decree, removal of the vehicle is justified."
    },
    "ELECTRIC_CHARGING": {
        "nl": "Derhalve is sprake van een overtreding van artikel 24, eerste lid, onder d, van het RVV 1990. Op grond van de gemeentelijke verordening is verwijdering van het voertuig gerechtvaardigd.",
        "en": "Therefore, this constitutes a violation of Article 24, paragraph 1, under d, of RVV 1990. Under the municipal ordinance, removal of the vehicle is justified."
    }
}


def generate_legal_statement(
    violation_code: str,
    context: dict = None,
    language: str = "nl",
    include_conclusion: bool = True
) -> str:
    """
    Generate a legal statement using templates and context data.

    Args:
        violation_code: The violation type (E6, E7, E9, G7, ELECTRIC_CHARGING)
        context: Dictionary with values to fill in template placeholders
        language: "nl" for Dutch, "en" for English
        include_conclusion: Whether to include the legal conclusion

    Returns:
        Formatted legal statement string

    Example:
        >>> context = {"observation_time": "10", "sub_sign_text": "autodate GreenWheels"}
        >>> statement = generate_legal_statement("E9", context, "nl")
    """
    template_data = LEGAL_TEMPLATES.get(violation_code)

    if not template_data:
        return f"[Unknown violation code: {violation_code}]"

    # Get the template for the requested language
    template_key = f"template_{language}"
    template = template_data.get(template_key)

    if not template:
        # Fallback to Dutch if requested language not available
        template = template_data.get("template_nl", "")

    # Merge default context with provided context
    merged_context = template_data.get("default_context", {}).copy()
    if context:
        merged_context.update(context)

    # Process sub-clauses if applicable
    merged_context = _process_sub_clauses(violation_code, merged_context, language, template_data)

    # Format the template with context
    try:
        statement = template.format(**merged_context)
    except KeyError as e:
        statement = f"[Template error: missing key {e}]"

    # Add legal conclusion if requested
    if include_conclusion:
        conclusion = get_legal_conclusion(violation_code, language)
        if conclusion:
            statement = f"{statement}\n\n{conclusion}"

    return statement.strip()


def _process_sub_clauses(
    violation_code: str,
    context: dict,
    language: str,
    template_data: dict
) -> dict:
    """
    Process and fill in sub-clauses based on context.

    Internal function to handle clause selection logic.
    """
    lang_suffix = f"_{language}"

    # Handle E6 sub-sign clauses
    if violation_code == "E6":
        sub_sign_clauses = template_data.get("sub_sign_clauses", {})
        card_reason_clauses = template_data.get("card_reason_clauses", {})

        # Determine sub-sign clause
        if context.get("reserved_plate"):
            clause_key = f"reserved{lang_suffix}"
            context["sub_sign_clause"] = sub_sign_clauses.get(clause_key, "").format(**context)
        else:
            clause_key = f"general{lang_suffix}"
            context["sub_sign_clause"] = sub_sign_clauses.get(clause_key, "")

        # Determine card reason clause
        card_status = context.get("card_status", "no_card")
        reason_key = f"{card_status}{lang_suffix}"
        context["card_reason"] = card_reason_clauses.get(reason_key, "")

    # Handle E7 time restriction clauses
    elif violation_code == "E7":
        time_clauses = template_data.get("time_restriction_clauses", {})
        if context.get("time_window"):
            clause_key = f"with_times{lang_suffix}"
            context["time_restriction_clause"] = time_clauses.get(clause_key, "").format(**context)
        else:
            clause_key = f"no_times{lang_suffix}"
            context["time_restriction_clause"] = time_clauses.get(clause_key, "").format(**context)

    # Handle E9 sub-sign clauses
    elif violation_code == "E9":
        sub_sign_clauses = template_data.get("sub_sign_clauses", {})
        if context.get("sub_sign_text"):
            clause_key = f"with_subsign{lang_suffix}"
            context["sub_sign_clause"] = sub_sign_clauses.get(clause_key, "").format(**context)
        else:
            clause_key = f"no_subsign{lang_suffix}"
            context["sub_sign_clause"] = sub_sign_clauses.get(clause_key, "")

    # Handle G7 time restriction clauses
    elif violation_code == "G7":
        time_clauses = template_data.get("time_restriction_clauses", {})
        if context.get("time_window"):
            clause_key = f"with_times{lang_suffix}"
            context["time_restriction_clause"] = time_clauses.get(clause_key, "").format(**context)
        else:
            clause_key = f"no_times{lang_suffix}"
            context["time_restriction_clause"] = time_clauses.get(clause_key, "")

    return context


def get_legal_conclusion(violation_code: str, language: str = "nl") -> Optional[str]:
    """
    Get the legal conclusion statement for a violation type.

    Args:
        violation_code: The violation type
        language: "nl" or "en"

    Returns:
        Legal conclusion string or None
    """
    conclusions = LEGAL_CONCLUSION_TEMPLATES.get(violation_code, {})
    return conclusions.get(language)


def generate_full_legal_output(
    violation_code: str,
    mllm_output: dict,
    officer_observation: str = None,
    language: str = "nl"
) -> dict:
    """
    Generate the complete legal statement output structure.

    Args:
        violation_code: The violation type
        mllm_output: MLLM analysis output (Layer 2 format)
        officer_observation: Original officer observation text
        language: "nl" or "en"

    Returns:
        Dictionary with nl and en versions of the legal statement
    """
    # Extract context from MLLM output
    context = _extract_context_from_mllm(mllm_output)

    # Generate statements in both languages
    statement_nl = generate_legal_statement(violation_code, context, "nl", include_conclusion=True)
    statement_en = generate_legal_statement(violation_code, context, "en", include_conclusion=True)

    return {
        "nl": statement_nl,
        "en": statement_en,
        "violation_code": violation_code,
        "context_used": context,
        "based_on_officer_observation": officer_observation is not None
    }


def _extract_context_from_mllm(mllm_output: dict) -> dict:
    """
    Extract template context values from MLLM output.

    Args:
        mllm_output: MLLM analysis output (Layer 2 format)

    Returns:
        Context dictionary for template filling
    """
    context = {
        "observation_time": "5"  # Default observation time
    }

    # Extract vehicle plate
    plate_info = mllm_output.get("vehicle", {}).get("license_plate", {})
    if isinstance(plate_info, dict):
        context["vehicle_plate"] = plate_info.get("value", "[KENTEKEN]")
    elif isinstance(plate_info, str):
        context["vehicle_plate"] = plate_info

    # Extract traffic sign info
    sign_info = mllm_output.get("traffic_sign", {})
    if sign_info.get("sub_sign_text"):
        context["sub_sign_text"] = sign_info["sub_sign_text"]
        # Check if sub-sign contains a license plate (for reserved spaces)
        sub_text = str(sign_info["sub_sign_text"]).upper()
        if any(c.isdigit() for c in sub_text) and "-" in sub_text:
            context["reserved_plate"] = sign_info["sub_sign_text"]

    # Determine card status for E6
    windshield = mllm_output.get("windshield_items", {})
    disability_card = windshield.get("disability_card", "not_visible")
    if disability_card == "no":
        context["card_status"] = "no_card"
    elif disability_card == "not_visible":
        context["card_status"] = "not_visible"
    else:
        context["card_status"] = "no_card"

    return context


def get_available_templates() -> list:
    """
    Get a list of all available template violation codes.

    Returns:
        List of violation code strings
    """
    return list(LEGAL_TEMPLATES.keys())
