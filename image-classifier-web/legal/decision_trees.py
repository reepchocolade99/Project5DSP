"""
Legal Decision Trees for Amsterdam Parking Enforcement System.
Based on Legal Reasoning Architecture v2.0

This module contains deterministic decision trees for each violation type,
mapping MLLM observations to legal requirements with article references.

Legal Sources:
- RVV 1990 (BWBR0004825): Traffic rules, sign definitions
- Besluit wegslepen (BWBR0012649): Towing authority, registration requirements
- Wegenverkeerswet 1994 (BWBR0006622): Parent law

Version: 2.0
"""

LEGAL_DECISION_TREES = {
    "E6": {
        "name": "Gehandicaptenparkeerplaats",
        "name_en": "Disabled parking space",
        "required_checks": [
            {
                "check_id": "E6_SIGN",
                "description": "Sign E6 present and visible",
                "description_nl": "Bord E6 aanwezig en zichtbaar",
                "source_field": "traffic_sign.sign_code",
                "expected_value": "E6",
                "legal_reference": "RVV 1990 Bijlage 1, Bord E6",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Bijlage1"
            },
            {
                "check_id": "E6_NO_CARD",
                "description": "No valid disability parking card",
                "description_nl": "Geen geldige gehandicaptenparkeerkaart",
                "source_field": "windshield_items.disability_card",
                "expected_value": "no",
                "legal_reference": "RVV 1990 Article 26 paragraph 1b",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel26"
            },
            {
                "check_id": "E6_NO_DRIVER",
                "description": "No driver present (parking, not loading/unloading)",
                "description_nl": "Geen bestuurder aanwezig (parkeren, niet laden/lossen)",
                "source_field": "environment.driver_present",
                "expected_value": False,
                "legal_reference": "RVV 1990 Article 1 (definition of parking)",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel1"
            }
        ],
        "violation_article": "RVV 1990 Article 26",
        "violation_article_url": "https://wetten.overheid.nl/BWBR0004825#Artikel26",
        "violation_text_nl": "De bestuurder mag zijn voertuig niet parkeren op een gehandicaptenparkeerplaats, aangeduid door verkeersbord E6, indien hij niet in het bezit is van een geldige gehandicaptenparkeerkaart.",
        "violation_text_en": "The driver may not park their vehicle in a disabled parking space, indicated by traffic sign E6, if they do not possess a valid disability parking card.",
        "wegslepen_basis": "Besluit wegslepen Article 2e",
        "wegslepen_url": "https://wetten.overheid.nl/BWBR0012649#Artikel2",
        "feit_code": "R402C"
    },

    "E6_RESERVED": {
        "name": "Gereserveerde gehandicaptenparkeerplaats",
        "name_en": "Reserved disability parking space",
        "required_checks": [
            {
                "check_id": "E6R_SIGN",
                "description": "Sign E6 with license plate sub-sign present",
                "description_nl": "Bord E6 met kenteken-onderbord aanwezig",
                "source_field": "traffic_sign.sign_code",
                "expected_value": "E6",
                "legal_reference": "RVV 1990 Bijlage 1, Bord E6 with sub-sign",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Bijlage1"
            },
            {
                "check_id": "E6R_WRONG_PLATE",
                "description": "Vehicle license plate does not match sub-sign",
                "description_nl": "Voertuigkenteken komt niet overeen met onderbord",
                "source_field": "vehicle.license_plate.value",
                "compare_with": "traffic_sign.sub_sign_text",
                "expected_result": "mismatch",
                "legal_reference": "RVV 1990 Article 26 paragraph 1c",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel26"
            }
        ],
        "violation_article": "RVV 1990 Article 26 paragraph 1c",
        "violation_article_url": "https://wetten.overheid.nl/BWBR0004825#Artikel26",
        "violation_text_nl": "De bestuurder mag zijn voertuig niet parkeren op een gereserveerde gehandicaptenparkeerplaats indien het kenteken van zijn voertuig niet overeenkomt met het kenteken op het onderbord.",
        "violation_text_en": "The driver may not park their vehicle in a reserved disability parking space if their vehicle's license plate does not match the license plate on the sub-sign.",
        "wegslepen_basis": "Besluit wegslepen Article 2e",
        "wegslepen_url": "https://wetten.overheid.nl/BWBR0012649#Artikel2",
        "feit_code": "R402C"
    },

    "E7": {
        "name": "Laden en lossen",
        "name_en": "Loading and unloading zone",
        "required_checks": [
            {
                "check_id": "E7_SIGN",
                "description": "Sign E7 present and visible",
                "description_nl": "Bord E7 aanwezig en zichtbaar",
                "source_field": "traffic_sign.sign_code",
                "expected_value": "E7",
                "legal_reference": "RVV 1990 Bijlage 1, Bord E7",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Bijlage1"
            },
            {
                "check_id": "E7_NO_PERMIT",
                "description": "No valid exemption visible",
                "description_nl": "Geen geldige ontheffing zichtbaar",
                "source_field": "windshield_items.permit",
                "expected_value": "no",
                "legal_reference": "RVV 1990 Article 87",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel87"
            },
            {
                "check_id": "E7_NO_ACTIVITY",
                "description": "No loading/unloading activity observed",
                "description_nl": "Geen laad/los activiteit waargenomen",
                "source_field": "environment.loading_activity",
                "expected_value": False,
                "legal_reference": "RVV 1990 Article 24 paragraph 1f",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel24"
            },
            {
                "check_id": "E7_NO_DRIVER",
                "description": "No driver present",
                "description_nl": "Geen bestuurder aanwezig",
                "source_field": "environment.driver_present",
                "expected_value": False,
                "legal_reference": "RVV 1990 Article 1 (definition of parking)",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel1"
            }
        ],
        "violation_article": "RVV 1990 Article 24 paragraph 1f",
        "violation_article_url": "https://wetten.overheid.nl/BWBR0004825#Artikel24",
        "violation_text_nl": "Het is verboden een voertuig te parkeren op een gelegenheid bestemd voor het onmiddellijk laden en lossen van goederen.",
        "violation_text_en": "It is prohibited to park a vehicle in an area designated for immediate loading and unloading of goods.",
        "wegslepen_basis": "Besluit wegslepen Article 2f",
        "wegslepen_url": "https://wetten.overheid.nl/BWBR0012649#Artikel2",
        "feit_code": "R397H"
    },

    "E9": {
        "name": "Vergunninghouders",
        "name_en": "Permit holders parking space",
        "required_checks": [
            {
                "check_id": "E9_SIGN",
                "description": "Sign E9 present and visible",
                "description_nl": "Bord E9 aanwezig en zichtbaar",
                "source_field": "traffic_sign.sign_code",
                "expected_value": "E9",
                "legal_reference": "RVV 1990 Bijlage 1, Bord E9",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Bijlage1"
            },
            {
                "check_id": "E9_NO_PERMIT",
                "description": "No valid permit visible",
                "description_nl": "Geen geldige vergunning zichtbaar",
                "source_field": "windshield_items.permit",
                "expected_value": "no",
                "legal_reference": "RVV 1990 Article 24 paragraph 1g",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel24"
            },
            {
                "check_id": "E9_NO_DRIVER",
                "description": "No driver present",
                "description_nl": "Geen bestuurder aanwezig",
                "source_field": "environment.driver_present",
                "expected_value": False,
                "legal_reference": "RVV 1990 Article 1 (definition of parking)",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel1"
            }
        ],
        "violation_article": "RVV 1990 Article 24 paragraph 1g",
        "violation_article_url": "https://wetten.overheid.nl/BWBR0004825#Artikel24",
        "violation_text_nl": "De bestuurder mag zijn voertuig niet parkeren op een parkeerplaats voor vergunninghouders, aangeduid door verkeersbord E9, indien voor zijn voertuig geen vergunning tot parkeren op die plaats is verleend.",
        "violation_text_en": "The driver may not park their vehicle in a permit holders parking space, indicated by traffic sign E9, if no parking permit has been granted for that vehicle at that location.",
        "wegslepen_basis": "Besluit wegslepen Article 2h",
        "wegslepen_url": "https://wetten.overheid.nl/BWBR0012649#Artikel2",
        "feit_code": "R397i"
    },

    "G7": {
        "name": "Voetgangersgebied",
        "name_en": "Pedestrian area",
        "required_checks": [
            {
                "check_id": "G7_SIGN",
                "description": "Sign G7 present",
                "description_nl": "Bord G7 aanwezig",
                "source_field": "traffic_sign.sign_code",
                "expected_value": "G7",
                "legal_reference": "RVV 1990 Bijlage 1, Bord G7",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Bijlage1"
            },
            {
                "check_id": "G7_NO_PERMIT",
                "description": "No valid exemption for pedestrian area",
                "description_nl": "Geen geldige ontheffing voor voetgangersgebied",
                "source_field": "windshield_items.permit",
                "expected_value": "no",
                "legal_reference": "RVV 1990 Article 87",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel87"
            }
        ],
        "violation_article": "RVV 1990 Article 87",
        "violation_article_url": "https://wetten.overheid.nl/BWBR0004825#Artikel87",
        "violation_text_nl": "Het is verboden een voertuig te parkeren in een voetgangersgebied zonder geldige ontheffing.",
        "violation_text_en": "It is prohibited to park a vehicle in a pedestrian area without a valid exemption.",
        "wegslepen_basis": "Besluit wegslepen Article 2i",
        "wegslepen_url": "https://wetten.overheid.nl/BWBR0012649#Artikel2",
        "feit_code": "R584"
    },

    "ELECTRIC_CHARGING": {
        "name": "Elektrisch oplaadpunt",
        "name_en": "Electric charging point",
        "required_checks": [
            {
                "check_id": "EC_SIGN",
                "description": "Electric charging sign present",
                "description_nl": "Elektrisch oplaadpunt bord aanwezig",
                "source_field": "traffic_sign.sign_code",
                "expected_value": "E4_ELECTRIC",
                "legal_reference": "RVV 1990 Bijlage 1, Bord E4 with charging symbol",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Bijlage1"
            },
            {
                "check_id": "EC_NOT_CONNECTED",
                "description": "Vehicle not connected to charging point",
                "description_nl": "Voertuig niet aangesloten op oplaadpunt",
                "source_field": "environment.charging_connected",
                "expected_value": False,
                "legal_reference": "RVV 1990 Article 24 paragraph 1d",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel24"
            }
        ],
        "violation_article": "RVV 1990 Article 24 paragraph 1d",
        "violation_article_url": "https://wetten.overheid.nl/BWBR0004825#Artikel24",
        "violation_text_nl": "Het is verboden een voertuig te parkeren op een oplaadpunt voor elektrische voertuigen zonder daarvan gebruik te maken.",
        "violation_text_en": "It is prohibited to park a vehicle at an electric vehicle charging point without using it.",
        "wegslepen_basis": "Municipal ordinance",
        "wegslepen_url": None,
        "feit_code": "R397"
    }
}


# Mapping from detected sign codes to violation types
SIGN_CODE_TO_VIOLATION = {
    "E6": "E6",
    "E7": "E7",
    "E9": "E9",
    "G7": "G7",
    "E4": "ELECTRIC_CHARGING",
    "E4_ELECTRIC": "ELECTRIC_CHARGING"
}


# Legal source URLs for reference
LEGAL_SOURCES = {
    "RVV_1990": {
        "name": "Reglement verkeersregels en verkeerstekens 1990",
        "name_en": "Traffic Rules and Traffic Signs Regulation 1990",
        "code": "BWBR0004825",
        "url": "https://wetten.overheid.nl/BWBR0004825"
    },
    "BESLUIT_WEGSLEPEN": {
        "name": "Besluit wegslepen van voertuigen",
        "name_en": "Vehicle Towing Decree",
        "code": "BWBR0012649",
        "url": "https://wetten.overheid.nl/BWBR0012649"
    },
    "WVW_1994": {
        "name": "Wegenverkeerswet 1994",
        "name_en": "Road Traffic Act 1994",
        "code": "BWBR0006622",
        "url": "https://wetten.overheid.nl/BWBR0006622"
    }
}


def get_decision_tree(violation_code: str) -> dict:
    """
    Get the decision tree for a given violation code.

    Args:
        violation_code: The violation type (E6, E7, E9, G7, ELECTRIC_CHARGING)

    Returns:
        Decision tree dictionary or None if not found
    """
    return LEGAL_DECISION_TREES.get(violation_code)


def get_violation_from_sign(sign_code: str) -> str:
    """
    Map a detected sign code to its corresponding violation type.

    Args:
        sign_code: The detected traffic sign code (e.g., "E9", "G7")

    Returns:
        Violation type string or None if not found
    """
    return SIGN_CODE_TO_VIOLATION.get(sign_code)


def get_all_violation_codes() -> list:
    """
    Get a list of all supported violation codes.

    Returns:
        List of violation code strings
    """
    return list(LEGAL_DECISION_TREES.keys())
