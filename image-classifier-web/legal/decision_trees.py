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
    # ─────────────────────────────────────────────────────────────────────────
    # E1 - Parkeerverbod (No Parking Zone)
    # ─────────────────────────────────────────────────────────────────────────
    "E1": {
        "name": "Parkeerverbod",
        "name_en": "No Parking Zone (E1)",
        "required_checks": [
            {
                "check_id": "E1_SIGN",
                "description": "E1 no parking sign is clearly visible",
                "description_nl": "Verkeersbord E1 (parkeerverbod) is duidelijk zichtbaar",
                "source_field": "traffic_sign.sign_code",
                "expected_value": "E1",
                "legal_reference": "RVV 1990 Bijlage 1, Bord E1",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Bijlage1"
            },
            {
                "check_id": "E1_NO_PERMIT",
                "description": "No valid exemption visible",
                "description_nl": "Geen geldige ontheffing zichtbaar",
                "source_field": "windshield_items.permit",
                "expected_value": "no",
                "legal_reference": "RVV 1990 Article 87",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel87"
            },
            {
                "check_id": "E1_VEHICLE_PARKED",
                "description": "Vehicle is parked in violation zone",
                "description_nl": "Voertuig is geparkeerd in verbodzone",
                "source_field": "vehicle.parked",
                "expected_value": True,
                "legal_reference": "RVV 1990 Article 1 (definition of parking)",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel1"
            },
            {
                "check_id": "E1_NO_DRIVER",
                "description": "Driver is not present in vehicle",
                "description_nl": "Bestuurder is niet aanwezig in voertuig",
                "source_field": "environment.driver_present",
                "expected_value": False,
                "legal_reference": "RVV 1990 Article 1 (definition of parking)",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel1"
            }
        ],
        "violation_article": "RVV 1990 Article 62 jo. bord E1",
        "violation_article_url": "https://wetten.overheid.nl/BWBR0004825#Artikel62",
        "violation_text_nl": "Het is verboden een voertuig te parkeren in een zone aangeduid door bord E1 (parkeerverbod).",
        "violation_text_en": "It is prohibited to park a vehicle in a zone indicated by sign E1 (no parking).",
        "wegslepen_basis": "Art. 170 lid 1 sub c WVW 1994",
        "wegslepen_url": "https://wetten.overheid.nl/BWBR0006622#Artikel170",
        "feit_code": "R394"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # E2 - Verbod stil te staan (No Stopping Zone)
    # ─────────────────────────────────────────────────────────────────────────
    "E2": {
        "name": "Verbod stil te staan",
        "name_en": "No Stopping Zone (E2)",
        "required_checks": [
            {
                "check_id": "E2_SIGN",
                "description": "E2 no stopping sign is clearly visible",
                "description_nl": "Verkeersbord E2 (stilstaan verboden) is duidelijk zichtbaar",
                "source_field": "traffic_sign.sign_code",
                "expected_value": "E2",
                "legal_reference": "RVV 1990 Bijlage 1, Bord E2",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Bijlage1"
            },
            {
                "check_id": "E2_NO_PERMIT",
                "description": "No valid exemption visible",
                "description_nl": "Geen geldige ontheffing zichtbaar",
                "source_field": "windshield_items.permit",
                "expected_value": "no",
                "legal_reference": "RVV 1990 Article 87",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel87"
            },
            {
                "check_id": "E2_VEHICLE_STOPPED",
                "description": "Vehicle is stopped in violation zone",
                "description_nl": "Voertuig staat stil in verbodzone",
                "source_field": "vehicle.stopped",
                "expected_value": True,
                "legal_reference": "RVV 1990 Article 1",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel1"
            }
        ],
        "violation_article": "RVV 1990 Article 62 jo. bord E2",
        "violation_article_url": "https://wetten.overheid.nl/BWBR0004825#Artikel62",
        "violation_text_nl": "Het is verboden een voertuig te laten stilstaan in een zone aangeduid door bord E2 (stilstaan verboden).",
        "violation_text_en": "It is prohibited to stop a vehicle in a zone indicated by sign E2 (no stopping).",
        "wegslepen_basis": "Art. 170 lid 1 sub c WVW 1994",
        "wegslepen_url": "https://wetten.overheid.nl/BWBR0006622#Artikel170",
        "feit_code": "R395"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # E4 - Parkeergelegenheid (Parking Facility with conditions)
    # ─────────────────────────────────────────────────────────────────────────
    "E4": {
        "name": "Parkeergelegenheid",
        "name_en": "Parking Facility (E4)",
        "required_checks": [
            {
                "check_id": "E4_SIGN",
                "description": "E4 parking sign is clearly visible",
                "description_nl": "Verkeersbord E4 (parkeergelegenheid) is duidelijk zichtbaar",
                "source_field": "traffic_sign.sign_code",
                "expected_value": "E4",
                "legal_reference": "RVV 1990 Bijlage 1, Bord E4",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Bijlage1"
            },
            {
                "check_id": "E4_CONDITIONS_VIOLATED",
                "description": "Parking conditions on sub-sign are not met",
                "description_nl": "Parkeervoorwaarden op onderbord worden niet nageleefd",
                "source_field": "traffic_sign.conditions_met",
                "expected_value": False,
                "legal_reference": "RVV 1990 Article 62",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel62"
            },
            {
                "check_id": "E4_NO_DRIVER",
                "description": "No driver present",
                "description_nl": "Geen bestuurder aanwezig",
                "source_field": "environment.driver_present",
                "expected_value": False,
                "legal_reference": "RVV 1990 Article 1 (definition of parking)",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel1"
            }
        ],
        "violation_article": "RVV 1990 Article 62 jo. bord E4",
        "violation_article_url": "https://wetten.overheid.nl/BWBR0004825#Artikel62",
        "violation_text_nl": "Het is verboden een voertuig te parkeren op een parkeergelegenheid aangeduid door bord E4 in strijd met de op het onderbord aangegeven voorwaarden.",
        "violation_text_en": "It is prohibited to park a vehicle in a parking facility indicated by sign E4 in violation of the conditions stated on the sub-sign.",
        "wegslepen_basis": "Art. 170 lid 1 sub c WVW 1994",
        "wegslepen_url": "https://wetten.overheid.nl/BWBR0006622#Artikel170",
        "feit_code": "R402a"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # E5 - Taxistandplaats (Taxi Stand)
    # ─────────────────────────────────────────────────────────────────────────
    "E5": {
        "name": "Taxistandplaats",
        "name_en": "Taxi Stand (E5)",
        "required_checks": [
            {
                "check_id": "E5_SIGN",
                "description": "E5 taxi stand sign is clearly visible",
                "description_nl": "Verkeersbord E5 (taxistandplaats) is duidelijk zichtbaar",
                "source_field": "traffic_sign.sign_code",
                "expected_value": "E5",
                "legal_reference": "RVV 1990 Bijlage 1, Bord E5",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Bijlage1"
            },
            {
                "check_id": "E5_NOT_TAXI",
                "description": "Vehicle is not a taxi",
                "description_nl": "Voertuig is geen taxi",
                "source_field": "vehicle.is_taxi",
                "expected_value": False,
                "legal_reference": "RVV 1990 Article 24",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel24"
            },
            {
                "check_id": "E5_NO_DRIVER",
                "description": "No driver present",
                "description_nl": "Geen bestuurder aanwezig",
                "source_field": "environment.driver_present",
                "expected_value": False,
                "legal_reference": "RVV 1990 Article 1 (definition of parking)",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel1"
            }
        ],
        "violation_article": "RVV 1990 Article 62 jo. bord E5",
        "violation_article_url": "https://wetten.overheid.nl/BWBR0004825#Artikel62",
        "violation_text_nl": "Het is verboden een voertuig anders dan een taxi te parkeren op een taxistandplaats aangeduid door bord E5.",
        "violation_text_en": "It is prohibited to park a vehicle other than a taxi at a taxi stand indicated by sign E5.",
        "wegslepen_basis": "Art. 170 lid 1 sub c WVW 1994",
        "wegslepen_url": "https://wetten.overheid.nl/BWBR0006622#Artikel170",
        "feit_code": "R402b"
    },

    # ─────────────────────────────────────────────────────────────────────────
    # E6 - Gehandicaptenparkeerplaats (Disabled Parking)
    # ─────────────────────────────────────────────────────────────────────────
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
    },

    "R396I": {
        "name": "Gele doorgetrokken streep",
        "name_en": "Yellow continuous line (no stopping)",
        "required_checks": [
            {
                "check_id": "R396I_YELLOW_LINE",
                "description": "Yellow continuous line visible on road",
                "description_nl": "Gele doorgetrokken streep zichtbaar op de weg",
                "source_field": "road_markings.yellow_line",
                "expected_value": True,
                "legal_reference": "RVV 1990 Article 23 paragraph 1c",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel23"
            },
            {
                "check_id": "R396I_NO_PERMIT",
                "description": "No valid exemption visible",
                "description_nl": "Geen geldige ontheffing zichtbaar",
                "source_field": "windshield_items.permit",
                "expected_value": "no",
                "legal_reference": "RVV 1990 Article 87",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel87"
            },
            {
                "check_id": "R396I_NO_DRIVER",
                "description": "No driver present in or near the vehicle",
                "description_nl": "Geen bestuurder aanwezig in of bij het voertuig",
                "source_field": "environment.driver_present",
                "expected_value": False,
                "legal_reference": "RVV 1990 Article 1 (definition of parking)",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel1"
            }
        ],
        "violation_article": "RVV 1990 Article 23 paragraph 1c",
        "violation_article_url": "https://wetten.overheid.nl/BWBR0004825#Artikel23",
        "violation_text_nl": "Het is verboden een voertuig te laten stilstaan langs een gele doorgetrokken streep.",
        "violation_text_en": "It is prohibited to stop a vehicle along a yellow continuous line.",
        "wegslepen_basis": "Besluit wegslepen Article 2",
        "wegslepen_url": "https://wetten.overheid.nl/BWBR0012649#Artikel2",
        "feit_code": "R396i"
    },

    "YELLOW_LINE": {
        "name": "Gele doorgetrokken streep",
        "name_en": "Yellow continuous line (no stopping)",
        "required_checks": [
            {
                "check_id": "YL_VISIBLE",
                "description": "Yellow continuous line visible on road",
                "description_nl": "Gele doorgetrokken streep zichtbaar op de weg",
                "source_field": "road_markings.yellow_line",
                "expected_value": True,
                "legal_reference": "RVV 1990 Article 23 paragraph 1c",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel23"
            },
            {
                "check_id": "YL_NO_PERMIT",
                "description": "No valid exemption visible",
                "description_nl": "Geen geldige ontheffing zichtbaar",
                "source_field": "windshield_items.permit",
                "expected_value": "no",
                "legal_reference": "RVV 1990 Article 87",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel87"
            },
            {
                "check_id": "YL_NO_DRIVER",
                "description": "No driver present",
                "description_nl": "Geen bestuurder aanwezig",
                "source_field": "environment.driver_present",
                "expected_value": False,
                "legal_reference": "RVV 1990 Article 1",
                "legal_url": "https://wetten.overheid.nl/BWBR0004825#Artikel1"
            }
        ],
        "violation_article": "RVV 1990 Article 23 paragraph 1c",
        "violation_article_url": "https://wetten.overheid.nl/BWBR0004825#Artikel23",
        "violation_text_nl": "Het is verboden een voertuig te laten stilstaan langs een gele doorgetrokken streep.",
        "violation_text_en": "It is prohibited to stop a vehicle along a yellow continuous line.",
        "wegslepen_basis": "Besluit wegslepen Article 2",
        "wegslepen_url": "https://wetten.overheid.nl/BWBR0012649#Artikel2",
        "feit_code": "R396i"
    }
}


# Mapping from detected sign codes to violation types
SIGN_CODE_TO_VIOLATION = {
    # E-codes (parking signs)
    "E1": "E1",
    "E2": "E2",
    "E3": "E3",
    "E4": "E4",
    "E4_ELECTRIC": "ELECTRIC_CHARGING",
    "E5": "E5",
    "E6": "E6",
    "E7": "E7",
    "E8": "E8",
    "E9": "E9",
    "E10": "E10",
    "E11": "E11",
    "E12": "E12",
    "E13": "E13",
    # G-codes (pedestrian areas)
    "G7": "G7",
    # R-codes (road markings and specific violations)
    "R394": "E1",
    "R395": "E2",
    "R396": "E3",
    "R396I": "R396I",
    "R396i": "R396I",
    "R397I": "E9",
    "R397i": "E9",
    "R397H": "E7",
    "R397h": "E7",
    "R402A": "E4",
    "R402a": "E4",
    "R402B": "E5",
    "R402b": "E5",
    "R402C": "E6",
    "R402c": "E6",
    "R402D": "E7",
    "R402d": "E7",
    "R402E": "E8",
    "R402e": "E8",
    "R584": "G7",
    # Special road markings
    "YELLOW_LINE": "YELLOW_LINE",
    "GELE_STREEP": "YELLOW_LINE",
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
