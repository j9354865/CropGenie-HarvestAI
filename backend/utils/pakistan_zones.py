"""
Pakistan Agro-Ecological Zone Reference Module
===============================================

Maps every administrative district of Pakistan to one of 10 official
Agro-Ecological Zones, each with distinct soil, climate, and crop
characteristics.

Data Source:
    Pakistan Agricultural Research Council (PARC) / FAO, 2024.

    The district-to-zone mapping was compiled as a **practical
    approximation** based on known geography, climate, and elevation of
    each district.  Administrative boundaries do not always align with
    agro-ecological boundaries, so some districts near zone borders are
    assigned to the zone that best represents the majority of their
    agricultural area.  This is intended as a useful practical tool, not
    a scientifically precise boundary map.

Coverage:
    ~170 districts across all provinces and territories:
    Punjab, Sindh, Khyber Pakhtunkhwa (incl. merged tribal districts),
    Balochistan, Islamabad Capital Territory, Gilgit-Baltistan,
    and Azad Jammu & Kashmir.
"""

from __future__ import annotations

from typing import Optional


# ── 10 Agro-Ecological Zones ──────────────────────────────────────────
#
# Each zone entry contains:
#   name       : official zone name
#   icon       : emoji for UI display
#   provinces  : primary provinces/areas covered
#   climate    : brief climate description
#   soil       : dominant soil types
#   crops      : typical / recommended crops
#   irrigation : water source & availability
#   challenges : key agricultural challenges

ZONES: dict[str, dict] = {
    "Zone 1": {
        "name": "Northern Dry Mountains",
        "icon": "🏔️",
        "provinces": "Gilgit-Baltistan, Chitral, upper KP",
        "climate": (
            "Cold, dry mountainous climate.  Harsh winters with heavy "
            "snowfall; short, cool summers.  Annual rainfall 200–500 mm."
        ),
        "soil": "Rocky, shallow, poor fertility; alluvial fans in valleys",
        "crops": [
            "Apples", "Apricots", "Cherries", "Walnuts", "Almonds",
            "Potatoes", "Maize (summer)", "Barley",
        ],
        "irrigation": "Glacial melt, springs; very limited canal irrigation",
        "challenges": [
            "Short growing season",
            "Limited arable land",
            "Poor road access to markets",
            "Frost damage risk",
        ],
    },
    "Zone 2": {
        "name": "Temperate Mountains",
        "icon": "🌲",
        "provinces": "Hazara division, Abbottabad, Mansehra, Murree, Swat upper valleys, AJK",
        "climate": (
            "Moderate hill climate.  Cool winters, pleasant summers.  "
            "Annual rainfall 1000–2000 mm (monsoon + winter rains)."
        ),
        "soil": "Loamy to clayey, moderately fertile; terraced slopes",
        "crops": [
            "Maize", "Wheat", "Rice (some areas)", "Potatoes",
            "Temperate fruits (peach, plum, pear)", "Vegetables",
        ],
        "irrigation": "Rain-fed (barani) + hill streams; some terrace irrigation",
        "challenges": [
            "Soil erosion on slopes",
            "Land sliding",
            "Limited mechanisation",
            "Market access",
        ],
    },
    "Zone 3": {
        "name": "Rainfed Plateau (Potohar)",
        "icon": "🌾",
        "provinces": "Rawalpindi, Chakwal, Attock, Jhelum, Islamabad, Gujrat",
        "climate": (
            "Sub-tropical, semi-arid.  Hot summers, cool winters.  "
            "Annual rainfall 500–1000 mm (mainly monsoon).  Drought-prone."
        ),
        "soil": "Loamy, sandy loam; moderately fertile; erosion-prone",
        "crops": [
            "Wheat (barani)", "Groundnut", "Pulses (chickpea, lentil)",
            "Maize", "Fodder crops", "Oilseeds",
        ],
        "irrigation": "Mostly rain-fed (barani); limited tube-wells and small dams",
        "challenges": [
            "Drought risk",
            "Groundwater depletion",
            "Low crop yields compared to irrigated areas",
            "Soil erosion",
        ],
    },
    "Zone 4": {
        "name": "Western Dry Mountains",
        "icon": "🏜️",
        "provinces": "Waziristan, western KP/Balochistan border, D.I.Khan hills",
        "climate": (
            "Arid mountainous.  Hot summers, cold winters.  "
            "Annual rainfall <300 mm.  Very dry."
        ),
        "soil": "Rocky, sandy, low organic matter; saline patches",
        "crops": [
            "Drought-resistant wheat", "Maize (rain-fed)", "Millets",
            "Dates", "Livestock fodder",
        ],
        "irrigation": "Karez systems, limited springs; mostly rain-fed",
        "challenges": [
            "Extreme water scarcity",
            "Poor soil fertility",
            "Limited infrastructure",
            "Security concerns",
        ],
    },
    "Zone 5": {
        "name": "Northern Irrigated Plain",
        "icon": "🌿",
        "provinces": (
            "Central/upper Punjab (Lahore, Gujranwala, Sialkot, "
            "Faisalabad, Sargodha, Sheikhupura, Hafizabad, Narowal)"
        ),
        "climate": (
            "Sub-tropical, semi-arid to sub-humid.  Hot summers "
            "(35–45 °C), mild winters.  Annual rainfall 300–800 mm."
        ),
        "soil": "Alluvial, loamy; highly fertile — Pakistan's breadbasket",
        "crops": [
            "Wheat", "Rice (basmati)", "Sugarcane", "Maize",
            "Citrus (kinnow)", "Vegetables", "Fodder", "Cotton (some areas)",
        ],
        "irrigation": "Extensive canal network (Indus Basin); tube-wells widely used",
        "challenges": [
            "Water-logging and salinity (some areas)",
            "Groundwater depletion",
            "Urban encroachment on farmland",
            "Pest pressure in rice/cotton",
        ],
    },
    "Zone 6": {
        "name": "Southern Irrigated Plain",
        "icon": "☀️",
        "provinces": (
            "Southern Punjab (Multan, Bahawalpur, Khanewal, Vehari, "
            "R.Y.Khan, Muzaffargarh, D.G.Khan) and Sindh irrigated belt "
            "(Hyderabad, Sukkur, Larkana, Nawabshah, etc.)"
        ),
        "climate": (
            "Hot, arid to semi-arid.  Very hot summers (40–50 °C), "
            "mild winters.  Annual rainfall 100–300 mm."
        ),
        "soil": "Alluvial, sandy loam to clay loam; fertile but salinity issues",
        "crops": [
            "Cotton", "Wheat", "Sugarcane", "Rice", "Maize",
            "Mangoes", "Dates", "Chillies", "Onion",
        ],
        "irrigation": "Major canal systems (Sukkur, Guddu, Kotri barrages); tube-wells",
        "challenges": [
            "Severe salinity and water-logging",
            "Extreme summer heat",
            "Water scarcity at tail-end canals",
            "Cotton pest outbreaks",
        ],
    },
    "Zone 7": {
        "name": "Piedmont Plain",
        "icon": "🏞️",
        "provinces": "Peshawar valley, Mardan, Charsadda, Nowshera, Swabi, D.I.Khan plains",
        "climate": (
            "Semi-arid.  Hot summers, cool winters.  "
            "Annual rainfall 300–700 mm.  Moderate climate."
        ),
        "soil": "Alluvial fans, sandy loam; moderately to highly fertile",
        "crops": [
            "Maize", "Wheat", "Sugarcane", "Tobacco",
            "Vegetables", "Fruits (plum, peach)", "Fodder",
        ],
        "irrigation": "Canal irrigation (Warsak, Upper Swat canals); tube-wells",
        "challenges": [
            "Water-logging in low-lying areas",
            "Flooding from hill torrents",
            "Urbanisation pressure",
        ],
    },
    "Zone 8": {
        "name": "Western Dry Plateau",
        "icon": "🗻",
        "provinces": "Balochistan highlands (Quetta, Kalat, Zhob, Loralai, Pishin, Mastung)",
        "climate": (
            "Cold, arid highland.  Very cold winters (sub-zero), "
            "hot summers in valleys.  Annual rainfall 100–300 mm."
        ),
        "soil": "Loamy, calcareous; low organic matter; saline in patches",
        "crops": [
            "Apples (Quetta valley)", "Grapes", "Apricots",
            "Wheat", "Barley", "Cumin", "Potatoes",
        ],
        "irrigation": "Karez, tube-wells, limited canal (Bolan area)",
        "challenges": [
            "Severe water scarcity",
            "Short growing season",
            "Harsh winters",
            "Limited infrastructure",
        ],
    },
    "Zone 9": {
        "name": "Sandy Desert",
        "icon": "🏜️",
        "provinces": "Tharparkar, Umerkot, Cholistan (Bahawalpur)",
        "climate": (
            "Hot arid desert.  Extremely hot summers (45–50 °C), "
            "mild winters.  Annual rainfall <200 mm (erratic)."
        ),
        "soil": "Sandy, very low fertility; wind erosion; saline patches",
        "crops": [
            "Millets (bajra)", "Guar (cluster bean)", "Sesame",
            "Pulses (moth bean)", "Livestock fodder",
        ],
        "irrigation": "Almost none; dependent on erratic rainfall",
        "challenges": [
            "Extreme water scarcity",
            "Drought and famine risk",
            "Sand dune movement",
            "Livestock-dependent livelihoods",
        ],
    },
    "Zone 10": {
        "name": "Coastal Zone",
        "icon": "🌊",
        "provinces": "Karachi, Thatta, Badin, Gwadar, Lasbela, coastal Balochistan",
        "climate": (
            "Hot, humid coastal.  Moderate temperatures year-round "
            "(20–35 °C).  Annual rainfall 100–250 mm (monsoon)."
        ),
        "soil": "Sandy, saline; deltaic alluvium near Indus mouth",
        "crops": [
            "Rice (delta areas)", "Coconut", "Banana", "Sugarcane",
            "Fodder", "Fisheries (dominant livelihood)",
        ],
        "irrigation": "Indus delta canals (limited); tidal flooding risk",
        "challenges": [
            "Seawater intrusion / salinity",
            "Cyclone risk",
            "Mangrove loss",
            "Limited arable land",
        ],
    },
}


# ── District → Zone Mapping ───────────────────────────────────────────
#
# Comprehensive mapping of ~170 administrative districts to their most
# appropriate agro-ecological zone.  Keys are lower-case district names
# for case-insensitive lookup.
#
# NOTE: Districts near zone boundaries are assigned to the zone that
# best represents the majority of their agricultural area.  This is a
# practical approximation, not a scientifically precise classification.

DISTRICT_ZONE_MAP: dict[str, str] = {
    # ── Punjab (36 districts) ─────────────────────────────────────
    "attock": "Zone 3",
    "bahawalnagar": "Zone 6",
    "bahawalpur": "Zone 6",
    "bhakkar": "Zone 5",
    "burewala": "Zone 5",
    "chakwal": "Zone 3",
    "chiniot": "Zone 5",
    "dera ghazi khan": "Zone 6",
    "faisalabad": "Zone 5",
    "gujranwala": "Zone 5",
    "gujrat": "Zone 3",
    "hafizabad": "Zone 5",
    "jhang": "Zone 5",
    "jhelum": "Zone 3",
    "kasur": "Zone 5",
    "khanewal": "Zone 6",
    "khushab": "Zone 3",
    "lahore": "Zone 5",
    "layyah": "Zone 6",
    "lodhran": "Zone 6",
    "mandi bahauddin": "Zone 5",
    "mianwali": "Zone 3",
    "multan": "Zone 6",
    "muzaffargarh": "Zone 6",
    "narowal": "Zone 5",
    "nankana sahib": "Zone 5",
    "okara": "Zone 5",
    "pakpattan": "Zone 5",
    "rahim yar khan": "Zone 6",
    "rajanpur": "Zone 6",
    "rawalpindi": "Zone 3",
    "sahiwal": "Zone 5",
    "sargodha": "Zone 5",
    "sheikhupura": "Zone 5",
    "sialkot": "Zone 5",
    "toba tek singh": "Zone 5",
    "vehari": "Zone 6",

    # ── Sindh (23 districts) ──────────────────────────────────────
    "badin": "Zone 10",
    "dadu": "Zone 6",
    "ghotki": "Zone 6",
    "hyderabad": "Zone 6",
    "jacobabad": "Zone 6",
    "jamshoro": "Zone 6",
    "karachi central": "Zone 10",
    "karachi east": "Zone 10",
    "karachi south": "Zone 10",
    "karachi west": "Zone 10",
    "kemari": "Zone 10",
    "korangi": "Zone 10",
    "malir": "Zone 10",
    "kashmore": "Zone 6",
    "khairpur": "Zone 6",
    "larkana": "Zone 6",
    "matiari": "Zone 6",
    "mirpur khas": "Zone 6",
    "naushahro feroze": "Zone 6",
    "sanghar": "Zone 6",
    "shaheed benazirabad": "Zone 6",
    "shikarpur": "Zone 6",
    "sukkur": "Zone 6",
    "tando allahyar": "Zone 6",
    "tando muhammad khan": "Zone 6",
    "tharparkar": "Zone 9",
    "thatta": "Zone 10",
    "umerkot": "Zone 9",

    # ── Khyber Pakhtunkhwa (36 districts incl. merged tribal areas) ─
    "abbottabad": "Zone 2",
    "allai": "Zone 2",
    "bajaur": "Zone 4",
    "bannu": "Zone 7",
    "battagram": "Zone 2",
    "buner": "Zone 7",
    "charsadda": "Zone 7",
    "chitral": "Zone 1",
    "dera ismail khan": "Zone 7",
    "dir lower": "Zone 2",
    "dir upper": "Zone 1",
    "hangu": "Zone 4",
    "haripur": "Zone 2",
    "karak": "Zone 4",
    "kohat": "Zone 7",
    "kolai palas": "Zone 2",
    "kurram": "Zone 4",
    "lakki marwat": "Zone 7",
    "lower chitral": "Zone 1",
    "lower dir": "Zone 2",
    "lower kohistan": "Zone 1",
    "malakand": "Zone 7",
    "mansehra": "Zone 2",
    "mardan": "Zone 7",
    "mohmand": "Zone 4",
    "north waziristan": "Zone 4",
    "nowshera": "Zone 7",
    "orakzai": "Zone 4",
    "peshawar": "Zone 7",
    "shanla": "Zone 2",
    "south waziristan": "Zone 4",
    "swabi": "Zone 7",
    "swat": "Zone 2",
    "tank": "Zone 7",
    "torghar": "Zone 2",
    "upper chitral": "Zone 1",
    "upper dir": "Zone 1",
    "upper kohistan": "Zone 1",

    # ── Balochistan (33 districts) ────────────────────────────────
    "awaran": "Zone 10",
    "barkhan": "Zone 8",
    "bolan": "Zone 8",
    "chagai": "Zone 4",
    "dera bugti": "Zone 6",
    "gwadar": "Zone 10",
    "harnai": "Zone 4",
    "hub": "Zone 10",
    "jafarabad": "Zone 6",
    "jhal magsi": "Zone 6",
    "kachi": "Zone 6",
    "kalat": "Zone 8",
    "kech": "Zone 4",
    "kharan": "Zone 4",
    "khuzdar": "Zone 4",
    "killa abdullah": "Zone 8",
    "killa saifullah": "Zone 8",
    "kohlu": "Zone 4",
    "lasbela": "Zone 10",
    "loralai": "Zone 8",
    "mastung": "Zone 8",
    "musakhel": "Zone 8",
    "nasirabad": "Zone 6",
    "nushki": "Zone 4",
    "panjgur": "Zone 4",
    "pishin": "Zone 8",
    "quetta": "Zone 8",
    "sherani": "Zone 4",
    "sibi": "Zone 4",
    "sohbatpur": "Zone 6",
    "washuk": "Zone 4",
    "zhob": "Zone 8",
    "ziarat": "Zone 8",

    # ── Gilgit-Baltistan (10 districts) ───────────────────────────
    "ghizer": "Zone 1",
    "gilgit": "Zone 1",
    "hunza": "Zone 1",
    "nagar": "Zone 1",
    "ghanchay": "Zone 1",
    "kharmang": "Zone 1",
    "skardu": "Zone 1",
    "shigar": "Zone 1",
    "diamer": "Zone 1",
    "astore": "Zone 1",

    # ── Azad Jammu & Kashmir (10 districts) ──────────────────────
    "bagh": "Zone 2",
    "bhimber": "Zone 2",
    "haveli": "Zone 2",
    "kotli": "Zone 2",
    "mirpur": "Zone 2",
    "muzaffarabad": "Zone 2",
    "neelum": "Zone 1",
    "poonch": "Zone 2",
    "sudhnoti": "Zone 2",
    "hattian bala": "Zone 2",

    # ── Islamabad Capital Territory (1 district) ──────────────────
    "islamabad": "Zone 3",
}


# ── Aliases for common city names ─────────────────────────────────────
# Maps alternative names / abbreviations to the canonical key used in
# DISTRICT_ZONE_MAP.  Covers common spellings and major city names that
# people might type instead of the formal district name.

_CITY_ALIASES: dict[str, str] = {
    # Major cities (common names people type)
    "karachi": "karachi central",
    "lahore": "lahore",
    "islamabad": "islamabad",
    "rawalpindi": "rawalpindi",
    "peshawar": "peshawar",
    "quetta": "quetta",
    "faisalabad": "faisalabad",
    "multan": "multan",
    "hyderabad": "hyderabad",
    "sukkur": "sukkur",
    "larkana": "larkana",
    "gilgit": "gilgit",
    "skardu": "skardu",
    "muzaffarabad": "muzaffarabad",

    # Common abbreviations and variants
    "d.g khan": "dera ghazi khan",
    "dg khan": "dera ghazi khan",
    "dgkhan": "dera ghazi khan",
    "d.i khan": "dera ismail khan",
    "di khan": "dera ismail khan",
    "dikhan": "dera ismail khan",
    "d.i.k": "dera ismail khan",
    "r.y khan": "rahim yar khan",
    "ryk": "rahim yar khan",
    "rahim yar khan": "rahim yar khan",

    # City vs district name variants
    "sahiwal": "sahiwal",
    "nawabshah": "shaheed benazirabad",
    "benazirabad": "shaheed benazirabad",
}


# ── Lookup Functions ──────────────────────────────────────────────────

def lookup_zone(city_or_district: str) -> tuple[str, dict] | None:
    """
    Look up a city or district name and return its zone.

    Tries, in order:
      1. Exact match (case-insensitive) in DISTRICT_ZONE_MAP
      2. Alias match in _CITY_ALIASES
      3. Partial / substring match against district names
      4. Returns None if no match found

    Parameters
    ----------
    city_or_district : str
        City or district name entered by the user.

    Returns
    -------
    tuple[str, dict] or None
        (zone_key, zone_data) if found, e.g. ("Zone 5", {...}),
        or None if no match.
    """
    if not city_or_district or not city_or_district.strip():
        return None

    query = city_or_district.strip().lower()

    # 1. Exact match
    if query in DISTRICT_ZONE_MAP:
        zone_key = DISTRICT_ZONE_MAP[query]
        return zone_key, ZONES[zone_key]

    # 2. Alias match
    if query in _CITY_ALIASES:
        canonical = _CITY_ALIASES[query]
        zone_key = DISTRICT_ZONE_MAP.get(canonical)
        if zone_key:
            return zone_key, ZONES[zone_key]

    # 3. Partial match — check if the query is a substring of any
    #    district name, or any district name is a substring of the query.
    #    If multiple matches, prefer the longest match (most specific).
    candidates: list[tuple[str, int]] = []
    for district_name, zone_key in DISTRICT_ZONE_MAP.items():
        if query in district_name or district_name in query:
            # Score by length — longer match is more specific
            match_len = min(len(query), len(district_name))
            candidates.append((zone_key, match_len))

    if candidates:
        # Pick the candidate with the longest match
        best_zone_key = max(candidates, key=lambda c: c[1])[0]
        return best_zone_key, ZONES[best_zone_key]

    # 4. No match
    return None


def get_zone(zone_key: str) -> dict | None:
    """Return zone data by key (e.g. 'Zone 5'), or None if not found."""
    return ZONES.get(zone_key)


def get_all_zones() -> dict[str, dict]:
    """Return the full ZONES dictionary."""
    return ZONES


def get_district_count() -> int:
    """Return the number of districts in the mapping."""
    return len(DISTRICT_ZONE_MAP)


def find_districts_in_zone(zone_key: str) -> list[str]:
    """Return a sorted list of district names for a given zone key."""
    return sorted(
        name.title()
        for name, zone in DISTRICT_ZONE_MAP.items()
        if zone == zone_key
    )
