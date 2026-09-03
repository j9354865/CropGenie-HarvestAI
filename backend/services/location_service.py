"""
Location insights service — resolves a city/district to its Pakistan
agro-ecological zone and formats data for the Gemini AI prompt.

Uses the reference data in ``backend.utils.pakistan_zones`` to map
~170 administrative districts to one of 10 zones.  When the city is
not an exact match, falls back to partial matching; when no match is
found at all, lets the weather service provide climate data so Gemini
can reason about the most likely zone.
"""

import logging
from typing import Optional

from backend.utils.pakistan_zones import lookup_zone

logger = logging.getLogger(__name__)


class LocationService:
    """Resolves a city/district to its agro-ecological zone."""

    def resolve_zone(
        self, city: str
    ) -> Optional[tuple[str, str, dict]]:
        """
        Look up a city/district and return its zone information.

        Parameters
        ----------
        city : str
            City or district name entered by the user.

        Returns
        -------
        tuple[str, str, dict] or None
            (matched_district, zone_key, zone_data) if found,
            or None if the city could not be matched to any zone.
        """
        result = lookup_zone(city)
        if result is None:
            logger.info("No zone match found for city: %s", city)
            return None

        zone_key, zone_data = result

        # Determine the best display name for the matched district.
        # lookup_zone returns the zone; we want to show which district
        # was actually matched (title-cased).
        matched_district = self._find_matched_district(city, zone_key)

        logger.info(
            "Zone match for '%s': %s (%s)",
            city,
            matched_district,
            zone_key,
        )
        return matched_district, zone_key, zone_data

    @staticmethod
    def _find_matched_district(city: str, zone_key: str) -> str:
        """
        Find the best district name to display for a given city input.

        Returns the title-cased district name from the mapping if an
        exact/partial match is found, otherwise returns the city as-is
        (title-cased).
        """
        query = city.strip().lower()

        from backend.utils.pakistan_zones import DISTRICT_ZONE_MAP

        # Exact match
        if query in DISTRICT_ZONE_MAP:
            return query.title()

        # Partial match — find the district that matched
        for district_name, zk in DISTRICT_ZONE_MAP.items():
            if zk == zone_key and (
                query in district_name or district_name in query
            ):
                return district_name.title()

        # Fallback — just title-case the input
        return city.strip().title()

    @staticmethod
    def format_zone_for_gemini(zone_data: dict) -> dict:
        """
        Extract and format zone characteristics for the Gemini prompt.

        Returns a dict with keys: zone_name, climate, soil, crops,
        irrigation, challenges — all as strings ready for the prompt.
        """
        return {
            "zone_name": zone_data["name"],
            "climate": zone_data["climate"],
            "soil": zone_data["soil"],
            "crops": ", ".join(zone_data.get("crops", [])),
            "irrigation": zone_data["irrigation"],
            "challenges": ", ".join(zone_data.get("challenges", [])),
        }
