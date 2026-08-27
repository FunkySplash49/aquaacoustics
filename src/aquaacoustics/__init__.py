"""
aquaacoustics - map + role-based UI layer that wraps the aeld leak detector.
===============================================================================
This package is presentation/integration only. All leak-detection physics
and maths live untouched in `aeld` (see src/aeld/). This layer's job is:

    sites          - preset pipe-monitoring locations + map geometry
    map_view       - the interactive survey map (Folium/OpenStreetMap)
    detection_view - the detailed "how it worked" page for one site's result

See docs/superpowers/specs/2026-08-27-aquaacoustics-integration-design.md
for the full design.
"""

from .sites import Site, SITES, get_site

__all__ = ["Site", "SITES", "get_site"]
