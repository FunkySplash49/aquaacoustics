"""
sites.py - preset pipe-monitoring sites for the AquaAcoustics survey map.
===========================================================================

Each Site bundles a map location (for drawing the survey polygon and sensor
markers) with the pipe parameters that drive the real aeld GCC-PHAT detector
(pipe length, wave velocity, noise level). Coordinates are city centres used
purely as illustrative, visually-distinct locations across India - they are
NOT real pipe survey data. See
docs/superpowers/specs/2026-08-27-aquaacoustics-integration-design.md,
"Non-goals".

Also here: the small amount of geometry needed to turn the detector's 1-D
"distance from Sensor A" answer into a 2-D point on the map. The on-map
distance between Sensor A and Sensor B is a fixed, purely visual constant
(ON_MAP_SENSOR_SEPARATION_M) - real pipe_length_m can be hundreds of metres
and is what the physics actually uses; the map only needs *something* to
draw a short line between two markers.
"""

from dataclasses import dataclass
from math import radians, sin, cos


# ============================================================================
# Site registry
# ============================================================================

@dataclass(frozen=True)
class Site:
    """One preset pipe-monitoring location."""

    name: str
    state: str
    lat: float                 # Sensor A's map latitude
    lng: float                 # Sensor A's map longitude
    heading_deg: float         # compass bearing, Sensor A -> Sensor B (0=N, 90=E)
    pipe_length_m: float       # REAL distance between Sensor A and B, fed to the physics
    wave_velocity_ms: float
    noise_level: float


# Six illustrative sites spanning different states and pipe scenarios, from a
# short quiet urban main to a long noisy rural trunk line. Coordinates are
# city centres used only as visually distinct map locations.
SITES: list[Site] = [
    Site("Jaipur Ring Main", "Rajasthan",
         26.9124, 75.7873, heading_deg=45.0,
         pipe_length_m=180.0, wave_velocity_ms=1250.0, noise_level=0.20),
    Site("Pune Industrial Line", "Maharashtra",
         18.5204, 73.8567, heading_deg=120.0,
         pipe_length_m=340.0, wave_velocity_ms=1300.0, noise_level=0.35),
    Site("Bengaluru Tech Corridor", "Karnataka",
         12.9716, 77.5946, heading_deg=200.0,
         pipe_length_m=95.0, wave_velocity_ms=1180.0, noise_level=0.10),
    Site("Chennai Coastal Main", "Tamil Nadu",
         13.0827, 80.2707, heading_deg=300.0,
         pipe_length_m=420.0, wave_velocity_ms=1400.0, noise_level=0.45),
    Site("Lucknow Old City Line", "Uttar Pradesh",
         26.8467, 80.9462, heading_deg=15.0,
         pipe_length_m=150.0, wave_velocity_ms=1150.0, noise_level=0.60),
    Site("Ahmedabad Riverside Trunk", "Gujarat",
         23.0225, 72.5714, heading_deg=160.0,
         pipe_length_m=260.0, wave_velocity_ms=1275.0, noise_level=0.25),
]

_SITES_BY_NAME = {site.name: site for site in SITES}


def get_site(name: str) -> Site:
    """Look up a preset site by name. Raises KeyError if it doesn't exist."""
    return _SITES_BY_NAME[name]


# ============================================================================
# Map geometry: turning a 1-D pipe distance into a 2-D lat/lng point
# ============================================================================

# Fixed, purely visual on-map separation between Sensor A and Sensor B, in
# metres. Real pipe_length_m (which can be hundreds of metres) is NOT drawn
# to scale on the map - it drives the physics, this drives the drawing.
ON_MAP_SENSOR_SEPARATION_M = 60.0

# Side length of the drawn "surveyed area" square, in metres. 110m x 110m =
# 12,100 sq m ~= 2.99 acres, matching the original doc's "~3-acre polygon".
SURVEY_POLYGON_SIDE_M = 110.0

# Metres per degree of latitude is effectively constant on Earth's surface.
_METERS_PER_DEG_LAT = 111_320.0


def _meters_per_deg_lng(lat: float) -> float:
    """Metres per degree of longitude shrinks toward the poles by cos(lat)."""
    return _METERS_PER_DEG_LAT * cos(radians(lat))


def _offset_latlng(lat: float, lng: float, east_m: float, north_m: float) -> tuple:
    """
    Move a point by a small east/north offset in metres.

    Flat-earth approximation: valid to well under a centimetre of error at
    the scale used here (tens to hundreds of metres), far smaller than a
    map marker's own pixel footprint.
    """
    dlat = north_m / _METERS_PER_DEG_LAT
    dlng = east_m / _meters_per_deg_lng(lat)
    return (lat + dlat, lng + dlng)


def sensor_coordinates(site: Site) -> tuple:
    """
    Where Sensor A and Sensor B are drawn on the map.

    Sensor A sits at the site's registered coordinate. Sensor B sits
    ON_MAP_SENSOR_SEPARATION_M away along the site's heading.

    Returns ((lat_a, lng_a), (lat_b, lng_b)).
    """
    heading_rad = radians(site.heading_deg)
    east_m = ON_MAP_SENSOR_SEPARATION_M * sin(heading_rad)
    north_m = ON_MAP_SENSOR_SEPARATION_M * cos(heading_rad)
    a = (site.lat, site.lng)
    b = _offset_latlng(site.lat, site.lng, east_m, north_m)
    return a, b


def interpolate_position(site: Site, computed_distance_m: float,
                         pipe_length_m: float = None) -> tuple:
    """
    Where the DETECTOR'S REAL RESULT is drawn on the map.

    Converts the detector's 1-D "distance from Sensor A" (in real metres,
    0..pipe_length_m) into a 2-D point along the Sensor-A-to-Sensor-B line
    drawn on the map, by straight linear interpolation on fraction of pipe
    length. This is never a random point - see the design doc's "honest
    marker" decision.

    `pipe_length_m` defaults to the site's preset (`site.pipe_length_m`) when
    not given. Pass it explicitly when the run that produced
    `computed_distance_m` used a DIFFERENT pipe length than the site's preset
    (e.g. the "Advanced: override this site's pipe and re-run" control) -
    otherwise the fraction is computed against the wrong denominator and the
    marker is placed incorrectly.
    """
    length = site.pipe_length_m if pipe_length_m is None else pipe_length_m
    fraction = computed_distance_m / length
    fraction = max(0.0, min(1.0, fraction))     # clamp for safety

    on_map_distance_m = fraction * ON_MAP_SENSOR_SEPARATION_M
    heading_rad = radians(site.heading_deg)
    east_m = on_map_distance_m * sin(heading_rad)
    north_m = on_map_distance_m * cos(heading_rad)
    return _offset_latlng(site.lat, site.lng, east_m, north_m)


def polygon_corners(site: Site) -> list:
    """
    Four corners of the "surveyed area" square drawn around the site,
    axis-aligned (N/E), centred on the site's coordinate. Purely visual
    context - see the design doc for why the leak marker itself is never
    placed randomly within it.
    """
    half = SURVEY_POLYGON_SIDE_M / 2.0
    offsets = [(-half, -half), (-half, half), (half, half), (half, -half)]
    return [_offset_latlng(site.lat, site.lng, east_m, north_m)
            for east_m, north_m in offsets]
