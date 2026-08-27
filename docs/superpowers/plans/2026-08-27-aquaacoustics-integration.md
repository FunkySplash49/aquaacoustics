# AquaAcoustics Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the standalone AELD leak-detection prototype into AquaAcoustics — one Streamlit app with an interactive survey map of preset monitoring sites, where triggering a site runs the real GCC-PHAT detector and draws the genuine result (never a random point) on the map.

**Architecture:** Single Python/Streamlit app. `src/aeld/` (physics + GCC-PHAT) is reused completely unmodified. A new `src/aquaacoustics/` layer adds a preset site registry, a `streamlit-folium` map page, and an adapted results page, wired together by a rewritten `app.py` router with a cosmetic Admin/Field-Staff role picker.

**Tech Stack:** Python, Streamlit, NumPy/SciPy (existing, untouched), Folium + streamlit-folium (new, for the OpenStreetMap-tiled map), pytest + Streamlit's `AppTest` framework (new, for end-to-end wiring tests without a browser).

## Global Constraints

- Single Streamlit app, Python only — no Next.js/TypeScript (design decision; see spec).
- `src/aeld/` logic, its own tests, and the standalone `aeld-prototype.streamlit.app` deployment are not touched by any task in this plan.
- New dependencies: `streamlit-folium`, `folium` — added to `requirements.txt`, nothing else.
- Map tiles: OpenStreetMap via Folium's default (no API key).
- The map's leak marker must always be the detector's real computed position (`aquaacoustics.sites.interpolate_position`), never a random point — this is a correctness rule, not a style preference; every task that draws a marker must respect it.
- Role picker ("Admin" / "Field Staff") is a plain selector — no passwords, no real authentication, ever.
- Six preset monitoring sites, one per state, each with distinct illustrative pipe parameters — not real survey data.
- No plaintext credentials or literal text from `AquaAcoustics_Documentation.md` are copied into any file.
- No `git push` and no deployment as part of this plan — this repo stays local until separately requested.
- Every new pure-logic function ships with a unit test in the same task that introduces it, per the project's existing "verifiable, not just plausible" testing philosophy.

---

## File Structure

```
aquaacoustics/
├── app.py                              # MODIFY — router + role picker (full rewrite)
├── requirements.txt                     # MODIFY — + streamlit-folium, folium
├── src/
│   ├── aeld/                            # UNCHANGED
│   └── aquaacoustics/                    # NEW package
│       ├── __init__.py                   # NEW
│       ├── sites.py                      # NEW — preset sites + map geometry
│       ├── map_view.py                   # NEW — Survey Map page
│       └── detection_view.py             # NEW — Leak Detection Detail page
├── tests/
│   ├── test_gccphat.py, test_signals.py  # UNCHANGED
│   ├── test_sites.py                     # NEW
│   ├── test_map_view.py                  # NEW
│   ├── test_detection_view.py            # NEW
│   └── test_app_flow.py                  # NEW — end-to-end AppTest
├── claude.md                             # MODIFY — rewritten for AquaAcoustics
├── context.md                            # MODIFY — rewritten progress log
└── EXPLANATION.txt                       # DELETE — describes the pre-integration app.py, would be stale
```

`src/aquaacoustics` depends on `src/aeld`, never the reverse. `detection_view.py` depends on `map_view.py` (reuses `pick_random_leak_position` — DRY, no duplicated randomness rule). `map_view.py` does not depend on `detection_view.py`. No circular imports.

---

### Task 1: Preset site registry and map geometry

**Files:**
- Create: `src/aquaacoustics/__init__.py`
- Create: `src/aquaacoustics/sites.py`
- Test: `tests/test_sites.py`

**Interfaces:**
- Consumes: nothing (pure Python + `math` stdlib only).
- Produces: `Site` (frozen dataclass: `name, state, lat, lng, heading_deg, pipe_length_m, wave_velocity_ms, noise_level`), `SITES: list[Site]` (6 entries), `get_site(name: str) -> Site`, `sensor_coordinates(site: Site) -> tuple[tuple[float,float], tuple[float,float]]`, `interpolate_position(site: Site, computed_distance_m: float) -> tuple[float,float]`, `polygon_corners(site: Site) -> list[tuple[float,float]]`, `ON_MAP_SENSOR_SEPARATION_M: float`. All consumed by Task 3 and Task 4.

- [ ] **Step 1: Write the failing test**

Create `tests/test_sites.py`:

```python
"""
test_sites.py - the preset site registry and its map geometry helpers.
========================================================================
These tests don't touch aeld's physics at all - they prove two separate
things: (1) every preset site is well-formed, and (2) the geometry that
turns a 1-D detector result into a 2-D map point is correct at its three
checkable landmarks: the two sensor endpoints, and the midpoint.
"""

from math import radians, cos, sqrt

import pytest

from aquaacoustics.sites import (
    SITES, get_site, sensor_coordinates, interpolate_position,
    polygon_corners, ON_MAP_SENSOR_SEPARATION_M,
)


# ============================================================================
# Registry sanity
# ============================================================================

def test_six_sites_registered():
    assert len(SITES) == 6


def test_site_names_are_unique():
    names = [site.name for site in SITES]
    assert len(names) == len(set(names))


def test_site_states_are_distinct():
    states = [site.state for site in SITES]
    assert len(states) == len(set(states))


@pytest.mark.parametrize("site", SITES, ids=lambda s: s.name)
def test_site_has_valid_coordinates(site):
    assert -90.0 <= site.lat <= 90.0
    assert -180.0 <= site.lng <= 180.0


@pytest.mark.parametrize("site", SITES, ids=lambda s: s.name)
def test_site_has_positive_pipe_parameters(site):
    assert site.pipe_length_m > 0
    assert site.wave_velocity_ms > 0
    assert site.noise_level >= 0


def test_get_site_returns_the_right_site():
    site = get_site("Jaipur Ring Main")
    assert site.name == "Jaipur Ring Main"
    assert site.state == "Rajasthan"


def test_get_site_unknown_name_raises():
    with pytest.raises(KeyError):
        get_site("Nonexistent Site")


# ============================================================================
# Map geometry: the three checkable landmarks
# ============================================================================

@pytest.mark.parametrize("site", SITES, ids=lambda s: s.name)
def test_interpolate_at_zero_matches_sensor_a(site):
    sensor_a, _sensor_b = sensor_coordinates(site)
    result = interpolate_position(site, 0.0)
    assert result[0] == pytest.approx(sensor_a[0], abs=1e-9)
    assert result[1] == pytest.approx(sensor_a[1], abs=1e-9)


@pytest.mark.parametrize("site", SITES, ids=lambda s: s.name)
def test_interpolate_at_full_length_matches_sensor_b(site):
    _sensor_a, sensor_b = sensor_coordinates(site)
    result = interpolate_position(site, site.pipe_length_m)
    assert result[0] == pytest.approx(sensor_b[0], abs=1e-9)
    assert result[1] == pytest.approx(sensor_b[1], abs=1e-9)


@pytest.mark.parametrize("site", SITES, ids=lambda s: s.name)
def test_interpolate_at_half_length_is_the_midpoint(site):
    sensor_a, sensor_b = sensor_coordinates(site)
    result = interpolate_position(site, site.pipe_length_m / 2.0)
    expected_lat = (sensor_a[0] + sensor_b[0]) / 2.0
    expected_lng = (sensor_a[1] + sensor_b[1]) / 2.0
    assert result[0] == pytest.approx(expected_lat, abs=1e-9)
    assert result[1] == pytest.approx(expected_lng, abs=1e-9)


def test_interpolate_clamps_beyond_pipe_length():
    site = SITES[0]
    beyond = interpolate_position(site, site.pipe_length_m * 2.0)
    _sensor_a, sensor_b = sensor_coordinates(site)
    assert beyond[0] == pytest.approx(sensor_b[0], abs=1e-9)
    assert beyond[1] == pytest.approx(sensor_b[1], abs=1e-9)


def test_interpolate_clamps_below_zero():
    site = SITES[0]
    below = interpolate_position(site, -50.0)
    sensor_a, _sensor_b = sensor_coordinates(site)
    assert below[0] == pytest.approx(sensor_a[0], abs=1e-9)
    assert below[1] == pytest.approx(sensor_a[1], abs=1e-9)


def test_sensor_separation_matches_constant():
    """Sensor A and B should be ON_MAP_SENSOR_SEPARATION_M apart on the map,
    regardless of the site's real (much larger) pipe_length_m."""
    site = SITES[0]
    sensor_a, sensor_b = sensor_coordinates(site)

    d_lat_m = (sensor_b[0] - sensor_a[0]) * 111_320.0
    d_lng_m = (sensor_b[1] - sensor_a[1]) * 111_320.0 * cos(radians(site.lat))
    separation_m = sqrt(d_lat_m ** 2 + d_lng_m ** 2)

    assert separation_m == pytest.approx(ON_MAP_SENSOR_SEPARATION_M, rel=1e-6)


def test_polygon_corners_returns_four_points_centred_on_site():
    site = SITES[0]
    corners = polygon_corners(site)
    assert len(corners) == 4

    avg_lat = sum(c[0] for c in corners) / 4.0
    avg_lng = sum(c[1] for c in corners) / 4.0
    assert avg_lat == pytest.approx(site.lat, abs=1e-9)
    assert avg_lng == pytest.approx(site.lng, abs=1e-9)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/aryansrivastva/Desktop/Main/aquaacoustics" && .venv/bin/pytest tests/test_sites.py -v`
Expected: FAIL / ERROR — `ModuleNotFoundError: No module named 'aquaacoustics'` (the package doesn't exist yet).

- [ ] **Step 3: Write the site registry and geometry module**

Create `src/aquaacoustics/sites.py`:

```python
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


def interpolate_position(site: Site, computed_distance_m: float) -> tuple:
    """
    Where the DETECTOR'S REAL RESULT is drawn on the map.

    Converts the detector's 1-D "distance from Sensor A" (in real metres,
    0..pipe_length_m) into a 2-D point along the Sensor-A-to-Sensor-B line
    drawn on the map, by straight linear interpolation on fraction of pipe
    length. This is never a random point - see the design doc's "honest
    marker" decision.
    """
    fraction = computed_distance_m / site.pipe_length_m
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
```

Create `src/aquaacoustics/__init__.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Users/aryansrivastva/Desktop/Main/aquaacoustics" && .venv/bin/pytest tests/test_sites.py -v`
Expected: PASS — all tests green (18 tests: 6 parametrized x 3 + 6 non-parametrized).

- [ ] **Step 5: Commit**

```bash
cd "/Users/aryansrivastva/Desktop/Main/aquaacoustics"
git add src/aquaacoustics/__init__.py src/aquaacoustics/sites.py tests/test_sites.py
git commit -m "feat: add preset site registry and map geometry helpers"
```

---

### Task 2: Add map dependencies

**Files:**
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: nothing.
- Produces: `folium` and `streamlit_folium` importable in the project's `.venv`, used by Task 3.

- [ ] **Step 1: Add the new dependencies to requirements.txt**

In `requirements.txt`, immediately after the `matplotlib>=3.8` line (in the "Everything the working system actually needs" section), add:

```
streamlit-folium>=0.23      # interactive Leaflet/OpenStreetMap map inside Streamlit
folium>=0.16                 # the underlying map-building library (OSM tiles, no API key)
```

- [ ] **Step 2: Install into the project's virtual environment**

Run: `cd "/Users/aryansrivastva/Desktop/Main/aquaacoustics" && .venv/bin/pip install streamlit-folium>=0.23 folium>=0.16`
Expected: `Successfully installed folium-... streamlit-folium-...` (plus any transitive deps like `branca`, `xyzservices`).

- [ ] **Step 3: Smoke-check the imports**

Run: `.venv/bin/python -c "import folium, streamlit_folium; print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
cd "/Users/aryansrivastva/Desktop/Main/aquaacoustics"
git add requirements.txt
git commit -m "chore: add streamlit-folium and folium dependencies"
```

---

### Task 3: Survey Map page

**Files:**
- Create: `src/aquaacoustics/map_view.py`
- Test: `tests/test_map_view.py`

**Interfaces:**
- Consumes: `aeld.SimulationConfig`, `aeld.run_detection` (exact signature: `run_detection(leak_position_m: float, config: SimulationConfig) -> DetectionResult`); `aquaacoustics.sites.{SITES, get_site, sensor_coordinates, interpolate_position, polygon_corners}` from Task 1.
- Produces: `render_map_page(role: str) -> None` (used by Task 5's `app.py`); `pick_random_leak_position(pipe_length_m: float, rng: np.random.Generator) -> float` and `build_history_entry(site: Site, result: DetectionResult, timestamp: datetime) -> dict` (both reused by Task 4's `detection_view.py` and covered by this task's tests). Session-state keys this task owns: `site_results` (`dict[str, DetectionResult]`), `history` (`list[dict]`), `selected_site` (`str`).

- [ ] **Step 1: Write the failing tests for the pure helpers**

Create `tests/test_map_view.py`:

```python
"""
test_map_view.py - pure helper functions behind the survey map page.
========================================================================
The Streamlit rendering itself (the folium map, buttons, layout) is covered
by the end-to-end AppTest in tests/test_app_flow.py, matching how the rest
of this project tests maths and logic directly and leaves UI layout to be
exercised by running the app. These tests cover the two testable pieces of
logic map_view.py owns: picking a leak position, and shaping a history row.
"""

from datetime import datetime, timezone

import numpy as np
import pytest

from aeld import SimulationConfig, run_detection
from aquaacoustics.map_view import pick_random_leak_position, build_history_entry
from aquaacoustics.sites import SITES


def test_pick_random_leak_position_stays_within_the_safe_range():
    rng = np.random.default_rng(7)
    pipe_length_m = 200.0
    for _ in range(200):
        position = pick_random_leak_position(pipe_length_m, rng)
        assert 0.05 * pipe_length_m <= position <= 0.95 * pipe_length_m


def test_pick_random_leak_position_is_reproducible_with_a_seeded_rng():
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    assert (pick_random_leak_position(200.0, rng_a)
            == pick_random_leak_position(200.0, rng_b))


def test_build_history_entry_has_the_expected_shape():
    site = SITES[0]
    config = SimulationConfig(
        pipe_length_m=site.pipe_length_m,
        wave_velocity_ms=site.wave_velocity_ms,
        noise_level=site.noise_level,
        random_seed=42,
    )
    result = run_detection(site.pipe_length_m / 2.0, config)
    timestamp = datetime(2026, 8, 27, 12, 0, 0, tzinfo=timezone.utc)

    entry = build_history_entry(site, result, timestamp)

    assert entry["site"] == site.name
    assert entry["state"] == site.state
    assert entry["estimated_position_m"] == pytest.approx(result.estimated_position_m)
    assert entry["true_position_m"] == pytest.approx(result.true_position_m)
    assert entry["error_m"] == pytest.approx(result.error_m)
    assert entry["is_confident"] == result.is_confident
    assert entry["timestamp"] == "2026-08-27 12:00:00 UTC"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/aryansrivastva/Desktop/Main/aquaacoustics" && .venv/bin/pytest tests/test_map_view.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aquaacoustics.map_view'`.

- [ ] **Step 3: Write the Survey Map page module**

Create `src/aquaacoustics/map_view.py`:

```python
"""
map_view.py - the interactive Survey Map page.
==================================================
Shows every preset monitoring site on a Folium/OpenStreetMap map of India.
Admin can select a site and trigger detection, which runs the REAL aeld
GCC-PHAT pipeline (src/aeld/detector.py) against that site's preset pipe and
draws the result - never a random point - on the map. Field Staff see the
same map read-only.

See docs/superpowers/specs/2026-08-27-aquaacoustics-integration-design.md
for the full design and the reasoning behind the "honest marker" rule.
"""

from datetime import datetime, timezone

import folium
import numpy as np
import streamlit as st
from streamlit_folium import st_folium

from aeld import SimulationConfig, run_detection
from .sites import (
    SITES, get_site, sensor_coordinates, interpolate_position, polygon_corners,
)


INDIA_CENTER = (22.9734, 78.6569)     # geographic centre, initial map view
INDIA_DEFAULT_ZOOM = 5
SITE_ZOOM = 17


# ============================================================================
# Pure helpers (unit tested in tests/test_map_view.py)
# ============================================================================

def pick_random_leak_position(pipe_length_m: float,
                              rng: np.random.Generator) -> float:
    """
    Pick an unknown-to-the-detector leak position, same rule the original
    AELD app used: stay slightly inside the ends, since a leak sitting
    exactly on a sensor is a degenerate edge case, not a useful demo.
    """
    return float(rng.uniform(0.05 * pipe_length_m, 0.95 * pipe_length_m))


def build_history_entry(site, result, timestamp: datetime) -> dict:
    """Shape one row of the session's trigger history for display."""
    return {
        "site": site.name,
        "state": site.state,
        "estimated_position_m": result.estimated_position_m,
        "true_position_m": result.true_position_m,
        "error_m": result.error_m,
        "is_confident": result.is_confident,
        "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
    }


# ============================================================================
# Session state
# ============================================================================

def _ensure_session_state():
    if "site_results" not in st.session_state:
        st.session_state["site_results"] = {}
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "selected_site" not in st.session_state:
        st.session_state["selected_site"] = SITES[0].name


def _trigger_detection(site) -> None:
    """Run the real detector on `site`'s preset pipe and record the result."""
    rng = np.random.default_rng()
    leak_position = pick_random_leak_position(site.pipe_length_m, rng)

    config = SimulationConfig(
        pipe_length_m=site.pipe_length_m,
        wave_velocity_ms=site.wave_velocity_ms,
        noise_level=site.noise_level,
    )

    with st.spinner(f"Listening at {site.name}..."):
        result = run_detection(leak_position, config)

    st.session_state["site_results"][site.name] = result

    timestamp = datetime.now(timezone.utc)
    st.session_state["history"].append(build_history_entry(site, result, timestamp))


# ============================================================================
# The page
# ============================================================================

def render_map_page(role: str) -> None:
    """Render the Survey Map page. `role` is 'Admin' or 'Field Staff'."""
    _ensure_session_state()

    st.title("Survey Map")
    st.caption("Pick a monitoring site to see or run its leak detection.")

    site_names = [site.name for site in SITES]
    selected_name = st.selectbox(
        "Monitoring site",
        site_names,
        index=site_names.index(st.session_state["selected_site"]),
    )
    st.session_state["selected_site"] = selected_name
    site = get_site(selected_name)

    st.caption(f"{site.state} - preset pipe: {site.pipe_length_m:.0f} m, "
               f"{site.wave_velocity_ms:.0f} m/s, "
               f"noise level {site.noise_level:.2f}")

    if role == "Admin":
        if st.button("Trigger Detection", type="primary"):
            _trigger_detection(site)
    else:
        st.caption(
            "Field Staff view: read-only. Ask an Admin to trigger detection."
        )

    result = st.session_state["site_results"].get(site.name)

    if result is not None:
        map_center = (site.lat, site.lng)
        zoom = SITE_ZOOM
    else:
        map_center = INDIA_CENTER
        zoom = INDIA_DEFAULT_ZOOM

    fmap = folium.Map(location=map_center, zoom_start=zoom, tiles="OpenStreetMap")

    for other_site in SITES:
        folium.CircleMarker(
            location=(other_site.lat, other_site.lng),
            radius=6,
            color="#34495E",
            fill=True,
            fill_opacity=0.8,
            tooltip=f"{other_site.name} ({other_site.state})",
        ).add_to(fmap)

    if result is not None:
        folium.Polygon(
            locations=polygon_corners(site),
            color="#34495E",
            weight=2,
            fill=True,
            fill_opacity=0.08,
            tooltip="Surveyed area",
        ).add_to(fmap)

        sensor_a, sensor_b = sensor_coordinates(site)
        folium.Marker(
            location=sensor_a, tooltip="Sensor A (0 m)",
            icon=folium.Icon(color="blue", icon="microphone", prefix="fa"),
        ).add_to(fmap)
        folium.Marker(
            location=sensor_b,
            tooltip=f"Sensor B ({site.pipe_length_m:.0f} m)",
            icon=folium.Icon(color="orange", icon="microphone", prefix="fa"),
        ).add_to(fmap)

        # The honest marker: always the detector's real computed position,
        # never a random point. See the design doc for why this matters.
        leak_point = interpolate_position(site, result.estimated_position_m)
        marker_color = "green" if result.is_confident else "gray"
        folium.Marker(
            location=leak_point,
            tooltip=(f"Detected leak: {result.estimated_position_m:.1f} m "
                     f"from Sensor A"
                     + ("" if result.is_confident else " (low confidence)")),
            icon=folium.Icon(color=marker_color, icon="exclamation-triangle",
                             prefix="fa"),
        ).add_to(fmap)

    st_folium(fmap, width=None, height=520, returned_objects=[])

    if result is not None:
        confidence_note = ("" if result.is_confident else
                           " (flagged low-confidence - see Leak Detection Detail)")
        st.info(
            f"Last result for **{site.name}**: leak found "
            f"{result.estimated_position_m:.1f} m from Sensor A"
            f"{confidence_note}. Open **Leak Detection Detail** in the "
            f"sidebar for the full explanation."
        )
    else:
        st.info("No detection has been triggered for this site yet.")

    st.divider()
    st.subheader("Session history")
    history = st.session_state["history"]
    if not history:
        st.caption("Nothing triggered yet this session.")
    else:
        st.dataframe(
            [
                {
                    "Site": row["site"],
                    "State": row["state"],
                    "Found at (m)": round(row["estimated_position_m"], 2),
                    "Actual (m)": round(row["true_position_m"], 2),
                    "Error (m)": round(row["error_m"], 3),
                    "Confident?": "Yes" if row["is_confident"] else "No",
                    "When": row["timestamp"],
                }
                for row in reversed(history)
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "In-memory only for this session - resets if the server "
            "restarts. This is a prototype, not a durable log."
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Users/aryansrivastva/Desktop/Main/aquaacoustics" && .venv/bin/pytest tests/test_map_view.py -v`
Expected: PASS — 3 tests green.

- [ ] **Step 5: Commit**

```bash
cd "/Users/aryansrivastva/Desktop/Main/aquaacoustics"
git add src/aquaacoustics/map_view.py tests/test_map_view.py
git commit -m "feat: add Survey Map page with honest-marker detection trigger"
```

---

### Task 4: Leak Detection Detail page

**Files:**
- Create: `src/aquaacoustics/detection_view.py`
- Test: `tests/test_detection_view.py`

**Interfaces:**
- Consumes: `aeld.SimulationConfig`, `aeld.run_detection`; `aquaacoustics.sites.get_site` (Task 1); `aquaacoustics.map_view.pick_random_leak_position` (Task 3); reads session-state keys `selected_site` and `site_results` (owned by Task 3).
- Produces: `render_detection_page() -> None` (used by Task 5's `app.py`); `build_override_config(base_result: DetectionResult, pipe_length_m: float, wave_velocity_ms: float, noise_level: float) -> SimulationConfig` (covered by this task's tests).

- [ ] **Step 1: Write the failing test for the pure helper**

Create `tests/test_detection_view.py`:

```python
"""
test_detection_view.py - pure helper behind the Leak Detection Detail page's
"Advanced: override this site's pipe" re-run control.
==============================================================================
Rendering itself is covered by the end-to-end AppTest in
tests/test_app_flow.py. This test covers the one piece of standalone logic
detection_view.py owns: building a re-run config that changes only the three
exposed sliders and carries every other setting over from the previous run.
"""

import pytest

from aeld import SimulationConfig, run_detection
from aquaacoustics.detection_view import build_override_config
from aquaacoustics.sites import SITES


def test_build_override_config_changes_only_the_three_exposed_fields():
    site = SITES[0]
    base_config = SimulationConfig(
        pipe_length_m=site.pipe_length_m,
        wave_velocity_ms=site.wave_velocity_ms,
        noise_level=site.noise_level,
        attenuation_per_m=0.007,
        sample_rate_hz=16000,
    )
    base_result = run_detection(site.pipe_length_m / 2.0, base_config)

    new_config = build_override_config(base_result, 999.0, 1500.0, 0.9)

    assert new_config.pipe_length_m == 999.0
    assert new_config.wave_velocity_ms == 1500.0
    assert new_config.noise_level == 0.9
    assert new_config.attenuation_per_m == 0.007
    assert new_config.sample_rate_hz == 16000
    assert new_config.leak_band_hz == base_config.leak_band_hz
    assert new_config.confidence_threshold == base_config.confidence_threshold
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/aryansrivastva/Desktop/Main/aquaacoustics" && .venv/bin/pytest tests/test_detection_view.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'aquaacoustics.detection_view'`.

- [ ] **Step 3: Write the Leak Detection Detail page module**

Create `src/aquaacoustics/detection_view.py`:

```python
"""
detection_view.py - the Leak Detection Detail page for one site.
====================================================================
This is the AELD prototype's existing "how it figured that out" narrative
(waveforms, the GCC-PHAT correlation spike, the arithmetic, the honesty
checks), unchanged in substance, now scoped to whichever site was last
triggered on the Survey Map page.
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from aeld import SimulationConfig, run_detection
from .sites import get_site
from .map_view import pick_random_leak_position


COLOR_A = "#2E86DE"
COLOR_B = "#EE5A24"
COLOR_CORR = "#8854D0"
COLOR_TRUE = "#20BF6B"


def style_axes(ax):
    """Apply one consistent, low-clutter look to a matplotlib axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    return ax


def describe_noise(level: float) -> str:
    """Translate the abstract noise number into something physically meaningful."""
    if level <= 0.05:
        return "Silent - a perfect laboratory"
    if level <= 0.25:
        return "Quiet - a rural pipe at night"
    if level <= 0.5:
        return "Normal - a typical residential street"
    if level <= 0.8:
        return "Noisy - traffic passing overhead"
    return "Very noisy - this will probably defeat the system"


def human_scale_comparison(error_m: float, pipe_length_m: float) -> str:
    """Put the accuracy in a form a person can picture."""
    if error_m <= 0:
        return "The estimate landed on the exact answer."

    ratio = pipe_length_m / error_m
    reference_km = 344.0
    equivalent_m = reference_km * 1000.0 / ratio

    if equivalent_m < 0.01:
        equivalent = f"{equivalent_m * 1000:.1f} millimetres"
    elif equivalent_m < 1.0:
        equivalent = f"{equivalent_m * 100:.1f} centimetres"
    elif equivalent_m < 1000:
        equivalent = f"{equivalent_m:.0f} metres"
    else:
        equivalent = f"{equivalent_m / 1000:.1f} kilometres"

    return (f"That is an accuracy of **1 part in {ratio:,.0f}** - the "
            f"equivalent of measuring the {reference_km:,.0f} km from London "
            f"to Paris to within **{equivalent}**.")


def format_distance(metres: float) -> str:
    """Pick readable units for an error that may span many orders of magnitude."""
    if metres < 0.001:
        return f"{metres * 1000:.3f} mm"
    if metres < 1.0:
        return f"{metres * 100:.2f} cm"
    return f"{metres:.2f} m"


def build_override_config(base_result, pipe_length_m: float,
                          wave_velocity_ms: float,
                          noise_level: float) -> SimulationConfig:
    """
    Build a re-run config for the "Advanced: override this site's pipe"
    control. Changes only the three exposed sliders; every other setting
    (attenuation, sample rate, frequency band, confidence threshold) carries
    over unchanged from the previous run at this site.
    """
    base_config = base_result.config
    return SimulationConfig(
        pipe_length_m=pipe_length_m,
        wave_velocity_ms=wave_velocity_ms,
        noise_level=noise_level,
        attenuation_per_m=base_config.attenuation_per_m,
        sample_rate_hz=base_config.sample_rate_hz,
        leak_band_hz=base_config.leak_band_hz,
        confidence_threshold=base_config.confidence_threshold,
    )


def render_detection_page() -> None:
    """The answer in plain language, then the story of how it was reached,
    for whichever site was last selected on the Survey Map page."""

    st.title("Leak Detection Detail")

    selected_name = st.session_state.get("selected_site")
    site_results = st.session_state.get("site_results", {})

    if not selected_name or selected_name not in site_results:
        st.info(
            "Nothing has been triggered yet. Go to **Survey Map**, pick a "
            "site, and press **Trigger Detection**."
        )
        return

    site = get_site(selected_name)
    result = site_results[selected_name]
    config = result.config

    st.caption(f"{site.name}, {site.state}")

    # ------------------------------------------------------------------
    # 1. THE ANSWER, in plain language
    # ------------------------------------------------------------------
    if result.is_confident:
        st.markdown(
            f"## The leak is {result.estimated_position_m:.2f} metres "
            f"from Sensor A."
        )
        st.markdown(
            f"It was actually at **{result.true_position_m:.2f} m**, so the "
            f"system was off by **{format_distance(result.error_m)}** - "
            f"found in **{result.time_total_ms:.1f} milliseconds**, using "
            f"nothing but two sensors and no digging."
        )
    else:
        st.markdown("## Something is leaking, but I cannot tell you where.")
        st.markdown(
            f"The background noise was too high to pick the leak's hiss "
            f"out reliably. The system's best guess would have been "
            f"**{result.estimated_position_m:.2f} m** (the leak was really "
            f"at **{result.true_position_m:.2f} m**), but it has flagged "
            f"that answer as untrustworthy rather than sending you to dig "
            f"there."
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("Where the system said", f"{result.estimated_position_m:.2f} m")
    m2.metric("Where it actually was", f"{result.true_position_m:.2f} m")
    m3.metric("How far off", format_distance(result.error_m))

    if result.is_confident:
        st.info(
            f"**Why this is remarkable.** "
            f"{human_scale_comparison(result.error_m, config.pipe_length_m)} "
            f"And it took {result.time_total_ms:.1f} milliseconds - roughly "
            f"{max(1, int(150 / max(result.time_total_ms, 0.01)))}x faster "
            f"than a blink of an eye."
        )
    else:
        st.error(
            "**The system knew it was unsure - and said so.** A tool that "
            "confidently reports a wrong location sends a crew to dig up "
            "the wrong road. One that admits uncertainty tells them to come "
            "back with better equipment."
        )

    st.divider()

    # ------------------------------------------------------------------
    # 2. HOW IT WORKS
    # ------------------------------------------------------------------
    st.header("How it figured that out, in three steps")

    st.subheader("Step 1: Two sensors listen to the same hiss")
    st.markdown(
        """
    A leak is not silent - escaping water makes a constant hissing sound.
    Both sensors pick up **the very same hiss**, because there is only one
    leak making it. Notice only this: the closer sensor's trace is
    **taller**, because sound fades as it travels.
    """
    )

    zoom_ms = st.slider(
        "How much of the recording to show (milliseconds)",
        min_value=2, max_value=100, value=20, step=1,
        key="detail_zoom_ms",
    )
    n_show = min(int(config.sample_rate_hz * zoom_ms / 1000.0),
                 result.time_s.size)

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    ax_a.plot(result.time_s[:n_show] * 1000, result.sensor_a[:n_show],
              color=COLOR_A, linewidth=0.9)
    ax_a.set_ylabel("Loudness")
    ax_a.set_title(
        f"Sensor A  -  {result.true_position_m:.0f} m from the leak",
        loc="left", fontsize=11, fontweight="bold")
    style_axes(ax_a)

    distance_b = config.pipe_length_m - result.true_position_m
    ax_b.plot(result.time_s[:n_show] * 1000, result.sensor_b[:n_show],
              color=COLOR_B, linewidth=0.9)
    ax_b.set_ylabel("Loudness")
    ax_b.set_xlabel("Time (milliseconds)")
    ax_b.set_title(
        f"Sensor B  -  {distance_b:.0f} m from the leak",
        loc="left", fontsize=11, fontweight="bold")
    style_axes(ax_b)

    limit = max(np.abs(result.sensor_a[:n_show]).max(),
                np.abs(result.sensor_b[:n_show]).max()) * 1.1
    ax_a.set_ylim(-limit, limit)
    ax_b.set_ylim(-limit, limit)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Step 2: One of them heard it a fraction sooner")
    delay_ms = abs(result.true_tau_s) * 1000.0
    nearer = "A" if result.true_tau_s < 0 else "B"
    st.markdown(
        f"""
    Sound takes time to travel. The leak is closer to **Sensor {nearer}**,
    so that sensor heard the hiss **{delay_ms:.1f} milliseconds** before
    the other one. That gap is the entire secret - measure it accurately
    and you know the position.

    The catch: **{delay_ms:.1f} ms is far too small to spot by eye.** So
    the computer does something cleverer.
    """
    )

    st.subheader(
        "Step 3: Slide one recording against the other until they match"
    )
    st.markdown(
        """
    Slide the second recording back and forth against the first. At most
    positions the two disagree - noise against noise. But at **one**
    position, the shared hiss lines up and they agree strongly. The tall
    spike below is that moment. **That spike is the measurement.**
    """
    )

    fig2, ax = plt.subplots(figsize=(12, 4.5))
    lags_ms = result.lags_s * 1000.0
    ax.plot(lags_ms, result.correlation, color=COLOR_CORR, linewidth=1.0,
            label="How well the two recordings agree")
    peak_lag_ms = result.estimated_tau_s * 1000.0
    ax.axvline(peak_lag_ms, color=COLOR_CORR, linewidth=1.8, alpha=0.9,
               label=f"Best match at {peak_lag_ms:+.3f} ms")
    ax.plot([peak_lag_ms], [result.correlation[result.peak_index]],
            marker="v", markersize=11, color=COLOR_CORR, zorder=5)
    true_lag_ms = result.true_tau_s * 1000.0
    ax.axvline(true_lag_ms, color=COLOR_TRUE, linestyle="--", linewidth=1.8,
               label=f"The true answer: {true_lag_ms:+.3f} ms")
    ax.axvline(0.0, color="grey", linestyle=":", linewidth=1.2, alpha=0.7,
               label="No slide (leak would be dead centre)")
    ax.set_xlabel("How far the second recording was slid (milliseconds)")
    ax.set_ylabel("Agreement")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    style_axes(ax)
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.divider()

    # ------------------------------------------------------------------
    # 3. CAN WE TRUST IT?
    # ------------------------------------------------------------------
    st.header("Can we trust this answer?")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Spike height above background",
                  f"{result.peak_sharpness:.1f}x",
                  delta=f"needs {config.confidence_threshold:.0f}x or more",
                  delta_color="off")
    with c2:
        if result.is_confident:
            st.success(
                f"**Trustworthy.** The spike was "
                f"{result.peak_sharpness:.1f} times the background level, "
                f"comfortably past the {config.confidence_threshold:.0f}x "
                f"bar."
            )
        else:
            st.error(
                f"**Not trustworthy.** The spike was only "
                f"{result.peak_sharpness:.1f} times the background, below "
                f"the {config.confidence_threshold:.0f}x bar. Do not dig on "
                f"this reading."
            )

    st.divider()

    # ------------------------------------------------------------------
    # 4. TRY A DIFFERENT SCENARIO FOR THIS SITE
    # ------------------------------------------------------------------
    with st.expander("Advanced: override this site's pipe and re-run"):
        st.caption(
            f"{site.name}'s preset is {site.pipe_length_m:.0f} m, "
            f"{site.wave_velocity_ms:.0f} m/s, noise "
            f"{site.noise_level:.2f}. Adjust below and re-run to try a "
            f"different scenario at this same site."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            override_length = st.slider(
                "Pipe length (m)", min_value=20.0, max_value=1000.0,
                value=float(config.pipe_length_m), step=10.0)
        with col2:
            override_velocity = st.slider(
                "Wave velocity (m/s)", min_value=300.0, max_value=2000.0,
                value=float(config.wave_velocity_ms), step=10.0)
        with col3:
            override_noise = st.slider(
                "Noise level", min_value=0.0, max_value=2.0,
                value=float(config.noise_level), step=0.05)
            st.caption(describe_noise(override_noise))

        if st.button("Re-run detection with these settings"):
            new_config = build_override_config(
                result, override_length, override_velocity, override_noise)
            rng = np.random.default_rng()
            leak_position = pick_random_leak_position(override_length, rng)
            new_result = run_detection(leak_position, new_config)
            st.session_state["site_results"][selected_name] = new_result
            st.rerun()

    # ------------------------------------------------------------------
    # 5. FOR THE CURIOUS
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("For the curious")

    with st.expander("How long each part took"):
        t1, t2, t3 = st.columns(3)
        t1.metric("Generating the recordings", f"{result.time_acquire_ms:.2f} ms")
        t2.metric("Finding the match", f"{result.time_gccphat_ms:.2f} ms")
        t3.metric("Total", f"{result.time_total_ms:.2f} ms")

    with st.expander("The arithmetic that turns a time into a distance"):
        st.markdown(
            "With `L` the pipe length, `v` the speed of sound in the pipe, "
            "and `t` the measured time gap:"
        )
        st.latex(r"\text{distance from Sensor A} = \frac{L + v \cdot t}{2}")
        st.code(
            f"t = {result.estimated_tau_s:+.8f} s\n"
            f"L = {config.pipe_length_m:.2f} m\n"
            f"v = {config.wave_velocity_ms:.2f} m/s\n\n"
            f"distance = ({config.pipe_length_m:.2f} + "
            f"{config.wave_velocity_ms:.2f} x {result.estimated_tau_s:+.8f}) / 2\n"
            f"         = {result.estimated_position_m:.4f} m\n\n"
            f"Real answer: {result.true_position_m:.4f} m\n"
            f"Off by:      {result.error_m:.6f} m",
            language="text",
        )

    with st.expander("Honest limitations"):
        st.markdown(
            f"""
        - **This is a simulation.** No real pipe was recorded.
        - **One straight pipe per site.** No junctions, branches or bends.
        - **The speed of sound is assumed known.** The answer scales
          directly with it.
        - **The {config.confidence_threshold:.0f}x trust bar was calibrated
          on this simulator, not a proof.**
        - **The map's on-screen sensor spacing is illustrative, not to
          scale.** This site's real pipe length is
          {site.pipe_length_m:.0f} m; only the leak marker's *position along
          the line* is real, proportional to the detector's actual result.
        """
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Users/aryansrivastva/Desktop/Main/aquaacoustics" && .venv/bin/pytest tests/test_detection_view.py -v`
Expected: PASS — 1 test green.

- [ ] **Step 5: Run the full test suite so far**

Run: `.venv/bin/pytest tests/ -v`
Expected: PASS — all tests in `test_gccphat.py`, `test_signals.py`, `test_sites.py`, `test_map_view.py`, `test_detection_view.py` green.

- [ ] **Step 6: Commit**

```bash
cd "/Users/aryansrivastva/Desktop/Main/aquaacoustics"
git add src/aquaacoustics/detection_view.py tests/test_detection_view.py
git commit -m "feat: add Leak Detection Detail page with scenario override"
```

---

### Task 5: App router, role picker, and end-to-end wiring

**Files:**
- Modify: `app.py` (full rewrite)
- Test: `tests/test_app_flow.py`

**Interfaces:**
- Consumes: `aquaacoustics.map_view.render_map_page(role: str)` (Task 3), `aquaacoustics.detection_view.render_detection_page()` (Task 4).
- Produces: the runnable app. Session-state key `role` (`"Admin"` or `"Field Staff"`), owned by this task.

- [ ] **Step 1: Write the end-to-end test against the CURRENT (old) app.py**

Create `tests/test_app_flow.py`:

```python
"""
test_app_flow.py - end-to-end wiring, run without a browser.
================================================================
Uses Streamlit's official AppTest framework to drive the real app.py: role
selection, site selection, triggering detection, and switching pages -
proving the pieces built in isolation (sites.py, map_view.py,
detection_view.py) are actually wired together correctly. This does not
replace tests/test_gccphat.py etc. - it proves the INTEGRATION, not the
maths (which is already proven elsewhere).
"""

from streamlit.testing.v1 import AppTest


def _run_app():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert not at.exception
    return at


def test_app_starts_on_survey_map_with_no_result_yet():
    at = _run_app()
    assert any("Survey Map" in t.value for t in at.title)
    assert at.session_state["role"] == "Admin"


def test_admin_can_trigger_detection_for_the_selected_site():
    at = _run_app()

    assert at.session_state["role"] == "Admin"
    selected_site = at.session_state["selected_site"]

    at.button[0].click().run(timeout=30)
    assert not at.exception

    assert selected_site in at.session_state["site_results"]
    result = at.session_state["site_results"][selected_site]
    assert result.estimated_position_m >= 0.0
    assert len(at.session_state["history"]) == 1


def test_field_staff_sees_no_trigger_button():
    at = _run_app()
    at.sidebar.radio[0].set_value("Field Staff").run(timeout=30)
    assert not at.exception
    assert at.session_state["role"] == "Field Staff"
    assert len(at.button) == 0


def test_detection_detail_page_renders_the_triggered_result():
    at = _run_app()
    at.button[0].click().run(timeout=30)
    assert not at.exception

    at.sidebar.radio[1].set_value("Leak Detection Detail").run(timeout=30)
    assert not at.exception
    assert any("Leak Detection Detail" in t.value for t in at.title)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "/Users/aryansrivastva/Desktop/Main/aquaacoustics" && .venv/bin/pytest tests/test_app_flow.py -v`
Expected: FAIL — the current `app.py` has no `role` or `selected_site` in `session_state`, and no "Survey Map" title, so the first assertion fails (`KeyError` or `AssertionError`).

- [ ] **Step 3: Rewrite app.py**

Replace the full contents of `app.py` with:

```python
"""
============================================================================
 app.py - AquaAcoustics: map-driven acoustic leak detection
============================================================================
Run with:
    streamlit run app.py

Two integrated screens:
    Survey Map              - pick a monitoring site, trigger real GCC-PHAT
                              detection, see the honest result on a map.
    Leak Detection Detail   - the full "how it figured that out" story for
                              whichever site was last triggered.

A role picker ("Admin" / "Field Staff") is a plain, passwordless selector -
cosmetic for this prototype, not a security boundary. Admin can trigger
detection; Field Staff can only view results.

All leak-detection physics live untouched in src/aeld/. All map + role
integration logic lives in src/aquaacoustics/. This file is routing only.
============================================================================
"""

import sys
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aquaacoustics.map_view import render_map_page              # noqa: E402
from aquaacoustics.detection_view import render_detection_page  # noqa: E402


st.set_page_config(
    page_title="AquaAcoustics",
    page_icon="~",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("AquaAcoustics")
st.sidebar.caption("Leak detection (AELD) across monitoring sites")

role = st.sidebar.radio("View as", ["Admin", "Field Staff"])
st.session_state["role"] = role

st.sidebar.divider()

page = st.sidebar.radio(
    "Go to",
    ["Survey Map", "Leak Detection Detail"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.markdown(
    """
**The idea in three lines**

A leak hisses continuously.

A sensor at each end of the pipe hears that hiss - but the nearer one hears
it a fraction of a second sooner.

That fraction tells you where the leak is.
    """
)

if page == "Survey Map":
    render_map_page(role)
else:
    render_detection_page()

st.sidebar.divider()
st.sidebar.caption(
    "Prototype. The physics simulation and the leak-locating maths are real "
    "and unit-tested; sites and their pipe parameters are illustrative "
    "presets, and the role picker is not a security boundary."
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "/Users/aryansrivastva/Desktop/Main/aquaacoustics" && .venv/bin/pytest tests/test_app_flow.py -v`
Expected: PASS — all 4 tests green.

If `at.title` / `at.sidebar.radio` attribute names don't match exactly (Streamlit's `AppTest` API can shift slightly between versions), consult `.venv/bin/python -c "from streamlit.testing.v1 import AppTest; help(AppTest)"` and adjust the test's element accessors to match — the underlying assertions (role defaults to Admin, triggering populates `site_results`/`history`, Field Staff has no button, detail page renders) stay the same regardless.

- [ ] **Step 5: Run the full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: PASS — every test across all six test files green.

- [ ] **Step 6: Manual smoke check**

Run: `.venv/bin/streamlit run app.py` and in the browser:
1. Confirm the sidebar shows "View as: Admin / Field Staff" and "Go to: Survey Map / Leak Detection Detail".
2. On Survey Map as Admin, pick a site, click "Trigger Detection" — confirm the map re-centers on the site, draws the polygon, two sensor markers, and one leak marker, and that a row appears in "Session history".
3. Switch to Leak Detection Detail — confirm it shows the same site's result with waveforms and the correlation plot.
4. Switch role to Field Staff on Survey Map — confirm the "Trigger Detection" button is gone.
5. Stop the server (Ctrl+C) once confirmed.

- [ ] **Step 7: Commit**

```bash
cd "/Users/aryansrivastva/Desktop/Main/aquaacoustics"
git add app.py tests/test_app_flow.py
git commit -m "feat: wire role picker + map/detail router into app.py"
```

---

### Task 6: Documentation and cleanup

**Files:**
- Modify: `claude.md` (full rewrite)
- Modify: `context.md` (full rewrite)
- Delete: `EXPLANATION.txt`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: nothing consumed by other tasks — this is the final task.

- [ ] **Step 1: Remove the stale explanation file**

`EXPLANATION.txt` documents the pre-integration, single-page `app.py` in detail; after Task 5 it no longer matches the code, and leaving it would misrepresent what's actually in the repo.

```bash
cd "/Users/aryansrivastva/Desktop/Main/aquaacoustics"
git rm EXPLANATION.txt
```

- [ ] **Step 2: Rewrite claude.md**

Replace the full contents of `claude.md` with:

```markdown
# claude.md — AquaAcoustics: Project Brief & Goals

This file stores the originating context and goals for AquaAcoustics, so any
future session (human or model) can pick the work up with full context.

---

## Project

**AquaAcoustics** integrates two things into one Streamlit app:

1. **AELD (Acoustic Emission Leak Detection)** — locates a leak in a
   pressurised pipe via GCC-PHAT Time Difference of Arrival, reused
   unmodified from the standalone prototype at
   `github.com/FunkySplash49/aeld-prototype`
   (`aeld-prototype.streamlit.app`). Its own history is in that repo's
   `claude.md`.
2. **A map layer** — an interactive `streamlit-folium` survey map of India
   showing preset monitoring sites, inspired by a documented (never locally
   built) Next.js geospatial survey tool the user described. That tool's
   own documentation file is not part of this repo; only its product
   concept (map, Admin/Field-Staff roles, polygon + pointer, session
   history) informed this design.

Leak detection is the sub-component; the map is the new integration layer
on top of it.

## Origin

Built during a chat session that started from the existing, deployed AELD
prototype. The user asked to integrate it with a separately-documented map
tool under the combined name AquaAcoustics, explicitly leaving the
implementation approach open ("no need to exactly copy instructions,
whatever suits best for the project works").

Full design reasoning — including why a single Streamlit app was chosen
over a Next.js/TypeScript split, and the "honest marker" rule (the map's
leak marker must always be the detector's real computed position, never a
random point) — is in
`docs/superpowers/specs/2026-08-27-aquaacoustics-integration-design.md`.
The implementation plan is in
`docs/superpowers/plans/2026-08-27-aquaacoustics-integration.md`.

## Goals

Inherited from AELD, unchanged:

1. **Locate a leak accurately.** Sub-metre accuracy at realistic noise
   levels — unchanged, since the physics is reused verbatim.
2. **Be explainable live.** Every stage visible and narratable.
3. **Be honest.** Never present a stub as a real measurement, and never
   report a location the underlying data does not support. This is why the
   map's leak marker is always the real computed result, never the
   original map tool's random pointer-in-polygon.
4. **Be understandable without background knowledge.**
5. **Be verifiable.** Covered by tests that would fail if the maths — or
   now, the map geometry and the app's wiring — broke.

New for this integration:

6. **Multi-site, not single-pipe.** The prototype now represents several
   independently-configured monitoring sites rather than one global pipe.
7. **No fake precision.** Preset sites and their pipe parameters are
   illustrative, clearly presented as such — not real survey data.

## Scope decisions

Agreed with the user during brainstorming before implementation:

1. **Single Streamlit stack, not Next.js.** Rebuilding the map tool's UI in
   Next.js would have meant either porting the already-tested GCC-PHAT
   maths to TypeScript (correctness risk) or running two separate
   services. The user chose one Streamlit-deployable app instead.
2. **streamlit-folium on OpenStreetMap tiles**, not a separate "OpenMaps
   API" — Folium's default tiles already are OpenStreetMap.
3. **The map's leak marker is always the real computed result.** The
   original map tool's documentation described a *random* pointer dropped
   in a polygon on "Trigger Survey" — a cosmetic flourish with no data
   behind it. Reusing that literally once the trigger is wired to real
   physics would violate Goal 3. The polygon stays (visual context); the
   marker does not.
4. **Cosmetic role picker, no passwords.** The source documentation
   included plaintext demo credentials; none were copied into this
   project. "View as: Admin / Field Staff" is a plain, passwordless
   selector, explicitly not a security boundary.
5. **This project lives in its own directory** (`aquaacoustics/`, renamed
   from a since-superseded backup of the AELD repo) as a fresh, standalone
   git repository with no push or deployment yet.
6. **Six preset sites, one per state**, each with distinct illustrative
   pipe parameters, rather than a single global pipe configuration.

## Non-goals

- No real GPS/pipe survey data — sites and their pipe parameters are
  illustrative presets.
- No real authentication — the role picker is cosmetic.
- No persistence across sessions or server restarts — session history
  lives only in `st.session_state`.
- No Next.js/TypeScript port.
- No change to `src/aeld`'s logic, tests, or the standalone AELD
  deployment, which remains untouched and live.
```

- [ ] **Step 3: Rewrite context.md**

Replace the full contents of `context.md` with:

```markdown
# context.md — AquaAcoustics progress tracker

## Status: implemented, locally tested, not yet pushed or deployed

## File structure

```
aquaacoustics/
├── app.py                          # router: role picker + Survey Map / Leak Detection Detail
├── requirements.txt                 # aeld deps + streamlit-folium, folium
├── src/
│   ├── aeld/                        # unchanged, reused from the standalone AELD prototype
│   └── aquaacoustics/
│       ├── sites.py                  # 6 preset sites + map geometry (interpolate_position, etc.)
│       ├── map_view.py               # Survey Map page (streamlit-folium, trigger detection)
│       └── detection_view.py         # Leak Detection Detail page (adapted from AELD's result page)
├── tests/
│   ├── test_gccphat.py, test_signals.py   # unchanged, from AELD
│   ├── test_sites.py                       # site registry + geometry
│   ├── test_map_view.py                    # pick_random_leak_position, build_history_entry
│   ├── test_detection_view.py              # build_override_config
│   └── test_app_flow.py                    # end-to-end AppTest (role, trigger, page switch)
└── docs/superpowers/
    ├── specs/2026-08-27-aquaacoustics-integration-design.md
    └── plans/2026-08-27-aquaacoustics-integration.md
```

## Completed steps

1. Repo renamed from a backup of the AELD project to `aquaacoustics/`,
   reinitialized as a fresh standalone git repo (no shared history, no
   remote).
2. Design spec written and committed.
3. Implementation plan written and committed.
4. `src/aquaacoustics/sites.py` — 6 preset sites, map geometry helpers,
   fully unit tested.
5. `streamlit-folium` + `folium` added as dependencies.
6. `src/aquaacoustics/map_view.py` — Survey Map page: site selector,
   Admin-only trigger, honest leak marker, session history table.
7. `src/aquaacoustics/detection_view.py` — Leak Detection Detail page,
   adapted from AELD's result page, plus a per-site scenario override.
8. `app.py` rewritten as the router: role picker, page navigation, wiring
   both pages together. End-to-end `AppTest` coverage added.
9. Docs (`claude.md`, this file) rewritten for the merged project; the
   stale `EXPLANATION.txt` (described the pre-integration `app.py`) was
   removed.

## Not done (explicitly out of scope for now)

- No `git push` — this repo has no remote configured.
- No deployment (Streamlit Community Cloud or otherwise).
- No Next.js/TypeScript component of any kind.
- No real authentication, no real GPS survey data, no cross-session
  persistence — see `claude.md` "Non-goals".
```

- [ ] **Step 4: Run the full test suite one last time**

Run: `cd "/Users/aryansrivastva/Desktop/Main/aquaacoustics" && .venv/bin/pytest tests/ -v`
Expected: PASS — every test green.

- [ ] **Step 5: Commit**

```bash
cd "/Users/aryansrivastva/Desktop/Main/aquaacoustics"
git add claude.md context.md
git commit -m "docs: rewrite claude.md and context.md for AquaAcoustics; remove stale EXPLANATION.txt"
```
