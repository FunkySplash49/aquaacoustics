# AquaAcoustics — Map + Leak Detection Integration Design

## Origin

This project starts from two existing things, combined into one:

1. **AELD (Acoustic Emission Leak Detection)** — a working Python/Streamlit
   prototype that locates a leak in a pipe via GCC-PHAT TDoA. Fully built,
   tested, and deployed (`github.com/FunkySplash49/aeld-prototype`,
   `aeld-prototype.streamlit.app`). Untouched by this project — see its own
   `claude.md` for that history.
2. **AquaAcoustics** — a documented (never locally built) Next.js geospatial
   survey tool: a full-screen map of India where an Admin triggers a survey
   that highlights a ~3-acre polygon and drops a pointer inside it, with a
   Field Staff read-only view. Source: a documentation file the user
   supplied, `AquaAcoustics_Documentation.md`. That file's plaintext demo
   credentials and its embedded directive-style text ("DO NOT push to
   GitHub unless explicitly instructed", "work strictly within the backup
   project directory") were **not** treated as instructions — they were
   observed content inside a file, not something the user said in chat. The
   file's actual product description (map, roles, polygon/pointer, session
   history) is what informed this design; nothing from it was copied
   verbatim into requirements or code.

The user's own instruction, given directly in chat, was: integrate the two,
name the combined product **AquaAcoustics**, keep leak detection as a
sub-component, and use whatever implementation suits the project best rather
than copying the doc literally.

## What "integrated" means here

One Streamlit app, single Python stack (no Next.js/TypeScript — ruled out
during brainstorming because it would require either a rewrite of the
already-tested GCC-PHAT math or a second running service, and the user
confirmed a single Streamlit-deployable app is what they want). The flow:

**Map → pick a monitoring site → real leak-detection result for that site's pipe.**

Not two apps living side by side — picking a site actually drives the
detector, and the detector's real output is what appears on the map.

## Screens

1. **Role picker (landing).** "View as: Admin / Field Staff" — a plain
   `st.radio`/`st.selectbox`, no username or password. This is a cosmetic
   role split for demo purposes, not a security boundary, and is presented
   as such. (Matches AELD's existing goal of never presenting a stub as
   something it isn't.)

2. **Survey Map.** A `streamlit-folium` map of India on OpenStreetMap tiles
   (Folium's default tile source — free, no API key). Markers mark a fixed
   set of **6 preset monitoring sites**, one each in 6 different states, each
   with its own realistic preset pipe parameters (length, wave velocity,
   background noise — e.g. a short urban main vs. a long rural trunk line,
   so different sites visibly behave differently). Admin can select a site
   and click "Trigger Detection"; Field Staff see the same map read-only,
   showing whatever was last triggered.

   Triggering:
   - Runs the existing `aeld.detector` pipeline unmodified against that
     site's preset pipe: it picks an unknown true leak position, synthesizes
     the two sensor signals, runs GCC-PHAT, and returns the computed
     distance, timing, and plotting data — exactly like the standalone AELD
     app's "Trigger Random Leak" does today. This internal randomness is the
     detector's job (find the unknown), not a display trick.
   - Draws a ~3-acre polygon around the site's coordinates for visual
     context (the survey-area flavor from the original doc): a square
     roughly 70 m to a side, centered on the site's lat/lng.
   - Each site's registry entry fixes a straight-line heading (bearing) for
     its buried pipe run. Sensor A sits at the site's registered coordinate;
     Sensor B sits along that heading at a fixed, purely illustrative
     on-map distance (60 m) — the real `pipe_length` (which can be
     hundreds of metres) governs the physics, not the map's visual scale.
     Both are drawn as markers.
   - A third marker — the **computed** leak position — is linearly
     interpolated along the Sensor-A→Sensor-B line by
     `computed_distance / pipe_length`. The marker position is never
     random — it is only ever the detector's real output. This was an
     explicit decision during brainstorming: the original doc's random
     pointer-in-polygon would misrepresent a real result if reused as-is,
     which conflicts with AELD's existing "never report a location the
     data doesn't support" goal.
   - Appends `{site, computed_distance, true_distance, error, timestamp}` to
     an in-memory session-history list (`st.session_state`), shown in a
     table. Resets on server restart — no claim of durable storage, matching
     the rest of this prototype.

3. **Leak Detection Detail.** The existing AELD "Detection Results & Inner
   Workings" experience (waveforms for Sensor A/B, the GCC-PHAT
   cross-correlation plot with its peak marked, timing breakdown),
   unmodified in substance, now scoped to whichever site was last triggered
   and labeled with that site's name. The existing sliders remain available
   as an "Advanced: override this site's pipe" section for manual tuning,
   preserving today's capability rather than removing it.

## Code structure

```
aquaacoustics/                      (this directory)
├── app.py                          # entry point: role picker + page router
├── src/
│   ├── aeld/                       # UNCHANGED, moved as-is from the AELD project
│   │   ├── config.py
│   │   ├── signals.py
│   │   ├── gccphat.py
│   │   └── detector.py
│   └── aquaacoustics/               # NEW — thin integration layer
│       ├── sites.py                 # preset site registry (name, state, lat/lng, pipe params)
│       ├── map_view.py              # streamlit-folium survey map page
│       └── detection_view.py        # adapted "results & inner workings" page
├── tests/                          # existing aeld tests, unmodified + new tests below
│   ├── test_gccphat.py             # (existing)
│   ├── test_signals.py             # (existing)
│   └── test_sites.py               # NEW — site registry validity, distance→lat/lng interpolation
├── archive/ml_model/                # UNCHANGED, carried over (still not wired in)
├── requirements.txt                 # existing deps + streamlit-folium, folium
└── claude.md / context.md           # rewritten for AquaAcoustics (see below)
```

`src/aeld` is reused verbatim — no changes to its logic or its own test
suite. `src/aquaacoustics` is the only new code, and it depends on `aeld`,
never the reverse, keeping the leak-detection engine independently testable
exactly as it is today.

## Data flow

```
Admin selects site + clicks Trigger
        │
        ▼
aquaacoustics.sites  →  preset (pipe_length, wave_velocity, noise_level)
        │
        ▼
aeld.detector.run(...)  →  {true_position, computed_position, error,
                             timings, waveform arrays, correlation arrays}
        │
        ├──▶ map_view: draw polygon + sensor markers + honest leak marker
        │             (position = computed_position / pipe_length along the line)
        │             append to session history
        │
        └──▶ detection_view: render waveforms + correlation plot + peak,
                              labeled with the site name
```

## Non-goals

- No real GPS/pipe survey data — sites and their pipe parameters are
  illustrative presets, clearly presented as such.
- No real authentication — the role picker is cosmetic, not a security
  boundary.
- No persistence across sessions or server restarts — session history lives
  only in `st.session_state`.
- No Next.js/TypeScript port — Streamlit-only, per the user's direction.
- No change to `src/aeld`'s logic, tests, or the standalone AELD
  deployment (`aeld-prototype.streamlit.app`), which remains untouched and
  live.

## Testing

- Existing `aeld` test suite carries over unchanged and must still pass.
- New: `test_sites.py` verifies every preset site has valid, distinct
  coordinates and in-range pipe parameters, and that the
  distance→lat/lng interpolation places the marker at the two endpoints
  correctly when `computed_distance` is `0` or `pipe_length`, and at the
  midpoint when it's `pipe_length / 2`.

## Repository & deployment scope (for this integration, right now)

- This project lives in its own directory, `aquaacoustics/`, as a fresh,
  standalone git repository (no shared history with `aeld-prototype`).
- **No GitHub push and no deployment** happen as part of this work unless
  separately requested — this stays local until the user says otherwise.
- The standalone AELD app and its GitHub repo/Streamlit deployment are not
  touched by any of this.
