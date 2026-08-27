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
