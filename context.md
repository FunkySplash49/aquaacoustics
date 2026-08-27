# context.md — AquaAcoustics progress tracker

## Status: implemented, tested, reviewed (task-level + final whole-branch,
including one fix-and-re-review cycle), pushed to GitHub. Not yet deployed.

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
10. Final whole-branch review (opus) found one Critical issue (the
    override re-run broke the honest-marker rule — `interpolate_position`
    divided by the site's preset pipe length instead of the run's actual
    length) and several Important ones (`setup.py` missing the new map
    deps, Field Staff able to trigger from the Detail page, override
    re-runs invisible in session history, a stale `PROGRESS.md`). Fixed in
    one commit; re-review found the fix's math correct but introduced a
    new regression (the map rendered one interaction stale right after
    clicking Trigger Detection); fixed in a second commit; final re-review
    passed clean.
11. Pushed to GitHub: `github.com/FunkySplash49/aquaacoustics` (private),
    branch `main`.

## Not done (explicitly out of scope for now)

- No deployment (Streamlit Community Cloud or otherwise).
- No Next.js/TypeScript component of any kind.
- No real authentication, no real GPS survey data, no cross-session
  persistence — see `claude.md` "Non-goals".
- A handful of Minor findings from the final review were deliberately
  deferred (not correctness bugs): loose type hints in `sites.py`, small
  DRY duplication in its heading math, no `__post_init__` guard on
  `pipe_length_m`, the Survey Map's "preset pipe:" caption still says
  "preset" even when describing an overridden run, `build_override_config`
  hand-lists fields instead of using `dataclasses.replace`, override
  history rows are visually indistinguishable from preset-trigger rows,
  and two tests in `test_sites.py` only exercise one of the six sites. See
  `.superpowers/sdd/progress.md` for the full list.
