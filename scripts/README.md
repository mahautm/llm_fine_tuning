# Scripts Index

## Canonical Python entrypoints

- `analysis/` — analysis/evaluation/data-prep scripts
- `plots/` — plotting scripts
- `tests/` — validation/test scripts

## Canonical shell entrypoints

- `entrypoints/jobs/` — batch job launchers
- `entrypoints/maintenance/` — maintenance/status helpers
- `entrypoints/tests/` — shell test launchers

## Compatibility note

Root-level `.sh` files are wrappers that forward to `scripts/entrypoints/...`.
Prefer invoking canonical scripts directly in new docs and automation.
