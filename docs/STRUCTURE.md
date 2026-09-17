# Data-generation structure

## Active dependency direction

```text
run_data_generation.py
  -> domain_packages/<domain>/
  -> generation_core/
       -> records and contracts
       -> topology and temporal models
       -> scheduler
       -> deterministic validation
       -> optional semantic judge/correction
       -> QA and training export
       -> writer
```

`run_parameter_calibration.py` is the only other active entrypoint. It uses the
same mechanism and reference-data contracts from `generation_core/`.

The active code uses `data_generation.*` imports. It must not import anything
under `legacy/` or read anything from `artifacts/`.

## Relocation map

| Previous location | Current location | Status |
|---|---|---|
| versioned large-network prototype package and wrappers | `legacy/prototype_v1/` | historical |
| `legacy_stpp/` and root-level STPP entrypoints | `legacy/stpp/` | historical |
| four v27 source/release snapshots | `legacy/stpp/snapshots/` | historical |
| original ST-Bench scenario and QA scripts | `legacy/stbench/` | historical |
| `GENERATION_CORE_DESIGN.md` | `docs/GENERATION_CORE_DESIGN.md` | active documentation |
| generated experiment and test outputs | `artifacts/` | preserved, ignored by Git |
| old nested `data_generation/.git` | `artifacts/legacy_data_generation_git_metadata/` | recoverable backup |

## Rules for future changes

1. Add domain-independent behavior only in `generation_core/`.
2. Add domain event types, entity types, mechanisms, priors, and wording only in
   `domain_packages/<domain>/`.
3. Keep one active generation CLI. Do not create version-numbered entrypoints.
4. Store generated data outside the source tree or under an ignored output
   directory.
5. Keep historical reconstruction code under `legacy/`; never import it into
   the active runtime.
6. Run `test_active_layout.py` whenever files or imports are moved.

