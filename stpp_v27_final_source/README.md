# STPP v27 final readable source

This directory is a faithful, readable source snapshot of the v27 generator
that previously completed successfully. It contains only the transitive local
Python modules and prompt files imported by v27.

No contract fix, Judge override, frozen import hook, or compressed source
payload is used. Runtime behavior remains the original v27 behavior. The only
packaging adjustment is the relative path used to locate the repository-level
`external/Spatio-Temporal-Point-Process-Simulator/` directory.

The launcher explicitly loads `data_generation/.env`.

Server checks:

```bash
python data_generation/stpp_v27_final_source/run.py --check-source-bundle
python data_generation/stpp_v27_final_source/run.py --check-env
python data_generation/stpp_v27_final_source/run.py --check-imports
```
