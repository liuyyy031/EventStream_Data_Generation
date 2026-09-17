# STPP v27 release

This directory is a complete, readable copy of the validated v27 generator.
It adds a narrow route-contract safety guard discovered from seed 20260720:
an event full path containing a repeated node is rejected as a deterministic
contract issue instead of reaching a negative path-stage lookup.

The lag-fidelity audit follows the same declared-route semantics as the
simulator. When declared event routes are enforced, only edges on those routes
must have propagated events and satisfy the configured lag tolerance. Extra
graph-connectivity edges remain visible in the report as ignored configured
edges. A declared route edge with no propagated event still fails closed.

Raw Agent 1 section parsing accepts Markdown bullets, mixed-case headings and
multiple `PROPAGATED ARRIVALS` subsections. All repeated subsections are
combined before comparing arrival event families with `EDGE MODULATION`;
missing or extra event families remain blocking.

No Judge override, frozen import hook, or compressed source payload is used.
Valid v27 scenarios retain the same behavior. The route guard routes malformed
Agent 2 output back through the existing retry/escalation loop. The packaging
adjustment for locating the repository-level
`external/Spatio-Temporal-Point-Process-Simulator/` directory is retained.

The launcher explicitly loads `data_generation/.env`.

Server checks:

```bash
python data_generation/stpp_v27_release/run.py --check-source-bundle
python data_generation/stpp_v27_release/run.py --check-route-guard
python data_generation/stpp_v27_release/run.py --check-lag-scope
python data_generation/stpp_v27_release/run.py --check-raw-section-parser
python data_generation/stpp_v27_release/run.py --check-env
python data_generation/stpp_v27_release/run.py --check-imports
```
