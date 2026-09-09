# EventFlow v1.4.0

EventFlow v1 is a clean Transportation-first replacement for the matrix-led
STPP v27 data contract. The v27 files remain untouched for comparison.

## Current scope

- Fully implemented: Transportation end-to-end generation.
- Event records use one namespaced `event_type_id`; definitions, categories,
  participant roles, attribute contracts, and standards mappings live in a
  versioned registry copied into every output directory.
- Event-to-entity links are qualified (`participants: [{entity_id, role}]`)
  rather than stored as an ambiguous list of bare IDs.
- A run can create multiple sparse networks with different structure families;
  episodes are distributed across them and only reference their local entities.
- Generated topologies have bounded out-degree, and each event has a separate
  hard propagation-child limit.
- Event-limit truncation is recorded and rejected by default instead of being
  silently accepted.
- Canonical event text verbalizes every current domain attribute; relation text
  includes its exact event-time lag.
- Placeholders only: OpenTelemetry and Business Process.
- Not included: LLM training, next-event prediction, candidate relations,
  fixed time bins, magnitude matrices, or dense adjacency matrices.
- Propagation lag now carries a reproducible FHWA BPR reference-formula trace.
  Synthetic demand and recovery priors remain explicitly marked as pending
  empirical fitting; a fitting entry point and held-out error report are included.

## Run locally or on a server

From the repository root:

```bash
python data_generation/run_eventflow_v1.py \
  --episode-count 100 \
  --nodes-per-network 1000 \
  --judge-mode heuristic \
  --output-dir data_generation/output_eventflow_v1_test
```

Use `--judge-mode llm` only after setting `LLM_API_KEY`, `LLM_BASE_URL`, and
`LLM_MODEL`. The LLM judge uses text and structured records only; it does not
require a multimodal model.

LLM Judge protocol errors are retried against the same episode. Empty reasons,
invalid JSON, and API errors never trigger a new generation seed. Semantic
rejection can regenerate an episode; exhausted protocol retries stop the run
and write `judge_protocol_failure_*.json` with bounded response previews.

`--nodes-per-network` is the road-segment count for each generated network.
The old `--node-count` spelling remains as a deprecated compatibility alias.

To use an external sparse network:

```bash
python data_generation/run_eventflow_v1.py \
  --network-file /path/to/network.json \
  --episode-count 100
```

See `config/external_network_example.json` for the accepted input format.

## Evidence experiments

Run an engineering-stability experiment over independent seeds:

```bash
python data_generation/run_eventflow_stability.py \
  --seed-count 5 \
  --episodes-per-seed 100 \
  --nodes-per-network 1000 \
  --judge-mode heuristic \
  --output-dir data_generation/output_eventflow_v13_stability
```

The root directory contains `stability_report.json` and
`stability_runs.csv`, plus one normalized dataset directory per seed.

Run the layered counterexample challenge with the real LLM Judge:

```bash
python data_generation/run_eventflow_judge_challenge.py \
  --judge-mode llm \
  --output-dir data_generation/output_eventflow_v13_judge_challenge
```

Fit a profile from reference observations, then use it for generation:

```bash
python data_generation/run_eventflow_calibration.py \
  --input-csv /path/to/transport_reference.csv \
  --output-profile data_generation/calibration/transport_fitted.json

python data_generation/run_eventflow_v1.py \
  --calibration-profile data_generation/calibration/transport_fitted.json \
  --episode-count 100 \
  --nodes-per-network 1000
```

`config/calibration_input_schema.csv` documents the accepted columns using
illustrative synthetic rows. It is not a real calibration dataset.

## Output

The run writes normalized streaming files:

- `manifest.json`: configuration, versions, limitations, and provenance.
- `event_type_registry.json`: the released event vocabulary and per-type
  semantic/attribute contract referenced by `event_type_id`.
- `networks.jsonl`: network metadata.
- `entities.jsonl`: reusable network entities.
- `network_edges.jsonl`: sparse propagation-eligibility edges.
- `episodes.jsonl`: EpisodeProgram and per-episode counts.
- `episode_entities.jsonl`: local entity subset for each episode.
- `events.jsonl`: exact-time events.
- `relations.jsonl`: the sole source of truth for event parent relations.
- `mechanisms.jsonl`: explicit multi-parent `all_of` / `any_of` groups.
- `texts.jsonl`: event-, relation-, mechanism-, and episode-aligned text.
- `validation.jsonl`: deterministic, domain, and semantic Judge results.
- `quality_report.json`: streaming distribution and coverage report.
- `calibration_profile.json`: the exact formulas, coefficients, sources,
  limitations and empirical-fit status used by the run.

`quality_report.json` also reports topology degree, relation-specific lag
distributions, maximum observed event branching, truncation count, recovery
coverage, and network-entity utilization.

Judge routing counters are separated into `generation_retry_count`,
`semantic_rejection_count`, `judge_protocol_retry_count`, and
`judge_protocol_failure_count`. `rejected_attempt_count` excludes protocol
retries because they do not reject or regenerate the episode.

Network connectivity is only propagation eligibility. A parent relation is
written only after a named DomainSpec rule executes. `parent_event_ids` is not
stored as a second source of truth.

## Tests

```bash
python -m unittest discover -s data_generation/eventflow_v1/tests -v
```
