# Reference-distribution calibration

The mentor repository implements four useful operations: extracting sequence
statistics, fitting parametric delay regimes, generating from a fitted prior,
and comparing generic versus calibrated samples. The active generator reuses
the statistically valid parts without treating sequential association as a
causal edge.

## What is applied to generation

- Event and transition frequencies, first/second-order transitions, sequence
  lengths, inter-event gaps, burstiness, and repeated-event ratios are
  extracted separately for `train`, `validation`, and `holdout`.
- Scenario weights are applied only when the normalized reference windows
  explicitly carry a `scenario_family` label.
- Explicit parent links that resolve to one declared mechanism support
  descriptive completed-link delay fits. Exponential, gamma, log-normal,
  Weibull and deterministic fits are compared by BIC. These fits do not account
  for selection or censoring and are NOT automatically applied to generation.
  Overrides require `eligible_for_generation_override: true`; the current
  completed-link fitter intentionally never grants this flag.
- Ordinary adjacent transitions remain non-causal statistics. They may
  reweight an already-declared, already-eligible continuous-time mechanism,
  but they never create a candidate or parent edge. Sequence lengths remain
  held-out evaluation statistics.

This distinction is necessary because the active generator uses typed
mechanisms, candidate lifecycles, competing risks, and state preconditions.
A Markov transition observed in a sequence is not sufficient evidence that the
first event caused the second.

## Reference input

There are now two equivalent entry paths.  A raw CSV, JSONL, or Parquet event
table is the convenient path.  The normalized three-file directory remains
the internal contract and an optional audit snapshot; it no longer has to be
prepared by hand.

For a raw table, provide a mapping JSON using
`generic-tabular-reference-mapping-v1`.  At minimum it maps a sequence/stream
identifier, occurrence timestamp, and event type.  It may additionally map a
context/subject identifier, source event ID, observation and recording times,
participants, an explicit parent event ID, trace ID, scenario family, split,
event attributes, and context-state fields.  The example is
`reference_adapters/examples/healthcare_event_table_mapping.json`; its schema
is `generation_core/schemas/generic_tabular_reference_mapping_v1.schema.json`.

The adapter performs the following deterministic normalization:

- one source stream becomes one observation window;
- timestamps become offsets from a declared observation-window start, or from
  the first occurrence when no bounds are supplied (an inferred event span);
- streams for the same context are assigned to the same split;
- raw event labels are checked against the explicit type mapping;
- only a populated `parent_event_id` field becomes an explicit parent link;
- adjacency in the source sequence is never promoted to causality;
- a source fingerprint, mapping fingerprint, validation result, and conversion
  counts are retained for audit.

The canonical audit directory contains:

- `reference_manifest.json`: domain, source identity, version, and split policy.
- `reference_windows.jsonl`: observation windows with disjoint `train`, `validation`, and
  `holdout` assignments.
- `reference_events.jsonl`: normalized events. Use `explicit_parent_record_ids` only when
  the source actually provides a defensible parent link.

Its schema is
`generation_core/schemas/reference_data_contract_v1.schema.json`. Zero-event
windows cannot be recovered from an event-only table. They can now be supplied
through `window.records` in the raw-table mapping, or through the normalized
contract. Root-rate fitting requires complete windows, mechanism-specific
at-risk exposure, and explicit event-mechanism attribution. See
[generation contract revision](generation_contract_revision.md) for examples.

## Server workflow

Fit a pending healthcare bundle directly from a raw event table and optionally
write the normalized audit snapshot:

```bash
python data_generation/run_parameter_calibration.py \
  --mode sequence-prior \
  --mechanism-spec data_generation/domain_packages/healthcare/mechanism_specs.json \
  --input-file data_generation/reference_raw/healthcare/events.csv \
  --mapping-config data_generation/reference_adapters/healthcare_mapping.json \
  --normalized-output-dir data_generation/reference/healthcare_v1 \
  --min-transition-samples 20 \
  --output data_generation/artifacts/calibration/healthcare_sequence_bundle_v1.json
```

The same command can still consume an already-normalized directory:

```bash
python data_generation/run_parameter_calibration.py \
  --mode sequence-prior \
  --mechanism-spec data_generation/domain_packages/healthcare/mechanism_specs.json \
  --reference-dir data_generation/reference/healthcare_v1 \
  --min-transition-samples 20 \
  --output data_generation/artifacts/calibration/healthcare_sequence_bundle_v1.json
```

Generate a generic baseline and an evaluation-only calibrated sample with the
same seed and run size:

```bash
python data_generation/run_data_generation.py \
  --domain healthcare \
  --episode-count 100 \
  --nodes-per-context 1000 \
  --seed 20260918 \
  --judge-mode none \
  --output-dir data_generation/output_healthcare_generic

python data_generation/run_data_generation.py \
  --domain healthcare \
  --episode-count 100 \
  --nodes-per-context 1000 \
  --seed 20260918 \
  --judge-mode none \
  --calibration-bundle data_generation/artifacts/calibration/healthcare_sequence_bundle_v1.json \
  --allow-pending-calibration \
  --calibration-components scenario,temporal,mechanism_reweighting \
  --mechanism-reweighting-mode first_second_order \
  --mechanism-reweighting-strength 1.0 \
  --mechanism-reweighting-min-history-count 20 \
  --output-dir data_generation/output_healthcare_calibrated
```

Compare both outputs against the untouched holdout split:

```bash
python data_generation/run_parameter_calibration.py \
  --mode compare \
  --reference-dir data_generation/reference/healthcare_v1 \
  --reference-split holdout \
  --generic-output-dir data_generation/output_healthcare_generic \
  --calibrated-output-dir data_generation/output_healthcare_calibrated \
  --output data_generation/artifacts/calibration/healthcare_comparison_v1.json
```

For direct raw input, replace `--reference-dir ...` in the comparison command
with the same `--input-file ... --mapping-config ...` pair.  The seeded mapping
recreates the same context-level split.

Lower calibrated distances support only the reported distributional claims.
They do not prove causal truth or universal realism. A fitted bundle remains
`empirically_estimated_pending_domain_acceptance`; the
`--allow-pending-calibration` flag exists for controlled evaluation, not for
silently declaring the parameters accepted. `quality_report.json` reports how
many candidates were reweighted, why other candidates were left unchanged,
and the applied multiplier range.

## Legal-mechanism reweighting

For one observed history with `K` observed next-event types, the reference
probability of the candidate target is compared with the uniform observed
target probability:

```text
multiplier = clip((p_transition / (1 / K)) ^ strength)
```

The multiplier is applied through proportional hazards, so the sampled clock,
survival probability, hazard, and risk-set probability remain mathematically
consistent. First/second-order statistics are used only when all of the
following hold:

- the domain package has already emitted the candidate;
- the candidate has exactly one realized parent;
- the parent and target types match the declared mechanism;
- the temporal model exposes a supported continuous-time distribution;
- the history sample count reaches the configured minimum;
- the target transition was observed in the training split.

Unobserved legal transitions, multi-parent mechanisms, deterministic rules,
and scheduled policy events retain their original priors. This prevents a
finite reference sample from deleting rare but valid behavior.

## Component ablations

Use the same bundle, seed, and episode count and change only
`--calibration-components`:

```text
scenario
temporal
mechanism_reweighting
scenario,temporal
scenario,temporal,mechanism_reweighting
```

The generic baseline omits `--calibration-bundle`. This allows the scenario,
time, and legal-mechanism effects to be measured independently without editing
the fingerprinted calibration artifact.

## Remaining limitations

- Sequence length is not forced as a hard event-count target; it remains an
  evaluation metric until a defensible observation-window or termination
  model is added.
- Multi-parent transition reweighting is intentionally disabled because an
  adjacent sequence does not identify joint-parent contribution weights.
- Reweighting improves agreement with measured transition distributions; it
  does not prove that those transitions are causal.
