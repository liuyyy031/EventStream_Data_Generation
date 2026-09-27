# Data Generation

This directory has one active generation path. Historical implementations are
kept under `legacy/` and are not imported by the active runtime.

## Active layout

```text
data_generation/
  run_data_generation.py       main generation CLI
  run_parameter_calibration.py parameter-calibration CLI
  run_continuations.py         shared-prefix stochastic continuations
  llm_client.py                shared OpenAI-compatible LLM client
  generation_core/             domain-neutral records, scheduler, validation,
                               semantic review/correction, QA and writers
  domain_packages/             executable domain definitions and priors
    transportation/
    healthcare/
    distributed_systems/       structural placeholder, not yet executable
  tests/                       tests for the active runtime only
  docs/                        active design documentation
  legacy/                      isolated historical implementations
  artifacts/                   preserved local outputs; never imported
```

The active runtime uses absolute `data_generation.*` imports. Code in
`generation_core/`, `domain_packages/`, the active CLIs, and active tests
must not import modules from `legacy/` or `artifacts/`.

For scientific time axes, variable horizons, scoped history, explicit observation
windows and continuation tests, see [generation contract revision](docs/generation_contract_revision.md).

## Generate data

Run from the repository root:

```bash
python data_generation/run_data_generation.py \
  --domain transportation \
  --episode-count 10 \
  --nodes-per-context 1000 \
  --seed 20260917 \
  --judge-mode none \
  --output-dir data_generation/output_transportation_smoke
```

Healthcare uses the same engine and output contract:

```bash
python data_generation/run_data_generation.py \
  --domain healthcare \
  --episode-count 10 \
  --nodes-per-context 1000 \
  --seed 20260917 \
  --judge-mode none \
  --output-dir data_generation/output_healthcare_smoke
```

For server-side semantic review and correction, keep credentials in the
repository `.env` and specify the model explicitly:

```bash
python data_generation/run_data_generation.py \
  --domain transportation \
  --episode-count 10 \
  --nodes-per-context 1000 \
  --seed 20260917 \
  --judge-mode llm \
  --judge-model deepseek-v4-flash \
  --judge-workers 2 \
  --semantic-correction-mode llm \
  --output-dir data_generation/output_transportation_llm_smoke
```

The LLM may review or correct grounded text. It cannot change event records,
relations, timestamps, entities, candidates, risk sets, or deterministic QA
answers.

For a formal healthcare run plus an evaluation-only demonstration of the
semantic correction loop, follow
[`docs/HEALTHCARE_TEST_PROTOCOL.md`](docs/HEALTHCARE_TEST_PROTOCOL.md). The
controlled challenge artifacts must never be mixed into training data.

For reference-sequence statistics, explicit-parent delay fitting, calibrated
evaluation runs, and generic-versus-calibrated holdout comparison, follow
[`docs/DISTRIBUTION_CALIBRATION.md`](docs/DISTRIBUTION_CALIBRATION.md).
The calibration CLI accepts either an already-normalized reference directory
or a raw CSV/JSONL/Parquet event table plus an explicit mapping JSON.  The raw
path can also emit the normalized three-file audit snapshot automatically.
The complete baseline and advanced clock library, including periodic,
self-exciting, refractory, mixture, queue, calendar, interval, and batch
models, is documented in [`docs/TEMPORAL_MODELS.md`](docs/TEMPORAL_MODELS.md).

## Main outputs

| File | Purpose |
|---|---|
| `episodes.jsonl` | Episode summaries and generation metadata |
| `events.jsonl` | Timestamped structured events |
| `event_relations.jsonl` | Realized typed event-to-event relations |
| `candidates.jsonl` | Fired, cancelled, and censored candidate lifecycles |
| `risk_sets.jsonl` | Pre-event time/type/entity competing-risk evidence |
| `episode_texts.jsonl` | Final grounded event and relation text |
| `qa_pairs.jsonl` | Ground-truth-derived event-stream understanding QA |
| `training/qa_instruction.jsonl` | Instruction-format training view |
| `training/qa_chat.jsonl` | Chat-format training view |
| `semantic_corrections.jsonl` | Auditable semantic correction rounds |
| `quality_report.json` | Validation, distribution, and limitation summary |

## Test the active runtime

```bash
python -m unittest discover -s data_generation/tests -p "test_*.py" -q
python data_generation/run_data_generation.py --help
python data_generation/run_parameter_calibration.py --help
```

Historical commands are documented inside [`legacy/`](legacy/README.md). They
are retained for provenance only and are not part of current acceptance tests.
The complete relocation map and dependency rules are in
[`docs/STRUCTURE.md`](docs/STRUCTURE.md).
