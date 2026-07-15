# STPP alternative for Stage 1

This is a non-destructive alternative to the SDE Stage 1 pipeline. The
original `demo_sts_sde.py` and `run_pipeline.py` are unchanged.

## Files

- `../external/Spatio-Temporal-Point-Process-Simulator/`: unmodified upstream
  repository at commit `be65d949e475c636a34fc6216044be94e139f50f`.
- `stpp_adapter.py`: runs STPPG and maps continuous events to STReasoner nodes
  and time windows.
- `demo_sts_stpp.py`: single-scenario Stage 1 entry point.
- `run_pipeline_stpp.py`: batch entry point corresponding to
  `run_pipeline.py`.
- `requirements_stpp_extra.txt`: STPPG-only dependencies.

## Data contract

Each pickle contains both representations:

```text
agent5_event_stream
  [{t, x, y, node_id, time_index}, ...]  # native point events

agent5_simulation_data
  float[num_nodes, seq_len]              # compatibility matrix
```

By default, `agent5_simulation_data[node, time]` is the number of events
assigned to that node and time window. Assignment uses the nearest node in the
scenario's normalized `spatial_layout`. Alternative aggregations are
`rolling_count` and `cumulative_count`.

The artifact deliberately uses `agent3_point_process_parameters`, not the old
`agent3_sde_parameters`. Existing reasoning and forecasting QA scripts read
`agent5_simulation_data` and remain usable. The old temporal-alignment QA
templates that ask for SDE drift parameters will produce no such samples;
they should not be used to describe STPP data.

`agent4_time_varying_adjacency` is retained as compatibility metadata for
graph questions. It is parsed from the scenario, but has
`used_by_simulator=false`: the upstream STPPG model is a continuous-space,
univariate Hawkes process and does not condition generation on STReasoner's
directed node graph.

## Commands (not executed during this change)

From the repository root:

```bash
pip install -r data_generation/requirements_stpp_extra.txt

python data_generation/demo_sts_stpp.py \
  --num_nodes 3 --domain Transportation --judges 1 \
  --aggregation count

python data_generation/run_pipeline_stpp.py \
  --num_tasks 100 --node_counts 3,5,10 --max_workers 8 \
  --aggregation count --out_dir data_generation/batch_output_stpp

python data_generation/generate_reasoning_QA.py \
  --data_dir data_generation/batch_output_stpp \
  --output_dir data/reasoning_before_filter_stpp

python data_generation/generate_reasoning_forecasting_QA.py \
  --data_dir data_generation/batch_output_stpp \
  --output_dir data/reasoning_before_filter_stpp
```

Judge 1 (scenario/parse consistency) is supported. Judge 2 is intentionally
disabled because it validates SDE parameters and an SDE visualization.

## Important limitations

1. The upstream simulator returns `(t, x, y)` events. It does not implement a
   multivariate/node-marked Hawkes process, despite the broad "marked" wording
   in its README.
2. The graph stored in the artifact is therefore descriptive, not the causal
   mechanism that generated the events. Graph-reasoning labels and the event
   data can disagree causally.
3. Point counts are sparse and nonnegative integers, unlike the smoother
   continuous SDE sequences. Forecasting difficulty and token statistics will
   change, so dataset filters and baselines should be recalibrated.
4. The upstream repository does not contain a `LICENSE` file. Confirm
   redistribution permission before publishing a repository or dataset that
   bundles its source.

