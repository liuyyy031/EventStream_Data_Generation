# Data Generation

Pipeline used to (re)generate [`data/ST-Bench`](../data/ST-Bench).

## Pipeline

```
Stage 1  Synthesize STS scenarios + run SDE simulation
            └─> data_generation/batch_output/task_*.{pkl,json}

Stage 2  Generate QA pairs from .pkl files
            ├─ alignment_QA       → data/alignment/*.jsonl
            ├─ reasoning_QA       → data/reasoning_before_filter/{entity,etiological,correlation}_*.jsonl
            └─ forecasting_QA     → data/reasoning_before_filter/forecasting_*.jsonl

Stage 3  Filter samples by length / token count
            data/reasoning_before_filter/  →  data/reasoning/

Stage 4  CoT rejection sampling (after running inference once)
            data/reasoning/*_finetune.jsonl  →  *_cot.jsonl, *_rl_new.jsonl

Stage 5  Convert to text / image variants
            data/reasoning/  →  data/reasoning_text/
                            →  data/reasoning_image/
```

`Stage 1` and the reasoning part of `Stage 2` need an LLM API. All LLM calls
go through [`llm_client.py`](llm_client.py), which speaks the
OpenAI-compatible chat-completions protocol. Configure it via environment
variables:

```bash
export LLM_API_KEY=<your_api_key>                       # required
export LLM_BASE_URL=https://api.openai.com/v1           # optional, default shown
export LLM_MODEL=gpt-4o-mini                            # optional, default shown
```

Any provider exposing an OpenAI-style `/chat/completions` endpoint (OpenAI,
OpenRouter, DeepSeek, Together AI, vLLM, Ollama, …) works by overriding
`LLM_BASE_URL` and `LLM_MODEL`. All other scripts are pure data processing
and do not need any API.

## Commands

Run from the repository root after `conda activate stt`.

```bash
# Stage 1 — raw STS scenarios (LLM API required)
python data_generation/run_pipeline.py \
    --num_tasks 100 --node_counts 3,5,10 --max_workers 8

# Stage 2 — QA generation
python data_generation/generate_alignment_QA.py
python data_generation/generate_reasoning_QA.py \
    --data_dir data_generation/batch_output \
    --output_dir data/reasoning_before_filter         # LLM API required
python data_generation/generate_reasoning_forecasting_QA.py \
    --data_dir data_generation/batch_output \
    --output_dir data/reasoning_before_filter

# Stage 3 — filter
python data/filter.py

# Stage 4 — CoT rejection sampling
#   Prerequisite: run inference first to produce exp/<exp>/generated_answer.json
for task in reasoning_forecasting reasoning_entity reasoning_etiological reasoning_correlation; do
    python data_generation/generate_cot.py --task $task --exp <exp_name>
done

# Stage 5 — text / image variants
python data/convert_to_text.py  --input_dir data/reasoning --output_dir data/reasoning_text
python data/convert_to_image.py --input_dir data/reasoning --output_dir data/reasoning_image
```

## File reference

| Script                                  | Stage | LLM API | Inputs → Outputs |
|-----------------------------------------|-------|---------|------------------|
| `llm_client.py`                         | -     | -       | shared OpenAI-compatible client used by every LLM-calling script |
| `run_pipeline.py`                       | 1     | yes     | parallel driver that runs `demo_sts_sde.py` N times → `batch_output/task_*.pkl` |
| `demo_sts_sde.py`                       | 1     | yes     | full 6-Agent + 2-Judge pipeline for a single scenario; importable as a library |
| `generate_file_list.py`                 | 1     | no      | `batch_output/` → `list_files.json` |
| `generate_alignment_QA.py`              | 2     | no      | `batch_output/*.pkl` → `data/alignment/` |
| `generate_reasoning_QA.py`              | 2     | yes     | `batch_output/*.pkl` → `data/reasoning_before_filter/` |
| `generate_reasoning_forecasting_QA.py`  | 2     | no      | `batch_output/*.pkl` → `data/reasoning_before_filter/` |
| `../data/filter.py`                     | 3     | no      | `reasoning_before_filter/` → `reasoning/` |
| `generate_cot.py`                       | 4     | no      | `reasoning/*_finetune.jsonl` + inference outputs → `*_cot.jsonl`, `*_rl_new.jsonl` |
| `../data/convert_to_text.py`            | 5     | no      | `reasoning/` → `reasoning_text/` |
| `../data/convert_to_image.py`           | 5     | no      | `reasoning/` → `reasoning_image/` |

## ST-Bench mapping

| Subset            | Source files                                |
|-------------------|---------------------------------------------|
| `ST-Align/`       | `data/alignment/alignment_{train,test}.jsonl` |
| `ST-SFT/`         | `data/reasoning/*_finetune.jsonl`           |
| `ST-CoT/`         | `data/reasoning/*_cot.jsonl`                |
| `ST-RL/`          | `data/reasoning/*_rl.jsonl`                 |
| `ST-Test/`        | `data/reasoning/*_test.jsonl`               |
| `ST-CoT-Text/`    | `data/reasoning_text/*_cot.jsonl`           |
| `ST-RL-Text/`     | `data/reasoning_text/*_rl.jsonl`            |
| `ST-CoT-Image/`   | `data/reasoning_image/*_cot.jsonl`          |
| `ST-RL-Image/`    | `data/reasoning_image/*_rl.jsonl`           |

Prompt templates live in [`prompts/`](prompts/).
