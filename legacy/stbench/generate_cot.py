

import argparse
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from loguru import logger

DEFAULT_TASK_CONFIG = {
    "alignment": {
        "dataset": os.path.join("data", "alignment", "alignment_finetune.jsonl"),
    },
    "reasoning_forecasting": {
        "dataset": os.path.join("data", "reasoning", "forecasting_finetune.jsonl"),
    },
    "reasoning_entity": {
        "dataset": os.path.join("data", "reasoning", "entity_finetune.jsonl"),
    },
    "reasoning_etiological": {
        "dataset": os.path.join("data", "reasoning", "etiological_finetune.jsonl"),
    },
    "reasoning_correlation": {
        "dataset": os.path.join("data", "reasoning", "correlation_finetune.jsonl"),
    },
}

ANSWER_TAG_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
CHOICE_PATTERN = re.compile(r"^\s*([A-Z])")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Perform rejection sampling to attach CoT answers to datasets.")
    parser.add_argument(
        "--task",
        type=str,
        default="reasoning_entity",
        choices=sorted(DEFAULT_TASK_CONFIG.keys()),
        help="Task name used to pick the default dataset path.",
    )
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        help="Experiment name under exp/ that contains generated_answer.json.",
    )
    parser.add_argument(
        "--answers",
        type=str,
        default=None,
        help="Optional explicit path to generated answers JSON. Overrides --exp.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset path overriding the task default.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSONL path. Defaults to replacing *_finetune.jsonl with *_cot.jsonl in the dataset directory.",
    )
    return parser.parse_args()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            obj.setdefault("idx", idx)
            records.append(obj)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def save_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_generated_answers(path: str) -> Dict[int, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Generated answers in {path} must be a list.")
    mapping: Dict[int, Dict[str, Any]] = {}
    for entry in data:
        idx = entry.get("idx")
        if isinstance(idx, int):
            mapping[idx] = entry
    return mapping


def extract_answer(response_text: str) -> Optional[str]:
    if not response_text:
        return None
    match = ANSWER_TAG_PATTERN.search(response_text)
    if not match:
        return None
    answer = match.group(1).strip()
    if not answer:
        return None
    return answer


def normalize_choice(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    stripped = text.strip()
    extracted = extract_answer(stripped)
    if extracted:
        stripped = extracted
    match = CHOICE_PATTERN.match(stripped)
    if not match:
        stripped_upper = stripped.upper()
        if len(stripped_upper) == 1 and stripped_upper.isalpha():
            return stripped_upper
        return None
    return match.group(1).upper()


def _parse_series(text: Any) -> List[float]:
    if text is None:
        return []
    if isinstance(text, list):
        return [float(v) for v in text]
    if isinstance(text, (int, float)):
        return [float(text)]
    text_str = str(text).strip()
    if not text_str:
        return []
    try:
        parsed = json.loads(text_str)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [float(v) for v in parsed]
    if isinstance(parsed, (int, float)):
        return [float(parsed)]
    numbers = re.findall(r"-?\d+\.?\d*", text_str)
    return [float(num) for num in numbers]


def _align_series(pred_series: List[float], target_series: List[float]) -> Optional[List[float]]:
    if not target_series or not pred_series:
        return None
    aligned = list(pred_series)
    if len(aligned) < len(target_series):
        pad_val = aligned[-1]
        aligned.extend([pad_val] * (len(target_series) - len(aligned)))
    elif len(aligned) > len(target_series):
        aligned = aligned[: len(target_series)]
    return aligned


def _compute_mae(pred_series: List[float], target_series: List[float]) -> Optional[Tuple[float, float]]:
    aligned = _align_series(pred_series, target_series)
    if aligned is None:
        return None
    absolute_errors = [abs(aligned[i] - target_series[i]) for i in range(len(target_series))]
    if not absolute_errors:
        return None
    mae = sum(absolute_errors) / len(absolute_errors)
    target_scale = sum(abs(v) for v in target_series) / len(target_series) if target_series else 0.0
    if target_scale <= 1e-6:
        target_scale = 1.0
    normalized_mae = mae / target_scale
    return mae, normalized_mae


def _rejection_sampling_reasoning(
    dataset: List[Dict[str, Any]],
    answers: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    matched_records: List[Dict[str, Any]] = []
    matched_count = 0
    matched_samples = 0
    unmatched_indices: List[int] = []
    unmatched_samples: List[Dict[str, Any]] = []

    for idx, sample in enumerate(dataset):
        entry = answers.get(idx)
        correct_choice = normalize_choice(sample.get("output"))
        has_match = False

        if entry:
            responses = sorted(entry.get("responses", []), key=lambda r: r.get("attempt", 0))
            for resp in responses:
                candidate = extract_answer(resp.get("response", ""))
                if candidate is None:
                    continue
                if correct_choice and normalize_choice(candidate) == correct_choice:
                    has_match = True
                    matched_sample = dict(sample)
                    matched_sample["output"] = resp.get("response", "")
                    attempt = resp.get("attempt")
                    if attempt is not None:
                        matched_sample["cot_attempt"] = attempt
                    matched_records.append(matched_sample)
                    matched_count += 1

        if not has_match:
            unmatched_indices.append(idx)
            unmatched_samples.append(dict(sample))
        else:
            matched_samples += 1

    return {
        "matched_records": matched_records,
        "matched_count": matched_count,
        "matched_samples": matched_samples,
        "total": len(dataset),
        "unmatched_indices": unmatched_indices,
        "unmatched_samples": unmatched_samples,
        "metric": "exact_match",
        "threshold": None,
    }


def _rejection_sampling_forecasting(
    dataset: List[Dict[str, Any]],
    answers: Dict[int, Dict[str, Any]],
) -> Dict[str, Any]:
    matched_records: List[Dict[str, Any]] = []
    matched_count = 0
    matched_samples = 0
    unmatched_indices: List[int] = []
    unmatched_samples: List[Dict[str, Any]] = []

    per_response_scores: List[float] = []
    per_sample_candidates: List[Tuple[int, Dict[str, Any], List[Tuple[Dict[str, Any], float, float]]]] = []

    for idx, sample in enumerate(dataset):
        target_series = _parse_series(sample.get("output"))
        entry = answers.get(idx)
        candidates: List[Tuple[Dict[str, Any], float, float]] = []

        if entry and target_series:
            responses = sorted(entry.get("responses", []), key=lambda r: r.get("attempt", 0))
            for resp in responses:
                pred_series = _parse_series(extract_answer(resp.get("response", "")))
                computed = _compute_mae(pred_series, target_series)
                if computed is None:
                    continue
                mae, normalized_mae = computed
                per_response_scores.append(normalized_mae)
                candidates.append((resp, mae, normalized_mae))

        per_sample_candidates.append((idx, sample, candidates))

    if per_response_scores:
        sorted_scores = sorted(per_response_scores)
        top_k = max(1, int(len(sorted_scores) * 0.2))
        threshold = sorted_scores[top_k - 1]
    else:
        threshold = None

    for sample_idx, sample, candidates in per_sample_candidates:
        accepted = []
        for resp, mae, normalized_mae in candidates:
            if threshold is not None and normalized_mae <= threshold:
                accepted.append((resp, mae, normalized_mae))

        if accepted:
            matched_samples += 1
            for resp, mae, normalized_mae in accepted:
                new_sample = dict(sample)
                new_sample["output"] = resp.get("response", "")
                attempt = resp.get("attempt")
                if attempt is not None:
                    new_sample["cot_attempt"] = attempt
                new_sample["cot_mae"] = mae
                new_sample["cot_normalized_mae"] = normalized_mae
                matched_records.append(new_sample)
                matched_count += 1
        else:
            unmatched_indices.append(sample.get("idx", sample_idx))
            unmatched_samples.append(dict(sample))

    return {
        "matched_records": matched_records,
        "matched_count": matched_count,
        "matched_samples": matched_samples,
        "total": len(dataset),
        "unmatched_indices": unmatched_indices,
        "unmatched_samples": unmatched_samples,
        "metric": "normalized_mae",
        "threshold": threshold,
    }


def rejection_sampling_cot(
    dataset: List[Dict[str, Any]],
    answers: Dict[int, Dict[str, Any]],
    task: str,
) -> Dict[str, Any]:
    if task.lower() == "reasoning_forecasting":
        return _rejection_sampling_forecasting(dataset, answers)
    return _rejection_sampling_reasoning(dataset, answers)


def resolve_paths(args: argparse.Namespace) -> Dict[str, str]:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    dataset_path = args.dataset
    if dataset_path is None:
        task_cfg = DEFAULT_TASK_CONFIG.get(args.task)
        if not task_cfg:
            available = ", ".join(sorted(DEFAULT_TASK_CONFIG.keys()))
            raise ValueError(f"Unknown task '{args.task}'. Available: {available}")
        dataset_path = os.path.join(repo_root, task_cfg["dataset"])
    elif not os.path.isabs(dataset_path):
        dataset_path = os.path.join(repo_root, dataset_path)

    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset_dir = os.path.dirname(dataset_path)
    dataset_name = os.path.basename(dataset_path)
    base = dataset_name[: -len(".jsonl")] if dataset_name.endswith(".jsonl") else dataset_name
    for suffix in ("_finetune", "_rl", "_test"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    answers_path = args.answers
    if answers_path is None:
        answers_path = os.path.join(repo_root, "exp", args.exp, "generated_answer.json")
    elif not os.path.isabs(answers_path):
        answers_path = os.path.join(repo_root, answers_path)

    if not os.path.isfile(answers_path):
        raise FileNotFoundError(f"Generated answers not found: {answers_path}")

    output_path = args.output
    if output_path is None:
        output_name = f"{base}_cot.jsonl"
        output_path = os.path.join(dataset_dir, output_name)
    elif not os.path.isabs(output_path):
        output_path = os.path.join(repo_root, output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    rl_path = os.path.join(dataset_dir, f"{base}_rl.jsonl")
    rl_new_path = os.path.join(dataset_dir, f"{base}_rl_new.jsonl")

    return {
        "dataset": dataset_path,
        "answers": answers_path,
        "output": output_path,
        "rl": rl_path,
        "rl_new": rl_new_path,
    }


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)

    logger.info(f"Loading dataset from {paths['dataset']}")
    dataset = load_jsonl(paths["dataset"])

    logger.info(f"Loading generated answers from {paths['answers']}")
    answers = load_generated_answers(paths["answers"])

    result = rejection_sampling_cot(dataset, answers, args.task)
    save_jsonl(paths["output"], result["matched_records"])

    logger.info(
        f"Saved {len(result['matched_records'])} matched CoT responses "
        f"(covering {result['matched_samples']} / {result['total']} samples) to {paths['output']}"
    )
    if result.get("metric") == "normalized_mae" and result.get("threshold") is not None:
        logger.info(
            "Normalized MAE threshold for top 20%% responses: %.6f",
            result["threshold"],
        )

    # Merge unmatched samples with existing RL split and save to new file
    rl_records: List[Dict[str, Any]] = []
    if os.path.isfile(paths["rl"]):
        try:
            rl_records = load_jsonl(paths["rl"])
        except Exception as err:
            logger.warning(f"Failed to load RL dataset {paths['rl']}: {err}. Treating as empty.")
            rl_records = []
    else:
        logger.warning(f"RL dataset not found at {paths['rl']}. Starting from empty list.")

    merged_rl = rl_records + result["unmatched_samples"]
    save_jsonl(paths["rl_new"], merged_rl)

    if result["unmatched_indices"]:
        logger.warning(
            "Samples with no correct responses (moved to RL): %s",
            ", ".join(map(str, result["unmatched_indices"][:20]))
            + (" ..." if len(result["unmatched_indices"]) > 20 else ""),
        )
    logger.info(
        f"Saved combined RL dataset with {len(merged_rl)} samples to {paths['rl_new']}"
    )


if __name__ == "__main__":
    main()

