# Healthcare generation and semantic-correction test protocol

This protocol separates two questions that must not be conflated.

1. The regular generation run tests whether healthcare episodes satisfy the
   structural, temporal, text-alignment, Judge, QA, and export contracts.
2. The controlled correction challenge tests whether an explicitly known
   text-only semantic defect is detected and removed without changing any
   structured event-stream fact.

The controlled challenge output is evaluation-only and must not be included in
training data.

## A. Regular healthcare run

Use at least 100 episodes and the normal 1,000-entity context. Enable the LLM
Judge and correction loop:

```bash
python data_generation/run_data_generation.py \
  --domain healthcare \
  --episode-count 100 \
  --nodes-per-context 1000 \
  --seed 20260919 \
  --judge-mode llm \
  --judge-model deepseek-v4-flash \
  --judge-base-url https://api.deepseek.com \
  --judge-workers 4 \
  --semantic-correction-mode llm \
  --semantic-correction-max-rounds 2 \
  --qa-export-mode ground_truth_template \
  --output-dir data_generation/output_healthcare_formal_100
```

Passing requires `quality_report.json` to report `passed: true`, zero
deterministic and text-alignment failures, zero QA validation failures, and
zero Judge/correction protocol failures. A correction-call count of zero is
not a failure: deterministic grounded text may contain no text defect.

## B. Controlled correction challenge

Run five evaluation-only cases. Each starts from a baseline that the Judge has
passed. The evaluator then appends a deliberately unsupported diagnostic and
treatment-effect claim while keeping the claim-to-record alignment and the
entire structured episode unchanged.

```bash
python data_generation/evaluation/run_healthcare_correction_challenge.py \
  --challenge-count 5 \
  --nodes-per-context 1000 \
  --seed 20260920 \
  --judge-model deepseek-v4-flash \
  --judge-base-url https://api.deepseek.com \
  --protocol-max-attempts 3 \
  --output-dir data_generation/artifacts/healthcare_correction_challenge
```

The report checks four separate properties:

- the Judge classifies the injected error as an addressable text issue;
- the corrector produces a protocol-valid rewrite;
- the corrected text passes deterministic alignment and the second Judge;
- the structured-truth SHA-256 remains identical before and after correction.

The aggregate pass threshold is 80% for detection, successful correction, and
post-correction Judge recovery, plus 100% structured-truth preservation.

