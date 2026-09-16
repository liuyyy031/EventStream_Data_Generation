# Healthcare domain package

This package implements synthetic inpatient monitoring workflows for LLM event
stream understanding research. It is executable, but it is not a clinical
simulation, diagnostic system, treatment recommender, or empirical patient
outcome model.

## Implemented workflows

- scheduled temperature observation -> configured-threshold flag -> alert ->
  clinical workflow review -> non-prescriptive care-plan adjustment ->
  follow-up observation -> status assessment;
- scheduled oxygen-saturation observation through the same reviewed workflow;
- specimen collection -> laboratory result -> configured-threshold flag ->
  reviewed workflow;
- two independently observed abnormal vital signs -> one explicit two-parent
  alert -> reviewed workflow;
- routine normal observation -> scheduled repeat observation -> status
  assessment.

The threshold flag is explicitly a rule-derived finding, not a diagnosis. The
care-plan event contains no drug, dose, procedure, or efficacy claim. A
follow-up status means only that a generated value is inside the configured
synthetic reference band; it does not assert that an intervention caused an
improvement.

## Typed context instead of one generic grid

The context graph contains patients, encounters, devices, specimens,
practitioners, and locations. Its sparse typed layers express only admissible
care context: encounter--patient, encounter--location, device--patient,
specimen--patient, and practitioner--patient. Global connectivity and spatial
coordinates are not required, and a context edge never becomes an event parent
unless an executed mechanism names actual parent events.

At least one complete synthetic care bundle is guaranteed per generated
context so every configured workflow remains executable even in a small unit
test. This supplemental structure is reported in the topology profile rather
than hidden.

## Time semantics

Configured observations and routine repeats use deterministic scheduled-time
atoms. Laboratory turnaround, review, care-plan, follow-up, observation, and
recording delays use bounded conditional log-normal models. The model family is
standard; the bundled coefficients and bounds are transparent synthetic priors
and are not claimed as clinically estimated parameters. The occurrence,
observation, and recording timestamps remain separate.

## Structural sources and limits

- HL7 FHIR Observation motivates keeping the subject, encounter, device,
  specimen, clinically relevant occurrence time, and issued/recording context
  distinct: https://hl7.org/fhir/observation.html
- NEWS2 motivates the high-level pattern of physiological measurement,
  abnormality recognition, escalation, review, and changed monitoring
  frequency. This package does not reproduce NEWS2 scoring or prescribe its
  clinical response thresholds:
  https://www.rcp.ac.uk/media/a4ibkkbf/news2-final-report_0_0.pdf
- The shared time--type--entity risk-set factorization is inspired by flexible
  marked spatio-temporal point-process work, but the healthcare mechanisms and
  JSON audit contract are this project's implementation design:
  https://arxiv.org/abs/2103.04647

## Run

From the repository root, deterministic validation only:

```bash
python data_generation/run_data_generation.py \
  --domain healthcare \
  --episode-count 20 \
  --nodes-per-context 1000 \
  --seed 20260916 \
  --output-dir data_generation/output_healthcare_20
```

With the server-side semantic Judge (credentials and service URL are loaded
from `.env`):

```bash
python data_generation/run_data_generation.py \
  --domain healthcare \
  --episode-count 20 \
  --nodes-per-context 1000 \
  --seed 20260916 \
  --judge-mode llm \
  --judge-model deepseek-v4-flash \
  --judge-workers 4 \
  --output-dir data_generation/output_healthcare_llm_20
```

Use `--scenario-family` with one of `temperature_escalation`,
`oxygen_desaturation_response`, `laboratory_abnormality_review`,
`combined_vital_sign_escalation`, or `routine_observation` for a focused test.
