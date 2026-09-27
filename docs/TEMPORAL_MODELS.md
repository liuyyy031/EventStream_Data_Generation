# Temporal model library

The generator separates a legal domain mechanism from its clock.  A temporal
model determines when an already-eligible candidate fires; it does not create
an event type, parent edge, participant, or domain rule.

## Baseline models

| Family | Purpose |
|---|---|
| `deterministic_delay` | Immediate or fixed delay |
| `scheduled_time` | Declared absolute offset |
| `exponential_arrival` | Memoryless random arrival |
| `conditional_lognormal` | Positive right-skewed conditional delay |
| `conditional_gamma` | Accumulated multi-stage conditional delay |
| `conditional_weibull` | Increasing or decreasing duration-dependent risk |
| `empirical_delay` | Resampling from observed delays |
| `piecewise_exponential_hazard` | State-dependent hazard with preserved cumulative risk |
| `exponential_backoff` | Retry policy with jitter and a cap |

## Additional models

| Family | Required domain inputs | Semantics |
|---|---|---|
| `periodic_piecewise_hazard` | A configured repeating rate profile | Non-homogeneous periodic or seasonal arrivals |
| `self_exciting_hazard` | `history_event_offsets` | Exponential-kernel Hawkes clock conditional on supplied history |
| `refractory_hazard` | Optional `last_event_offset_seconds` | Hard cooldown followed by an exponential clock |
| `conditional_lognormal_mixture` | Optional component covariates | Multimodal positive delay |
| `resource_queue_delay` | Resource availability and optional queue delay | FCFS wait plus Gamma or explicit service time |
| `availability_window_delay` | `required_work_seconds`, availability intervals | Work accumulates only in business/availability windows |
| `interval_time` | Earliest and latest offsets | Latent occurrence within a reported interval |
| `batch_scheduled_time` | Batch ID and shared offset | Multiple candidates receive the exact same timestamp |

The advanced classes live in `generation_core/advanced_temporal.py`.  They are
parameterized and therefore are not inserted into the neutral registry with
invented defaults.  A domain package must register the models it can justify
and reference their IDs from declared mechanisms.

## Important scope boundaries

- The periodic model supplies a repeating piecewise-constant intensity.  A
  domain decides whether bins represent hours, weekdays, seasons, or another
  calendar scale.
- Hawkes and refractory clocks support dynamic history updates without
  resetting accumulated hazard or the random threshold. Candidates can declare
  `history_spec` for scoped, observable history. The scheduler refreshes it at
  realized events and declared availability changes; models never invent events.
- The queue model consumes resource availability supplied by the domain.
  Resource ownership and capacity transitions remain domain state, not hidden
  temporal-model state.
- The availability model consumes offset intervals.  Time-zone conversion,
  holidays, and local business calendars must be converted to those intervals
  before simulation.
- The interval model samples a latent occurrence for generation while keeping
  the original interval in provenance.  It must not be used to pretend that a
  reference dataset supplied an exact timestamp.
- Batch candidates share a timestamp and batch identity.  The scheduler still
  uses declared priorities to serialize state updates at that timestamp; this
  is deterministic same-time execution, not an unordered database transaction.

## Validation status

Every additional family has tests for support, hazard/survival or point-mass
semantics, provenance, and registry descriptors.  This establishes software
correctness for the implemented formulas.  It does not establish that any one
parameterization fits a real domain.  Parameter acceptance still requires
reference-data fitting or an explicitly justified policy/configuration.
