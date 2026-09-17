# Healthcare domain structure

This directory reserves the domain contract only. The first implementation
should cover one auditable loop: latent patient state, scheduled observation,
derived abnormality, alert, review, intervention, and follow-up observation.
Raw measurements must remain distinct from semantic events. No clinical
probability or time parameter is implied by the current placeholder.

Its reserved topology uses the shared sparse heterogeneous-graph engine. Device,
specimen, practitioner, encounter, and patient populations remain typed; global
connectivity and spatial coordinates are not required. Multi-party clinical
contexts must be represented through encounter or workflow entities rather than
by turning every participant pair into an edge.

`mechanism_specs.json` is intentionally empty. It verifies the shared mechanism
contract without inventing clinical events, relationships, or probabilities
before a reviewed reference dataset is selected.
