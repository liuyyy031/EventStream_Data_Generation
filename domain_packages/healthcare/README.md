# Healthcare domain structure

This directory reserves the domain contract only. The first implementation
should cover one auditable loop: latent patient state, scheduled observation,
derived abnormality, alert, review, intervention, and follow-up observation.
Raw measurements must remain distinct from semantic events. No clinical
probability or time parameter is implied by the current placeholder.
