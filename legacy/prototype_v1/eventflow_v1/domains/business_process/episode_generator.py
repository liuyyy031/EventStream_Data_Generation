"""Future business-process EventFlow implementation."""


class BusinessProcessEpisodeGenerator:
    def generate(self, *_: object, **__: object) -> None:
        raise NotImplementedError(
            "Business-process generation is deferred until the Transportation pipeline is validated"
        )

