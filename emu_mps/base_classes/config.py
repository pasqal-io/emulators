from pulser.noise_model import NoiseModel


class BackendConfig:
    """The base backend configuration.

    Args:
        observables: a list of callbacks to compute observables
        with_modulation: whether or not run the sequence with hardware modulation
        noise_model: The pulser.NoiseModel to use in the simulation.
    """

    def __init__(
        self,
        *,
        # "Callback" is a forward type reference because of the circular import otherwise.
        observables: list["Callback"] = [],  # type: ignore # noqa: F821
        with_modulation: bool = False,
        noise_model: NoiseModel = None
    ):
        self.callbacks = (
            observables  # we can add other types of callbacks, and just stack them
        )
        self.with_modulation = with_modulation
        self.noise_model = noise_model
