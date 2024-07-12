from emu_ct.base_classes.callbacks import Callback
from pulser.noise_model import NoiseModel


class BackendConfig:
    """The base backend configuration.

    Attributes:
        observables: a list of callbacks to compute observables
        with_modulation: whether or not run the sequence with hardware modulation
    """

    def __init__(
        self,
        *,
        observables: list[Callback] = [],
        with_modulation: bool = False,
        noise_model: NoiseModel = None
    ):
        self.callbacks = (
            observables  # we can add other types of callbacks, and just stack them
        )
        self.with_modulation = with_modulation
        self.noise_model = noise_model
