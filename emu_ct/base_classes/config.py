from .callbacks import Callback


class BackendConfig:
    """The base backend configuration.

    Attributes:
        backend_options: A dictionary of backend specific options.
        with_modulation: whether or not run the sequence with hardware modulation
    """

    def __init__(
        self, *, observables: list[Callback] = [], with_modulation: bool = False
    ):
        self.callbacks = (
            observables  # we can add other types of callbacks, and just stack them
        )
        self.with_modulation = with_modulation
