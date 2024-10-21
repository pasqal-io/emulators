from __future__ import annotations
from pulser.noise_model import NoiseModel
import logging
import sys


class BackendConfig:
    """The base backend configuration.

    Args:
        observables: a list of callbacks to compute observables
        with_modulation: if True, run the sequence with hardware modulation
        noise_model: The pulser.NoiseModel to use in the simulation.
        interaction_matrix: when specified, override the interaction terms in the Hamiltonian
        log_level: The output verbosity. Should be one of the constants from logging.
    """

    def __init__(
        self,
        *,
        # "Callback" is a forward type reference because of the circular import otherwise.
        observables: list["Callback"] | None = None,  # type: ignore # noqa: F821
        with_modulation: bool = False,
        noise_model: NoiseModel = None,
        interaction_matrix: list[list[float]] | None = None,
        log_level: int = logging.INFO,
    ):
        if observables is None:
            observables = []
        self.callbacks = (
            observables  # we can add other types of callbacks, and just stack them
        )
        self.with_modulation = with_modulation
        self.noise_model = noise_model
        self.interaction_matrix = interaction_matrix
        self.logger = logging.getLogger("global_logger")
        logging.basicConfig(
            level=log_level, format="%(message)s", stream=sys.stdout, force=True
        )  # default to stream = sys.stderr
        if noise_model is not None and (
            noise_model.runs != 1 or noise_model.samples_per_run != 1
        ):
            self.logger.warning(
                "Warning: The runs and samples_per_run values of the NoiseModel are ignored!"
            )
