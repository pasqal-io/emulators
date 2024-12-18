from __future__ import annotations
from pulser.noise_model import NoiseModel
import logging
import sys
import pathlib


class BackendConfig:
    """The base backend configuration.

    Args:
        observables: a list of callbacks to compute observables
        with_modulation: if True, run the sequence with hardware modulation
        noise_model: The pulser.NoiseModel to use in the simulation.
        interaction_matrix: When specified, override the interaction terms in the Hamiltonian.
            This corresponds to the $U_{ij}$ terms in the documentation. Must be symmetric.
        interaction_cutoff: set interaction coefficients smaller than this to 0.
            This can improve the memory profile of the application for some backends.
        log_level: The output verbosity. Should be one of the constants from logging.
        log_file: a path to a file where to store the log, instead of printing to stdout

    Examples:
        >>> observables = [BitStrings(400, 100)] #compute 100 bitstrings at 400ns
        >>> noise_model = pulser.noise_model.NoiseModel()
        >>> interaction_matrix = [[1 for _ in range(nqubits)] for _ in range(nqubits)]
        >>> interaction_cutoff = 2.0 #this will turn off all the above interactions again
        >>> log_level = logging.warn
    """

    def __init__(
        self,
        *,
        # "Callback" is a forward type reference because of the circular import otherwise.
        observables: list["Callback"] | None = None,  # type: ignore # noqa: F821
        with_modulation: bool = False,
        noise_model: NoiseModel = None,
        interaction_matrix: list[list[float]] | None = None,
        interaction_cutoff: float = 0.0,
        log_level: int = logging.INFO,
        log_file: pathlib.Path | None = None,
    ):
        if observables is None:
            observables = []
        self.callbacks = (
            observables  # we can add other types of callbacks, and just stack them
        )
        self.with_modulation = with_modulation
        self.noise_model = noise_model

        if interaction_matrix is not None and (
            not isinstance(interaction_matrix, list)
            or not isinstance(interaction_matrix[0], list)
        ):
            raise ValueError(
                "Interaction matrix must be provided as a Python list of lists of floats"
            )

        self.interaction_matrix = interaction_matrix
        self.interaction_cutoff = interaction_cutoff
        self.logger = logging.getLogger("global_logger")
        if log_file is None:
            logging.basicConfig(
                level=log_level, format="%(message)s", stream=sys.stdout, force=True
            )  # default to stream = sys.stderr
        else:
            logging.basicConfig(
                level=log_level,
                format="%(message)s",
                filename=str(log_file),
                filemode="w",
                force=True,
            )
        if noise_model is not None and (
            noise_model.runs != 1
            or noise_model.samples_per_run != 1
            or noise_model.runs is not None
            or noise_model.samples_per_run is not None
        ):
            self.logger.warning(
                "Warning: The runs and samples_per_run values of the NoiseModel are ignored!"
            )
