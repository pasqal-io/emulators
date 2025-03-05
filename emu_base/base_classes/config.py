from __future__ import annotations
from pulser.noise_model import NoiseModel
import logging
import sys
import pathlib
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from emu_base.base_classes.callback import Callback


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
        observables: list[Callback] | None = None,
        with_modulation: bool = False,
        noise_model: NoiseModel | None = None,
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

        if interaction_matrix is not None and not (
            isinstance(interaction_matrix, list)
            and isinstance(interaction_matrix[0], list)
            and isinstance(interaction_matrix[0][0], float)
        ):
            raise ValueError(
                "Interaction matrix must be provided as a Python list of lists of floats"
            )

        if interaction_matrix is not None:
            int_mat = torch.tensor(interaction_matrix)
            tol = 1e-10
            if not (
                int_mat.numel() != 0
                and torch.all(torch.isreal(int_mat))
                and int_mat.dim() == 2
                and int_mat.shape[0] == int_mat.shape[1]
                and torch.allclose(int_mat, int_mat.T, atol=tol)
                and torch.norm(torch.diag(int_mat)) < tol
            ):
                raise ValueError("Interaction matrix is not symmetric and zero diag")

        self.interaction_matrix = interaction_matrix
        self.interaction_cutoff = interaction_cutoff
        self.logger = logging.getLogger("global_logger")
        self.log_file = log_file
        self.log_level = log_level

        self.init_logging()

        if noise_model is not None and (
            noise_model.runs != 1
            or noise_model.samples_per_run != 1
            or noise_model.runs is not None
            or noise_model.samples_per_run is not None
        ):
            self.logger.warning(
                "Warning: The runs and samples_per_run values of the NoiseModel are ignored!"
            )

    def init_logging(self) -> None:
        if self.log_file is None:
            logging.basicConfig(
                level=self.log_level, format="%(message)s", stream=sys.stdout, force=True
            )  # default to stream = sys.stderr
        else:
            logging.basicConfig(
                level=self.log_level,
                format="%(message)s",
                filename=str(self.log_file),
                filemode="w",
                force=True,
            )
