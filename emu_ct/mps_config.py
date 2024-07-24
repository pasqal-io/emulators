from typing import Any
from emu_ct.base_classes.config import BackendConfig
from emu_ct.base_classes.state import State
from .utils import DEVICE_COUNT
import torch


class MPSConfig(BackendConfig):
    """
    The configuration of the emu-ct MPSBackend.
    """

    def __init__(
        self,
        *,
        initial_state: State | None = None,
        dt: int = 10,
        precision: float = 1e-5,
        max_bond_dim: int = 1024,
        max_krylov_dim: int = 100,
        extra_krylov_tolerance: float = 1e-3,
        num_devices_to_use: int = DEVICE_COUNT,
        interaction_matrix: torch.Tensor | None = None,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.initial_state = initial_state
        self.dt = dt
        self.precision = precision
        self.max_bond_dim = max_bond_dim
        self.max_krylov_dim = max_krylov_dim
        self.num_devices_to_use = num_devices_to_use
        self.interaction_matrix = interaction_matrix
        self.extra_krylov_tolerance = extra_krylov_tolerance

        if self.noise_model is not None:
            if not set(self.noise_model.noise_types).issubset({"SPAM"}):
                raise NotImplementedError(
                    "Unsupported noise type(s): "
                    + str(set(self.noise_model.noise_types) - {"SPAM"})
                )

            if self.noise_model.p_false_pos > 0.0 or self.noise_model.p_false_neg > 0.0:
                raise NotImplementedError(
                    "Unsupported: measurement errors - set p_false_pos=0. and p_false_neg=0."
                )
