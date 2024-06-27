from typing import Any
from .base_classes.config import BackendConfig
from .utils import DEVICE_COUNT


class MPSConfig(BackendConfig):
    def __init__(
        self,
        *,
        initial_state: str = "ground",
        dt: int = 10,
        precision: float = 1e-10,
        max_bond_dim: int = 1024,
        max_krylov_dim: int = 100,
        num_devices_to_use: int = DEVICE_COUNT,
        **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.initial_state = initial_state
        self.dt = dt
        self.precision = precision
        self.max_bond_dim = max_bond_dim
        self.max_krylov_dim = max_krylov_dim
        self.num_devices_to_use = num_devices_to_use
