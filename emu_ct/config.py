from typing import Any
import torch


class SingletonMeta(type):

    _instances: dict = {}

    def __call__(cls: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Config(metaclass=SingletonMeta):
    """
    This is a singleton config class, i.e. Config() always returns the same object
    which will be created when Config() is first called,
    and will exist for the lifetime of the program.
    It stores the config values, to be retrieved/modified at will via
    Config().some_getter_or_setter(...)
    """

    def __init__(self) -> None:
        self._num_devices_actual = torch.cuda.device_count()
        self._num_devices_to_use = self._num_devices_actual
        self._max_bond_dim = 1024
        self.bond_precision = 1e-8

    def set_num_devices_to_use(self, devices: int) -> None:
        self._num_devices_to_use = min(self._num_devices_actual, devices)

    def get_num_devices_to_use(self) -> int:
        # mypy does not understand that this is an int
        return self._num_devices_to_use  # type: ignore[no-any-return]

    def set_max_bond_dim(self, max_bond_dim: int) -> None:
        self._max_bond_dim = max_bond_dim

    def get_max_bond_dim(self) -> int:
        # mypy does not understand that this is an int
        return self._max_bond_dim  # type: ignore[no-any-return]

    def set_bond_precision(self, precision: float) -> None:
        self.bond_precision = precision

    def get_bond_precision(self) -> float:
        return self.bond_precision
