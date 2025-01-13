import math
from typing import Any

from emu_base.base_classes.config import BackendConfig
from emu_base.base_classes.default_callbacks import QubitDensity
from emu_base.base_classes.operator import Operator

from emu_sv import StateVector


def custom_qubit_density(
    self: QubitDensity, config: BackendConfig, t: int, state: StateVector, H: Operator
) -> Any:

    num_qubits = int(math.log2(len(state.vector)))
    return [
        state.vector.reshape((2,) * num_qubits).select(i, 1).norm() ** 2
        for i in range(num_qubits)
    ]
