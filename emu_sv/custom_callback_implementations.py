import math
from typing import Any

import torch

from emu_base.base_classes.config import BackendConfig
from emu_base.base_classes.default_callbacks import Energy, QubitDensity
from emu_base.base_classes.operator import Operator

from emu_sv import StateVector
from emu_sv.hamiltonian import RydbergHamiltonian


def custom_qubit_density(
    self: QubitDensity, config: BackendConfig, t: int, state: StateVector, H: Operator
) -> Any:

    num_qubits = int(math.log2(len(state.vector)))
    state_tensor = state.vector.reshape((2,) * num_qubits)
    return [(state_tensor.select(i, 1).norm() ** 2).item() for i in range(num_qubits)]


# feeding RydbergHamiltonian class as an Operator
# for performance reasons
def custom_energy(
    self: Energy, config: BackendConfig, t: int, state: StateVector, H: RydbergHamiltonian
) -> Any:
    return torch.vdot(state, H * state).item()

    # TODO: make a test for custom energy
