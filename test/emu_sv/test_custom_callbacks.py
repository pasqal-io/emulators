from emu_sv.state_vector import StateVector
from emu_sv.dense_operator import DenseOperator
from emu_sv.sv_config import SVConfig
from emu_sv.custom_callback_implementations import custom_qubit_density
from emu_base.base_classes.default_callbacks import QubitDensity

from pytest import approx

from unittest.mock import MagicMock


def test_custom_qubit_density():
    #set up for state
    basis = ("r", "g")
    num_qubits = 4
    strings = {"rrrr": 1.0, "ggggg": 1.0}
    state = StateVector.from_state_string(
        basis=basis, nqubits=num_qubits, strings=strings
    )

    config = SVConfig()
    
    MockOperator = MagicMock(spec=DenseOperator)

    H_mock = MockOperator.return_value

    MockQubitDensity = MagicMock(spec=QubitDensity)

    qubitDenistyMock = MockQubitDensity.return_value 

    t = 1

    qubit_density = custom_qubit_density(qubitDenistyMock, config, t, state, H_mock)
    expected = [0.5] * num_qubits
    assert qubit_density == approx(expected, abs=1e-8)
