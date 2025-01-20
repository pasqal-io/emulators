from emu_sv.state_vector import StateVector
from emu_sv.dense_operator import DenseOperator
from emu_sv.sv_config import SVConfig
from emu_sv.custom_callback_implementations import (
    custom_qubit_density,
    custom_correlation_matrix,
)
from emu_base.base_classes.default_callbacks import QubitDensity, CorrelationMatrix

from pytest import approx

from unittest.mock import MagicMock


def test_custom_qubit_density():
    # set up for state
    basis = ("r", "g")
    num_qubits = 4
    strings = {"rrrr": 1.0, "ggggg": 1.0}
    state = StateVector.from_state_string(
        basis=basis, nqubits=num_qubits, strings=strings
    )

    config = SVConfig()

    operator_mock = MagicMock(spec=DenseOperator)

    H_mock = operator_mock.return_value

    MockQubitDensity = MagicMock(spec=QubitDensity)

    qubit_density_mock = MockQubitDensity.return_value

    t = 1

    qubit_density = custom_qubit_density(qubit_density_mock, config, t, state, H_mock)
    expected = [0.5] * num_qubits
    assert qubit_density == approx(expected, abs=1e-8)


def test_custom_correlation():
    # set up for state
    basis = ("r", "g")
    num_qubits = 4
    strings = {"rrrr": 1.0, "gggg": 1.0}
    state = StateVector.from_state_string(
        basis=basis, nqubits=num_qubits, strings=strings
    )
    config = SVConfig()
    operator_mock = MagicMock(spec=DenseOperator)
    H_mock = operator_mock.return_value
    MockCorrelation = MagicMock(spec=CorrelationMatrix)
    correlation_mock = MockCorrelation.return_value
    t = 1
    correlation = custom_correlation_matrix(
        correlation_mock, config, t, state, H_mock)
    print(correlation)
    expected = [[0.5] * num_qubits]*num_qubits
    for i, row in enumerate(correlation):
        for j, col in enumerate(row):
            assert col == approx(expected[i][j], abs=1e-8)

