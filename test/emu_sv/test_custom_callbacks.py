import torch
from emu_sv.state_vector import StateVector
from emu_sv.dense_operator import DenseOperator
from emu_sv.sv_config import SVConfig
from emu_sv.custom_callback_implementations import (
    custom_qubit_density,
    custom_correlation_matrix,
    custom_energy,
    custom_energy_variance,
    custom_second_momentum_energy,
)
from emu_base.base_classes.default_callbacks import (
    QubitDensity,
    CorrelationMatrix,
    Energy,
    EnergyVariance,
    SecondMomentOfEnergy,
)

from pytest import approx

from unittest.mock import MagicMock

from emu_sv.hamiltonian import RydbergHamiltonian

# device = "cuda"
device = "cpu"


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
    strings = {"rgrg": 1.0, "grgr": 1.0}
    state = StateVector.from_state_string(
        basis=basis, nqubits=num_qubits, strings=strings
    )
    config = SVConfig()
    operator_mock = MagicMock(spec=DenseOperator)
    H_mock = operator_mock.return_value
    MockCorrelation = MagicMock(spec=CorrelationMatrix)
    correlation_mock = MockCorrelation.return_value
    t = 1
    correlation = custom_correlation_matrix(correlation_mock, config, t, state, H_mock)

    expected = []
    for qubiti in range(num_qubits):
        correlation_one = []
        for qubitj in range(num_qubits):
            if (qubiti + qubitj) % 2 == 0:
                correlation_one.append(0.5)
            else:
                correlation_one.append(0.0)
        expected.append(correlation_one)

    for i, row in enumerate(correlation):
        for j, col in enumerate(row):
            assert col == approx(expected[i][j], abs=1e-8)


def test_custom_energy_and_variance_and_second():

    torch.manual_seed(1337)
    dtype = torch.float64

    basis = ("r", "g")
    num_qubits = 4
    strings = {"rgrg": 1.0, "grgr": 1.0}
    state = StateVector.from_state_string(
        basis=basis, nqubits=num_qubits, strings=strings
    )
    config = SVConfig()

    omega = torch.randn(num_qubits, dtype=dtype, device=device)
    delta = torch.randn(num_qubits, dtype=dtype, device=device)
    interaction_matrix = torch.randn((num_qubits, num_qubits))
    h_rydberg = RydbergHamiltonian(
        omegas=omega, deltas=delta, interaction_matrix=interaction_matrix, device=device
    )

    MockEnergy = MagicMock(spec=Energy)
    energy_mock = MockEnergy.return_value
    t = 1
    energy = custom_energy(energy_mock, config, t, state, h_rydberg)
    expected_energy = 0.73826361936

    assert energy == approx(expected_energy, abs=1e-8)

    MockEnergy = MagicMock(spec=EnergyVariance)
    energy_variance_mock = MockEnergy.return_value

    energy_variance = custom_energy_variance(
        energy_variance_mock, config, t, state, h_rydberg
    )
    expected_varaince = 3.67378968943955
    assert energy_variance == approx(expected_varaince, abs=1e-8)

    MockEnergy = MagicMock(spec=SecondMomentOfEnergy)
    second_momentum_mock = MockEnergy.return_value

    second_momentum = custom_second_momentum_energy(
        second_momentum_mock, config, t, state, h_rydberg
    )
    expected_second = 4.2188228611101

    assert second_momentum == approx(expected_second, abs=1e-8)
