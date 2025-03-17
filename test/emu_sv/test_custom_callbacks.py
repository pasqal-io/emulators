import torch

from pytest import approx
from unittest.mock import MagicMock

from emu_base import DEVICE_COUNT

from emu_sv.custom_callback_implementations import (
    correlation_matrix_sv_impl,
    energy_variance_sv_impl,
<<<<<<< HEAD
    qubit_density_sv_impl,
    second_moment_sv_impl,
=======
    qubit_occupation_sv_impl,
    energy_second_moment_sv_impl,
>>>>>>> kb/emu-sv-API
)
from emu_sv.dense_operator import DenseOperator
from emu_sv.hamiltonian import RydbergHamiltonian
from emu_sv.sv_config import SVConfig
from emu_sv.state_vector import StateVector

from pulser.backend.default_observables import (
    CorrelationMatrix,
    EnergySecondMoment,
    EnergyVariance,
    Occupation,
)

device = "cuda" if DEVICE_COUNT > 0 else "cpu"


<<<<<<< HEAD
def test_custom_qubit_density() -> None:
    # set up for state
    basis = ("r", "g")
    num_qubits = 4
    strings = {"rrrr": 1.0, "gggg": 1.0}
=======
def test_custom_occupation() -> None:
    # set up for state
    basis = ("r", "g")
    num_qubits = 4
    strings = {"rrrr": 1.0, "ggggg": 1.0}
>>>>>>> kb/emu-sv-API
    state = StateVector.from_state_amplitudes(eigenstates=basis, amplitudes=strings)

    operator_mock = MagicMock(spec=DenseOperator)

    H_mock = operator_mock.return_value

    MockOccupation = MagicMock(spec=Occupation)

<<<<<<< HEAD
    qubit_density_mock = MockOccupation.return_value
=======
    occupation_mock = MockOccupation.return_value
>>>>>>> kb/emu-sv-API

    t = 1

    config = SVConfig()

<<<<<<< HEAD
    qubit_density = qubit_density_sv_impl(qubit_density_mock, config, t, state, H_mock)
=======
    occupation = qubit_occupation_sv_impl(occupation_mock, config, t, state, H_mock)
>>>>>>> kb/emu-sv-API
    expected = [0.5] * num_qubits
    assert occupation.cpu() == approx(expected, abs=1e-8)


def test_custom_correlation() -> None:
    # set up for state
    basis = ("r", "g")
    num_qubits = 4
    strings = {"rgrg": 1.0, "grgr": 1.0}
    state = StateVector.from_state_amplitudes(eigenstates=basis, amplitudes=strings)
    config = SVConfig()
    operator_mock = MagicMock(spec=DenseOperator)
    H_mock = operator_mock.return_value
    correlation_matrix_mock = MagicMock(spec=CorrelationMatrix)
    correlation_mock = correlation_matrix_mock.return_value
    t = 1
    correlation = correlation_matrix_sv_impl(correlation_mock, config, t, state, H_mock)

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
            assert col.cpu() == approx(expected[i][j], abs=1e-8)


def test_custom_energy_and_variance_and_second() -> None:

    torch.manual_seed(1337)
    dtype = torch.complex128

    basis = ("r", "g")
    num_qubits = 4
    strings = {"rgrg": 1.0, "grgr": 1.0}
    state = StateVector.from_state_amplitudes(
        eigenstates=basis, amplitudes=strings, gpu=device == "cuda"
    )
    config = SVConfig()

    omegas = torch.randn(num_qubits, dtype=dtype).to(device)
    deltas = torch.randn(num_qubits, dtype=dtype).to(device)
    phis = torch.zeros_like(omegas)
    interaction_matrix = torch.randn((num_qubits, num_qubits))
    h_rydberg = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interaction_matrix,
        device=device,
    )

    t = 1

    energy_mock = MagicMock(spec=EnergyVariance)
    energy_variance_mock = energy_mock.return_value

    energy_variance = energy_variance_sv_impl(
        energy_variance_mock, config, t, state, h_rydberg
    )
    expected_varaince = 3.67378968943955

    assert energy_variance.cpu() == approx(expected_varaince, abs=4e-7)

    second_moment_energy_mock = MagicMock(spec=EnergySecondMoment)
    second_moment_mock = second_moment_energy_mock.return_value

    second_moment = energy_second_moment_sv_impl(
        second_moment_mock, config, t, state, h_rydberg
    )
    expected_second = 4.2188228611101

    assert second_moment.cpu() == approx(expected_second, abs=2e-7)
