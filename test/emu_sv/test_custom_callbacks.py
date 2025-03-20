import torch

from pytest import approx
from unittest.mock import MagicMock

from emu_base import DEVICE_COUNT

from emu_sv.custom_callback_implementations import (
    correlation_matrix_sv_impl,
    energy_variance_sv_impl,
    qubit_occupation_sv_impl,
    energy_second_moment_sv_impl,
)
from emu_sv import (
    DenseOperator,
    SVConfig,
    StateVector,
)
from emu_sv.hamiltonian import RydbergHamiltonian

from pulser.backend import (
    CorrelationMatrix,
    EnergySecondMoment,
    EnergyVariance,
    Occupation,
)

device = "cuda" if DEVICE_COUNT > 0 else "cpu"


def test_custom_occupation() -> None:
    # set up for state
    basis = ("r", "g")
    num_qubits = 4
    strings = {"rrrr": 1.0, "ggggg": 1.0}
    state = StateVector.from_state_amplitudes(eigenstates=basis, amplitudes=strings)

    operator_mock = MagicMock(spec=DenseOperator)

    H_mock = operator_mock.return_value

    MockOccupation = MagicMock(spec=Occupation)

    occupation_mock = MockOccupation.return_value

    config = SVConfig()

    occupation = qubit_occupation_sv_impl(
        occupation_mock, config=config, state=state, hamiltonian=H_mock
    )
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
    correlation = correlation_matrix_sv_impl(
        correlation_mock, config=config, state=state, hamiltonian=H_mock
    )

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

    basis = ("r", "g")
    num_qubits = 4
    strings = {"rgrg": 1.0, "grgr": 1.0}
    state = StateVector.from_state_amplitudes(
        eigenstates=basis, amplitudes=strings, gpu=device == "cuda"
    )
    config = SVConfig()

    omegas = torch.randn(num_qubits, dtype=torch.float64).to(
        device=device, dtype=torch.complex128
    )
    deltas = torch.randn(num_qubits, dtype=torch.float64).to(
        device=device, dtype=torch.complex128
    )
    phis = torch.zeros_like(omegas)
    interaction_matrix = torch.randn((num_qubits, num_qubits))
    h_rydberg = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interaction_matrix,
        device=device,
    )

    energy_mock = MagicMock(spec=EnergyVariance)
    energy_variance_mock = energy_mock.return_value

    energy_variance = energy_variance_sv_impl(
        energy_variance_mock, config=config, state=state, hamiltonian=h_rydberg
    )
    expected_varaince = 3.67378968943955

    assert energy_variance.cpu() == approx(expected_varaince, abs=4e-7)

    second_moment_energy_mock = MagicMock(spec=EnergySecondMoment)
    second_moment_mock = second_moment_energy_mock.return_value

    second_moment = energy_second_moment_sv_impl(
        second_moment_mock, config=config, state=state, hamiltonian=h_rydberg
    )
    expected_second = 4.2188228611101

    assert second_moment.cpu() == approx(expected_second, abs=2e-7)
