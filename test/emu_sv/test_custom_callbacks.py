import pytest
import torch

from pytest import approx
from unittest.mock import MagicMock

from emu_base import DEVICE_COUNT

from emu_sv.hamiltonian import RydbergHamiltonian
from emu_sv import (
    DenseOperator,
    SVConfig,
    StateVector,
    DensityMatrix,
    CorrelationMatrix,
    EnergySecondMoment,
    EnergyVariance,
    Occupation,
)

from emu_sv.custom_callback_implementations import (
    correlation_matrix_sv_impl,
    correlation_matrix_sv_den_mat_impl,
    energy_second_moment_sv_impl,
    energy_variance_sv_impl,
    qubit_occupation_sv_impl,
    qubit_occupation_sv_den_mat_impl,
)

device = "cuda" if DEVICE_COUNT > 0 else "cpu"  # some observables are on cpu always
gpu = False if device == "cpu" else True

dtype_fl = torch.float64
dtype = torch.complex128


@pytest.fixture(params=[True, False])
def noise(request):
    return request.param


class TestObservables:

    def test_custom_occupation(self, noise: bool) -> None:
        basis = ("r", "g")
        num_qubits = 4
        strings = {"rrrr": 1.0, "gggg": 1.0}

        operator_mock = MagicMock(spec=DenseOperator)
        H_mock = operator_mock.return_value
        MockOccupation = MagicMock(spec=Occupation)
        occupation_mock = MockOccupation.return_value

        config = SVConfig(gpu=gpu)
        state = StateVector.from_state_amplitudes(eigenstates=basis, amplitudes=strings)

        if noise:
            state = DensityMatrix.from_state_vector(state)
            occupation = qubit_occupation_sv_den_mat_impl(
                occupation_mock, config=config, state=state, hamiltonian=H_mock
            )
        else:
            occupation = qubit_occupation_sv_impl(
                occupation_mock, config=config, state=state, hamiltonian=H_mock
            )

        expected = torch.tensor([0.5] * num_qubits, dtype=dtype_fl, device="cpu")
        assert torch.allclose(occupation, expected)

    def test_custom_correlation(self, noise: bool) -> None:
        basis = ("r", "g")
        num_qubits = 4
        strings = {"rgrg": 1.0, "grgr": 1.0}

        state = StateVector.from_state_amplitudes(eigenstates=basis, amplitudes=strings)
        config = SVConfig(gpu=gpu)
        operator_mock = MagicMock(spec=DenseOperator)
        H_mock = operator_mock.return_value
        correlation_matrix_mock = MagicMock(spec=CorrelationMatrix)
        correlation_mock = correlation_matrix_mock.return_value

        if noise:
            state = DensityMatrix.from_state_vector(state)
            correlation = correlation_matrix_sv_den_mat_impl(
                correlation_mock, config=config, state=state, hamiltonian=H_mock
            )
        else:
            correlation = correlation_matrix_sv_impl(
                correlation_mock, config=config, state=state, hamiltonian=H_mock
            )

        expected = torch.zeros(num_qubits, num_qubits, dtype=dtype_fl, device="cpu")
        for i in range(num_qubits):
            for j in range(num_qubits):
                if (i + j) % 2 == 0:
                    expected[i, j] = 0.5

        assert torch.allclose(correlation, expected)


def test_custom_energy_and_variance_and_second() -> None:

    torch.manual_seed(1337)

    basis = ("r", "g")
    num_qubits = 4
    strings = {"rgrg": 1.0, "grgr": 1.0}
    state = StateVector.from_state_amplitudes(eigenstates=basis, amplitudes=strings)
    config = SVConfig(gpu=gpu)

    omegas = torch.randn(num_qubits, dtype=dtype_fl).to(device=device, dtype=dtype)
    deltas = torch.randn(num_qubits, dtype=dtype_fl).to(device=device, dtype=dtype)
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
