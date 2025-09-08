import pytest
import torch

from pytest import approx
from unittest.mock import MagicMock
import pulser
from emu_base import DEVICE_COUNT

from emu_sv.hamiltonian import RydbergHamiltonian
from emu_sv.lindblad_operator import RydbergLindbladian
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
    energy_second_moment_den_mat_impl,
    energy_variance_sv_impl,
    energy_variance_sv_den_mat_impl,
    qubit_occupation_sv_impl,
    qubit_occupation_sv_den_mat_impl,
)
import pulser.noise_model

device = "cuda" if DEVICE_COUNT > 0 else "cpu"  # some observables are on cpu always
gpu = False if device == "cpu" else True

dtype_fl = torch.float64
dtype = torch.complex128


@pytest.mark.parametrize(
    "noise",
    [
        True,
        False,
    ],
)
def test_custom_occupation(noise: bool) -> None:
    # set up for state
    basis = ("r", "g")
    num_qubits = 10
    strings = {"r" * num_qubits: 1.0, "g" * num_qubits: 1.0}
    state = StateVector.from_state_amplitudes(eigenstates=basis, amplitudes=strings)
    operator_mock = MagicMock(spec=DenseOperator)
    H_mock = operator_mock.return_value
    MockOccupation = MagicMock(spec=Occupation)
    occupation_mock = MockOccupation.return_value
    config = SVConfig(gpu=gpu)
    expected = torch.tensor([0.5] * num_qubits, dtype=dtype_fl, device="cpu")
    if not noise:
        occupation = qubit_occupation_sv_impl(
            occupation_mock, config=config, state=state, hamiltonian=H_mock
        )

        assert torch.allclose(occupation, expected)
    else:
        state = DensityMatrix.from_state_vector(state)
        occupation = qubit_occupation_sv_den_mat_impl(
            occupation_mock, config=config, state=state, hamiltonian=H_mock
        )

        assert torch.allclose(occupation, expected)


@pytest.mark.parametrize(
    "noise",
    [
        True,
        False,
    ],
)
def test_custom_correlation(noise: bool) -> None:
    # set up for state
    basis = ("r", "g")
    num_qubits = 6
    strings = {"rg" * int(num_qubits / 2): 1.0, "gr" * int(num_qubits / 2): 1.0}
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
    for qubiti in range(num_qubits):
        for qubitj in range(num_qubits):
            if (qubiti + qubitj) % 2 == 0:
                expected[qubiti, qubitj] = 0.5

    assert torch.allclose(expected, correlation)


@pytest.mark.parametrize(
    "noise",
    [
        True,
        False,
    ],
)
def test_custom_energy_and_variance_and_second(noise) -> None:

    torch.manual_seed(1337)

    basis = ("r", "g")
    num_qubits = 4  # only even number for this test
    strings = {"rg" * int(num_qubits / 2): 1.0, "gr" * int(num_qubits / 2): 1.0}
    state = StateVector.from_state_amplitudes(eigenstates=basis, amplitudes=strings)
    config = SVConfig(gpu=gpu)

    omegas = torch.randn(num_qubits, dtype=dtype_fl).to(device=device, dtype=dtype)
    deltas = torch.randn(num_qubits, dtype=dtype_fl).to(device=device, dtype=dtype)
    phis = torch.zeros_like(omegas)
    interaction_matrix = torch.randn((num_qubits, num_qubits))

    energy_mock = MagicMock(spec=EnergyVariance)
    energy_variance_mock = energy_mock.return_value

    second_moment_energy_mock = MagicMock(spec=EnergySecondMoment)
    second_moment_mock = second_moment_energy_mock.return_value
    expected_variance = 3.67378968943955
    expected_second = 4.2188228611101
    if not noise:
        hamiltonian = RydbergHamiltonian(
            omegas=omegas,
            deltas=deltas,
            phis=phis,
            interaction_matrix=interaction_matrix,
            device=device,
        )
        energy_variance = energy_variance_sv_impl(
            energy_variance_mock, config=config, state=state, hamiltonian=hamiltonian
        )
        second_moment = energy_second_moment_sv_impl(
            second_moment_mock, config=config, state=state, hamiltonian=hamiltonian
        )
        assert energy_variance.cpu() == approx(expected_variance, abs=4e-7)
        assert second_moment.cpu() == approx(expected_second, abs=2e-7)

    if noise:
        state = DensityMatrix.from_state_vector(state)

        pulser_linblads = pulser.noise_model.NoiseModel(
            depolarizing_rate=0.1,
        )
        hamiltonian = RydbergLindbladian(
            omegas=omegas,
            deltas=deltas,
            phis=phis,
            pulser_linblads=pulser_linblads,
            interaction_matrix=interaction_matrix,
            device=device,
        )
        energy_variance = energy_variance_sv_den_mat_impl(
            energy_variance_mock, config=config, state=state, hamiltonian=hamiltonian
        )

        second_moment = energy_second_moment_den_mat_impl(
            second_moment_mock, config=config, state=state, hamiltonian=hamiltonian
        )

        assert energy_variance.cpu() == approx(expected_variance, abs=4e-7)
        assert second_moment.cpu() == approx(expected_second, abs=2e-7)
