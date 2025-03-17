import torch

from pytest import approx
from unittest.mock import MagicMock

from emu_base import DEVICE_COUNT

from emu_mps.custom_callback_implementations import (
    correlation_matrix_mps_impl,
    energy_variance_mps_impl,
    qubit_occupation_mps_impl,
    energy_second_moment_mps_impl,
    energy_mps_impl
)
from emu_mps.hamiltonian import make_H, update_H, HamiltonianType
from emu_mps import MPSConfig
from emu_mps import MPS, MPO

from pulser.backend.default_observables import (
    CorrelationMatrix,
    EnergySecondMoment,
    EnergyVariance,
    Occupation,
    Energy
)

device = "cuda" if DEVICE_COUNT > 0 else "cpu"


def test_custom_qubit_density() -> None:
    # set up for state
    basis = ("r", "g")
    num_qubits = 4
    strings = {"rrrr": 1.0, "gggg": 1.0}
    state = MPS.from_state_amplitudes(eigenstates=basis, amplitudes=strings)
    config = MPSConfig()
    operator_mock = MagicMock(spec=MPO)

    H_mock = operator_mock.return_value

    MockOccupation = MagicMock(spec=Occupation)

    qubit_density_mock = MockOccupation.return_value

    t = 1

    qubit_density = qubit_occupation_mps_impl(qubit_density_mock, config=config, state=state, H=H_mock)
    expected = [0.5] * num_qubits
    assert qubit_density.cpu() == approx(expected, abs=1e-8)


def test_custom_correlation() -> None:
    # set up for state
    basis = ("r", "g")
    num_qubits = 4
    strings = {"rgrg": 1.0, "grgr": 1.0}
    state = MPS.from_state_amplitudes(eigenstates=basis, amplitudes=strings)
    config = MPSConfig()
    operator_mock = MagicMock(spec=MPO)
    H_mock = operator_mock.return_value
    correlation_matrix_mock = MagicMock(spec=CorrelationMatrix)
    correlation_mock = correlation_matrix_mock.return_value
    t = 1
    correlation = correlation_matrix_mps_impl(correlation_mock, config=config, state=state, H=H_mock)

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
    state = MPS.from_state_amplitudes(
        eigenstates=basis, amplitudes=strings, num_gpus_to_use=DEVICE_COUNT
    )
    config = MPSConfig()

    omegas = torch.randn(num_qubits, dtype=torch.float64).to(torch.complex128)
    deltas = torch.randn(num_qubits, dtype=torch.float64).to(torch.complex128)
    phis = torch.zeros_like(omegas)
    interaction_matrix = torch.randn((num_qubits, num_qubits))
    interaction_matrix = (interaction_matrix + interaction_matrix.T)*0.5
    h_rydberg = make_H(
        interaction_matrix=interaction_matrix, num_gpus_to_use=DEVICE_COUNT, hamiltonian_type=HamiltonianType.Rydberg
    )
    update_H(hamiltonian=h_rydberg, omega=omegas, delta=deltas, phi=phis)

    t = 1

    energy_obj = Energy()
    base_energy = energy_obj.apply(state=state, hamiltonian=h_rydberg).real.to("cpu")
    energy = energy_mps_impl(energy_obj, config=config, state=state, H=h_rydberg)
    assert torch.allclose(base_energy, energy)

    variance_obj = EnergyVariance()
    base_variance = variance_obj.apply(state=state, hamiltonian=h_rydberg).real
    variance = energy_variance_mps_impl(variance_obj, config=config, state=state, H=h_rydberg)
    assert variance.item() == approx(base_variance)
    
    square_obj = EnergySecondMoment()
    base_square = square_obj.apply(state=state, hamiltonian=h_rydberg).real
    square = energy_second_moment_mps_impl(square_obj, config=config, state=state, H=h_rydberg)
    assert square.item() == approx(base_square)