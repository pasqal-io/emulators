from emu_mps.mps_backend_impl import MPSBackendImpl, NoisyMPSBackendImpl
from emu_mps.mps_config import MPSConfig
from pulser import NoiseModel
import math
import cmath
from unittest.mock import MagicMock, patch
from emu_mps.mps import MPS
import torch
import pytest


_ATOL = 1e-10


QUBIT_COUNT = 5


def _create_victim(constructor, dt, noise_model):
    config = MPSConfig(dt=dt, noise_model=noise_model)
    mock_pulser_data = MagicMock()
    mock_pulser_data.qubit_count = QUBIT_COUNT
    mock_pulser_data.slm_end_time = 10.0
    victim = constructor(config, mock_pulser_data)

    assert victim.qubit_count == QUBIT_COUNT
    assert victim.current_time == 0.0

    return victim


def create_victim(dt=10, noise_model=None):
    if noise_model is None:
        noise_model = NoiseModel()
    victim = _create_victim(constructor=MPSBackendImpl, dt=dt, noise_model=noise_model)
    victim.has_lindblad_noise = False
    return victim


def create_noisy_victim(dt=10, noise_model=None):
    if noise_model is None:
        noise_model = NoiseModel()
    victim = _create_victim(
        constructor=NoisyMPSBackendImpl, dt=dt, noise_model=noise_model
    )
    victim.has_lindblad_noise = True
    return victim


@patch("emu_mps.mps_backend_impl.pick_well_prepared_qubits")
def test_init_dark_qubits_without_state_prep_error(pick_well_prepared_qubits_mock):
    noise_model = MagicMock(spec=NoiseModel)
    noise_model.runs = 1
    noise_model.samples_per_run = 1
    noise_model.noise_types = []
    noise_model.state_prep_error = 0.0
    victim = create_victim(noise_model=noise_model)

    victim.full_interaction_matrix = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
            [30, 31, 32, 33, 34],
            [40, 41, 42, 43, 44],
        ]
    )
    victim.masked_interaction_matrix = victim.full_interaction_matrix
    victim.omega = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        ]
    )

    victim.init_dark_qubits()
    pick_well_prepared_qubits_mock.assert_not_called()

    assert victim.well_prepared_qubits_filter is None

    assert torch.allclose(
        victim.masked_interaction_matrix,
        torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [10, 11, 12, 13, 14],
                [20, 21, 22, 23, 24],
                [30, 31, 32, 33, 34],
                [40, 41, 42, 43, 44],
            ]
        ),
    )
    assert torch.allclose(
        victim.masked_interaction_matrix, victim.full_interaction_matrix
    )

    assert torch.allclose(
        victim.omega,
        torch.tensor(
            [
                [0, 1, 2, 3, 4],
                [0, 1, 2, 3, 4],
            ]
        ),
    )


@patch("emu_mps.mps_backend_impl.pick_well_prepared_qubits")
def test_init_dark_qubits_with_state_prep_error(pick_well_prepared_qubits_mock):
    noise_model = MagicMock(spec=NoiseModel)
    noise_model.runs = 1
    noise_model.samples_per_run = 1
    noise_model.noise_types = []
    noise_model.state_prep_error = 0.123
    victim = create_victim(noise_model=noise_model)

    pick_well_prepared_qubits_mock.return_value = [True, False, True, True, False]

    victim.full_interaction_matrix = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [10, 11, 12, 13, 14],
            [20, 21, 22, 23, 24],
            [30, 31, 32, 33, 34],
            [40, 41, 42, 43, 44],
        ]
    )
    victim.masked_interaction_matrix = victim.full_interaction_matrix
    victim.omega = torch.tensor(
        [
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
        ]
    )
    victim.delta = torch.tensor(
        [
            [10, 11, 12, 13, 14],
            [10, 11, 12, 13, 14],
        ]
    )
    victim.phi = torch.tensor(
        [
            [20, 21, 22, 23, 24],
            [20, 21, 22, 23, 24],
        ]
    )

    victim.init_dark_qubits()
    pick_well_prepared_qubits_mock.assert_called_once_with(0.123, QUBIT_COUNT)

    assert victim.well_prepared_qubits_filter == [True, False, True, True, False]

    assert torch.allclose(
        victim.masked_interaction_matrix,
        torch.tensor(
            [
                [0, 2, 3],
                [20, 22, 23],
                [30, 32, 33],
            ]
        ),
    )
    assert torch.allclose(
        victim.masked_interaction_matrix, victim.full_interaction_matrix
    )

    assert torch.allclose(
        victim.omega,
        torch.tensor(
            [
                [0, 2, 3],
                [0, 2, 3],
            ]
        ),
    )
    assert torch.allclose(
        victim.delta,
        torch.tensor(
            [
                [10, 12, 13],
                [10, 12, 13],
            ]
        ),
    )
    assert torch.allclose(
        victim.phi,
        torch.tensor(
            [
                [20, 22, 23],
                [20, 22, 23],
            ]
        ),
    )


@patch("emu_mps.mps_backend_impl.compute_noise_from_lindbladians")
def test_init_lindblad_noise_with_lindbladians(compute_noise_from_lindbladians_mock):
    victim = create_noisy_victim()
    lindbladian1 = torch.tensor([[0, 1], [2, 3j]], dtype=torch.complex128)
    lindbladian2 = torch.tensor([[4j, 5j], [6, 7]], dtype=torch.complex128)
    victim.lindblad_ops = [lindbladian1, lindbladian2]
    victim.has_lindblad_noise = True

    noise_mock = MagicMock(spec=NoiseModel)
    noise_mock.noise_types = []
    noise_mock.runs = 1
    noise_mock.samples_per_run = 1
    compute_noise_from_lindbladians_mock.return_value = noise_mock
    victim.init_lindblad_noise()
    assert torch.allclose(
        victim.aggregated_lindblad_ops,
        torch.tensor(
            [
                [
                    [4, 6j],
                    [-6j, 10],
                ],
                [[52, 62], [62, 74]],
            ],
            dtype=torch.complex128,
        ),
    )

    compute_noise_from_lindbladians_mock.assert_called_with([lindbladian1, lindbladian2])
    assert victim.lindblad_noise is noise_mock


@patch("emu_mps.mps_backend_impl.random.uniform")
def test_set_jump_threshold(random_mock):
    victim = create_noisy_victim()
    victim.state = MPS.make(QUBIT_COUNT)
    random_mock.return_value = 0.123
    victim.set_jump_threshold(1.0)
    random_mock.assert_called_once()
    assert victim.jump_threshold == 0.123
    assert math.isclose(victim.norm_gap_before_jump, 0.877)


@patch("emu_mps.mps_backend_impl.pick_well_prepared_qubits")
def test_init_initial_state_default(pick_well_prepared_qubits_mock):
    pick_well_prepared_qubits_mock.return_value = [True, False, False, True, True]

    noise_model = MagicMock(spec=NoiseModel)
    noise_model.noise_types = []
    noise_model.runs = 1
    noise_model.samples_per_run = 1
    noise_model.state_prep_error = 0.1

    victim = create_victim(noise_model=noise_model)

    victim.config.precision = 0.001
    victim.config.max_bond_dim = 100
    victim.init_dark_qubits()
    pick_well_prepared_qubits_mock.assert_called_once_with(0.1, QUBIT_COUNT)
    victim.init_initial_state()

    expected = MPS.make(3)

    assert len(victim.state.factors) == 3
    assert all(
        torch.allclose(factor1, factor2)
        for factor1, factor2 in zip(expected.factors, victim.state.factors)
    )
    assert victim.state.config.precision == 0.001
    assert victim.state.config.max_bond_dim == 100


def test_init_initial_state_provided_filter():
    victim = create_victim()

    victim.well_prepared_qubits_filter = [True, True, False, True, True]

    with pytest.raises(NotImplementedError) as e:
        victim.init_initial_state(MagicMock())

    assert (
        str(e.value) == "Specifying the initial state in the presence "
        "of state preparation errors is currently not implemented."
    )


def test_init_initial_state_provided_normalized():
    victim = create_victim()

    victim.well_prepared_qubits_filter = None

    up = torch.tensor([[[0], [1]]], dtype=torch.complex128)
    down = torch.tensor([[[1], [0]]], dtype=torch.complex128)
    victim.init_initial_state(
        0.123
        * MPS(
            [up, up, down, up, down],
            eigenstates=("0", "1"),
        )
    )
    assert cmath.isclose(
        victim.state.inner(
            MPS(
                [up, up, down, up, down],
                eigenstates=("0", "1"),
            )
        ),
        1,
    )


@patch("emu_mps.mps_backend_impl.make_H")
@patch("emu_mps.mps_backend_impl.update_H")
def test_init_hamiltonian(update_H_mock, make_H_mock):
    victim = create_victim()
    victim.init_hamiltonian()
    assert make_H_mock.call_count == 1
    assert update_H_mock.call_count == 1
