from emu_mps.mps_backend_impl import (
    MPSBackendImpl,
    NoisyMPSBackendImpl,
    DMRGBackendImpl,
    SwipeDirection,
)
from emu_mps.mps_backend_impl import (
    permute_atom_order,
    permute_bitstrings,
    permute_occupations_and_correlations,
)
from emu_mps.mps_config import MPSConfig
from pulser import NoiseModel
import math
import cmath
from unittest.mock import MagicMock, patch
from emu_mps.mps import MPS
import torch
import pytest
from collections import Counter


_ATOL = 1e-10


QUBIT_COUNT = 5

dtype = torch.complex128


def _create_victim(constructor, dt, noise_model):
    config = MPSConfig(
        dt=dt,
        noise_model=noise_model,
        # no optimisation for Mock; full_interaction_matrix doesn't exist
        optimize_qubit_ordering=False,
    )
    mock_pulser_data = MagicMock()
    mock_pulser_data.qubit_count = QUBIT_COUNT
    mock_pulser_data.qubit_ids = tuple([i for i in range(QUBIT_COUNT)])
    mock_pulser_data.full_interaction_matrix = torch.eye(QUBIT_COUNT)
    mock_pulser_data.masked_interaction_matrix = torch.eye(QUBIT_COUNT)
    mock_pulser_data.slm_end_time = 10.0
    mock_pulser_data.dim = 2
    mock_pulser_data.noise_model = noise_model
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


def create_dmrg_mock(constructor=DMRGBackendImpl, dt=10):
    config = MPSConfig(dt=dt, optimize_qubit_ordering=False)
    mock_pulser_data = MagicMock()
    mock_pulser_data.qubit_count = QUBIT_COUNT
    mock_pulser_data.qubit_ids = tuple([i for i in range(QUBIT_COUNT)])
    mock_pulser_data.full_interaction_matrix = torch.eye(QUBIT_COUNT)
    mock_pulser_data.masked_interaction_matrix = torch.eye(QUBIT_COUNT)
    mock_pulser_data.has_lindblad_noise = False
    mock_pulser_data.slm_end_time = 10.0
    mock_pulser_data.eigenstates = ("g", "r")
    mock_pulser_data.noise_model = NoiseModel()

    dmrg_obj = constructor(config, mock_pulser_data)

    assert dmrg_obj.qubit_count == QUBIT_COUNT
    assert dmrg_obj.current_time == 0.0
    return dmrg_obj


def test_init_dark_qubits_without_state_prep_error():
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


def test_init_dark_qubits_with_state_prep_error():
    noise_model = MagicMock(spec=NoiseModel)
    noise_model.runs = 1
    noise_model.samples_per_run = 1
    noise_model.noise_types = []
    noise_model.state_prep_error = 0.123
    victim = create_victim(noise_model=noise_model)

    victim.pulser_data.bad_atoms = {
        "q0": False,
        "q1": True,
        "q2": False,
        "q3": False,
        "q4": True,
    }

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

    assert torch.equal(
        victim.well_prepared_qubits_filter,
        torch.tensor([True, False, True, True, False]),
    )

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
    lindbladian1 = torch.tensor([[0, 1], [2, 3j]], dtype=dtype)
    lindbladian2 = torch.tensor([[4j, 5j], [6, 7]], dtype=dtype)
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
                    [4.0, 6.0j],
                    [-6.0j, 10.0],
                ],
                [[52.0, 62.0], [62.0, 74.0]],
            ],
            dtype=dtype,
        ),
    )

    compute_noise_from_lindbladians_mock.assert_called_with(
        [lindbladian1, lindbladian2], 2
    )
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


def test_init_initial_state_default():

    noise_model = MagicMock(spec=NoiseModel)
    noise_model.noise_types = []
    noise_model.runs = 1
    noise_model.samples_per_run = 1
    noise_model.state_prep_error = 0.1

    victim = create_victim(noise_model=noise_model)
    victim.pulser_data.bad_atoms = {
        "q0": False,
        "q1": True,
        "q2": True,
        "q3": False,
        "q4": False,
    }
    victim.eigenstates = ("g", "r")

    victim.config = victim.config.with_changes(precision=0.001, max_bond_dim=100)
    victim.resolved_num_gpus = 0
    victim.init_dark_qubits()
    victim.init_initial_state()

    expected = MPS.make(3, num_gpus_to_use=0)

    assert len(victim.state.factors) == 3
    assert all(
        torch.allclose(factor1, factor2)
        for factor1, factor2 in zip(expected.factors, victim.state.factors)
    )
    assert victim.state.precision == 0.001
    assert victim.state.max_bond_dim == 100


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

    up = torch.tensor([[[0], [1]]], dtype=dtype)
    down = torch.tensor([[[1], [0]]], dtype=dtype)
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


@pytest.mark.parametrize(
    "list",
    [
        False,
        True,
    ],
)
def test_permute_results(list: bool) -> None:
    obs = ["bitstrings", "occupation", "correlation_matrix"]
    # I use list of for obs to mimic [obs_t0 and obs_t1]
    bitstrings = [Counter({"110": 1, "010": 2}), Counter({"110": 3, "010": 4})]
    occup = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5, 0.6])]
    corr = [torch.outer(occup[0], occup[0]), torch.outer(occup[1], occup[1])]

    if list:
        occup = [x.tolist() for x in occup]
        corr = [x.tolist() for x in corr]

    mock_results = MagicMock()
    mock_results.atom_order = ("q0", "q1", "q2")
    mock_results._results = [bitstrings, occup, corr]
    mock_results.get_result_tags.return_value = set(obs)
    mock_results._find_uuid.side_effect = lambda key: obs.index(key)

    assert mock_results.get_result_tags() == set(obs)
    assert mock_results._find_uuid("bitstrings") == 0
    assert mock_results._find_uuid("occupation") == 1
    assert mock_results._find_uuid("correlation_matrix") == 2

    perm = torch.tensor([2, 0, 1])  # makes "abc" to "cab"

    permute_atom_order(mock_results, perm)
    assert mock_results.atom_order == ("q2", "q0", "q1")

    permute_bitstrings(mock_results, perm)
    expected = [Counter({"001": 2, "011": 1}), Counter({"001": 4, "011": 3})]
    assert mock_results._results[0] == expected

    permute_occupations_and_correlations(mock_results, perm)
    expected_occup = [torch.tensor([0.3, 0.1, 0.2]), torch.tensor([0.6, 0.4, 0.5])]
    assert all(
        torch.equal(actual, expe)
        for actual, expe in zip(mock_results._results[1], expected_occup)
    )

    expected_corr = [
        torch.outer(expected_occup[0], expected_occup[0]),
        torch.outer(expected_occup[1], expected_occup[1]),
    ]
    assert all(
        torch.equal(actual, expe)
        for actual, expe in zip(mock_results._results[2], expected_corr)
    )


@patch("emu_mps.mps_backend_impl.new_left_bath")
@patch("emu_mps.mps_backend_impl.minimize_energy_pair")
@patch("emu_mps.mps_backend_impl.update_H")
@patch("emu_mps.mps_backend_impl.make_H")
@patch("emu_mps.mps_backend_impl.right_baths")
def test_progress_at_random_middle_mpssite(
    mock_right_baths,
    mock_make_H,
    mock_update_H,
    mock_minimize,
    mock_new_left_bath,
):
    mock_make_H.return_value = MagicMock(factors=[None] * QUBIT_COUNT)
    mock_update_H.return_value = None
    mock_right_baths.return_value = [torch.zeros(1, 1, 1)] * (QUBIT_COUNT - 1)
    mock_new_left_bath.return_value = torch.zeros(1)

    dmrg = create_dmrg_mock()
    dmrg.timestep_index = 0
    dmrg.timestep_count = 2

    dmrg.hamiltonian = MagicMock(factors=[None] * QUBIT_COUNT)
    dmrg.state = MagicMock(factors=[None] * QUBIT_COUNT, orthogonality_center=1)
    dmrg.sweep_index = dmrg.state.orthogonality_center

    # at the 2nd MPS site (orthogonality_center = 1), the left bath must be a
    # list of 1 element
    # while the right bath being a list of 3 elements
    dmrg.left_baths = [torch.zeros(1)]
    dmrg.right_baths = [torch.zeros(1)] * 3

    new_left_factor = torch.tensor([[1.0]])
    new_right_factor = torch.tensor([[2.0]])
    current_energy = 2.3
    mock_minimize.return_value = (new_left_factor, new_right_factor, current_energy)

    dmrg.progress()

    mock_minimize.assert_called_once()
    assert dmrg.current_energy == current_energy
    assert dmrg.sweep_index == 2
    assert dmrg.state.orthogonality_center == 2
    assert len(dmrg.left_baths) == 2
    assert len(dmrg.right_baths) == 2
    assert dmrg.swipe_direction == SwipeDirection.LEFT_TO_RIGHT


@patch("emu_mps.mps_backend_impl.new_left_bath")
@patch("emu_mps.mps_backend_impl.minimize_energy_pair")
@patch("emu_mps.mps_backend_impl.update_H")
@patch("emu_mps.mps_backend_impl.make_H")
@patch("emu_mps.mps_backend_impl.right_baths")
def test_progress_at_right_mps_boundary(
    mock_right_baths, mock_make_H, mock_update_H, mock_minimize, mock_new_left_bath
):
    mock_make_H.return_value = MagicMock(factors=[None] * QUBIT_COUNT)
    mock_update_H.return_value = None
    mock_right_baths.return_value = [torch.zeros(1, 1, 1)] * (QUBIT_COUNT - 1)
    mock_new_left_bath.return_value = torch.zeros(1)

    dmrg = create_dmrg_mock()
    dmrg.timestep_index = 0
    dmrg.timestep_count = 2

    dmrg.state = MagicMock(factors=[None] * QUBIT_COUNT, orthogonality_center=2)
    dmrg.sweep_index = dmrg.state.orthogonality_center

    dmrg.hamiltonian = MagicMock(factors=[None] * QUBIT_COUNT)
    # at the 3rd MPS site (orthogonality_center = 2), the left bath must be a
    # list of 2 elements
    # while the right bath being a list of two elements
    dmrg.left_baths = [torch.zeros(1)] * 2
    dmrg.right_baths = [torch.zeros(1)] * 2

    new_left_factor = torch.tensor([[1.0]])
    new_right_factor = torch.tensor([[2.0]])
    current_energy = 0.5
    mock_minimize.return_value = (new_left_factor, new_right_factor, current_energy)

    dmrg.progress()

    mock_minimize.assert_called_once()
    assert dmrg.current_energy == pytest.approx(0.5)
    assert dmrg.sweep_index == 3
    assert dmrg.state.orthogonality_center == 3
    assert len(dmrg.left_baths) == 3
    assert len(dmrg.right_baths) == 1
    assert dmrg.swipe_direction == SwipeDirection.RIGHT_TO_LEFT


@patch("emu_mps.mps_backend_impl.new_left_bath")
@patch("emu_mps.mps_backend_impl.minimize_energy_pair")
@patch("emu_mps.mps_backend_impl.update_H")
@patch("emu_mps.mps_backend_impl.make_H")
@patch("emu_mps.mps_backend_impl.right_baths")
def test_left_to_right_update(
    mock_right_baths, mock_make_H, mock_update_H, mock_minimize, mock_new_left
):
    mock_make_H.return_value = MagicMock(factors=[None] * QUBIT_COUNT)
    mock_update_H.return_value = None
    mock_right_baths.return_value = [torch.zeros(1)] * (QUBIT_COUNT - 1)
    mock_new_left.return_value = torch.zeros(1)
    mock_minimize.return_value = (torch.tensor([[1.0]]), torch.tensor([[2.0]]), 0.5)

    dmrg = create_dmrg_mock()
    dmrg.init()
    dmrg.sweep_index = 1
    dmrg.swipe_direction = SwipeDirection.LEFT_TO_RIGHT
    dmrg.left_baths = [torch.zeros(1)]
    dmrg.right_baths = [torch.zeros(1)] * 3

    dmrg._left_to_right_update(idx=1)

    assert dmrg.sweep_index == 2
    assert len(dmrg.left_baths) == 2
    assert len(dmrg.right_baths) == 2
    dmrg.swipe_direction = SwipeDirection.LEFT_TO_RIGHT


@patch("emu_mps.mps_backend_impl.new_left_bath")
@patch("emu_mps.mps_backend_impl.minimize_energy_pair")
@patch("emu_mps.mps_backend_impl.update_H")
@patch("emu_mps.mps_backend_impl.make_H")
@patch("emu_mps.mps_backend_impl.right_baths")
def test_right_to_left_update(
    mock_right_baths, mock_make_H, mock_update_H, mock_minimize, mock_new_left
):
    mock_make_H.return_value = MagicMock(
        factors=[torch.ones(1, 1, 1, 1, dtype=dtype) for i in range(QUBIT_COUNT)]
    )

    mock_update_H.return_value = None
    mock_right_baths.return_value = [torch.zeros(1, dtype=dtype)] * (QUBIT_COUNT - 1)
    mock_new_left.return_value = torch.zeros(1, 1, 1, dtype=dtype)

    mock_minimize.return_value = (
        torch.zeros(1, 1, dtype=dtype),
        torch.zeros(1, 1, dtype=dtype),
        0.5,
    )

    dmrg = create_dmrg_mock()
    dmrg.init()
    dmrg.sweep_index = 2
    dmrg.swipe_direction = SwipeDirection.RIGHT_TO_LEFT
    dmrg.left_baths = [
        torch.zeros(1, 1, 1, dtype=dtype, device=dmrg.state.factors[0].device)
    ] * 2
    dmrg.right_baths = [
        torch.zeros(1, 1, 1, dtype=dtype, device=dmrg.state.factors[0].device)
    ]

    dmrg._right_to_left_update(idx=1)

    assert len(dmrg.left_baths) == 1
    assert len(dmrg.right_baths) == 2
    assert dmrg.sweep_index == 1
    assert dmrg.swipe_direction == SwipeDirection.RIGHT_TO_LEFT
