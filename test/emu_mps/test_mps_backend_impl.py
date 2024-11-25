from emu_mps.mps_backend_impl import MPSBackendImpl
from emu_mps.mps_config import MPSConfig
import math
import cmath
from unittest.mock import MagicMock, patch, ANY, call
from emu_mps.mps import MPS
import torch
import pytest


_ATOL = 1e-10


QUBIT_COUNT = 5


def create_victim(dt=10, noise_model=None):
    config = MPSConfig(dt=dt, noise_model=noise_model)
    sequence = MagicMock()
    sequence.register.qubit_ids = ["whatever"] * QUBIT_COUNT
    mock_pulser_data = MagicMock()
    mock_pulser_data.slm_end_time = 10.0
    with patch("emu_mps.mps_backend_impl.PulserData.__new__") as mock_new:
        mock_new.return_value = mock_pulser_data
        victim = MPSBackendImpl(sequence, config)

    assert victim.qubit_count == QUBIT_COUNT
    assert victim.current_time == 0.0

    return victim


@patch("emu_mps.mps_backend_impl.pick_well_prepared_qubits")
def test_init_dark_qubits_without_state_prep_error(pick_well_prepared_qubits_mock):
    noise_model = MagicMock()
    noise_model.depolarizing_rate = 0.123
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
    noise_model = MagicMock()
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
@patch("emu_mps.mps_backend_impl.random.random")
def test_init_lindblad_noise_without_lindbladians(
    random_mock, compute_noise_from_lindbladians_mock
):
    victim = create_victim()
    victim.state = MPS.make(QUBIT_COUNT)
    victim.lindblad_ops = []

    random_mock.return_value = 0.123
    victim.init_lindblad_noise()
    assert not victim.has_lindblad_noise
    assert victim.aggregated_lindblad_ops is None

    compute_noise_from_lindbladians_mock.assert_called_with([])
    random_mock.assert_called_once()
    assert victim.jump_threshold == 0.123
    assert math.isclose(victim.norm_gap_before_jump, 0.877)


@patch("emu_mps.mps_backend_impl.compute_noise_from_lindbladians")
@patch("emu_mps.mps_backend_impl.random.random")
def test_init_lindblad_noise_with_lindbladians(
    random_mock, compute_noise_from_lindbladians_mock
):
    victim = create_victim()
    victim.state = MPS.make(QUBIT_COUNT)
    lindbladian1 = torch.tensor([[0, 1], [2, 3j]], dtype=torch.complex128)
    lindbladian2 = torch.tensor([[4j, 5j], [6, 7]], dtype=torch.complex128)
    victim.lindblad_ops = [lindbladian1, lindbladian2]

    noise_mock = MagicMock()
    compute_noise_from_lindbladians_mock.return_value = noise_mock
    random_mock.return_value = 0.123
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
    random_mock.assert_called_once()
    assert victim.jump_threshold == 0.123
    assert math.isclose(victim.norm_gap_before_jump, 0.877)


def test_init_initial_state_default():
    victim = create_victim()

    victim.well_prepared_qubits_filter = [True, False, False, True, True]
    victim.config.precision = 0.001
    victim.config.max_bond_dim = 100
    victim.config.num_gpus_to_use = 0
    victim.init_initial_state(None)

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

    up = torch.tensor([[[0], [1]]], dtype=torch.complex128)
    down = torch.tensor([[[1], [0]]], dtype=torch.complex128)
    victim.init_initial_state(MPS([up, up, down, up, down]))
    assert cmath.isclose(victim.state.inner(MPS([up, up, down, up, down])), 1)


@patch("emu_mps.mps_backend_impl.make_H")
def test_init_hamiltonian(make_H_mock):
    victim = create_victim()
    victim.init_hamiltonian()
    assert make_H_mock.call_count == 1


@patch("emu_mps.mps_backend_impl.evolve_tdvp")
@patch("emu_mps.mps_backend_impl.make_H")
@patch("emu_mps.mps_backend_impl.update_H")
def test_do_time_step_without_noise(update_H_mock, make_H_mock, evolve_tdvp_mock):
    victim = create_victim()

    victim.jump_threshold = 0.8
    victim.norm_gap_before_jump = 0.2
    victim.has_lindblad_noise = False
    victim.lindblad_noise = torch.zeros(2, 2, dtype=torch.complex128)
    victim.state = MPS.make(5)
    victim.init_hamiltonian()

    def evolve_tdvp_mock_side_effect(
        t, state, hamiltonian, extra_krylov_tolerance, max_krylov_dim, is_hermitian
    ):
        assert state.orthogonality_center is not None
        # Drastic norm decrease.
        state.factors[state.orthogonality_center] = (
            0.6 * state.factors[state.orthogonality_center]
        )

    evolve_tdvp_mock.side_effect = evolve_tdvp_mock_side_effect
    victim.do_time_step(0)  # No quantum jump attempted in the absence of noise.

    assert evolve_tdvp_mock.call_count == 1
    assert victim.is_masked is True
    assert math.isclose(victim.state.norm(), 0.6)
    assert math.isclose(victim.norm_gap_before_jump, -0.44)
    assert update_H_mock.call_count == 1
    assert make_H_mock.call_count == 1

    victim.do_time_step(1)  # No quantum jump attempted in the absence of noise.
    assert update_H_mock.call_count == 2
    assert make_H_mock.call_count == 2
    assert victim.is_masked is False


@patch("emu_mps.mps_backend_impl.evolve_tdvp")
@patch("emu_mps.mps_backend_impl.make_H")
@patch("emu_mps.mps_backend_impl.update_H")
@patch("emu_mps.mps_backend_impl.find_root_brents")
@patch("emu_base.pulser_adapter._get_all_lindblad_noise_operators")
@patch("emu_mps.mps_backend_impl.MPSBackendImpl.do_random_quantum_jump")
def test_do_time_step_with_noise(
    do_random_quantum_jump_mock,
    get_all_lindblad_noise_operators_mock,
    find_root_brents_mock,
    update_H_mock,
    make_H_mock,
    evolve_tdvp_mock,
):
    victim = create_victim(dt=12)
    victim.init_hamiltonian()

    test_jump_threshold = 0.8
    victim.jump_threshold = test_jump_threshold
    victim.norm_gap_before_jump = 0.2
    victim.has_lindblad_noise = False
    victim.lindblad_noise = torch.ones(2, 2, dtype=torch.complex128)
    victim.slm_end_time = 100
    victim.state = MPS.make(5)
    victim.has_lindblad_noise = True

    # Test data for how much the state's norm decreases over time.
    norms_per_time = {
        0: 1,
        3: 0.9,
        6: test_jump_threshold,  # t=6 is where the collapse should happen.
        11: 0.7,
        12: 0.6,  # dt, i.e. the target time of do_time_step
    }

    def find_root_brents_mock_side_effect(f, start, end, f_start, f_end, tolerance):
        assert start == 0
        assert end == 12
        assert math.isclose(f_start, 1**2 - test_jump_threshold)
        assert math.isclose(f_end, norms_per_time[end] ** 2 - test_jump_threshold)

        # Root finding iterations
        f(3)
        f(11)
        f(6)
        return 6  # norms_per_time[6] is the given collapse_threshold.

    find_root_brents_mock.side_effect = find_root_brents_mock_side_effect

    def evolve_tdvp_mock_side_effect(
        t, state, hamiltonian, extra_krylov_tolerance, max_krylov_dim, is_hermitian
    ):
        delta_t = t
        assert state.orthogonality_center is not None
        assert delta_t.real == 0
        target_t = victim.current_time - delta_t.imag * 1000
        assert 3 - _ATOL <= target_t <= 12 + _ATOL
        assert round(target_t) in norms_per_time
        assert math.isclose(state.norm(), norms_per_time[int(victim.current_time)])
        state.factors[state.orthogonality_center] = (
            norms_per_time[round(target_t)]
            * MPS.make(QUBIT_COUNT).factors[state.orthogonality_center]
        )

    evolve_tdvp_mock.side_effect = evolve_tdvp_mock_side_effect

    def do_random_quantum_jump_side_effect():
        victim.jump_threshold = 0.2  # Reset after jump at t=6.
        victim.norm_gap_before_jump = 0.8
        # The state should also be normalized here but we don't need it for this test.

    do_random_quantum_jump_mock.side_effect = do_random_quantum_jump_side_effect
    victim.do_time_step(0)
    do_random_quantum_jump_mock.assert_called_once()
    find_root_brents_mock.assert_called_once()
    evolve_tdvp_mock.assert_has_calls(
        [
            # Initial call: t=0 -> t=12
            call(
                t=pytest.approx(-0.012j, abs=_ATOL),
                state=ANY,
                hamiltonian=ANY,
                extra_krylov_tolerance=ANY,
                max_krylov_dim=ANY,
                is_hermitian=False,
            ),
            # Root finding call: t=12 -> t=3
            call(
                t=pytest.approx(0.009j, abs=_ATOL),
                state=ANY,
                hamiltonian=ANY,
                extra_krylov_tolerance=ANY,
                max_krylov_dim=ANY,
                is_hermitian=False,
            ),
            # Root finding call: t=3 -> t=11
            call(
                t=pytest.approx(-0.008j, abs=_ATOL),
                state=ANY,
                hamiltonian=ANY,
                extra_krylov_tolerance=ANY,
                max_krylov_dim=ANY,
                is_hermitian=False,
            ),
            # Root finding call: t=11 -> t=6
            call(
                t=pytest.approx(0.005j, abs=_ATOL),
                state=ANY,
                hamiltonian=ANY,
                extra_krylov_tolerance=ANY,
                max_krylov_dim=ANY,
                is_hermitian=False,
            ),
            # Post jump: t=6 -> t=dt == 12
            call(
                t=pytest.approx(-0.006j, abs=_ATOL),
                state=ANY,
                hamiltonian=ANY,
                extra_krylov_tolerance=ANY,
                max_krylov_dim=ANY,
                is_hermitian=False,
            ),
        ]
    )
    assert evolve_tdvp_mock.call_count == 5
    assert update_H_mock.call_count == 1
    assert make_H_mock.call_count == 1
