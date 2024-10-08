from emu_mps.mps_backend_impl import MPSBackendImpl
from emu_mps.mps_config import MPSConfig
import math
from unittest.mock import MagicMock, patch, ANY, call
from unittest import TestCase
from emu_mps.mps import MPS


_ATOL = 1e-10


class Approximately:
    def __init__(self, expected_value: complex):
        self.expected_value = expected_value

    def __eq__(self, other):
        return abs(self.expected_value - other) < _ATOL


@patch("emu_mps.mps_backend_impl.make_H")
class MPSBackendImplTest(TestCase):
    def _patch(self, to_patch):
        patcher = patch(to_patch)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def create_victim(self, dt=10, noise_model=None):
        self.config = MPSConfig(dt=dt, noise_model=noise_model)
        self.qubit_count = 5
        self.mock_sequence = MagicMock()
        self.mock_sequence.register.qubit_ids = ["whatever"] * self.qubit_count
        adressed_basis = "ground-rydberg"
        self.mock_sequence.get_addressed_bases.return_value = [adressed_basis]

        self.omega, self.delta, self.phi = MagicMock(), MagicMock(), MagicMock()

        self.extract_omega_delta_phi_mock = self._patch(
            "emu_mps.pulser_adapter._extract_omega_delta_phi"
        )
        self.extract_omega_delta_phi_mock.return_value = self.omega, self.delta, self.phi

        self.rydberg_interaction_mock = self._patch(
            "emu_mps.pulser_adapter._rydberg_interaction"
        )

        self.victim = MPSBackendImpl(self.mock_sequence, self.config)
        self.extract_omega_delta_phi_mock.assert_called_once_with(
            self.mock_sequence, ANY, ANY
        )

        self.evolve_tdvp_mock = self._patch("emu_mps.mps_backend_impl.evolve_tdvp")

        assert self.victim.current_time == 0.0

    @patch("emu_mps.mps_backend_impl.pick_well_prepared_qubits")
    @patch("emu_mps.mps_backend_impl.MPSBackendImpl._init_lindblad_noise")
    def test_init_dark_qubits(
        self, _init_lindblad_noise_mock, pick_well_prepared_qubits_mock, make_H_mock
    ):
        noise_model = MagicMock()
        noise_model.noise_types = ("depolarizing", "SPAM", "dephasing")
        noise_model.state_prep_error = 0.1
        noise_model.hyperfine_dephasing_rate = 0.0

        pick_well_prepared_qubits_mock.return_value = [True, False, True, True, False]

        self.create_victim(noise_model=noise_model)

        pick_well_prepared_qubits_mock.assert_called_once()

    def test_do_time_step_without_noise(self, make_H_mock):
        self.create_victim()

        self.victim.collapse_threshold = (
            0.8  # Normally randomly initialized in MPSBackendImpl's constructor.
        )
        assert not self.victim.is_noisy

        def evolve_tdvp_mock_side_effect(
            t, state, hamiltonian, extra_krylov_tolerance, max_krylov_dim, is_hermitian
        ):
            assert state.orthogonality_center is not None
            # Drastic norm decrease.
            state.factors[state.orthogonality_center] = (
                0.6 * state.factors[state.orthogonality_center]
            )

        self.evolve_tdvp_mock.side_effect = evolve_tdvp_mock_side_effect
        self.victim.do_time_step(0)  # No quantum jump attempted in the absence of noise.
        self.evolve_tdvp_mock.assert_called_once()

        assert math.isclose(self.victim.state.norm(), 0.6)

    @patch("emu_mps.mps_backend_impl.find_root_brents")
    @patch("emu_mps.pulser_adapter._get_all_lindblad_noise_operators")
    @patch("emu_mps.mps_backend_impl.compute_noise_from_lindbladians")
    @patch("emu_mps.mps_backend_impl.torch.stack")
    @patch("emu_mps.mps_backend_impl.MPSBackendImpl.random_noise_collapse")
    def test_do_time_step_with_noise(
        self,
        random_noise_collapse_mock,
        torch_stack,
        compute_noise_from_lindbladians_mock,
        get_all_lindblad_noise_operators_mock,
        find_root_brents_mock,
        make_H_mock,
    ):
        self.create_victim(dt=12, noise_model=MagicMock())

        test_collapse_threshold = 0.8
        self.victim.collapse_threshold = test_collapse_threshold
        assert self.victim.is_noisy

        # Test data for how much the state's norm decreases over time.
        norms_per_time = {
            0: 1,
            3: 0.9,
            6: test_collapse_threshold,  # t=6 is where the collapse should happen.
            11: 0.7,
            12: 0.6,  # dt, i.e. the target time of do_time_step
        }

        def find_root_brents_mock_side_effect(f, start, end, f_start, f_end, tolerance):
            assert start == 0
            assert end == 12
            assert math.isclose(f_start, 1**2 - test_collapse_threshold)
            assert math.isclose(f_end, norms_per_time[end] ** 2 - test_collapse_threshold)

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
            target_t = self.victim.current_time - delta_t.imag * 1000
            assert 3 - _ATOL <= target_t <= 12 + _ATOL
            assert round(target_t) in norms_per_time
            assert math.isclose(
                state.norm(), norms_per_time[int(self.victim.current_time)]
            )
            state.factors[state.orthogonality_center] = (
                norms_per_time[round(target_t)]
                * MPS.make(self.qubit_count).factors[state.orthogonality_center]
            )

        self.evolve_tdvp_mock.side_effect = evolve_tdvp_mock_side_effect

        def random_noise_collapse_mock_side_effect():
            self.victim.collapse_threshold = 0.2  # Reset after jump at t=6.
            # The state should also be normalized here but we don't need it for this test.

        random_noise_collapse_mock.side_effect = random_noise_collapse_mock_side_effect
        self.victim.do_time_step(0)
        random_noise_collapse_mock.assert_called_once()
        find_root_brents_mock.assert_called_once()
        self.evolve_tdvp_mock.assert_has_calls(
            [
                # Initial call: t=0 -> t=12
                call(
                    t=Approximately(-0.012j),
                    state=ANY,
                    hamiltonian=ANY,
                    extra_krylov_tolerance=ANY,
                    max_krylov_dim=ANY,
                    is_hermitian=False,
                ),
                # Root finding call: t=12 -> t=3
                call(
                    t=Approximately(0.009j),
                    state=ANY,
                    hamiltonian=ANY,
                    extra_krylov_tolerance=ANY,
                    max_krylov_dim=ANY,
                    is_hermitian=False,
                ),
                # Root finding call: t=3 -> t=11
                call(
                    t=Approximately(-0.008j),
                    state=ANY,
                    hamiltonian=ANY,
                    extra_krylov_tolerance=ANY,
                    max_krylov_dim=ANY,
                    is_hermitian=False,
                ),
                # Root finding call: t=11 -> t=6
                call(
                    t=Approximately(0.005j),
                    state=ANY,
                    hamiltonian=ANY,
                    extra_krylov_tolerance=ANY,
                    max_krylov_dim=ANY,
                    is_hermitian=False,
                ),
                # Post jump: t=6 -> t=dt == 12
                call(
                    t=Approximately(-0.006j),
                    state=ANY,
                    hamiltonian=ANY,
                    extra_krylov_tolerance=ANY,
                    max_krylov_dim=ANY,
                    is_hermitian=False,
                ),
            ]
        )
        assert self.evolve_tdvp_mock.call_count == 5
