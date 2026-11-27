import inspect
from collections.abc import Callable
from unittest.mock import MagicMock, patch
import torch
from emu_sv import SVConfig, SVBackend, Occupation, Solver, StateVector, DensityMatrix
from emu_sv.dense_state import DenseState
from emu_sv.sv_backend_impl import SVBackendImpl
from emu_sv.time_evolution import (
    BaseStepper,
    EvolveDensityMatrix,
    EvolveMonteCarlo,
    EvolveStateVector,
    Chebyshev,
    Lanczos,
)
from emu_sv.enums import Exponentiator
from emu_sv.hamiltonians import (
    FusedRydbergHamiltonian,
    RydbergHamiltonian,
)
import emu_sv.time_evolution.exponentiation as time_evolution_exponentiation
import emu_sv.time_evolution.torch as time_evolution_torch
from emu_base.math.double_krylov import double_krylov
from emu_base.math.krylov_exp import DEFAULT_MAX_KRYLOV_DIM, krylov_exp, krylov_exp_impl
from test.utils_testing import randn_interaction_matrix
from emu_base import SequenceData, HamiltonianType
from pulser.backend import Results
import pytest
import logging

device = "cpu"


def test_sv_impl():
    """test that index_add is called in a no_grad context in forward"""
    config = SVConfig(gpu=False if device == "cpu" else True)
    pulser_data = MagicMock(
        spec=SequenceData,
        omega=torch.tensor([[1.0]], requires_grad=True),
        delta=torch.tensor([[1.0]]),
        phi=torch.tensor([[1.0]]),
        interaction_matrix=lambda t: torch.zeros((1, 1, 1), dtype=torch.float64),
        state_prep_error=0.0,
        target_times=[0.0, 1.0],
        qubit_ids=(),
        lindblad_ops=[],
    )
    bknd_impl = SVBackendImpl(config, pulser_data)
    bknd_impl._evolve_step(1.0, 0, 1)


@pytest.mark.parametrize(
    "state, stochastic_noise, lindblad_noise",
    [
        (
            DenseState.from_state_amplitudes(
                eigenstates=("r", "g"), amplitudes={"r": 1.0}
            ),
            False,
            False,
        ),
        (
            StateVector.from_state_amplitudes(
                eigenstates=("r", "g"), amplitudes={"r": 1.0}
            ),
            False,
            False,
        ),
        (
            DenseState.from_state_amplitudes(
                eigenstates=("r", "g"), amplitudes={"r": 1.0}
            ),
            True,
            False,
        ),
        (
            StateVector.from_state_amplitudes(
                eigenstates=("r", "g"), amplitudes={"r": 1.0}
            ),
            True,
            False,
        ),
        (
            DensityMatrix.from_state_amplitudes(
                eigenstates=("r", "g"), amplitudes={"r": 1.0}
            ),
            False,
            True,
        ),
    ],
)
def test_init_initial_state(state, stochastic_noise, lindblad_noise):
    """test the SVBackendImpl correctly handles the fed initial state"""
    config = SVConfig(gpu=False if device == "cpu" else True, initial_state=state)
    pulser_data = MagicMock(
        spec=SequenceData,
        omega=torch.tensor([[1.0]], requires_grad=True),
        delta=torch.tensor([[1.0]]),
        phi=torch.tensor([[1.0]]),
        interaction_matrix=lambda t: torch.zeros((1, 1, 1), dtype=torch.float64),
        state_prep_error=0.0,
        target_times=[1.0],
        qubit_ids=(),
        lindblad_ops=[torch.eye(2, dtype=torch.complex128)] if lindblad_noise else [],
        has_stochastic_noise=stochastic_noise,
    )
    bknd_impl = SVBackendImpl(config, pulser_data)
    if lindblad_noise and not stochastic_noise:
        state_type = DensityMatrix
    else:
        state_type = StateVector
    assert type(bknd_impl.state) is state_type


def test_run_from_sequence_data():
    duration = 100
    dt = 10

    occup = Occupation(
        evaluation_times=[dt * x / duration for x in range(duration // dt + 1)]
    )

    config = SVConfig(
        observables=[occup],
        log_level=logging.WARN,
        interaction_cutoff=1e-10,
    )

    omega = torch.tensor(
        [
            [0.7092 + 0.0j, 0.7092 + 0.0j, 0.7092 + 0.0j],
            [7.8431 + 0.0j, 7.8431 + 0.0j, 7.8431 + 0.0j],
            [26.2913 + 0.0j, 26.2913 + 0.0j, 26.2913 + 0.0j],
            [53.0011 + 0.0j, 53.0011 + 0.0j, 53.0011 + 0.0j],
            [73.0656 + 0.0j, 73.0656 + 0.0j, 73.0656 + 0.0j],
            [71.8630 + 0.0j, 71.8630 + 0.0j, 71.8630 + 0.0j],
            [50.3238 + 0.0j, 50.3238 + 0.0j, 50.3238 + 0.0j],
            [23.9187 + 0.0j, 23.9187 + 0.0j, 23.9187 + 0.0j],
            [6.6745 + 0.0j, 6.6745 + 0.0j, 6.6745 + 0.0j],
            [0.4483 + 0.0j, 0.4483 + 0.0j, 0.4483 + 0.0j],
        ],
        dtype=torch.complex128,
    )

    seq_data = SequenceData(
        omega=omega,
        delta=torch.zeros_like(omega),
        phi=torch.zeros_like(omega),
        interaction_matrix=lambda x: torch.zeros((1, 3, 3), dtype=torch.float64),
        qubit_ids=("q0", "q1", "a2"),
        bad_atoms=(False, False, False),
        lindblad_ops=[],
        state_prep_error=0.0,
        target_times=[dt * x for x in range(duration // dt + 1)],
        eigenstates=["r", "g"],
        hamiltonian_type=HamiltonianType.Rydberg,
        has_stochastic_noise=False,
    )

    results = SVBackend._run_from_sequence_data(seq_data, config)

    assert set(results._tagmap.keys()) == {"occupation", "statistics"}
    assert results.get_result_times("occupation") == occup.evaluation_times.tolist()
    # statistics are stored after every time step
    target_times = seq_data.target_times
    assert results.get_result_times("statistics") == pytest.approx(
        [t / target_times[-1] for t in target_times[1:]]
    )
    assert results.occupation[0] == pytest.approx(0.0)
    assert results.occupation[-1] == pytest.approx(1.0)


def test_backend_impl_stepper_logic():
    mock_config = MagicMock()
    mock_config.gpu = False
    mock_config.initial_state = None
    mock_config.solver = Solver.DEFAULT

    mock_sequence = MagicMock()
    mock_sequence.state_prep_error = 0.0
    mock_sequence.lindblad_ops = [1.0, 2.0]
    mock_sequence.has_stochastic_noise = False
    backend_impl = SVBackendImpl(mock_config, mock_sequence)
    assert type(backend_impl.stepper) is EvolveDensityMatrix

    mock_sequence = MagicMock()
    mock_sequence.state_prep_error = 0.0
    mock_sequence.lindblad_ops = []
    mock_sequence.has_stochastic_noise = False
    backend_impl = SVBackendImpl(mock_config, mock_sequence)
    assert type(backend_impl.stepper) is EvolveStateVector

    mock_sequence = MagicMock()
    mock_sequence.state_prep_error = 0.0
    mock_sequence.lindblad_ops = [1.0, 2.0]
    mock_sequence.has_stochastic_noise = True
    backend_impl = SVBackendImpl(mock_config, mock_sequence)
    assert type(backend_impl.stepper) is EvolveMonteCarlo

    mock_config.solver = Solver.LINDBLAD
    mock_sequence = MagicMock()
    mock_sequence.state_prep_error = 0.0
    mock_sequence.lindblad_ops = [1.0, 2.0]
    mock_sequence.has_stochastic_noise = True
    backend_impl = SVBackendImpl(mock_config, mock_sequence)
    assert type(backend_impl.stepper) is EvolveDensityMatrix

    mock_config.solver = Solver.MONTECARLO
    mock_sequence = MagicMock()
    mock_sequence.state_prep_error = 0.0
    mock_sequence.lindblad_ops = [1.0, 2.0]
    mock_sequence.has_stochastic_noise = False
    backend_impl = SVBackendImpl(mock_config, mock_sequence)
    assert type(backend_impl.stepper) is EvolveMonteCarlo

    mock_config.solver = Solver.MONTECARLO
    mock_sequence = MagicMock()
    mock_sequence.state_prep_error = 0.0
    mock_sequence.lindblad_ops = []
    mock_sequence.has_stochastic_noise = False
    backend_impl = SVBackendImpl(mock_config, mock_sequence)
    assert type(backend_impl.stepper) is EvolveStateVector

    mock_config.solver = Solver.LINDBLAD
    mock_sequence = MagicMock()
    mock_sequence.state_prep_error = 0.0
    mock_sequence.lindblad_ops = []
    mock_sequence.has_stochastic_noise = False
    backend_impl = SVBackendImpl(mock_config, mock_sequence)
    assert type(backend_impl.stepper) is EvolveStateVector


@pytest.mark.parametrize("kernel_available", [False, True])
@pytest.mark.parametrize("stepper_type", [EvolveStateVector, EvolveMonteCarlo])
def test_backend_impl_hamiltonian_type(
    kernel_available: bool, stepper_type: type[BaseStepper]
) -> None:
    mock_config = MagicMock(initial_state=None, solver=Solver.MONTECARLO, gpu=False)
    mock_sequence = MagicMock(
        state_prep_error=0.0,
        lindblad_ops=[1.0, 2.0] if stepper_type is EvolveMonteCarlo else [],
        has_stochastic_noise=False,
    )
    with patch.object(
        FusedRydbergHamiltonian, "is_available", return_value=kernel_available
    ):
        backend_impl = SVBackendImpl(mock_config, mock_sequence)
    assert type(backend_impl.stepper) is stepper_type
    expected = FusedRydbergHamiltonian if kernel_available else RydbergHamiltonian
    assert backend_impl.stepper.hamiltonian_type is expected


@pytest.mark.parametrize(
    "requested, lindblad_ops, solver, kernel_available, expected",
    [
        (Exponentiator.DEFAULT, [], Solver.DEFAULT, False, Chebyshev),
        (Exponentiator.DEFAULT, [], Solver.DEFAULT, True, Chebyshev),
        (Exponentiator.LANCZOS, [], Solver.DEFAULT, True, Lanczos),
        (Exponentiator.CHEBYSHEV, [], Solver.DEFAULT, False, Chebyshev),
        (Exponentiator.DEFAULT, [1.0], Solver.MONTECARLO, True, Chebyshev),
        (Exponentiator.DEFAULT, [1.0], Solver.LINDBLAD, True, Lanczos),
        (Exponentiator.CHEBYSHEV, [1.0], Solver.MONTECARLO, True, Chebyshev),
        (Exponentiator.CHEBYSHEV, [1.0], Solver.LINDBLAD, True, ValueError),
    ],
)
def test_backend_impl_exponentiator(
    requested, lindblad_ops, solver, kernel_available, expected
) -> None:
    mock_config = MagicMock(
        initial_state=None,
        solver=solver,
        gpu=False,
        max_krylov_dim=MAX_KRYLOV_DIM,
        exponentiator=requested,
    )
    mock_sequence = MagicMock(
        state_prep_error=0.0, lindblad_ops=lindblad_ops, has_stochastic_noise=False
    )
    with patch.object(
        FusedRydbergHamiltonian, "is_available", return_value=kernel_available
    ):
        if expected is ValueError:
            with pytest.raises(ValueError, match="CHEBYSHEV"):
                SVBackendImpl(mock_config, mock_sequence)
            return
        backend_impl = SVBackendImpl(mock_config, mock_sequence)
    exponentiator = backend_impl.stepper.exponentiator
    assert type(exponentiator) is expected
    if expected is Lanczos:
        assert exponentiator.max_krylov_dim == MAX_KRYLOV_DIM


def test_monte_carlo_stepper_rejects_state_requiring_grad():
    """the stepper itself refuses, also when used without SVBackendImpl"""
    state = torch.tensor([0.0, 1.0], dtype=torch.complex128, requires_grad=True)
    zero = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(NotImplementedError, match="Monte Carlo"):
        EvolveMonteCarlo(Lanczos(), DEFAULT_MAX_KRYLOV_DIM, RydbergHamiltonian).apply(
            0.1,
            zero,
            zero,
            zero,
            torch.zeros(1, 1, dtype=torch.float64),
            state,
            1e-10,
            [torch.eye(2, dtype=torch.complex128)],
        )


# SVConfig.max_krylov_dim used to be ignored: every Krylov call ran with the default of
# 100, so a step needing more Lanczos vectors failed with RecursionError whatever the
# config said. 17 differs from that default and is enough to span the largest space used
# below (a 2-qubit density matrix, 4**2 = 16), so convergence never depends on it.
MAX_KRYLOV_DIM = 17
assert MAX_KRYLOV_DIM != DEFAULT_MAX_KRYLOV_DIM


def received_max_krylov_dims(spy: MagicMock, real: Callable) -> list[int]:
    """The max_krylov_dim of every recorded call, defaults applied."""
    dims = []
    for call in spy.call_args_list:
        bound = inspect.signature(real).bind(*call.args, **call.kwargs)
        bound.apply_defaults()
        dims.append(bound.arguments["max_krylov_dim"])
    return dims


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize(
    "solver, lindblad_ops, stepper_type",
    [
        (Solver.DEFAULT, [], EvolveStateVector),
        (Solver.LINDBLAD, [torch.eye(2, dtype=torch.complex128)], EvolveDensityMatrix),
        (Solver.MONTECARLO, [torch.eye(2, dtype=torch.complex128)], EvolveMonteCarlo),
    ],
)
def test_max_krylov_dim_reaches_krylov_exp(solver, lindblad_ops, stepper_type):
    """SVConfig.max_krylov_dim reaches every krylov_exp call of a time step."""
    config = SVConfig(
        gpu=False,
        max_krylov_dim=MAX_KRYLOV_DIM,
        solver=solver,
        exponentiator=Exponentiator.LANCZOS,
    )
    pulser_data = MagicMock(
        spec=SequenceData,
        omega=torch.tensor([[1.0, 1.0]]),
        delta=torch.tensor([[1.0, 1.0]]),
        phi=torch.tensor([[0.0, 0.0]]),
        interaction_matrix=lambda t: torch.zeros((1, 2, 2), dtype=torch.float64),
        state_prep_error=0.0,
        target_times=[0.0, 1.0],
        qubit_ids=(),
        lindblad_ops=lindblad_ops,
    )
    bknd_impl = SVBackendImpl(config, pulser_data)
    assert type(bknd_impl.stepper) is stepper_type

    with (
        patch.object(
            time_evolution_torch, "krylov_exp", autospec=True, side_effect=krylov_exp
        ) as torch_spy,
        patch.object(
            time_evolution_exponentiation,
            "krylov_exp_impl",
            autospec=True,
            side_effect=krylov_exp_impl,
        ) as impl_spy,
    ):
        bknd_impl._evolve_step(1.0, 0, 1)

    dims = received_max_krylov_dims(torch_spy, krylov_exp)
    dims += received_max_krylov_dims(impl_spy, krylov_exp_impl)
    assert len(dims) >= 1
    assert dims == [MAX_KRYLOV_DIM] * len(dims)


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("exponentiator_type", [Lanczos, Chebyshev])
def test_max_krylov_dim_reaches_backward(exponentiator_type):
    """The gradient pass uses the stepper's max_krylov_dim, for the gradient of the
    input state (krylov_exp) and of the pulse parameters (double_krylov), whatever
    the exponentiator of the forward step."""
    exponentiator = (
        Lanczos(MAX_KRYLOV_DIM) if exponentiator_type is Lanczos else Chebyshev()
    )
    torch.manual_seed(1337)
    nqubits = 2
    omegas = torch.rand(nqubits, dtype=torch.float64, requires_grad=True)
    deltas = torch.rand(nqubits, dtype=torch.float64, requires_grad=True)
    phis = torch.rand(nqubits, dtype=torch.float64, requires_grad=True)
    interaction = randn_interaction_matrix(nqubits, requires_grad=True)
    state = torch.randn(2**nqubits, dtype=torch.complex128)
    state = (state / state.norm()).requires_grad_()

    with (
        patch.object(
            time_evolution_torch, "krylov_exp", autospec=True, side_effect=krylov_exp
        ) as krylov_spy,
        patch.object(
            time_evolution_exponentiation,
            "krylov_exp_impl",
            autospec=True,
            side_effect=krylov_exp_impl,
        ) as forward_spy,
        patch.object(
            time_evolution_torch,
            "double_krylov",
            autospec=True,
            side_effect=double_krylov,
        ) as double_spy,
    ):
        state_out, _ = EvolveStateVector(
            exponentiator, MAX_KRYLOV_DIM, RydbergHamiltonian
        ).apply(1.0, omegas, deltas, phis, interaction, state, 1e-10, None)
        torch.autograd.grad(
            state_out.real.sum(), (omegas, deltas, phis, interaction, state)
        )

    # forward step (no Krylov call for Chebyshev)
    expected_forward = [MAX_KRYLOV_DIM] if exponentiator_type is Lanczos else []
    assert received_max_krylov_dims(forward_spy, krylov_exp_impl) == expected_forward
    # backward for the input state
    assert received_max_krylov_dims(krylov_spy, krylov_exp) == [MAX_KRYLOV_DIM]
    # backward for omegas, deltas, phis and the interaction matrix
    assert received_max_krylov_dims(double_spy, double_krylov) == [MAX_KRYLOV_DIM]


def make_sequence_data(
    omega: torch.Tensor,
    interaction_matrix: Callable[[float], torch.Tensor] | None = None,
    **kwargs,
) -> SequenceData:
    nsteps, nqubits = omega.shape
    if interaction_matrix is None:
        constant = torch.zeros((1, nqubits, nqubits), dtype=torch.float64)
        interaction_matrix = lambda t: constant  # noqa: E731
    defaults = dict(
        omega=omega,
        delta=torch.zeros_like(omega),
        phi=torch.zeros_like(omega),
        interaction_matrix=interaction_matrix,
        qubit_ids=tuple(f"q{i}" for i in range(nqubits)),
        bad_atoms=(False,) * nqubits,
        lindblad_ops=[],
        state_prep_error=0.0,
        target_times=[10.0 * x for x in range(nsteps + 1)],
        eigenstates=["r", "g"],
        hamiltonian_type=HamiltonianType.Rydberg,
        has_stochastic_noise=False,
    )
    return SequenceData(**(defaults | kwargs))


def test_fused_intervals():
    """Steps are fused while the pulses are equal, the interaction matrix is the
    same object, and no observable is evaluated in between."""
    omega = torch.tensor([[1.0, 1.0]] * 4 + [[2.0, 1.0]] * 4, dtype=torch.float64)
    first = torch.zeros((1, 2, 2), dtype=torch.float64)
    second = torch.zeros((1, 2, 2), dtype=torch.float64)  # equal content, new object
    # an observable at the end of step 1 (t = 20 of 80); the matrix changes at t = 60
    occupation = Occupation(evaluation_times=[0.25, 1.0])
    config = SVConfig(observables=[occupation], gpu=False, log_level=logging.WARN)
    data = make_sequence_data(omega, lambda t: first if t < 60.0 else second)

    impl = SVBackendImpl(config, data)

    assert impl.fused_intervals() == [(0, 2), (2, 4), (4, 6), (6, 8)]


def test_fused_intervals_not_with_gradients():
    omega = torch.ones((4, 2), dtype=torch.float64, requires_grad=True)
    config = SVConfig(observables=[], gpu=False, log_level=logging.WARN)
    impl = SVBackendImpl(config, make_sequence_data(omega))

    assert impl.fused_intervals() == [(0, 1), (1, 2), (2, 3), (3, 4)]
    with torch.no_grad():
        assert impl.fused_intervals() == [(0, 4)]


@pytest.mark.parametrize(
    "exponentiator", [Exponentiator.CHEBYSHEV, Exponentiator.LANCZOS]
)
def test_fused_run_matches_unfused(exponentiator: Exponentiator):
    """A quench evolved as one fused step gives the same state as step by step."""
    torch.manual_seed(1337)
    nsteps, nqubits = 10, 3
    omega = torch.ones((nsteps, nqubits), dtype=torch.float64) * 5.0
    interaction = torch.rand((1, nqubits, nqubits), dtype=torch.float64)
    interaction = interaction + interaction.transpose(1, 2)
    data = make_sequence_data(omega, lambda t: interaction)

    def run(times: list[float]) -> Results:
        config = SVConfig(
            observables=[Occupation(evaluation_times=times)],
            gpu=False,
            exponentiator=exponentiator,
            log_level=logging.WARN,
        )
        return SVBackend._run_from_sequence_data(data, config)

    fused = run([1.0])
    unfused = run([x / nsteps for x in range(1, nsteps + 1)])

    assert torch.allclose(fused.occupation[-1], unfused.occupation[-1], atol=1e-9)
    # statistics are stored after every step actually taken
    assert fused.get_result_times("statistics") == [1.0]
    assert len(unfused.get_result_times("statistics")) == nsteps


def test_evolve_step_reads_interaction_at_midpoint():
    omega = torch.ones((4, 2), dtype=torch.float64)
    constant = torch.zeros((1, 2, 2), dtype=torch.float64)
    times = []

    def interaction_matrix(t: float) -> torch.Tensor:
        times.append(t)
        return constant

    config = SVConfig(observables=[], gpu=False, log_level=logging.WARN)
    impl = SVBackendImpl(config, make_sequence_data(omega, interaction_matrix))

    impl._evolve_step(10.0, 1, 2)
    impl._evolve_step(30.0, 1, 4)
    assert times == [15.0, 25.0]


def test_dark_qubit_interaction_matrix_is_memoised():
    """With state preparation errors, the masked matrix is built once per matrix
    object of the adapter, so that steps on the same side of an SLM end fuse."""
    omega = torch.ones((4, 3), dtype=torch.float64)
    full = torch.ones((1, 3, 3), dtype=torch.float64)
    config = SVConfig(observables=[], gpu=False, log_level=logging.WARN)
    data = make_sequence_data(
        omega,
        lambda t: full,
        bad_atoms=(False, True, False),
        state_prep_error=0.1,
    )
    impl = SVBackendImpl(config, data)

    masked = impl.interaction_matrix(5.0)
    assert impl.interaction_matrix(25.0) is masked
    assert masked[0, 1].tolist() == [0.0] * 3 and masked[0, :, 1].tolist() == [0.0] * 3
    assert masked[0, 0, 2] == 1.0
    assert impl.fused_intervals() == [(0, 4)]


@pytest.mark.usefixtures("restore_rng_state")
def test_monte_carlo_jump_time_tolerance_from_config_dt():
    """Fused steps do not coarsen the jump times: the tolerance follows config.dt."""
    omega = torch.ones((4, 2), dtype=torch.float64)
    config = SVConfig(
        observables=[], gpu=False, dt=10, solver=Solver.MONTECARLO, log_level=logging.WARN
    )
    data = make_sequence_data(
        omega,
        lindblad_ops=[torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.complex128)],
        has_stochastic_noise=True,
    )
    impl = SVBackendImpl(config, data)

    assert isinstance(impl.stepper, EvolveMonteCarlo)
    assert impl.stepper.jump_time_tolerance == pytest.approx(0.1 * 10 * 1e-3)
