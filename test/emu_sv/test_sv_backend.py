import inspect
from collections.abc import Callable
from unittest.mock import MagicMock, patch
import torch
from emu_sv import SVConfig, SVBackend, Occupation, Solver, StateVector, DensityMatrix
from emu_sv.dense_state import DenseState
from emu_sv.sv_backend_impl import SVBackendImpl
from emu_sv.time_evolution import EvolveDensityMatrix, EvolveMonteCarlo, EvolveStateVector
import emu_sv.time_evolution.implementations as time_evolution_impl
import emu_sv.time_evolution.torch as time_evolution_torch
from emu_base.math.double_krylov import double_krylov
from emu_base.math.krylov_exp import DEFAULT_MAX_KRYLOV_DIM, krylov_exp
from test.utils_testing import randn_interaction_matrix
from emu_base import SequenceData, HamiltonianType
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
        target_times=[1.0],
        qubit_ids=(),
        lindblad_ops=[],
    )
    bknd_impl = SVBackendImpl(config, pulser_data)
    bknd_impl._evolve_step(1.0, 0)


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
    assert results.occupation[0] == pytest.approx(0.0)
    assert results.occupation[-1] == pytest.approx(1.0)


def test_backend_impl_stepper_logic():
    mock_config = MagicMock()
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
    config = SVConfig(gpu=False, max_krylov_dim=MAX_KRYLOV_DIM, solver=solver)
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
            time_evolution_impl, "krylov_exp", autospec=True, side_effect=krylov_exp
        ) as impl_spy,
    ):
        bknd_impl._evolve_step(1.0, 0)

    dims = received_max_krylov_dims(torch_spy, krylov_exp)
    dims += received_max_krylov_dims(impl_spy, krylov_exp)
    assert len(dims) >= 1
    assert dims == [MAX_KRYLOV_DIM] * len(dims)


@pytest.mark.usefixtures("restore_rng_state")
def test_max_krylov_dim_reaches_backward():
    """The gradient pass uses the stepper's max_krylov_dim, for the gradient of the
    input state (krylov_exp) and of the pulse parameters (double_krylov)."""
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
            time_evolution_torch,
            "double_krylov",
            autospec=True,
            side_effect=double_krylov,
        ) as double_spy,
    ):
        state_out, _ = EvolveStateVector(MAX_KRYLOV_DIM).apply(
            1.0, omegas, deltas, phis, interaction, state, 1e-10, None
        )
        torch.autograd.grad(
            state_out.real.sum(), (omegas, deltas, phis, interaction, state)
        )

    # forward step, then backward for the input state
    assert received_max_krylov_dims(krylov_spy, krylov_exp) == [MAX_KRYLOV_DIM] * 2
    # backward for omegas, deltas, phis and the interaction matrix
    assert received_max_krylov_dims(double_spy, double_krylov) == [MAX_KRYLOV_DIM]
