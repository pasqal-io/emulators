import time
from unittest.mock import ANY, MagicMock, patch
from collections import Counter
from typing import Any

import pulser
import pytest
import torch
import random
from pytest import approx

import emu_mps
from emu_mps import (
    MPS,
    BitStrings,
    Fidelity,
    MPSBackend,
    MPSConfig,
    StateResult,
    Occupation,
    Energy,
    EnergyVariance,
    EnergySecondMoment,
    CorrelationMatrix,
)

from emu_base import unix_like
import pulser.noise_model
from pulser.backend import Results

from emu_mps.mps_backend_impl import MPSBackendImpl
from emu_mps.solver import Solver
from emu_mps.solver_utils import right_baths
from test.utils_testing import (
    pulser_afm_sequence_grid,
    pulser_afm_sequence_ring,
    pulser_XY_sequence_slm_mask,
)

seed = 1337
device = "cpu"  # "cuda"


def create_antiferromagnetic_mps(num_qubits: int):
    str = ""
    for i in range(num_qubits):
        if i % 2:
            str += "g"
        else:
            str += "r"
    amplitudes = {str: 1.0}
    return MPS.from_state_amplitudes(eigenstates=["r", "g"], amplitudes=amplitudes)


def simulate(
    seq: pulser.Sequence,
    *,
    dt: int = 100,
    noise_model: Any | None = None,
    state_prep_error: float = 0,
    p_false_pos: float = 0,
    p_false_neg: float = 0,
    initial_state: Any | None = None,
    given_fidelity_state: bool = True,
    interaction_cutoff: float = 0,
    eval_times: list[float] = [1.0],
    solver: Solver = Solver.TDVP,
) -> Results:
    if given_fidelity_state:
        fidelity_state = create_antiferromagnetic_mps(len(seq.register.qubit_ids))
    else:
        fidelity_state = MPS.make(len(seq.register.qubit_ids))

    if state_prep_error > 0.0 or p_false_pos > 0.0 or p_false_neg > 0.0:
        assert noise_model is None, "Provide either noise_model or SPAM values"

        runs_args = (
            {"runs": 1, "samples_per_run": 1} if state_prep_error > 0.0 else {}
        )  # Avoid Pulser warning

        noise_model = pulser.noise_model.NoiseModel(
            **runs_args,
            state_prep_error=state_prep_error,
            p_false_pos=p_false_pos,
            p_false_neg=p_false_neg,
        )
    else:
        if noise_model is None:
            noise_model = pulser.noise_model.NoiseModel()

    mps_config = MPSConfig(
        initial_state=initial_state,
        dt=dt,
        precision=1e-5,
        observables=[
            Occupation(evaluation_times=eval_times),
            BitStrings(evaluation_times=eval_times, num_shots=1000),
            Energy(evaluation_times=eval_times),
            EnergyVariance(evaluation_times=eval_times),
            EnergySecondMoment(evaluation_times=eval_times),
            CorrelationMatrix(evaluation_times=eval_times),
            StateResult(evaluation_times=eval_times),
            Fidelity(evaluation_times=eval_times, state=fidelity_state, tag_suffix="1"),
        ],
        noise_model=noise_model,
        interaction_cutoff=interaction_cutoff,
        optimize_qubit_ordering=False,
        solver=solver,
    )

    backend = MPSBackend(seq, config=mps_config)
    result = backend.run()

    return result


def simulate_line(n, **kwargs):
    seq = pulser_afm_sequence_grid(
        rows=1,
        columns=n,
        Omega_max=Omega_max,
        U=U,
        delta_0=delta_0,
        delta_f=delta_f,
        t_rise=t_rise,
        t_fall=t_fall,
    )
    return simulate(seq, **kwargs)


def get_proba(state: MPS, bitstring: str):
    # FIXME: use MPS factory method from bitstring
    one = torch.tensor([[[0], [1]]], dtype=torch.complex128)
    zero = torch.tensor([[[1], [0]]], dtype=torch.complex128)

    factors = [one if bitstring[i] == "1" else zero for i in range(state.num_sites)]

    return abs(state.inner(MPS(factors, eigenstates=("0", "1"))).item()) ** 2


Omega_max = 4 * 2 * torch.pi
U = Omega_max / 2
delta_0 = -6 * U
delta_f = 2 * U
t_rise = 500
t_fall = 1000


def test_default_MPSConfig_ctr() -> None:
    mps_config = MPSConfig()
    assert len(mps_config.observables) == 1
    assert isinstance(mps_config.observables[0], BitStrings)
    assert mps_config.observables[0].evaluation_times == [
        1.0
    ]  # meaning very end of the simuation


def test_XY_3atoms() -> None:
    torch.manual_seed(seed)
    seq = pulser_XY_sequence_slm_mask(amplitude=25.0)

    result = simulate(seq, dt=10, given_fidelity_state=False)

    final_state: MPS = result.state[-1]
    final_vec = torch.einsum("abc,cde,efg->abdfg", *(final_state.factors)).reshape(8)

    expected_res = torch.tensor(
        [
            -0.0684 - 0.5677j,
            0.0202 - 0.0305j,
            -0.0313 + 0.0214j,
            -0.2322 + 0.3942j,
            0.0202 - 0.0305j,
            -0.2329 + 0.3709j,
            -0.2322 + 0.3942j,
            0.2344 - 0.0602j,
        ],
        device=final_state.factors[0].device,
        dtype=torch.complex128,
    )

    # pulser magnetization: [0.46024234949993825,0.4776498885102908,0.4602423494999386#
    q_density = result.occupation[-1]

    max_bond_dim = final_state.get_max_bond_dim()
    assert max_bond_dim == 2
    assert approx(q_density.tolist(), 1e-3) == [0.4610, 0.4786, 0.4610]
    assert torch.allclose(final_vec, expected_res, rtol=0, atol=1e-4)


def test_XY_3atomswith_slm() -> None:
    torch.manual_seed(seed)
    seq = pulser_XY_sequence_slm_mask(amplitude=0.0, slm_masked_atoms=(1, 2))

    result = simulate(seq, dt=10, given_fidelity_state=False)

    final_state: MPS = result.state[-1]
    final_vec = torch.einsum("abc,cde,efg->abdfg", *(final_state.factors)).reshape(8)
    # pulser vector: 0.707,(−0.171+0.182j),(0.449−0.103j),0.0,(0.138−0.455j),1.761×10 −12,
    # −1.873×10−12j,0.0

    expected_res = torch.tensor(
        [
            7.0711e-01 - 4.2972e-17j,
            -1.7133e-01 + 1.7989e-01j,
            4.4791e-01 - 1.0291e-01j,
            2.2578e-16 - 4.4738e-15j,
            1.3729e-01 - 4.5631e-01j,
            2.1802e-15 - 3.0011e-16j,
            -1.7551e-15 + 2.5736e-15j,
            2.6618e-15 + 5.4529e-16j,
        ],
        device=final_state.factors[0].device,
        dtype=torch.complex128,
    )
    # pulser magnetization: [0.22572457283642877,0.21208108307887844,0.06213666344288577
    q_density = result.occupation[-1]

    max_bond_dim = final_state.get_max_bond_dim()
    assert max_bond_dim == 2
    assert approx(q_density, 1e-3) == [0.2270, 0.2112, 0.0617]
    assert torch.allclose(
        final_vec, expected_res, rtol=0, atol=1e-4
    )  # todo, compare against pulser results


@pytest.mark.parametrize(
    "optimize_order",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "occupation",
    [
        False,
        True,
    ],
)
def test_end_to_end_domain_wall_ring(
    occupation: bool,
    optimize_order: bool,
) -> None:
    # This setup is sensitive to the permutation order in contrast to AFM state preparation
    torch.manual_seed(seed)

    num_qubits = 6
    seq = pulser_afm_sequence_ring(
        num_qubits=num_qubits,
        Omega_max=0,
        U=U,
        delta_0=delta_0,
        delta_f=delta_f,
        t_rise=1300,
        t_fall=1400,
    )

    initial_state = emu_mps.MPS.from_state_amplitudes(
        eigenstates=("r", "g"),
        amplitudes={(num_qubits // 2) * "r" + (num_qubits // 2) * "g": 1.0},
    )
    eval_times = [1 / 44, 1]  # 1/44 is 1 dt step
    observables = [
        BitStrings(evaluation_times=eval_times, num_shots=100),
        Energy(evaluation_times=eval_times),
        EnergyVariance(evaluation_times=eval_times),
        CorrelationMatrix(evaluation_times=eval_times),
    ]
    if occupation:
        observables.append(Occupation(evaluation_times=eval_times))
    # I want to test permutation results close to the initial state
    mps_config = MPSConfig(
        initial_state=initial_state,
        dt=100,
        precision=1e-5,
        observables=observables,
        optimize_qubit_ordering=optimize_order,
    )

    backend = MPSBackend(seq, config=mps_config)
    result = backend.run()

    ntime_step = 0
    bitstrings = result.bitstrings[ntime_step]
    energy = result.energy[ntime_step]
    energy_variance = result.energy_variance[ntime_step]
    expect_occup = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.float64)
    correlation_matrix = result.correlation_matrix[ntime_step]
    expect_corr = torch.outer(expect_occup, expect_occup).to(torch.complex128)

    assert result.atom_order == seq.register.qubit_ids
    assert bitstrings["111000"] == 100
    assert approx(energy, rel=1e-4) == 286.8718
    assert approx(energy_variance, abs=1e-5) == 0
    assert torch.allclose(expect_corr, correlation_matrix, atol=1e-3)
    if occupation:
        occupation = result.occupation[ntime_step]
        assert torch.allclose(expect_occup, occupation, atol=1e-3)


def test_end_to_end_afm_ring() -> None:
    torch.manual_seed(seed)

    num_qubits = 10
    seq = pulser_afm_sequence_ring(
        num_qubits=num_qubits,
        Omega_max=Omega_max,
        U=U,
        delta_0=delta_0,
        delta_f=delta_f,
        t_rise=t_rise,
        t_fall=t_fall,
    )

    result = simulate(seq)

    final_time = -1
    bitstrings = result.bitstrings[final_time]
    occupation = result.occupation[final_time]
    energy = result.energy[final_time]
    energy_variance = result.energy_variance[final_time]
    second_moment_energy = result.energy_second_moment[final_time]
    state_fin = result.state[final_time]
    fidelity_fin = result.fidelity_1[final_time]
    max_bond_dim = state_fin.get_max_bond_dim()
    fidelity_st = create_antiferromagnetic_mps(num_qubits)
    assert max_bond_dim == 29
    assert fidelity_st.overlap(state_fin) == approx(fidelity_fin, abs=1e-10)

    assert bitstrings["1010101010"] == 129
    assert bitstrings["0101010101"] == 135

    # Comparing against EMU-SV -- state vector emulator
    assert approx(occupation, abs=1e-3) == [0.5782] * 10
    assert approx(occupation, rel=1e-3) == [0.5782] * 10

    assert approx(energy, abs=1e-2) == -115.3455
    assert approx(energy, rel=1e-4) == -115.3455

    assert approx(energy_variance, abs=1e-1) == 45.91
    assert approx(energy_variance, rel=1e-2) == 45.911

    assert approx(second_moment_energy, abs=0.45) == 13350.5
    assert approx(second_moment_energy, rel=1e-4) == 13350.5


def test_dmrg_afm_ring() -> None:
    torch.manual_seed(seed)

    num_qubits = 10
    seq = pulser_afm_sequence_ring(
        num_qubits=num_qubits,
        Omega_max=Omega_max,
        U=U,
        delta_0=delta_0,
        delta_f=delta_f,
        t_rise=t_rise,
        t_fall=t_fall,
    )

    result = simulate(seq, solver=Solver.DMRG)

    final_time = -1
    bitstrings = result.bitstrings[final_time]
    occupation = result.occupation[final_time]
    energy = result.energy[final_time]
    energy_variance = result.energy_variance[final_time]
    second_moment_energy = result.energy_second_moment[final_time]
    state_fin = result.state[final_time]
    fidelity_fin = result.fidelity_1[final_time]
    max_bond_dim = state_fin.get_max_bond_dim()
    fidelity_st = create_antiferromagnetic_mps(num_qubits)
    occupation_even_sites = occupation[0::2]
    occupation_odd_sites = occupation[1::2]

    assert max_bond_dim == 4
    # check that the output state is the AFM state
    assert fidelity_st.overlap(state_fin) == approx(fidelity_fin, abs=1e-10)
    assert bitstrings["1010101010"] == 977
    assert torch.allclose(
        fidelity_fin, torch.tensor(0.9735, dtype=torch.float64), atol=1e-3
    )

    # check that the number operator should return 1 on even sites and 0 elsewhere
    assert torch.allclose(
        occupation_even_sites, torch.tensor(1, dtype=torch.float64), atol=1e-3
    )
    assert torch.allclose(
        occupation_odd_sites, torch.tensor(0, dtype=torch.float64), atol=1e-2
    )

    assert torch.allclose(energy, torch.tensor(-124.0612, dtype=torch.float64), atol=1e-4)

    # check that energy variance should effectively be 0
    assert torch.allclose(
        energy_variance, torch.tensor(0, dtype=torch.float64), atol=1e-5
    )

    assert torch.allclose(
        second_moment_energy, torch.tensor(15391, dtype=torch.float64), atol=1e-1
    )


def test_dmrg_afm_square_grid() -> None:
    torch.manual_seed(seed)

    seq = pulser_afm_sequence_grid(
        rows=3,
        columns=3,
        Omega_max=Omega_max,
        U=U,
        delta_0=delta_0,
        delta_f=delta_f,
        t_rise=t_rise,
        t_fall=t_fall,
    )

    result = simulate(seq, solver=Solver.DMRG)

    final_time = -1
    bitstrings = result.bitstrings[final_time]
    occupation = result.occupation[final_time]
    energy = result.energy[final_time]
    energy_variance = result.energy_variance[final_time]
    state_fin = result.state[final_time]
    fidelity_fin = result.fidelity_1[final_time]
    max_bond_dim = state_fin.get_max_bond_dim()
    fidelity_st = create_antiferromagnetic_mps(9)
    occupation_even_sites = occupation[0::2]
    occupation_odd_sites = occupation[1::2]

    assert max_bond_dim == 4
    # check that the output state is the AFM state
    assert fidelity_st.overlap(state_fin) == approx(fidelity_fin, abs=1e-10)
    assert bitstrings["101010101"] == 992
    assert torch.allclose(
        fidelity_fin, torch.tensor(0.9868, dtype=torch.float64), atol=1e-3
    )

    # check that the number operator should return 1 on even sites and 0 elsewhere
    assert torch.allclose(
        occupation_even_sites, torch.tensor(1, dtype=torch.float64), atol=1e-2
    )
    assert torch.allclose(
        occupation_odd_sites, torch.tensor(0, dtype=torch.float64), atol=1e-2
    )

    assert torch.allclose(energy, torch.tensor(-118.7510, dtype=torch.float64), atol=1e-4)

    # check that energy variance should effectively be 0
    assert torch.allclose(
        energy_variance, torch.tensor(0, dtype=torch.float64), atol=1e-5
    )


def test_dmrg_large_detuning() -> None:
    # at very large detuning, the state should be stuck in the initial state
    # DMRG should exactly capture the product state of 9 qubits
    torch.manual_seed(seed)

    seq = pulser_afm_sequence_grid(
        rows=3,
        columns=3,
        Omega_max=0,
        U=U,
        delta_0=-50,
        delta_f=delta_f,
        t_rise=t_rise,
        t_fall=t_fall,
    )

    result = simulate(seq, solver=Solver.DMRG)

    final_time = -1
    bitstrings = result.bitstrings[final_time]
    occupation = result.occupation[final_time]
    energy_variance = result.energy_variance[final_time]
    state_fin = result.state[final_time]
    fidelity_fin = result.fidelity_1[final_time]
    max_bond_dim = state_fin.get_max_bond_dim()

    # check that the final state remains classical
    assert max_bond_dim == 1
    # check that it remains the initial state ["000000000"]
    assert bitstrings["000000000"] == 1000
    assert torch.allclose(fidelity_fin, torch.tensor(0, dtype=torch.float64), atol=1e-3)

    assert torch.allclose(occupation, torch.tensor(0, dtype=torch.float64), atol=1e-2)

    # check that energy variance should effectively be 0
    assert torch.allclose(
        energy_variance, torch.tensor(0, dtype=torch.float64), atol=1e-5
    )


def test_end_to_end_afm_line_with_state_preparation_errors() -> None:
    torch.manual_seed(seed)

    with patch("emu_mps.mps_backend_impl.pick_dark_qubits") as pick_dark_qubits_mock:
        pick_dark_qubits_mock.return_value = torch.logical_not(
            torch.tensor([True, True, True, False])
        )
        result = simulate_line(4, state_prep_error=0.1)
        final_state = result.state[-1]
        pick_dark_qubits_mock.assert_called_with(0.1, 4)

    assert get_proba(final_state, "1110") == approx(0.56, abs=1e-2)
    assert get_proba(final_state, "1010") == approx(0.43, abs=1e-2)

    # A dark qubit at the end of the line gives the same result as a line with one less qubit.
    with patch("emu_mps.mps_backend_impl.pick_dark_qubits") as pick_dark_qubits_mock:
        result = simulate_line(3)
        final_state = result.state[-1]
        pick_dark_qubits_mock.assert_not_called()
        assert get_proba(final_state, "111") == approx(0.56, abs=1e-2)
        assert get_proba(final_state, "101") == approx(0.43, abs=1e-2)

    with patch("emu_mps.mps_backend_impl.pick_dark_qubits") as pick_dark_qubits_mock:
        pick_dark_qubits_mock.return_value = torch.logical_not(
            torch.tensor([True, False, True, True])
        )
        result = simulate_line(4, state_prep_error=0.1)
        final_state = result.state[-1]

    assert get_proba(final_state, "1011") == approx(0.95, abs=1e-2)

    # Results for a 2 qubit line.
    result = simulate_line(2)
    final_state = result.state[-1]
    assert get_proba(final_state, "11") == approx(0.95, abs=1e-2)

    with patch("emu_mps.mps_backend_impl.pick_dark_qubits") as pick_dark_qubits_mock:
        pick_dark_qubits_mock.return_value = torch.logical_not(
            torch.tensor([False, True, True, False])
        )
        result = simulate_line(4, state_prep_error=0.1)
        final_state = result.state[-1]

    assert get_proba(final_state, "0110") == approx(0.95, abs=1e-2)

    # FIXME: When n-1 qubits are dark, the simulation fails!
    with patch("emu_mps.mps_backend_impl.pick_dark_qubits") as pick_dark_qubits_mock:
        with pytest.raises(ValueError) as exception_info:
            pick_dark_qubits_mock.return_value = torch.logical_not(
                torch.tensor([False, False, True, False])
            )
            result = simulate_line(4, state_prep_error=0.1)
            final_state = result.state[-1]

    assert "For 1 qubit states, do state vector" in str(exception_info.value)


def test_end_to_end_afm_line_with_measurement_errors() -> None:
    with patch("emu_mps.mps.apply_measurement_errors") as apply_measurement_errors_mock:
        bitstrings = MagicMock()
        apply_measurement_errors_mock.return_value = bitstrings
        results = simulate_line(4, p_false_pos=0.0, p_false_neg=0.5)
        apply_measurement_errors_mock.assert_called_with(
            ANY, p_false_pos=0.0, p_false_neg=0.5
        )
        assert results.bitstrings[-1] is bitstrings


def test_initial_state() -> None:
    pulse = pulser.Pulse.ConstantAmplitude(
        0.0, pulser.waveforms.ConstantWaveform(10.0, 0.0), 0.0
    )
    reg = pulser.Register.rectangle(5, 1, spacing=1e10, prefix="q")
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(pulse, "ising_global")  # do nothing in the pulse

    state = emu_mps.MPS.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes={"rrrrr": 1.0}
    )
    assert state.norm() == approx(1.0)  # assert unit norm

    state_result = emu_mps.StateResult(evaluation_times=[1.0])
    config = emu_mps.MPSConfig(observables=[state_result], initial_state=state)
    backend = emu_mps.MPSBackend(seq, config=config)
    results = backend.run()
    # assert that the initial state was used by the emulator
    assert results.get_result(state_result, 1.0).inner(state).real == approx(1.0)
    # but that it's a copy
    assert results.get_result(state_result, 1.0) is not state


def test_initial_state_copy() -> None:
    duration = 10.0
    pulse = pulser.Pulse.ConstantAmplitude(
        Omega_max, pulser.waveforms.RampWaveform(duration, delta_0, delta_f), 0.0
    )
    reg = pulser.Register.rectangle(5, 1, spacing=1e10, prefix="q")
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(pulse, "ising_global")

    initial_state = emu_mps.MPS.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes={"rrrrr": 1.0}
    )

    config = emu_mps.MPSConfig(initial_state=initial_state)

    emu_mps.MPSBackend(seq, config=config).run()

    # Check the initial state's factors were not modified.
    assert all(
        torch.allclose(initial_state_factor, expected_initial_state_factor)
        for initial_state_factor, expected_initial_state_factor in zip(
            initial_state.factors,
            emu_mps.MPS.from_state_amplitudes(
                eigenstates=("r", "g"), amplitudes={"rrrrr": 1.0}
            ).factors,
        )
    )


def test_end_to_end_afm_ring_with_noise() -> None:
    if not unix_like:
        pytest.skip(reason="fails due to different RNG on windows")
    torch.manual_seed(seed)
    random.seed(0xDEADBEEF)

    num_qubits = 6
    seq = pulser_afm_sequence_ring(
        num_qubits=num_qubits,
        Omega_max=Omega_max,
        U=U,
        delta_0=delta_0,
        delta_f=delta_f,
        t_rise=t_rise,
        t_fall=t_fall,
    )

    noise_model = pulser.noise_model.NoiseModel(
        depolarizing_rate=0.3,  # High enough to trigger a jump.
    )

    result = simulate(seq, noise_model=noise_model)

    bitstrings = result.bitstrings[-1]
    final_state = result.state[-1]
    max_bond_dim = final_state.get_max_bond_dim()

    assert bitstrings["101010"] == 454
    assert bitstrings["010101"] == 500
    assert max_bond_dim == 8


def test_end_to_end_spontaneous_emission() -> None:
    if not unix_like:
        pytest.skip(reason="fails due to different RNG on windows")
    torch.manual_seed(seed)
    random.seed(0xDEADBEEF)

    # Sequence with no driving.
    duration = 10000
    rows, cols = 3, 4
    reg = pulser.Register.rectangle(
        rows, cols, pulser.devices.MockDevice.rydberg_blockade_radius(U), prefix="q"
    )
    seq = pulser.Sequence(reg, pulser.devices.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(
        pulser.Pulse.ConstantAmplitude(
            amplitude=0.0,
            detuning=pulser.waveforms.ConstantWaveform(duration=duration, value=0.0),
            phase=0.0,
        ),
        "ising_global",
    )

    noise_model = pulser.noise_model.NoiseModel(
        relaxation_rate=0.1,
    )

    initial_state = emu_mps.MPS.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes={"rrrrrrrrrrrr": 1.0}
    )

    def check_baths(impl: MPSBackendImpl):
        # Mocking MPSBackendImpl.get_current_right_bath() to check that
        # the right baths administration happens properly when a quantum jump occurs.

        assert len(impl.right_baths) in [
            impl.state.num_sites - impl.sweep_index,
            impl.state.num_sites - impl.sweep_index - 1,
        ]

        expected_right_baths = right_baths(
            impl.state,
            impl.hamiltonian,
            final_qubit=impl.state.num_sites - len(impl.right_baths) + 1,
        )
        assert all(
            torch.allclose(actual, expected)
            for actual, expected in zip(impl.right_baths, expected_right_baths)
        )

        return impl.right_baths[-1]

    with patch(
        "emu_mps.mps_backend_impl.MPSBackendImpl.get_current_right_bath", autospec=True
    ) as get_current_right_bath_mock:
        get_current_right_bath_mock.side_effect = check_baths

        result = simulate(seq, noise_model=noise_model, initial_state=initial_state)

    final_state = result.state[-1]

    assert get_proba(final_state, "100000110000") == approx(1, abs=1e-2)

    # Aggregating results of many runs to check the exponential decrease of qubit density
    # would be too much for this unit test.


def test_end_to_end_spontaneous_emission_rate() -> None:
    if not unix_like:
        pytest.skip(reason="fails due to different RNG on windows")
    torch.manual_seed(seed)
    random.seed(0xDEADBEEF)

    # Sequence with no driving.
    duration = 10000
    rows, cols = 1, 2
    reg = pulser.Register.rectangle(rows, cols, 1e10, prefix="q")
    seq = pulser.Sequence(reg, pulser.devices.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(
        pulser.Pulse.ConstantAmplitude(
            amplitude=0.0,
            detuning=pulser.waveforms.ConstantWaveform(duration=duration, value=0.0),
            phase=0.0,
        ),
        "ising_global",
    )

    noise_model = pulser.noise_model.NoiseModel(
        relaxation_rate=0.1,
    )

    initial_state = emu_mps.MPS.from_state_amplitudes(
        eigenstates=["r", "g"], amplitudes={"rr": 1.0}
    )
    results = []
    for _ in range(100):
        results.append(
            simulate(
                seq, noise_model=noise_model, initial_state=initial_state, dt=duration
            )
        )

    counts = {}
    # round probabilities to merge 1.0 and 0.9999999999999998 etc.
    for string in ["00", "01", "10", "11"]:
        counts[string] = Counter(
            [round(get_proba(result.state[-1], string)) for result in results]
        )[1]

    # the exact rates are {"11":0.135, "01":0.233, "10":0.233, "00":0.400}
    expected_counts = {"11": 16, "01": 17, "10": 23, "00": 44}

    assert counts == expected_counts

    occu_list = [
        result.occupation[-1] for result in results
    ]  # to trigger the occupation observable

    avg_occ = sum(occu_list) / len(occu_list)
    assert torch.allclose(
        avg_occ, torch.tensor([0.3900, 0.3300], dtype=torch.float64), atol=1e-3
    )


def test_laser_waist() -> None:
    duration = 1000
    reg = pulser.Register.from_coordinates(
        [(0.0, 0.0), (10.0, 0.0)], center=False, prefix="q"
    )
    seq = pulser.Sequence(reg, pulser.devices.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.declare_channel("ising_local", "rydberg_local")
    seq.add(
        pulser.Pulse.ConstantAmplitude(
            amplitude=torch.pi,
            detuning=pulser.waveforms.ConstantWaveform(duration=duration, value=0.0),
            phase=0.0,
        ),
        "ising_global",
    )
    e_inv = 0.36787944117144233
    seq.target(seq.register.qubit_ids[1], "ising_local")
    seq.add(
        pulser.Pulse.ConstantAmplitude(
            amplitude=torch.pi * (1 - e_inv),
            detuning=pulser.waveforms.ConstantWaveform(duration=duration, value=0.0),
            phase=0.0,
        ),
        "ising_local",
    )

    noise_model = pulser.noise_model.NoiseModel(
        laser_waist=10.0,
    )

    result = simulate(seq, noise_model=noise_model, dt=10, interaction_cutoff=100)

    final_state = result.state[-1]

    assert pytest.approx(final_state.norm()) == 1.0

    expected_state = emu_mps.MPS.from_state_amplitudes(
        eigenstates=("r", "g"), amplitudes={"rr": 1.0}
    )

    assert pytest.approx(final_state.inner(expected_state)) == -1.0


def test_autosave() -> None:
    duration = 300
    rows, cols = 2, 3
    reg = pulser.Register.rectangle(
        rows, cols, pulser.devices.MockDevice.rydberg_blockade_radius(U), prefix="q"
    )
    seq = pulser.Sequence(reg, pulser.devices.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(
        pulser.Pulse.ConstantAmplitude(
            amplitude=torch.pi,
            detuning=pulser.waveforms.ConstantWaveform(duration=duration, value=0.0),
            phase=0.0,
        ),
        "ising_global",
    )

    evaluation_times = [1.0 / 30.0, 1.0 / 3.0, 0.5]
    energy = Energy(evaluation_times=evaluation_times)

    save_simulation_original = MPSBackendImpl.save_simulation
    save_file = None

    counter = 100  # Number of simulation steps before crashing

    def save_simulation_mock_side_effect(self):
        nonlocal counter
        counter -= 1
        if counter > 0:
            self.last_save_time = time.time() + 999
            return save_simulation_original(self)

        assert self.timestep_index == 11

        self.last_save_time = 0  # Trigger saving regardless of time
        save_simulation_original(self)
        nonlocal save_file
        save_file = self.autosave_file
        raise Exception("Process killed!")

    with patch.object(
        MPSBackendImpl, "save_simulation", autospec=True
    ) as save_simulation_mock:
        save_simulation_mock.side_effect = save_simulation_mock_side_effect

        with pytest.raises(Exception) as e:
            MPSBackend(seq, config=MPSConfig(observables=[energy])).run()

        assert str(e.value) == "Process killed!"

    assert save_file is not None and save_file.is_file()
    results_after_resume = MPSBackend.resume(save_file)

    assert not save_file.is_file()

    results_expected = MPSBackend(seq, config=MPSConfig(observables=[energy])).run()

    for t in evaluation_times:
        assert torch.allclose(
            results_after_resume.get_result("energy", t),
            results_expected.get_result("energy", t),
        )


def test_obs_after_autosave() -> None:
    duration = 300
    rows, cols = 2, 3
    reg = pulser.Register.rectangle(
        rows, cols, pulser.devices.MockDevice.rydberg_blockade_radius(U), prefix="q"
    )
    seq = pulser.Sequence(reg, pulser.devices.MockDevice)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(
        pulser.Pulse.ConstantAmplitude(
            amplitude=torch.pi,
            detuning=pulser.waveforms.ConstantWaveform(duration=duration, value=0.0),
            phase=0.0,
        ),
        "ising_global",
    )

    evaluation_times = [1.0 / 30.0, 1.0 / 3.0, 0.5]
    energy = Energy(evaluation_times=evaluation_times)
    correlation = CorrelationMatrix(evaluation_times=evaluation_times)
    occupation = Occupation(evaluation_times=evaluation_times)

    save_simulation_original = MPSBackendImpl.save_simulation

    def save_simulation_mock_side_effect(self):
        self.last_save_time = 0  # save at each timestep
        return save_simulation_original(self)

    with patch.object(
        MPSBackendImpl, "save_simulation", autospec=True
    ) as save_simulation_mock:
        save_simulation_mock.side_effect = save_simulation_mock_side_effect

        results = MPSBackend(
            seq, config=MPSConfig(observables=[energy, correlation, occupation])
        ).run()
    assert all([isinstance(x, torch.Tensor) for x in results.energy])
    assert all([isinstance(x, torch.Tensor) for x in results.occupation])
    assert all([isinstance(x, torch.Tensor) for x in results.correlation_matrix])
