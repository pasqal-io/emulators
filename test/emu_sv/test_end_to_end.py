import math
import torch
from pytest import approx

import pulser
import pulser.noise_model

import emu_base.base_classes
import emu_base.base_classes.default_callbacks
from emu_base.base_classes import (
    BitStrings,
    CorrelationMatrix,
    Energy,
    EnergyVariance,
    Fidelity,
    QubitDensity,
    SecondMomentOfEnergy,
    StateResult,
)

from emu_sv.sv_backend import SVBackend
from emu_sv.sv_config import SVConfig, StateVector

from test.utils_testing import (
    pulser_afm_sequence_ring,
    pulser_blackman,
)


seed = 1337

sv_backend = SVBackend()


Omega_max = 4 * 2 * torch.pi
U = Omega_max / 2
delta_0 = -6 * U
delta_f = 2 * U
t_rise = 500
t_fall = 1000


def create_antiferromagnetic_state_vector(
    num_qubits: int, gpu: bool = True
) -> StateVector:
    factors = [torch.zeros(2, dtype=torch.complex128) for _ in range(num_qubits)]
    for i in range(num_qubits):
        if i % 2:
            factors[i][0] = 1.0
        else:
            factors[i][1] = 1.0

    afm_vec = factors[0]
    for i in factors[1:]:
        afm_vec = torch.kron(afm_vec, i)

    return StateVector(afm_vec, gpu=gpu)


def simulate(
    seq: pulser.Sequence,
    *,
    dt=100,
    noise_model=None,
    state_prep_error=0.0,
    p_false_pos=0.0,
    p_false_neg=0.0,
    initial_state=None,
    given_fidelity_state=True,
    interaction_cutoff=0.0,
    gpu=True,
):
    final_time = seq.get_duration()

    if given_fidelity_state:
        fidelity_state = create_antiferromagnetic_state_vector(
            len(seq.register.qubit_ids), gpu=gpu
        )
    else:
        fidelity_state = StateVector.make(len(seq.register.qubit_ids))

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

    nqubits = len(seq.register.qubit_ids)
    times = {final_time}

    sv_config = SVConfig(
        initial_state=initial_state,
        dt=dt,
        krylov_tolerance=1e-5,
        observables=[
            StateResult(evaluation_times=times),
            BitStrings(evaluation_times=times, num_shots=1000),
            Fidelity(evaluation_times=times, state=fidelity_state),
            QubitDensity(evaluation_times=times, basis={"r", "g"}, nqubits=nqubits),
            Energy(evaluation_times=times),
            EnergyVariance(evaluation_times=times),
            SecondMomentOfEnergy(evaluation_times=times),
            CorrelationMatrix(evaluation_times=times, basis={"r", "g"}, nqubits=nqubits),
        ],
        noise_model=noise_model,
        interaction_cutoff=interaction_cutoff,
        gpu=gpu,
    )

    result = sv_backend.run(seq, sv_config)

    return result


def test_end_to_end_afm_ring():
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

    result = simulate(
        seq, gpu=False
    )  # only run on cpu, bitstring sampling is device dependent

    final_time = seq.get_duration()
    bitstrings = result["bitstrings"][final_time]
    final_state = result["state"][final_time]
    final_fidelity = result[
        f"fidelity_{emu_base.base_classes.default_callbacks._fidelity_counter}"
    ][final_time]

    fidelity_state = create_antiferromagnetic_state_vector(num_qubits, gpu=False)

    assert bitstrings["1010101010"] == 136
    assert bitstrings["0101010101"] == 159
    assert fidelity_state.inner(final_state) == approx(final_fidelity, abs=1e-10)

    q_density = result["qubit_density"][final_time]

    assert torch.allclose(
        torch.tensor([0.578] * 10, dtype=torch.float64), q_density, atol=1e-3
    )

    energy = result["energy"][final_time]  # (-115.34554274708604-2.1316282072803006e-14j)
    assert approx(energy, 1e-7) == -115.34554479213088

    energy_variance = result["energy_variance"][final_time]  # 45.911110563993134
    assert approx(energy_variance, 1e-3) == 45.91111056399

    second_moment_energy = result["second_moment_of_energy"][
        final_time
    ]  # 13350.505342183847
    assert approx(second_moment_energy, 1e-6) == 13350.5053421


def test_end_to_end_pi_half_pulse():
    # π/2 pulse Blackman creates |ψ❭=(|0❭-1j|1❭)/sqrt(2)
    duration = 1000  # ns, so 1 μs
    area = math.pi / 2
    seq = pulser_blackman(duration, area)
    result = simulate(seq, dt=50)

    final_time = seq.get_duration()
    final_state = result["state"][final_time]

    expected = torch.tensor([1, -1j], dtype=torch.complex128) / math.sqrt(2)
    assert torch.allclose(final_state.vector.cpu(), expected, atol=1e-8)


def test_end_to_end_pi_half_pulse_with_phase():
    # π/2 pulse Blackman creates |ψ❭=(|0❭-1j|1❭)/sqrt(2)
    # with a phase factor exp(-1j*duration*φ) = 1j
    duration = 1000  # ns, so 1 μs
    area = math.pi / 2
    phase = math.pi / 2
    seq = pulser_blackman(duration, area, phase=phase)
    result = simulate(seq, dt=50)
    final_time = seq.get_duration()

    final_state = result["state"][final_time]
    # with the phase we expect |ψ❭=(|0❭+|1❭)/sqrt(2)
    expected = torch.tensor([1, 1], dtype=torch.complex128) / math.sqrt(2)

    assert torch.allclose(final_state.vector.cpu(), expected, atol=1e-8)
