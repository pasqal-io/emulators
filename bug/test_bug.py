from pytest import approx
import math
import pulser
import torch
import numpy as np

import emu_base.base_classes
import emu_base.base_classes.default_callbacks

from emu_base.base_classes import (
    BitStrings,
    Fidelity,
    StateResult,
    QubitDensity,
    Energy,
    EnergyVariance,
    SecondMomentOfEnergy,
    CorrelationMatrix,
)

from emu_sv.sv_config import SVConfig, StateVector
from emu_sv.sv_backend import SVBackend


import pulser.noise_model

seed = 1337

sv_backend = SVBackend()

time_factor = 20

Omega_max = 4 * 2 * torch.pi
U = Omega_max / (2 * time_factor)
delta_0 = -2 * U
delta_f = 2 * U
t_rise = 500 / time_factor
t_fall = 500 / time_factor

systemsize = 5


def pulser_afm_sequence_from_register(
    reg: pulser.Register,
    Omega_max: float,
    delta_0: float,
    delta_f: float,
    t_rise: float,
    t_fall: float,
    device: pulser.devices = pulser.devices.MockDevice,
):
    t_sweep = (delta_f - delta_0) / (2 * np.pi * 10) * 1000

    rise = pulser.Pulse.ConstantDetuning(
        pulser.waveforms.RampWaveform(t_rise, 0.0, Omega_max), delta_0, 0.0
    )
    sweep = pulser.Pulse.ConstantAmplitude(
        Omega_max, pulser.waveforms.RampWaveform(t_sweep, delta_0, delta_f), 0.0
    )
    fall = pulser.Pulse.ConstantDetuning(
        pulser.waveforms.RampWaveform(t_fall, Omega_max, 0.0), delta_f, 0.0
    )

    seq = pulser.Sequence(reg, device)
    seq.declare_channel("ising_global", "rydberg_global")
    seq.add(rise, "ising_global")
    seq.add(sweep, "ising_global")
    seq.add(fall, "ising_global")

    return seq


def pulser_afm_sequence_ring(
    num_qubits: int,
    Omega_max: float,
    U: float,
    delta_0: float,
    delta_f: float,
    t_rise: float,
    t_fall: float,
    device: pulser.devices = pulser.devices.MockDevice,
):
    # Define a ring of atoms distanced by a blockade radius distance:
    R_interatomic = device.rydberg_blockade_radius(U)
    coords = (
        R_interatomic
        / (2 * math.tan(math.pi / num_qubits))
        * torch.tensor(
            [
                [
                    math.cos(theta * 2 * math.pi / num_qubits),
                    math.sin(theta * 2 * math.pi / num_qubits),
                ]
                for theta in range(num_qubits)
            ]
        )
    )

    reg = pulser.Register.from_coordinates(coords, prefix="q")

    return pulser_afm_sequence_from_register(
        reg,
        Omega_max=Omega_max,
        delta_0=delta_0,
        delta_f=delta_f,
        t_rise=t_rise,
        t_fall=t_fall,
    )



def create_antiferromagnetic_state_vector(num_qubits: int) -> StateVector:
    factors = [torch.zeros(2, dtype=torch.complex128) for _ in range(num_qubits)]
    for i in range(num_qubits):
        if i % 2:
            factors[i][0] = 1.0
        else:
            factors[i][1] = 1.0

    afm_vec = factors[0]
    for i in factors[1:]:
        afm_vec = torch.kron(afm_vec, i)

    return StateVector(afm_vec)


def simulate(
    seq: pulser.Sequence,
    *,
    dt=10,
    noise_model=None,
    state_prep_error=0.0,
    p_false_pos=0.0,
    p_false_neg=0.0,
    initial_state=None,
    given_fidelity_state=True,
    interaction_cutoff=0.0,
):
    final_time = seq.get_duration()

    if given_fidelity_state:
        fidelity_state = create_antiferromagnetic_state_vector(
            len(seq.register.qubit_ids)
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

            Fidelity(evaluation_times=times, state=fidelity_state),
        ],
        noise_model=noise_model,
        interaction_cutoff=interaction_cutoff,
    )

    result = sv_backend.run(seq, sv_config)

    return result


def test_end_to_end_afm_ring():
    torch.manual_seed(seed)

    num_qubits = systemsize
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

    final_time = seq.get_duration()
    final_state = result["state"][final_time]
    final_fidelity = result[
        f"fidelity_{emu_base.base_classes.default_callbacks._fidelity_counter}"
    ][final_time]

    fidelity_state = create_antiferromagnetic_state_vector(num_qubits)

    assert fidelity_state.inner(final_state) == approx(final_fidelity, abs=1e-10)


test_end_to_end_afm_ring()