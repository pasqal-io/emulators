import pulser
from typing import Tuple
from warnings import warn
import torch
import math
from pulser.noise_model import NoiseModel
from emu_mps.lindblad_operators import get_lindblad_operators


def get_qubit_positions(
    register: pulser.Register,
) -> list[torch.Tensor]:
    if any(not isinstance(p, torch.Tensor) for p in register.qubits.values()):
        return [torch.tensor(position) for position in register.qubits.values()]

    return list(register.qubits.values())


def _convert_sequence_samples(
    rydberg_dict_numpy: dict,
) -> None:
    """Convert amp, det and phase with type numpy.ndarray to torch.Tensor"""
    for a_d_p in rydberg_dict_numpy.values():
        for key, value in a_d_p.items():
            if not isinstance(value, torch.Tensor):
                a_d_p[key] = torch.tensor(value)


def extract_omega_delta_phi(
    sequence: pulser.Sequence,
    dt: int,
    with_modulation: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples the Pulser sequence and returns a tuple of tensors (omega, delta, phi)
    containing:
    - omega[i, q] = amplitude at time i * dt for qubit q
    - delta[i, q] = detuning at time i * dt for qubit q
    - phi[i, q] = phase at time i * dt for qubit q
    """

    if with_modulation and sequence._slm_mask_targets:
        raise NotImplementedError(
            "Simulation of sequences combining an SLM mask and output "
            "modulation is not supported."
        )

    sequence_dict = pulser.sampler.sample(
        sequence,
        modulation=with_modulation,
        extended_duration=sequence.get_duration(include_fall_time=with_modulation),
    ).to_nested_dict(all_local=True)["Local"]

    # TODO: from here accept the XY by ["XY"]
    if "ground-rydberg" in sequence_dict and len(sequence_dict) == 1:
        locals_rydberg_a_d_p = sequence_dict["ground-rydberg"]
    else:
        raise ValueError("Emu-MPS only accepts ground-rydberg channels")

    _convert_sequence_samples(locals_rydberg_a_d_p)

    max_duration = sequence.get_duration(include_fall_time=with_modulation)

    nsamples = math.ceil(max_duration / dt - 1 / 2)
    omega = torch.zeros(
        nsamples,
        len(sequence.register.qubit_ids),
        dtype=torch.complex128,
    )
    delta = torch.zeros(
        nsamples,
        len(sequence.register.qubit_ids),
        dtype=torch.complex128,
    )
    phi = torch.zeros(
        nsamples,
        len(sequence.register.qubit_ids),
        dtype=torch.complex128,
    )

    step = 0
    t = int((step + 1 / 2) * dt)

    while t < max_duration:

        for q_pos, q_id in enumerate(sequence.register.qubit_ids):
            omega[step, q_pos] = locals_rydberg_a_d_p[q_id]["amp"][t]
            delta[step, q_pos] = locals_rydberg_a_d_p[q_id]["det"][t]
            phi[step, q_pos] = locals_rydberg_a_d_p[q_id]["phase"][t]
        step += 1
        t = int((step + 1 / 2) * dt)

    return omega, delta, phi


_NON_LINDBLADIAN_NOISE = ["SPAM", "doppler", "amplitude"]


def get_all_lindblad_noise_operators(noise_model: NoiseModel) -> list[torch.Tensor]:
    if noise_model is None:
        return []

    lindblad_operators = [
        op
        for noise_type in noise_model.noise_types
        if noise_type not in _NON_LINDBLADIAN_NOISE
        for op in get_lindblad_operators(noise_type=noise_type, noise_model=noise_model)
    ]
    if len(lindblad_operators) > 0:
        warn(
            "Monte Carlo based Lindbladt noise is currently undocumented. Use at your own risk!"
        )
    return lindblad_operators
