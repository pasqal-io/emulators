import pulser
from typing import Tuple
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
    sequence_samples: pulser.sampler.samples.SequenceSamples,
) -> None:
    for channel_samples in sequence_samples.samples_list:
        if not isinstance(channel_samples.amp, torch.Tensor):
            channel_samples.amp = torch.tensor(channel_samples.amp)
        if not isinstance(channel_samples.det, torch.Tensor):
            channel_samples.det = torch.tensor(channel_samples.det)
        if not isinstance(channel_samples.phase, torch.Tensor):
            channel_samples.phase = torch.tensor(channel_samples.phase)


def extract_omega_delta_phi(
    sequence: pulser.sequence.sequence.Sequence,
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

    sequence_samples = pulser.sampler.sampler.sample(
        sequence,
        modulation=with_modulation,
        extended_duration=sequence.get_duration(include_fall_time=with_modulation),
    )
    _convert_sequence_samples(sequence_samples)

    max_duration = sequence_samples.max_duration
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
    number_of_channels = len(sequence_samples.samples_list)
    current_slot_indices = [0] * number_of_channels
    step = 0
    t = int((step + 1 / 2) * dt)

    while t < max_duration:
        seen_qubits = set()
        for channel_index, slot_index in enumerate(current_slot_indices):
            channel_samples = sequence_samples.samples_list[channel_index]
            while (
                slot_index < len(channel_samples.slots)
                and t > channel_samples.slots[slot_index].tf
            ):
                slot_index += 1
            current_slot_indices[channel_index] = slot_index
            if (
                slot_index >= len(channel_samples.slots)
                or t < channel_samples.slots[slot_index].ti
            ):
                continue
            for qubit_id in channel_samples.slots[slot_index].targets:
                qubit_index = sequence.register.qubit_ids.index(qubit_id)

                if qubit_index in seen_qubits:
                    # FIXME: if amp or det are 0 just ignore??
                    raise NotImplementedError("multiple pulses acting on same qubit")

                seen_qubits.add(qubit_index)

                omega[step, qubit_index] = channel_samples.amp[t]
                delta[step, qubit_index] = channel_samples.det[t]
                phi[step, qubit_index] = channel_samples.phase[t]
        step += 1
        t = int((step + 1 / 2) * dt)

    return omega, delta, phi


_NON_LINDBLADIAN_NOISE = ["SPAM", "doppler", "amplitude"]


def get_all_lindblad_noise_operators(noise_model: NoiseModel) -> list[torch.Tensor]:
    if noise_model is None:
        return []

    return [
        op
        for noise_type in noise_model.noise_types
        if noise_type not in _NON_LINDBLADIAN_NOISE
        for op in get_lindblad_operators(noise_type=noise_type, noise_model=noise_model)
    ]
