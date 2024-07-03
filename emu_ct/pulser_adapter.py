import pulser

import torch


def get_qubit_positions(
    register: pulser.Register,
) -> list[torch.Tensor]:
    return [
        position._array if position.is_tensor else torch.tensor(position._array)
        for position in register.qubits.values()
    ]


def _convert_sequence_samples(
    sequence_samples: pulser.sampler.samples.SequenceSamples,
) -> None:
    for channel_samples in sequence_samples.samples_list:
        channel_samples.amp = (
            channel_samples.amp._array
            if channel_samples.amp.is_tensor
            else torch.tensor(channel_samples.amp._array)
        )
        channel_samples.det = (
            channel_samples.det._array
            if channel_samples.det.is_tensor
            else torch.tensor(channel_samples.det._array)
        )


def _extract_omega_delta(
    sequence: pulser.sequence.sequence.Sequence,
    dt: int,
    with_modulation: bool,
) -> torch.Tensor:
    """
    Samples the Pulser sequence and returns a tensor T containing:
    - T[0, i, q] = amplitude at time i * sampling_rate for qubit q
    - T[1, i, q] = detuning at time i * sampling_rate for qubit q
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

    result = torch.zeros(
        2,
        int(max_duration / dt) + 1,
        len(sequence.register.qubit_ids),
        dtype=torch.complex128,
    )
    number_of_channels = len(sequence_samples.samples_list)
    current_slot_indices = [0] * number_of_channels
    step = 0

    while step * dt < max_duration:
        t = step * dt
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

                result[0, step, qubit_index] = channel_samples.amp[t]
                result[1, step, qubit_index] = channel_samples.det[t]

        step += 1

    return result
