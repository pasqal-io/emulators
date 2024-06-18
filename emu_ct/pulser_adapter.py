import pulser
from emu_ct import QubitPosition, MPS, make_H, evolve_tdvp
import torch


def get_qubit_positions(
    register: pulser.Register,
) -> list[QubitPosition]:
    return [QubitPosition(*position) for position in register.qubits.values()]


def extract_omega_delta(
    seq: pulser.sequence.sequence.Sequence, sampling_rate: float
) -> torch.Tensor:
    """
    Samples the Pulser sequence and returns a tensor T containing:
    - T[0, i, q] = amplitude at time i * sampling_rate for qubit q
    - T[1, i, q] = detuning at time i * sampling_rate for qubit q
    """
    sequence_samples = pulser.sampler.sampler.sample(seq)

    assert 0.0 < sampling_rate <= 1.0

    result = torch.zeros(
        2,
        int(seq.get_duration() * sampling_rate) + 1,
        len(seq.register.qubit_ids),
        dtype=torch.complex128,
    )

    number_of_channels = len(sequence_samples.samples_list)

    current_slot_indices = [0] * number_of_channels

    step = 0
    dt = int(1 / sampling_rate)
    while step * dt < seq.get_duration():
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
                qubit_index = seq.register.qubit_ids.index(qubit_id)

                if qubit_index in seen_qubits:
                    # FIXME: if amp or det are 0 just ignore??
                    raise NotImplementedError("multiple pulses acting on same qubit")

                seen_qubits.add(qubit_index)

                result[0, step, qubit_index] = channel_samples.amp[t]
                result[1, step, qubit_index] = channel_samples.det[t]

        step += 1

    return result


def simulate_pulser_sequence(
    seq: pulser.sequence.sequence.Sequence,
) -> MPS:  # pass eta here?
    state = MPS(
        [(torch.tensor([1.0, 0.0], dtype=torch.complex128).reshape(1, 2, 1))]
        * len(seq.register.qubit_ids)
    )

    sampling_rate: float = 0.01
    coeff = 0.001  # Omega and delta are given in rad/microseconds, dt in nanoseconds
    omega_delta = extract_omega_delta(seq, sampling_rate=sampling_rate)

    emuct_register = get_qubit_positions(seq.register)

    for step in range(
        omega_delta.shape[1]
    ):  # TODO: this while should be converted into a run function as in Pulser
        mpo_t0 = make_H(emuct_register, omega_delta[0, step, :], omega_delta[1, step, :])
        evolve_tdvp(-coeff / sampling_rate * 1j, state, mpo_t0)

    return state
