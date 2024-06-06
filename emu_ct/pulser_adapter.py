import pulser
from emu_ct import Register
import torch


def registers_to_pyemunt(
    register: pulser.Register,
) -> list[Register]:  # Why is this not called "registers_to_emuct"?
    """Convert pulser registers into pyemunt registers"""
    return [Register(*position) for position in register.qubits.values()]


def slot_target_to_positions(slot_target: set, qubit_ids: tuple) -> list[int]:
    """Matches the target atom (the atom or atoms that will be implemented amp and det)
    with the position in the register"""
    return [i for i, qubit_id in enumerate(qubit_ids) if qubit_id in slot_target]


def extract_values_from_channel(
    channel: pulser.sampler.samples.SequenceSamples,
    register: pulser.Register,
    ret_amp: list[torch.Tensor],
    ret_det: list[torch.Tensor],
    t: int,
) -> None:
    """
    Extract amplitude and detuning from a channel
    """
    for slot in channel.slots:
        if slot.ti <= t < slot.tf:
            targets = slot_target_to_positions(slot.targets, register.qubit_ids)
            for i in targets:
                ret_amp[i] = channel.amp[t]
                ret_det[i] = channel.det[t]
            break


def extract_values_from_sequence(
    discretized_sequence: dict, register: pulser.Register, t: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Extract amplitude and detuning from the discretized sequence
    """
    ret_amp: list[float] = torch.zeros(
        len(register.qubit_ids), dtype=torch.complex128
    )  # initialize
    ret_det: list[float] = torch.zeros(
        len(register.qubit_ids), dtype=torch.complex128
    )  # initialize
    # create res_amp and result_detu
    for samples in discretized_sequence.values():
        extract_values_from_channel(samples, register, ret_amp, ret_det, t)
    return ret_amp, ret_det
