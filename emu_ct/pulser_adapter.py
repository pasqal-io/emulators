import pulser
from emu_ct import Register
import torch


def registers_to_pyemunt(register: pulser.Register) -> list[Register]:
    """Convert pulser registers into pyemunt registers"""
    reg_pyemutn = []
    for (_, position) in register.qubits.items():
        reg_pyemutn.append(Register(*position))

    return reg_pyemutn


def slot_target_to_positions(slot_target: set, qubit_ids: tuple) -> list[int]:
    """Matches the target atom (the atom or atoms that will be implemented amp and det)
    with the position in the register"""
    position_target: list[int] = []
    for i, j in enumerate(qubit_ids):
        for _, m in enumerate(slot_target):
            if j == m:
                position_target.append(i)

    return position_target


def extract_values_from_channel(
    channel: pulser.sampler.samples.SequenceSamples,
    register: pulser.Register,
    ret_amp: list[torch.Tensor],
    ret_det: list[torch.Tensor],
    t: int,
) -> None:
    """
    Extract amplitude and detening from a channel
    """
    for slot in channel.slots:
        if slot.ti <= t < slot.tf:
            targets = slot_target_to_positions(slot.targets, register.qubit_ids)
            for i in targets:
                ret_amp[i] = channel.amp[t]
                ret_det[i] = channel.det[t]
            break


def extract_values_from_sequence(
    discretize_sequence: dict, register: pulser.Register, t: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    Extract amplitude and detening from the discretize sequence
    """
    ret_amp: list[float] = torch.zeros(
        len(register.qubit_ids), dtype=torch.float64
    )  # initialize
    ret_det: list[float] = torch.zeros(
        len(register.qubit_ids), dtype=torch.float64
    )  # initialize
    # create res_amp and result_detu
    for (_, samples) in discretize_sequence.items():
        extract_values_from_channel(samples, register, ret_amp, ret_det, t)
    return ret_amp, ret_det
