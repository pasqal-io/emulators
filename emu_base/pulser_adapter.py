from typing import Sequence
from enum import Enum
import torch
import math
import pulser
from pulser.sampler import SequenceSamples
from pulser.noise_model import NoiseModel
from pulser.register.base_register import QubitId
from pulser.backend.config import EmulationConfig
from pulser._hamiltonian_data import HamiltonianData
from emu_base.jump_lindblad_operators import get_lindblad_operators


class HamiltonianType(Enum):
    Rydberg = 1
    XY = 2


_NON_LINDBLADIAN_NOISE = {"SPAM", "doppler", "amplitude", "detuning", "register"}


def _get_all_lindblad_noise_operators(
    noise_model: NoiseModel | None,
) -> list[torch.Tensor]:
    if noise_model is None:
        return []

    return [
        op
        for noise_type in noise_model.noise_types
        if noise_type not in _NON_LINDBLADIAN_NOISE
        for op in get_lindblad_operators(noise_type=noise_type, noise_model=noise_model)
    ]


def _get_target_times(
    sequence: pulser.Sequence, config: EmulationConfig, dt: int
) -> list[int]:
    sequence_duration = sequence.get_duration(include_fall_time=config.with_modulation)

    observable_times = set(range(0, sequence_duration, dt))
    observable_times.add(sequence_duration)
    for obs in config.observables:
        times: Sequence[float]
        if obs.evaluation_times is not None:
            times = obs.evaluation_times
        elif config.default_evaluation_times != "Full":
            times = config.default_evaluation_times.tolist()  # type: ignore[union-attr,assignment]
        observable_times |= set([round(time * sequence_duration) for time in times])

    target_times: list[int] = list(observable_times)
    target_times.sort()
    return target_times


def _extract_omega_delta_phi(
    noisy_samples: SequenceSamples,
    qubit_ids: tuple[str, ...],
    target_times: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequence_dict = noisy_samples.to_nested_dict(all_local=True, samples_type="tensor")[
        "Local"
    ]
    nsamples = len(target_times) - 1
    omega = torch.zeros(
        nsamples,
        len(qubit_ids),
        dtype=torch.complex128,
    )
    delta = torch.zeros(
        nsamples,
        len(qubit_ids),
        dtype=torch.complex128,
    )
    phi = torch.zeros(
        nsamples,
        len(qubit_ids),
        dtype=torch.complex128,
    )
    max_duration = noisy_samples.max_duration

    if "ground-rydberg" in sequence_dict and len(sequence_dict) == 1:
        locals_a_d_p = sequence_dict["ground-rydberg"]
    elif "XY" in sequence_dict and len(sequence_dict) == 1:
        locals_a_d_p = sequence_dict["XY"]
    else:
        raise ValueError("Only `ground-rydberg` and `mw_global` channels are supported.")
    for i in range(nsamples):
        t = (target_times[i] + target_times[i + 1]) / 2
        # The sampled values correspond to the start of each interval
        # To maximize the order of the solver, we need the values in the middle
        if math.ceil(t) < max_duration:
            # If we're not the final step, approximate this using linear
            # interpolation
            # Note that for dt even, t1=t2
            t1 = math.floor(t)
            t2 = math.ceil(t)
            for q_pos, q_id in enumerate(qubit_ids):
                omega[i, q_pos] = (
                    locals_a_d_p[q_id]["amp"][t1] + locals_a_d_p[q_id]["amp"][t2]
                ) / 2.0
                delta[i, q_pos] = (
                    locals_a_d_p[q_id]["det"][t1] + locals_a_d_p[q_id]["det"][t2]
                ) / 2.0
                phi[i, q_pos] = (
                    locals_a_d_p[q_id]["phase"][t1] + locals_a_d_p[q_id]["phase"][t2]
                ) / 2.0
        else:
            # We're in the final step and dt=1, approximate this using linear extrapolation
            # we can reuse omega_1 and omega_2 from before
            for q_pos, q_id in enumerate(qubit_ids):
                delta[i, q_pos] = (
                    3.0 * locals_a_d_p[q_id]["det"][t2] - locals_a_d_p[q_id]["det"][t1]
                ) / 2.0
                phi[i, q_pos] = (
                    3.0 * locals_a_d_p[q_id]["phase"][t2]
                    - locals_a_d_p[q_id]["phase"][t1]
                ) / 2.0
                omega[i, q_pos] = max(
                    (3.0 * locals_a_d_p[q_id]["amp"][t2] - locals_a_d_p[q_id]["amp"][t1])
                    / 2.0,
                    0.0,
                )
    return omega, delta, phi


class PulserData:
    slm_end_time: float
    full_interaction_matrix: torch.Tensor
    masked_interaction_matrix: torch.Tensor
    omega: torch.Tensor
    delta: torch.Tensor
    phi: torch.Tensor
    hamiltonian_type: HamiltonianType
    lindblad_ops: list[torch.Tensor]
    qubit_ids: tuple[QubitId, ...]

    def __init__(self, *, sequence: pulser.Sequence, config: EmulationConfig, dt: int):
        self.qubit_ids = sequence.register.qubit_ids
        self.qubit_count = len(self.qubit_ids)
        self.target_times = _get_target_times(sequence=sequence, config=config, dt=dt)
        self.hamiltonian = HamiltonianData.from_sequence(
            sequence,
            with_modulation=config.with_modulation,
            noise_model=config.noise_model,
        )

        self.omega, self.delta, self.phi = _extract_omega_delta_phi(
            self.hamiltonian.noisy_samples, self.qubit_ids, self.target_times
        )

        int_type = self.hamiltonian.interaction_type
        if int_type == "ising":  # for local and global
            self.hamiltonian_type = HamiltonianType.Rydberg
        elif int_type == "XY":
            self.hamiltonian_type = HamiltonianType.XY
        else:
            raise ValueError(f"Unsupported basis: {int_type}")

        self.lindblad_ops = _get_all_lindblad_noise_operators(config.noise_model)
        self.has_lindblad_noise: bool = self.lindblad_ops != []

        if config.interaction_matrix is not None:
            assert len(config.interaction_matrix) == self.qubit_count, (
                "The number of qubits in the register should be the same as the size of "
                "the interaction matrix"
            )

            self.full_interaction_matrix = config.interaction_matrix.as_tensor()
        else:
            self.full_interaction_matrix = (
                self.hamiltonian.noisy_interaction_matrix.as_tensor()
            )

        self.full_interaction_matrix[
            torch.abs(self.full_interaction_matrix) < config.interaction_cutoff
        ] = 0.0
        self.masked_interaction_matrix = self.full_interaction_matrix.clone()

        self.slm_end_time = (
            sequence._slm_mask_time[1] if len(sequence._slm_mask_time) > 1 else 0.0
        )

        # disable interaction for SLM masked qubits
        slm_targets = list(sequence._slm_mask_targets)
        for target in sequence.register.find_indices(slm_targets):
            self.masked_interaction_matrix[target] = 0.0
            self.masked_interaction_matrix[:, target] = 0.0
