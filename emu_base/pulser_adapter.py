import pulser
from typing import Tuple
import torch
import math
from pulser.noise_model import NoiseModel
from enum import Enum

from emu_base.base_classes.config import BackendConfig
from emu_base.lindblad_operators import get_lindblad_operators
from emu_base.utils import dist2, dist3


class HamiltonianType(Enum):
    Rydberg = 1
    XY = 2


def _get_qubit_positions(
    register: pulser.Register,
) -> list[torch.Tensor]:
    """Conversion from pulser Register to emu-mps register (torch type).
    Each element will be given as [Rx,Ry,Rz]"""

    positions = [position.as_tensor() for position in register.qubits.values()]

    if len(positions[0]) == 2:
        return [torch.cat((position, torch.zeros(1))) for position in positions]
    return positions


def _rydberg_interaction(sequence: pulser.Sequence) -> torch.Tensor:
    """
    Computes the Ising interaction matrix from the qubit positions.
    Hᵢⱼ=C₆/Rᵢⱼ⁶ (nᵢ⊗ nⱼ)
    """

    num_qubits = len(sequence.register.qubit_ids)

    c6 = sequence.device.interaction_coeff

    qubit_positions = _get_qubit_positions(sequence.register)
    interaction_matrix = torch.zeros(num_qubits, num_qubits)

    for numi in range(len(qubit_positions)):
        for numj in range(numi + 1, len(qubit_positions)):
            interaction_matrix[numi][numj] = (
                c6 / dist2(qubit_positions[numi], qubit_positions[numj]) ** 3
            )
            interaction_matrix[numj, numi] = interaction_matrix[numi, numj]
    return interaction_matrix


def _xy_interaction(sequence: pulser.Sequence) -> torch.Tensor:
    """
    Computes the XY interaction matrix from the qubit positions.
    C₃ (1−3 cos(𝜃ᵢⱼ)²)/ Rᵢⱼ³ (𝜎ᵢ⁺ 𝜎ⱼ⁻ +  𝜎ᵢ⁻ 𝜎ⱼ⁺)
    """
    num_qubits = len(sequence.register.qubit_ids)

    c3 = sequence.device.interaction_coeff_xy

    qubit_positions = _get_qubit_positions(sequence.register)
    interaction_matrix = torch.zeros(num_qubits, num_qubits)
    mag_field = torch.tensor(sequence.magnetic_field)  # by default [0.0,0.0,30.0]
    mag_norm = torch.norm(mag_field)

    for numi in range(len(qubit_positions)):
        for numj in range(numi + 1, len(qubit_positions)):
            cosine = 0
            if mag_norm >= 1e-8:  # selected by hand
                cosine = torch.dot(
                    (qubit_positions[numi] - qubit_positions[numj]), mag_field
                ) / (torch.norm(qubit_positions[numi] - qubit_positions[numj]) * mag_norm)

            interaction_matrix[numi][numj] = (
                c3  # check this value with pulser people
                * (1 - 3 * cosine**2)
                / dist3(qubit_positions[numi], qubit_positions[numj])
            )
            interaction_matrix[numj, numi] = interaction_matrix[numi, numj]

    return interaction_matrix


def _extract_omega_delta_phi(
    *,
    sequence: pulser.Sequence,
    dt: int,
    with_modulation: bool,
    laser_waist: float | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples the Pulser sequence and returns a tuple of tensors (omega, delta, phi)
    containing:
    - omega[i, q] = amplitude at time i * dt for qubit q
    - delta[i, q] = detuning at time i * dt for qubit q
    - phi[i, q] = phase at time i * dt for qubit q

    if laser_waist is w_0 != None, the omega values coming from the global pulse channel
    will me modulated as $\\Omega_i=\\Omega_i e^{-r_i^2/w_0^2}$
    """

    if with_modulation and sequence._slm_mask_targets:
        raise NotImplementedError(
            "Simulation of sequences combining an SLM mask and output "
            "modulation is not supported."
        )

    samples = pulser.sampler.sample(
        sequence,
        modulation=with_modulation,
        extended_duration=sequence.get_duration(include_fall_time=with_modulation),
    )
    sequence_dict = samples.to_nested_dict(all_local=True, samples_type="tensor")["Local"]

    if "ground-rydberg" in sequence_dict and len(sequence_dict) == 1:
        locals_a_d_p = sequence_dict["ground-rydberg"]
    elif "XY" in sequence_dict and len(sequence_dict) == 1:
        locals_a_d_p = sequence_dict["XY"]
    else:
        raise ValueError("Emu-MPS only accepts ground-rydberg or mw_global channels")

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

    if laser_waist:
        qubit_positions = _get_qubit_positions(sequence.register)
        waist_factors = torch.tensor(
            [math.exp(-((x[:2].norm() / laser_waist) ** 2)) for x in qubit_positions]
        )
    else:
        waist_factors = torch.ones(len(sequence.register.qubit_ids))

    global_times = set()
    for ch, ch_samples in samples.channel_samples.items():
        if samples._ch_objs[ch].addressing == "Global":
            for slot in ch_samples.slots:
                global_times |= set(i for i in range(slot.ti, slot.tf))

    step = 0
    t = int((step + 1 / 2) * dt)

    while t < max_duration:
        for q_pos, q_id in enumerate(sequence.register.qubit_ids):
            omega[step, q_pos] = locals_a_d_p[q_id]["amp"][t]
            delta[step, q_pos] = locals_a_d_p[q_id]["det"][t]
            phi[step, q_pos] = locals_a_d_p[q_id]["phase"][t]
        if t in global_times:
            omega[step] *= waist_factors
        step += 1
        t = int((step + 1 / 2) * dt)

    return omega, delta, phi


_NON_LINDBLADIAN_NOISE = {"SPAM", "doppler", "amplitude"}


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


class PulserData:
    slm_end_time: int
    full_interaction_matrix: torch.Tensor
    masked_interaction_matrix: torch.Tensor
    omega: torch.Tensor
    delta: torch.Tensor
    phi: torch.Tensor
    hamiltonian_type: HamiltonianType
    lindblad_ops: list[torch.Tensor]

    def __init__(self, *, sequence: pulser.Sequence, config: BackendConfig, dt: int):
        self.qubit_count = len(sequence.register.qubit_ids)

        laser_waist = (
            config.noise_model.laser_waist if config.noise_model is not None else None
        )
        self.omega, self.delta, self.phi = _extract_omega_delta_phi(
            sequence=sequence,
            dt=dt,
            with_modulation=config.with_modulation,
            laser_waist=laser_waist,
        )
        self.lindblad_ops = _get_all_lindblad_noise_operators(config.noise_model)
        self.has_lindblad_noise: bool = self.lindblad_ops != []

        addressed_basis = sequence.get_addressed_bases()[0]
        if addressed_basis == "ground-rydberg":  # for local and global
            self.hamiltonian_type = HamiltonianType.Rydberg
        elif addressed_basis == "XY":
            self.hamiltonian_type = HamiltonianType.XY
        else:
            raise ValueError(f"Unsupported basis: {addressed_basis}")

        if config.interaction_matrix is not None:
            assert len(config.interaction_matrix) == self.qubit_count, (
                "The number of qubits in the register should be the same as the size of "
                "the interaction matrix"
            )

            self.full_interaction_matrix = torch.tensor(
                config.interaction_matrix, dtype=torch.float64
            )
        elif self.hamiltonian_type == HamiltonianType.Rydberg:
            self.full_interaction_matrix = _rydberg_interaction(sequence)
        elif self.hamiltonian_type == HamiltonianType.XY:
            self.full_interaction_matrix = _xy_interaction(sequence)
        self.full_interaction_matrix[
            torch.abs(self.full_interaction_matrix) < config.interaction_cutoff
        ] = 0.0
        self.masked_interaction_matrix = self.full_interaction_matrix.clone()

        slm_targets = sequence._slm_mask_targets
        self.slm_end_time = (
            sequence._slm_mask_time[1] if len(sequence._slm_mask_time) > 1 else 0.0
        )

        for target in slm_targets:
            self.masked_interaction_matrix[target] = 0.0
            self.masked_interaction_matrix[:, target] = 0.0
