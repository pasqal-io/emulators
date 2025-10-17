import functools
import torch

from typing import List, Optional, Union
import numpy as np
import pulser
import math

# for testing purposes, reference to the real multinomial
_real_multinomial = torch.multinomial


def cpu_multinomial_wrapper(probs: torch.Tensor, num_samples: int, replacement=False):
    """
    For independent device *(cpu or gpu) tests. This is a function that
    intercepts calls to torch.multinomial, moves `probs` to CPU, applies the real
    torch.multinomial there, and moves the result back
    to the original device.
    """
    # Move to CPU (detach: no gradient tracking issues)
    probs_cpu = probs.detach().cpu()
    # real multinomial on CPU
    out_cpu = _real_multinomial(probs_cpu, num_samples, replacement)
    # Move back to the device of original `probs`
    out = out_cpu.to(probs.device)
    return out


def ghz_state_factors(
    nqubits: int,
    dim=2,
    dtype: torch.dtype = torch.complex128,
    device: Optional[Union[str, torch.device]] = None,
) -> List[torch.Tensor]:
    assert nqubits >= 2
    if dim == 2:
        core_1 = (
            1
            / torch.sqrt(torch.tensor([2.0], device=device, dtype=dtype))
            * torch.tensor(
                [
                    [
                        [1.0 + 0.0j, 0.0 + 0.0j],
                        [
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                        ],
                    ]
                ],
                dtype=dtype,
                device=device,
            )
        )
        core_mid = torch.tensor(
            [
                [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
                [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
            ],
            dtype=dtype,
            device=device,
        )
        # similar to core_mid, except no bond to the right
        core3 = torch.tensor(
            [[[1.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [1.0 + 0.0j]]],
            dtype=dtype,
            device=device,
        )

    if dim == 3:
        core_1 = (
            1
            / torch.sqrt(torch.tensor([2.0], device=device, dtype=dtype))
            * torch.tensor(
                [
                    [
                        [1.0 + 0.0j, 0.0 + 0.0j],
                        [
                            0.0 + 0.0j,
                            1.0 + 0.0j,
                        ],
                        [
                            0.0 + 0.0j,
                            0.0 + 0.0j,
                        ],
                    ]
                ],
                dtype=dtype,
                device=device,
            )
        )
        core_mid = torch.tensor(
            [
                [
                    [1.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j],
                ],
                [
                    [0.0 + 0.0j, 0.0 + 0.0j],
                    [0.0 + 0.0j, 1.0 + 0.0j],
                    [0.0 + 0.0j, 0.0 + 0.0j],
                ],
            ],
            dtype=dtype,
            device=device,
        )
        core3 = torch.tensor(
            [
                [
                    [1.0 + 0.0j],
                    [0.0 + 0.0j],
                    [0.0 + 0.0j],
                ],
                [
                    [0.0 + 0.0j],
                    [1.0 + 0.0j],
                    [0.0 + 0.0j],
                ],
            ],
            dtype=dtype,
            device=device,
        )

    cores = [core_1]
    for _ in range(nqubits - 2):
        cores.append(core_mid)
    cores.append(core3)
    return cores


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


def pulser_afm_sequence_grid(
    rows: int,
    columns: int,
    Omega_max: float,
    U: float,
    delta_0: float,
    delta_f: float,
    t_rise: float,
    t_fall: float,
    device: pulser.devices = pulser.devices.MockDevice,
):
    R_interatomic = device.rydberg_blockade_radius(U)
    reg = pulser.Register.rectangle(
        rows, columns, torch.tensor(R_interatomic), prefix="q"
    )

    return pulser_afm_sequence_from_register(
        reg,
        Omega_max=Omega_max,
        delta_0=delta_0,
        delta_f=delta_f,
        t_rise=t_rise,
        t_fall=t_fall,
    )


def pulser_quench_sequence_grid(nx: int, ny: int):
    def generate_square_lattice(nx, ny, R):
        coordinates = []
        for i in range(nx):
            for j in range(ny):
                x = i * R
                y = j * R
                coordinates.append((x, y))

        return coordinates

    # Hamiltonian parameters as ratios of J_max - i.e. hx/J_max, hz/J_max and t/J_max
    hx = 1.5  # hx/J_max
    hz = 0  # hz/J_max
    t = 1.5  # t/J_max

    # Set up pulser simulations
    R = 7  # Qubit seperation
    U = pulser.devices.AnalogDevice.interaction_coeff / R**6  # U_ij

    # Conversion from Rydberg Hamiltonian to Ising model
    NN_coeff = U / 4
    omega = 2 * hx * NN_coeff
    delta = -2 * hz * NN_coeff + 2 * U
    T = np.round(1000 * t / NN_coeff)

    # Set up qubit positions and register
    coords = generate_square_lattice(nx, ny, R)
    qubits = dict(enumerate(coords))

    reg = pulser.register.Register(qubits)
    seq = pulser.Sequence(reg, pulser.devices.AnalogDevice)
    seq.declare_channel("ising", "rydberg_global")

    # Add the main pulse to the pulse sequence
    simple_pulse = pulser.pulse.Pulse.ConstantPulse(T, omega, delta, 0)
    seq.add(simple_pulse, "ising")

    return seq


def pulser_XY_sequence_slm_mask(amplitude: float = 0.0, slm_masked_atoms: tuple = ()):
    """XY sequence with and without slm_masked atoms"""
    reg = pulser.Register.rectangle(3, 1, spacing=8.0, prefix="q")
    seq = pulser.Sequence(reg, pulser.MockDevice)
    seq.declare_channel("ch0", "mw_global")

    # State preparation using SLM mask
    if len(slm_masked_atoms) > 0:
        slm_masked_qubit_ids = [reg.qubit_ids[i] for i in slm_masked_atoms]
        seq.config_slm_mask(slm_masked_qubit_ids)
        masked_pulse = pulser.Pulse.ConstantDetuning(
            pulser.BlackmanWaveform(200, np.pi / 2), 0.0, 0
        )
        seq.add(masked_pulse, "ch0")

    # Simulation pulse
    simple_pulse = pulser.Pulse.ConstantPulse(500, amplitude, 0.0, 0)
    seq.add(simple_pulse, "ch0")

    return seq


def pulser_blackman(duration: float, area: float, phase: float = 0.0):
    """Sequence with just a single Blackman pulse"""
    reg = pulser.Register({"q0": (0, 0)})
    device = pulser.MockDevice
    seq = pulser.Sequence(reg, device)
    seq.declare_channel("ch0", "rydberg_global")

    pi2_wf = pulser.BlackmanWaveform(duration, area)
    pi_2 = pulser.Pulse.ConstantDetuning(pi2_wf, detuning=0, phase=phase)
    seq.add(pi_2, "ch0")

    return seq


def random_density_matrix(num_qubits: int):
    """Generates a random density matrix for a given number of qubits."""
    dim = 2**num_qubits

    # Generate a random complex matrix
    real_part = torch.randn(dim, dim)
    imag_part = torch.randn(dim, dim)
    A = real_part + 1j * imag_part

    # Create a Hermitian matrix
    state = A @ A.conj().T

    # Normalize the matrix
    return state / torch.trace(state)


def list_to_kron(list_tensors: list[torch.Tensor]):
    """Given a list of local torch tensor operators (nxn): [A1,A2,A3,...,An]

    convert to a matrix using the Kroneker product ⊗.

    Result: A1 ⊗ A2 ⊗ A3 ⊗ ... ⊗ An
    """
    return functools.reduce(torch.kron, list_tensors)
