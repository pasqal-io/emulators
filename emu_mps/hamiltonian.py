"""
This file deals with creation of the MPO corresponding
to the Hamiltonian of a neutral atoms quantum processor.
"""

from abc import abstractmethod, ABC
from typing import Iterator

from emu_base import HamiltonianType
import torch
from emu_mps.mpo import MPO

dtype = torch.complex128


class Operators:
    id = torch.eye(2, dtype=dtype)
    n = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)
    creation = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    sx = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=dtype)
    sy = torch.tensor([[0.0, -0.5j], [0.5j, 0.0]], dtype=dtype)
    pu = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)


class HamiltonianMPOFactors(ABC):
    def __init__(self, interaction_matrix: torch.Tensor):
        assert interaction_matrix.ndim == 2, "interaction matrix is not a matrix"
        assert (
            interaction_matrix.shape[0] == interaction_matrix.shape[1]
        ), "interaction matrix is not square"

        self.interaction_matrix = interaction_matrix.clone()
        self.interaction_matrix.fill_diagonal_(0.0)  # or assert
        self.qubit_count = self.interaction_matrix.shape[0]
        self.middle = self.qubit_count // 2

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield self.first_factor()

        for n in range(1, self.middle):
            yield self.left_factor(n)

        if self.qubit_count >= 3:
            yield self.middle_factor()

        for n in range(self.middle + 1, self.qubit_count - 1):
            yield self.right_factor(n)

        yield self.last_factor()

    @abstractmethod
    def first_factor(self) -> torch.Tensor:
        pass

    @abstractmethod
    def left_factor(self, n: int) -> torch.Tensor:
        pass

    @abstractmethod
    def middle_factor(self) -> torch.Tensor:
        pass

    @abstractmethod
    def right_factor(self, n: int) -> torch.Tensor:
        pass

    @abstractmethod
    def last_factor(self) -> torch.Tensor:
        pass


class RydbergHamiltonianMPOFactors(HamiltonianMPOFactors):
    def first_factor(self) -> torch.Tensor:
        has_right_interaction = self.interaction_matrix[0, 1:].any()
        fac = torch.zeros(1, 2, 2, 3 if has_right_interaction else 2, dtype=dtype)
        fac[0, :, :, 1] = Operators.id
        if has_right_interaction:
            fac[0, :, :, 2] = Operators.n

        return fac

    def left_factor(self, n: int) -> torch.Tensor:
        has_right_interaction = self.interaction_matrix[n, n + 1 :].any()
        current_left_interactions = self.interaction_matrix[:n, n:].any(dim=1)
        left_interactions_to_keep = self.interaction_matrix[:n, n + 1 :].any(dim=1)

        fac = torch.zeros(
            int(current_left_interactions.sum().item() + 2),
            2,
            2,
            int(left_interactions_to_keep.sum().item() + int(has_right_interaction) + 2),
            dtype=dtype,
        )

        fac[0, :, :, 0] = Operators.id
        fac[1, :, :, 1] = Operators.id
        if has_right_interaction:
            fac[1, :, :, -1] = Operators.n

        fac[2:, :, :, 0] = (
            self.interaction_matrix[:n][current_left_interactions, n, None, None]
            * Operators.n
        )

        i = 2
        j = 2
        for current_left_interaction in current_left_interactions.nonzero().flatten():
            if left_interactions_to_keep[current_left_interaction]:
                fac[i, :, :, j] = Operators.id
                j += 1
            i += 1
        return fac

    def middle_factor(self) -> torch.Tensor:
        n = self.middle
        current_left_interactions = self.interaction_matrix[:n, n:].any(dim=1)
        current_right_interactions = self.interaction_matrix[n + 1 :, : n + 1].any(dim=1)

        fac = torch.zeros(
            int(current_left_interactions.sum().item() + 2),
            2,
            2,
            int(current_right_interactions.sum().item() + 2),
            dtype=dtype,
        )

        fac[0, :, :, 0] = Operators.id
        fac[1, :, :, 1] = Operators.id

        fac[2:, :, :, 0] = (
            self.interaction_matrix[:n][current_left_interactions, n, None, None]
            * Operators.n
        )

        fac[1, :, :, 2:] = self.interaction_matrix[n + 1 :][
            None, None, current_right_interactions, n
        ] * Operators.n.unsqueeze(-1)

        fac[2:, :, :, 2:] = (
            self.interaction_matrix[:n, n + 1 :][current_left_interactions, :][
                :, None, None, current_right_interactions
            ]
            * Operators.id[None, ..., None]
        )

        return fac

    def right_factor(self, n: int) -> torch.Tensor:
        has_left_interaction = self.interaction_matrix[n, :n].any()
        current_right_interactions = self.interaction_matrix[n + 1 :, : n + 1].any(dim=1)
        right_interactions_to_keep = self.interaction_matrix[n + 1 :, :n].any(dim=1)

        fac = torch.zeros(
            int(right_interactions_to_keep.sum().item() + int(has_left_interaction) + 2),
            2,
            2,
            int(current_right_interactions.sum().item() + 2),
            dtype=dtype,
        )

        fac[0, :, :, 0] = Operators.id
        fac[1, :, :, 1] = Operators.id
        if has_left_interaction:
            fac[2, :, :, 0] = Operators.n

        fac[1, :, :, 2:] = self.interaction_matrix[n + 1 :][
            None, None, current_right_interactions, n
        ] * Operators.n.unsqueeze(-1)

        i = 3 if has_left_interaction else 2
        j = 2
        for current_right_interaction in current_right_interactions.nonzero().flatten():
            if right_interactions_to_keep[current_right_interaction]:
                fac[i, :, :, j] = Operators.id
                i += 1
            j += 1
        return fac

    def last_factor(self) -> torch.Tensor:
        has_left_interaction = self.interaction_matrix[-1, :-1].any()
        fac = torch.zeros(3 if has_left_interaction else 2, 2, 2, 1, dtype=dtype)
        fac[0, :, :, 0] = Operators.id
        if has_left_interaction:
            if self.qubit_count >= 3:
                fac[2, :, :, 0] = Operators.n
            else:
                fac[2, :, :, 0] = self.interaction_matrix[0, 1] * Operators.n

        return fac


class XYHamiltonianMPOFactors(HamiltonianMPOFactors):
    def first_factor(self) -> torch.Tensor:
        has_right_interaction = self.interaction_matrix[0, 1:].any()
        fac = torch.zeros(1, 2, 2, 4 if has_right_interaction else 2, dtype=dtype)
        fac[0, :, :, 1] = Operators.id
        if has_right_interaction:
            fac[0, :, :, 2] = Operators.creation
            fac[0, :, :, 3] = Operators.creation.T

        return fac

    def left_factor(self, n: int) -> torch.Tensor:
        has_right_interaction = self.interaction_matrix[n, n + 1 :].any()
        current_left_interactions = self.interaction_matrix[:n, n:].any(dim=1)
        left_interactions_to_keep = self.interaction_matrix[:n, n + 1 :].any(dim=1)

        fac = torch.zeros(
            int(2 * current_left_interactions.sum().item() + 2),
            2,
            2,
            int(
                2 * left_interactions_to_keep.sum().item()
                + 2 * int(has_right_interaction)
                + 2
            ),
            dtype=dtype,
        )

        fac[0, :, :, 0] = Operators.id
        fac[1, :, :, 1] = Operators.id
        if has_right_interaction:
            fac[1, :, :, -2] = Operators.creation
            fac[1, :, :, -1] = Operators.creation.T

        fac[2::2, :, :, 0] = (
            self.interaction_matrix[:n][current_left_interactions, n, None, None]
            * Operators.creation.T
        )
        fac[3::2, :, :, 0] = (
            self.interaction_matrix[:n][current_left_interactions, n, None, None]
            * Operators.creation
        )

        i = 2
        j = 2
        for current_left_interaction in current_left_interactions.nonzero().flatten():
            if left_interactions_to_keep[current_left_interaction]:
                fac[i, :, :, j] = Operators.id
                fac[i + 1, :, :, j + 1] = Operators.id
                j += 2
            i += 2
        return fac

    def middle_factor(self) -> torch.Tensor:
        n = self.middle
        current_left_interactions = self.interaction_matrix[:n, n:].any(dim=1)
        current_right_interactions = self.interaction_matrix[n + 1 :, : n + 1].any(dim=1)

        fac = torch.zeros(
            int(2 * current_left_interactions.sum().item() + 2),
            2,
            2,
            int(2 * current_right_interactions.sum().item() + 2),
            dtype=dtype,
        )

        fac[0, :, :, 0] = Operators.id
        fac[1, :, :, 1] = Operators.id

        fac[2::2, :, :, 0] = (
            self.interaction_matrix[:n][current_left_interactions, n, None, None]
            * Operators.creation.T
        )
        fac[3::2, :, :, 0] = (
            self.interaction_matrix[:n][current_left_interactions, n, None, None]
            * Operators.creation
        )

        fac[1, :, :, 2::2] = self.interaction_matrix[n + 1 :][
            None, None, current_right_interactions, n
        ] * Operators.creation.unsqueeze(-1)
        fac[1, :, :, 3::2] = self.interaction_matrix[n + 1 :][
            None, None, current_right_interactions, n
        ] * Operators.creation.T.unsqueeze(-1)

        fac[2::2, :, :, 2::2] = (
            self.interaction_matrix[:n, n + 1 :][current_left_interactions, :][
                :, None, None, current_right_interactions
            ]
            * Operators.id[None, ..., None]
        )
        fac[3::2, :, :, 3::2] = (
            self.interaction_matrix[:n, n + 1 :][current_left_interactions, :][
                :, None, None, current_right_interactions
            ]
            * Operators.id[None, ..., None]
        )

        return fac

    def right_factor(self, n: int) -> torch.Tensor:
        has_left_interaction = self.interaction_matrix[n, :n].any()
        current_right_interactions = self.interaction_matrix[n + 1 :, : n + 1].any(dim=1)
        right_interactions_to_keep = self.interaction_matrix[n + 1 :, :n].any(dim=1)

        fac = torch.zeros(
            int(
                2 * right_interactions_to_keep.sum().item()
                + 2 * int(has_left_interaction)
                + 2
            ),
            2,
            2,
            int(2 * current_right_interactions.sum().item() + 2),
            dtype=dtype,
        )

        fac[0, :, :, 0] = Operators.id
        fac[1, :, :, 1] = Operators.id
        if has_left_interaction:
            fac[2, :, :, 0] = Operators.creation.T
            fac[3, :, :, 0] = Operators.creation

        fac[1, :, :, 2::2] = self.interaction_matrix[n + 1 :][
            None, None, current_right_interactions, n
        ] * Operators.creation.unsqueeze(-1)
        fac[1, :, :, 3::2] = self.interaction_matrix[n + 1 :][
            None, None, current_right_interactions, n
        ] * Operators.creation.T.unsqueeze(-1)

        i = 4 if has_left_interaction else 2
        j = 2
        for current_right_interaction in current_right_interactions.nonzero().flatten():
            if right_interactions_to_keep[current_right_interaction]:
                fac[i, :, :, j] = Operators.id
                fac[i + 1, :, :, j + 1] = Operators.id
                i += 2
            j += 2
        return fac

    def last_factor(self) -> torch.Tensor:
        has_left_interaction = self.interaction_matrix[-1, :-1].any()
        fac = torch.zeros(4 if has_left_interaction else 2, 2, 2, 1, dtype=dtype)
        fac[0, :, :, 0] = Operators.id
        if has_left_interaction:
            if self.qubit_count >= 3:
                fac[2, :, :, 0] = Operators.creation.T
                fac[3, :, :, 0] = Operators.creation
            else:
                fac[2, :, :, 0] = self.interaction_matrix[0, 1] * Operators.creation.T
                fac[3, :, :, 0] = self.interaction_matrix[0, 1] * Operators.creation

        return fac


def make_H(
    *,
    interaction_matrix: torch.Tensor,  # depends on Hamiltonian Type
    hamiltonian_type: HamiltonianType,
    num_gpus_to_use: int | None,
) -> MPO:
    r"""
    Constructs and returns a Matrix Product Operator (MPO) representing the
    neutral atoms Hamiltonian, parameterized by `omega`, `delta`, and `phi`.

    The Hamiltonian H is given by:
    H = ∑ⱼΩⱼ[cos(ϕⱼ)σˣⱼ + sin(ϕⱼ)σʸⱼ] - ∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼC⁶/rᵢⱼ⁶ nᵢnⱼ

    If noise is considered, the Hamiltonian includes an additional term to support
    the Monte Carlo WaveFunction algorithm:
    H = ∑ⱼΩⱼ[cos(ϕⱼ)σˣⱼ + sin(ϕⱼ)σʸⱼ] - ∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼC⁶/rᵢⱼ⁶ nᵢnⱼ - 0.5i∑ₘ ∑ᵤ Lₘᵘ⁺ Lₘᵘ
    where Lₘᵘ are the Lindblad operators representing the noise, m for noise channel
    and u for the number of atoms

    make_H constructs an MPO of the appropriate size, but the single qubit terms are left at zero.
    To fill in the appropriate values, call update_H

    Args:
        interaction_matrix (torch.Tensor): The interaction matrix describing the interactions
        between qubits.
        num_gpus_to_use (int): how many gpus to put the Hamiltonian on. See utils.assign_devices
    Returns:
        MPO: A Matrix Product Operator (MPO) representing the specified Hamiltonian.

    Note:
    For more information about the Hamiltonian and its usage, refer to the
    [Pulser documentation](https://pulser.readthedocs.io/en/stable/conventions.html#hamiltonians).

    """
    if hamiltonian_type == HamiltonianType.Rydberg:
        return MPO(
            list(RydbergHamiltonianMPOFactors(interaction_matrix)),
            num_gpus_to_use=num_gpus_to_use,
        )

    if hamiltonian_type == HamiltonianType.XY:
        return MPO(
            list(XYHamiltonianMPOFactors(interaction_matrix)),
            num_gpus_to_use=num_gpus_to_use,
        )

    raise ValueError(f"Unsupported hamiltonian type {hamiltonian_type}")


def update_H(
    hamiltonian: MPO,
    omega: torch.Tensor,
    delta: torch.Tensor,
    phi: torch.Tensor,
    noise: torch.Tensor = torch.zeros(2, 2),
) -> None:
    """
    The single qubit operators in the Hamiltonian,
    corresponding to the omega, delta, phi parameters and the aggregated Lindblad operators
    have a well-determined position in the factors of the Hamiltonian.
    This function updates this part of the factors to update the
    Hamiltonian with new parameters without rebuilding the entire thing.
    See make_H for details about the Hamiltonian.

    This is an in-place operation, so this function returns nothing.

    Args:
        omega (torch.Tensor): Rabi frequency Ωⱼ for each qubit.
        delta (torch.Tensor): The detuning value Δⱼ for each qubit.
        phi (torch.Tensor): The phase ϕⱼ corresponding to each qubit.
        noise (torch.Tensor, optional): The single-qubit noise
        term -0.5i∑ⱼLⱼ†Lⱼ applied to all qubits.
        This can be computed using the `compute_noise_from_lindbladians` function.
        Defaults to a zero tensor.
    """

    assert noise.shape == (2, 2)
    nqubits = omega.size(dim=0)

    a = torch.tensordot(omega * torch.cos(phi), Operators.sx, dims=0)
    c = torch.tensordot(delta, Operators.pu, dims=0)
    b = torch.tensordot(omega * torch.sin(phi), Operators.sy, dims=0)

    single_qubit_terms = a + b - c + noise
    factors = hamiltonian.factors

    factors[0][0, :, :, 0] = single_qubit_terms[0]
    for i in range(1, nqubits):
        factors[i][1, :, :, 0] = single_qubit_terms[i]
