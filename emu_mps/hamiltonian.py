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
    id_3x3 = torch.eye(3, dtype=dtype)
    n = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)
    sx = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=dtype)
    sy = torch.tensor([[0.0, -0.5j], [0.5j, 0.0]], dtype=dtype)


class HamiltonianMPOFactors(ABC):
    """Abstract class for MPO factors of a two-body Hamiltonian.

    Subclasses implement the local MPO tensor at each position in the chain:
    first site, left half, middle, right half, and last site.
    """

    def __init__(self, interaction_matrix: torch.Tensor, dim: int = 2):
        self._validate_interaction_matrix(interaction_matrix)

        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        self.dim = dim

        self.interaction_matrix = interaction_matrix.clone()
        self.interaction_matrix.fill_diagonal_(0.0)  # or assert
        self.num_sites = self.interaction_matrix.shape[0]
        self.middle_site = self.num_sites // 2
        self.identity = Operators.id if self.dim == 2 else Operators.id_3x3

    @staticmethod
    def _validate_interaction_matrix(matrix: torch.Tensor) -> None:
        if matrix.ndim != 2:
            raise ValueError("interaction_matrix must be 2-dimensional.")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("interaction_matrix must be square.")

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield the full ordered list of MPO factors for the Hamiltonian."""
        yield self.first_factor()

        for n in range(1, self.middle_site):
            yield self.left_factor(n)

        if self.num_sites >= 3:
            yield self.middle_factor()

        for n in range(self.middle_site + 1, self.num_sites - 1):
            yield self.right_factor(n)

        yield self.last_factor()

    @abstractmethod
    def first_factor(self) -> torch.Tensor:
        """Return the MPO factor for the first site."""

    @abstractmethod
    def left_factor(self, n: int) -> torch.Tensor:
        """Return the MPO factor for site ``n`` in the left half of the chain."""

    @abstractmethod
    def middle_factor(self) -> torch.Tensor:
        """Return the MPO factor at the central site bridging both halves."""

    @abstractmethod
    def right_factor(self, n: int) -> torch.Tensor:
        """Return the MPO factor for site ``n`` in the right half of the chain."""

    @abstractmethod
    def last_factor(self) -> torch.Tensor:
        """Return the MPO factor for the last site."""

    def _has_right_interaction(self, site: int) -> bool:
        return bool(self.interaction_matrix[site, site + 1 :].any())

    def _has_left_interaction(self, site: int) -> bool:
        return bool(self.interaction_matrix[site, :site].any())

    def _empty_factor(self, left_bond_dim: int, right_bond_dim: int) -> torch.Tensor:
        return torch.zeros(
            left_bond_dim,
            self.dim,
            self.dim,
            right_bond_dim,
            dtype=dtype,
        )

    def _left_interaction_masks(self, site: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For a site in the left half:
        - current_left_interactions[i] tells whether site i < site interacts
          with current/future sites
        - left_interactions_to_keep[i] tells whether that interaction channel
          remains active after this site
        """
        current_left_interactions = self.interaction_matrix[:site, site:].any(dim=1)
        left_interactions_to_keep = self.interaction_matrix[:site, site + 1 :].any(dim=1)
        return current_left_interactions, left_interactions_to_keep

    def _right_interaction_masks(self, site: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For a site in the right half:
        - current_right_interactions[j] tells whether site j > site interacts
          with current/past sites
        - right_interactions_to_keep[j] tells whether that interaction channel
          remains active before this site
        """
        current_right_interactions = self.interaction_matrix[site + 1 :, : site + 1].any(
            dim=1
        )
        right_interactions_to_keep = self.interaction_matrix[site + 1 :, :site].any(dim=1)
        return current_right_interactions, right_interactions_to_keep

    def _left_interaction_coefficients(
        self, n: int, current_left_interactions: torch.Tensor
    ) -> torch.Tensor:
        return self.interaction_matrix[:n][current_left_interactions, n, None, None]

    def _right_interaction_coefficients(
        self, n: int, current_right_interactions: torch.Tensor
    ) -> torch.Tensor:
        return self.interaction_matrix[n + 1 :][None, None, current_right_interactions, n]

    def _middle_interaction_coefficients(
        self,
        n: int,
        current_left_interactions: torch.Tensor,
        current_right_interactions: torch.Tensor,
    ) -> torch.Tensor:
        return self.interaction_matrix[:n, n + 1 :][current_left_interactions, :][
            :, None, None, current_right_interactions
        ]


class RydbergHamiltonianMPOFactors(HamiltonianMPOFactors):
    def first_factor(self) -> torch.Tensor:
        has_right_interaction = self._has_right_interaction(site=0)

        left_bond_dim = 1
        right_bond_dim = 3 if has_right_interaction else 2
        factor = self._empty_factor(left_bond_dim, right_bond_dim)

        factor[0, :, :, 1] = self.identity
        if has_right_interaction:
            factor[0, :2, :2, 2] = Operators.n

        return factor

    def left_factor(self, n: int) -> torch.Tensor:
        has_right_interaction = self._has_right_interaction(site=n)
        current_left_interactions, left_interactions_to_keep = (
            self._left_interaction_masks(n)
        )

        left_bond_dim = int(current_left_interactions.sum().item() + 2)
        right_bond_dim = int(
            left_interactions_to_keep.sum().item() + int(has_right_interaction) + 2
        )
        factor = self._empty_factor(left_bond_dim, right_bond_dim)

        factor[0, :, :, 0] = self.identity
        factor[1, :, :, 1] = self.identity
        if has_right_interaction:
            factor[1, :2, :2, -1] = Operators.n

        coeff = self._left_interaction_coefficients(n, current_left_interactions)
        factor[2:, :2, :2, 0] = coeff * Operators.n

        i = 2
        j = 2
        for current_left_interaction in current_left_interactions.nonzero().flatten():
            if left_interactions_to_keep[current_left_interaction]:
                factor[i, :, :, j] = self.identity
                j += 1
            i += 1
        return factor

    def middle_factor(self) -> torch.Tensor:
        n = self.middle_site
        current_left_interactions, _ = self._left_interaction_masks(n)
        current_right_interactions, _ = self._right_interaction_masks(n)

        left_bond_dim = int(current_left_interactions.sum().item() + 2)
        right_bond_dim = int(current_right_interactions.sum().item() + 2)
        factor = self._empty_factor(left_bond_dim, right_bond_dim)

        factor[0, :, :, 0] = self.identity
        factor[1, :, :, 1] = self.identity

        coeff = self._left_interaction_coefficients(n, current_left_interactions)
        factor[2:, :2, :2, 0] = coeff * Operators.n

        coeff = self.interaction_matrix[n + 1 :][
            None, None, current_right_interactions, n
        ]
        factor[1, :2, :2, 2:] = coeff * Operators.n.unsqueeze(-1)

        coeff = self._middle_interaction_coefficients(
            n, current_left_interactions, current_right_interactions
        )
        factor[2:, :, :, 2:] = coeff * self.identity[None, ..., None]

        return factor

    def right_factor(self, n: int) -> torch.Tensor:
        has_left_interaction = self._has_left_interaction(site=n)
        current_right_interactions, right_interactions_to_keep = (
            self._right_interaction_masks(site=n)
        )

        left_bond_dim = int(
            right_interactions_to_keep.sum().item() + int(has_left_interaction) + 2
        )
        right_bond_dim = int(current_right_interactions.sum().item() + 2)
        factor = self._empty_factor(left_bond_dim, right_bond_dim)

        factor[0, :, :, 0] = self.identity
        factor[1, :, :, 1] = self.identity
        if has_left_interaction:
            factor[2, :2, :2, 0] = Operators.n

        coeff = self.interaction_matrix[n + 1 :][
            None, None, current_right_interactions, n
        ]
        factor[1, :2, :2, 2:] = coeff * Operators.n.unsqueeze(-1)

        i = 3 if has_left_interaction else 2
        j = 2
        for current_right_interaction in current_right_interactions.nonzero().flatten():
            if right_interactions_to_keep[current_right_interaction]:
                factor[i, :, :, j] = self.identity
                i += 1
            j += 1
        return factor

    def last_factor(self) -> torch.Tensor:
        has_left_interaction = self._has_left_interaction(site=-1)

        left_bond_dim = 3 if has_left_interaction else 2
        right_bond_dim = 1
        factor = self._empty_factor(left_bond_dim, right_bond_dim)
        factor[0, :, :, 0] = self.identity
        if has_left_interaction:
            coeff = self.interaction_matrix[0, 1] if self.num_sites == 2 else 1
            factor[2, :2, :2, 0] = coeff * Operators.n

        return factor


class XYHamiltonianMPOFactors(HamiltonianMPOFactors):
    def first_factor(self) -> torch.Tensor:
        has_right_interaction = self._has_right_interaction(site=0)

        left_bond_dim = 1
        right_bond_dim = 4 if has_right_interaction else 2
        factor = self._empty_factor(left_bond_dim, right_bond_dim)
        factor[0, :, :, 1] = self.identity
        if has_right_interaction:
            factor[0, :2, :2, 2] = Operators.sx
            factor[0, :2, :2, 3] = Operators.sy

        return factor

    def left_factor(self, n: int) -> torch.Tensor:
        has_right_interaction = self._has_right_interaction(site=n)
        current_left_interactions, left_interactions_to_keep = (
            self._left_interaction_masks(n)
        )

        left_bond_dim = int(2 * current_left_interactions.sum().item() + 2)
        right_bond_dim = int(
            2 * left_interactions_to_keep.sum().item()
            + 2 * int(has_right_interaction)
            + 2
        )
        factor = self._empty_factor(left_bond_dim, right_bond_dim)

        factor[0, :, :, 0] = self.identity
        factor[1, :, :, 1] = self.identity
        if has_right_interaction:
            factor[1, :2, :2, -2] = Operators.sx
            factor[1, :2, :2, -1] = Operators.sy

        coeff = self._left_interaction_coefficients(n, current_left_interactions)
        factor[2::2, :2, :2, 0] = coeff * 2 * Operators.sx
        factor[3::2, :2, :2, 0] = coeff * 2 * Operators.sy

        i = 2
        j = 2
        for current_left_interaction in current_left_interactions.nonzero().flatten():
            if left_interactions_to_keep[current_left_interaction]:
                factor[i, :, :, j] = self.identity
                factor[i + 1, :, :, j + 1] = self.identity
                j += 2
            i += 2
        return factor

    def middle_factor(self) -> torch.Tensor:
        n = self.middle_site
        current_left_interactions, _ = self._left_interaction_masks(n)
        current_right_interactions, _ = self._right_interaction_masks(n)

        left_bond_dim = int(2 * current_left_interactions.sum().item() + 2)
        right_bond_dim = int(2 * current_right_interactions.sum().item() + 2)

        factor = self._empty_factor(left_bond_dim, right_bond_dim)

        factor[0, :, :, 0] = self.identity
        factor[1, :, :, 1] = self.identity

        coeff = self._left_interaction_coefficients(n, current_left_interactions)
        factor[2::2, :2, :2, 0] = coeff * 2 * Operators.sx
        factor[3::2, :2, :2, 0] = coeff * 2 * Operators.sy

        coeff = self.interaction_matrix[n + 1 :][
            None, None, current_right_interactions, n
        ]
        factor[1, :2, :2, 2::2] = coeff * 2 * Operators.sx.unsqueeze(-1)
        factor[1, :2, :2, 3::2] = coeff * 2 * Operators.sy.unsqueeze(-1)

        coeff = self._middle_interaction_coefficients(
            n, current_left_interactions, current_right_interactions
        )
        factor[2::2, :, :, 2::2] = coeff * 2 * self.identity[None, ..., None]
        factor[3::2, :, :, 3::2] = coeff * 2 * self.identity[None, ..., None]

        return factor

    def right_factor(self, n: int) -> torch.Tensor:
        has_left_interaction = self._has_left_interaction(site=n)
        current_right_interactions, right_interactions_to_keep = (
            self._right_interaction_masks(site=n)
        )

        left_bond_dim = int(
            2 * right_interactions_to_keep.sum().item()
            + 2 * int(has_left_interaction)
            + 2
        )
        right_bond_dim = int(2 * current_right_interactions.sum().item() + 2)
        factor = self._empty_factor(left_bond_dim, right_bond_dim)

        factor[0, :, :, 0] = self.identity
        factor[1, :, :, 1] = self.identity
        if has_left_interaction:
            factor[2, :2, :2, 0] = Operators.sx
            factor[3, :2, :2, 0] = Operators.sy

        coeff = self.interaction_matrix[n + 1 :][
            None, None, current_right_interactions, n
        ]
        factor[1, :2, :2, 2::2] = coeff * 2 * Operators.sx.unsqueeze(-1)
        factor[1, :2, :2, 3::2] = coeff * 2 * Operators.sy.unsqueeze(-1)

        i = 4 if has_left_interaction else 2
        j = 2
        for current_right_interaction in current_right_interactions.nonzero().flatten():
            if right_interactions_to_keep[current_right_interaction]:
                factor[i, :, :, j] = self.identity
                factor[i + 1, :, :, j + 1] = self.identity
                i += 2
            j += 2
        return factor

    def last_factor(self) -> torch.Tensor:
        has_left_interaction = self._has_left_interaction(site=-1)

        left_bond_dim = 4 if has_left_interaction else 2
        right_bond_dim = 1
        factor = self._empty_factor(left_bond_dim, right_bond_dim)
        factor[0, :, :, 0] = self.identity
        if has_left_interaction:
            coeff = 2 * self.interaction_matrix[0, 1] if self.num_sites == 2 else 1
            factor[2, :2, :2, 0] = coeff * Operators.sx
            factor[3, :2, :2, 0] = coeff * Operators.sy

        return factor


def make_H(
    *,
    interaction_matrix: torch.Tensor,  # depends on Hamiltonian Type
    hamiltonian_type: HamiltonianType,
    dim: int = 2,
    num_gpus_to_use: int | None,
) -> MPO:
    r"""
    Constructs and returns a Matrix Product Operator (MPO) representing the
    neutral atoms Hamiltonian, parameterized by `omega`, `delta`, and `phi`.

    The linear terms of the Hamiltonian is
    H_0 = ∑ⱼΩⱼ[cos(ϕⱼ)σˣⱼ + sin(ϕⱼ)σʸⱼ] - ∑ⱼΔⱼnⱼ

    The Rydberg Hamiltonian H is given by:
    H_R = H_0 + ∑ᵢ﹥ⱼC⁶/rᵢⱼ⁶ nᵢnⱼ

    The XY Hamiltonian H is given by:
    H_XY = H_0 + ∑ᵢ﹥ⱼC₃/rᵢⱼ³ 2(SˣᵢSˣⱼ + SʸᵢSʸⱼ)

    If noise is considered, the Hamiltonian includes an additional term to
    support the Monte Carlo WaveFunction algorithm:
    H = H_{R|XY} - 0.5i∑ₘ ∑ᵤ Lₘᵘ⁺ Lₘᵘ
    where Lₘᵘ are the Lindblad operators representing the noise,
    m for noise channel and u for the number of atoms

    make_H constructs an MPO of the appropriate size, but the single qubit
    terms are left at zero.
    To fill in the appropriate values, call update_H

    Args:
        interaction_matrix (torch.Tensor): The interaction matrix describing
        the interactions between qubits.
        hamiltonian_type: whether to use XY or Rydberg interation
        dim: dimension of the basis (2 or 3)
        num_gpus_to_use (int): how many gpus to put the Hamiltonian on.
        See utils.assign_devices
    Returns:
        MPO: A Matrix Product Operator (MPO) representing the specified
        Hamiltonian.

    Note:
    For more information about the Hamiltonian and its usage, refer to the
    [Pulser documentation](https://pulser.readthedocs.io/en/stable/conventions.html#hamiltonians).

    """

    if hamiltonian_type == HamiltonianType.Rydberg:
        return MPO(
            list(RydbergHamiltonianMPOFactors(interaction_matrix, dim=dim)),
            num_gpus_to_use=num_gpus_to_use,
        )

    if hamiltonian_type == HamiltonianType.XY:
        return MPO(
            list(XYHamiltonianMPOFactors(interaction_matrix, dim=dim)),
            num_gpus_to_use=num_gpus_to_use,
        )

    raise ValueError(f"Unsupported hamiltonian_type: {hamiltonian_type}")


def update_H(
    hamiltonian: MPO,
    omega: torch.Tensor,
    delta: torch.Tensor,
    phi: torch.Tensor,
    noise: torch.Tensor,
) -> None:
    """
    The single qubit operators in the Hamiltonian,
    corresponding to the omega, delta, phi parameters and the aggregated
    Lindblad operators have a well-determined position in the factors of
    the Hamiltonian.
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
        This can be computed using the `compute_noise_from_lindbladians`
        function.
        Defaults to a zero tensor.
    """

    if noise.shape not in {(2, 2), (3, 3)}:
        raise ValueError(
            f"noise must have shape (2, 2) or (3, 3), got {tuple(noise.shape)}"
        )
    nqubits = omega.size(dim=0)

    a = torch.tensordot(omega * torch.cos(phi), Operators.sx, dims=0)
    c = torch.tensordot(delta, Operators.n, dims=0)
    b = torch.tensordot(omega * torch.sin(phi), Operators.sy, dims=0)

    factors = hamiltonian.factors

    single_qubit_terms = torch.stack(nqubits * [noise])

    single_qubit_terms[:, :2, :2] += a + b - c

    factors[0][0, :, :, 0] = single_qubit_terms[0]
    for i in range(1, nqubits):
        factors[i][1, :, :, 0] = single_qubit_terms[i]
