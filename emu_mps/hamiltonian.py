"""
This file deals with creation of the MPO corresponding
to the Hamiltonian of a neutral atoms quantum processor.
"""

from typing import Iterator, cast

import torch
from emu_mps.mpo import MPO

dtype = torch.complex128


class Operators:
    id = torch.eye(2, dtype=dtype)
    id_3x3 = torch.eye(3, dtype=dtype)
    n = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=dtype)
    sx = torch.tensor([[0.0, 0.5], [0.5, 0.0]], dtype=dtype)
    sy = torch.tensor([[0.0, -0.5j], [0.5j, 0.0]], dtype=dtype)


class HamiltonianMPOFactors:
    """Creates the MPO factors of a two-body Hamiltonian.

    Fills the single qubit terms with zeros for modifying in-place.
    """

    def __init__(
        self,
        interaction_matrix: torch.Tensor,
        interaction_ops: torch.Tensor,
        dim: int = 2,
    ):
        self._validate_interaction_matrix(interaction_matrix)

        if dim not in (2, 3):
            raise ValueError(f"dim must be 2 or 3, got {dim}")
        self.dim = dim

        self.interaction_matrix = interaction_matrix.clone().cpu()
        self.interaction_ops = interaction_ops
        self.num_sites = self.interaction_matrix.shape[1]
        self.middle_site = self.num_sites // 2
        self.identity = Operators.id if self.dim == 2 else Operators.id_3x3

    @staticmethod
    def _validate_interaction_matrix(matrix: torch.Tensor) -> None:
        if matrix.ndim != 3:
            raise ValueError(
                "interaction_matrix must be a stack of 2-dimensional matrices."
            )
        if matrix.shape[1] != matrix.shape[2]:
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

    def first_factor(self) -> torch.Tensor:
        """Return the MPO factor for the first site."""
        has_right_interaction = self._has_right_interaction(site=0)

        left_bond_dim = 1
        right_bond_dim = 2 + sum(has_right_interaction)
        factor = self._empty_factor(left_bond_dim, right_bond_dim)

        factor[0, :, :, 1] = self.identity
        for i, b in enumerate(has_right_interaction):
            if b:
                factor[0, :2, :2, 2 + i] = self.interaction_ops[i]

        return factor

    def left_factor(self, n: int) -> torch.Tensor:
        """Return the MPO factor for site ``n`` in the left half of the chain
        except the first factor."""
        has_right_interaction = self._has_right_interaction(site=n)
        current_left_interactions, left_interactions_to_keep = (
            self._left_interaction_masks(n)
        )

        left_bonds_per_int = current_left_interactions.sum(dim=1)
        left_bond_dim = int(left_bonds_per_int.sum().item() + 2)
        right_bonds_per_int = left_interactions_to_keep.sum(dim=1) + torch.tensor(
            has_right_interaction
        )
        right_bond_dim = int(right_bonds_per_int.sum().item() + 2)
        factor = self._empty_factor(left_bond_dim, right_bond_dim)

        factor[0, :, :, 0] = self.identity
        factor[1, :, :, 1] = self.identity
        ind = 1
        for i, b in enumerate(has_right_interaction):
            ind += int(right_bonds_per_int[i])
            if b:
                factor[1, :2, :2, ind] = self.interaction_ops[i]

        coeffs = self._left_interaction_coefficients(n, current_left_interactions)
        ind = 0
        for i, bonds in enumerate(left_bonds_per_int):
            factor[ind + 2 : ind + bonds + 2, :2, :2, 0] = (
                coeffs[ind : ind + bonds] * self.interaction_ops[i]
            )
            ind += bonds

        i = 2
        j = 2
        for ind, current_left_interactions_per_int in enumerate(
            current_left_interactions
        ):
            for (
                current_left_interaction
            ) in current_left_interactions_per_int.nonzero().flatten():
                if left_interactions_to_keep[ind][current_left_interaction]:
                    factor[i, :, :, j] = self.identity
                    j += 1
                i += 1
            if has_right_interaction[ind]:
                j += 1
        return factor

    def middle_factor(self) -> torch.Tensor:
        """Return the MPO factor at the central site bridging both halves."""
        n = self.middle_site
        current_left_interactions, _ = self._left_interaction_masks(n)
        current_right_interactions, _ = self._right_interaction_masks(n)

        left_bonds_per_int = current_left_interactions.sum(dim=1)
        left_bond_dim = int(left_bonds_per_int.sum().item() + 2)

        right_bonds_per_int = current_right_interactions.sum(dim=1)
        right_bond_dim = int(right_bonds_per_int.sum().item() + 2)
        factor = self._empty_factor(left_bond_dim, right_bond_dim)

        factor[0, :, :, 0] = self.identity
        factor[1, :, :, 1] = self.identity

        coeffs = self._left_interaction_coefficients(n, current_left_interactions)
        ind = 0
        for i, bonds in enumerate(left_bonds_per_int):
            factor[ind + 2 : ind + bonds + 2, :2, :2, 0] = (
                coeffs[ind : ind + bonds] * self.interaction_ops[i]
            )
            ind += bonds

        coeffs = self._right_interaction_coefficients(n, current_right_interactions)
        ind = 0
        for i, bonds in enumerate(right_bonds_per_int):
            factor[1, :2, :2, ind + 2 : ind + bonds + 2] = coeffs[
                :, :, ind : ind + bonds
            ] * self.interaction_ops[i].unsqueeze(-1)
            ind += bonds

        coeffs = self._middle_interaction_coefficients(
            n, current_left_interactions, current_right_interactions
        )
        factor[2:, :, :, 2:] = coeffs * self.identity[None, ..., None]

        return factor

    def right_factor(self, n: int) -> torch.Tensor:
        """Return the MPO factor for site ``n`` in the right half of the chain
        except the last factor."""
        has_left_interaction = self._has_left_interaction(site=n)
        current_right_interactions, right_interactions_to_keep = (
            self._right_interaction_masks(site=n)
        )

        left_bonds_per_int = right_interactions_to_keep.sum(dim=1) + torch.tensor(
            has_left_interaction
        )
        left_bond_dim = int(left_bonds_per_int.sum().item() + 2)

        right_bonds_per_int = current_right_interactions.sum(dim=1)
        right_bond_dim = int(right_bonds_per_int.sum().item() + 2)

        factor = self._empty_factor(left_bond_dim, right_bond_dim)

        factor[0, :, :, 0] = self.identity
        factor[1, :, :, 1] = self.identity

        ind = 2
        for i, b in enumerate(has_left_interaction):
            if b:
                factor[ind, :2, :2, 0] = self.interaction_ops[i]
            ind += int(left_bonds_per_int[i])

        coeffs = self._right_interaction_coefficients(n, current_right_interactions)
        ind = 0
        for i, bonds in enumerate(right_bonds_per_int):
            factor[1, :2, :2, ind + 2 : ind + bonds + 2] = coeffs[
                :, :, ind : ind + bonds
            ] * self.interaction_ops[i].unsqueeze(-1)
            ind += bonds

        i = 2
        j = 2
        for ind, current_right_interactions_per_int in enumerate(
            current_right_interactions
        ):
            if has_left_interaction[ind]:
                i += 1
            for (
                current_right_interaction
            ) in current_right_interactions_per_int.nonzero().flatten():
                if right_interactions_to_keep[ind][current_right_interaction]:
                    factor[i, :, :, j] = self.identity
                    i += 1
                j += 1

        return factor

    def last_factor(self) -> torch.Tensor:
        """Return the MPO factor for the last site."""
        has_left_interaction = self._has_left_interaction(site=-1)

        left_bond_dim = 2 + sum(has_left_interaction)
        right_bond_dim = 1
        factor = self._empty_factor(left_bond_dim, right_bond_dim)
        factor[0, :, :, 0] = self.identity

        for i, b in enumerate(has_left_interaction):
            if b:
                coeff = self.interaction_matrix[i, 0, 1] if self.num_sites == 2 else 1
                factor[2 + i, :2, :2, 0] = coeff * self.interaction_ops[i]
        return factor

    def _has_right_interaction(self, site: int) -> list[bool]:
        """Returns a list of booleans, one for each interaction term"""
        return list(bool(mat[site, site + 1 :].any()) for mat in self.interaction_matrix)

    def _has_left_interaction(self, site: int) -> list[bool]:
        """Returns a list of booleans, one for each interaction term"""
        return list(bool(mat[site, :site].any()) for mat in self.interaction_matrix)

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
          with current/next sites
        - left_interactions_to_keep[i] tells whether that interaction channel
          remains active after this site
        """
        current_left_interactions = self.interaction_matrix[:, :site, site:].any(dim=2)
        left_interactions_to_keep = self.interaction_matrix[:, :site, site + 1 :].any(
            dim=2
        )
        return current_left_interactions, left_interactions_to_keep

    def _right_interaction_masks(self, site: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        For a site in the right half:
        - current_right_interactions[j] tells whether site j > site interacts
          with current/previous sites
        - right_interactions_to_keep[j] tells whether that interaction channel
          remains active before this site
        """
        current_right_interactions = self.interaction_matrix[
            :, site + 1 :, : site + 1
        ].any(dim=2)
        right_interactions_to_keep = self.interaction_matrix[:, site + 1 :, :site].any(
            dim=2
        )
        return current_right_interactions, right_interactions_to_keep

    def _left_interaction_coefficients(
        self, n: int, current_left_interactions: torch.Tensor
    ) -> torch.Tensor:
        return self.interaction_matrix[:, :n][current_left_interactions, n, None, None]

    def _right_interaction_coefficients(
        self, n: int, current_right_interactions: torch.Tensor
    ) -> torch.Tensor:
        return self.interaction_matrix[:, n + 1 :][
            None, None, current_right_interactions, n
        ]

    def _middle_interaction_coefficients(
        self,
        n: int,
        current_left_interactions: torch.Tensor,
        current_right_interactions: torch.Tensor,
    ) -> torch.Tensor:
        return cast(
            torch.Tensor,
            torch.block_diag(
                *tuple(
                    self.interaction_matrix[i, :n, n + 1 :][
                        current_left_interactions[i], :
                    ][:, current_right_interactions[i]]
                    for i in range(self.interaction_matrix.shape[0])
                )
            )[:, None, None, :],
        )


def make_H(
    *,
    interaction_matrix: torch.Tensor,  # depends on Hamiltonian Type
    interaction_ops: torch.Tensor,
    dim: int = 2,
    num_gpus_to_use: int | None,
) -> MPO:
    r"""
    Constructs and returns a Matrix Product Operator (MPO) representing the
    neutral atoms Hamiltonian, parameterized by `omega`, `delta`, and `phi`.

    The linear term of the Hamiltonian is
    H_0 = ∑ⱼΩⱼ[cos(ϕⱼ)σˣⱼ + sin(ϕⱼ)σʸⱼ] - ∑ⱼΔⱼnⱼ

    The Interaction term is given by:
    H_int = H_0
      + ∑ₖ∑ᵢ﹥ⱼinteraction_matrix[k,i,j] interaction_ops[k]_i interaction_ops[k]_j

    If noise is considered, the Hamiltonian includes an additional term to
    support the Monte Carlo WaveFunction algorithm:
    H = H_int - 0.5i∑ₘ ∑ᵤ Lₘᵘ⁺ Lₘᵘ
    where Lₘᵘ are the Lindblad operators representing the noise,
    m for noise channel and u for the number of atoms

    make_H constructs an MPO of the appropriate size, but the single qubit
    terms are left at zero.
    To fill in the appropriate values, call update_H

    Args:
        interaction_matrix (torch.Tensor): The interaction matrix describing
        the interactions between qubits.
        interaction_ops: the interaction operator to use, see above.
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

    return MPO(
        list(HamiltonianMPOFactors(interaction_matrix, interaction_ops, dim=dim)),
        num_gpus_to_use=num_gpus_to_use,
    )


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

    if not (noise.shape == (2, 2) or noise.shape == (3, 3)):
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
