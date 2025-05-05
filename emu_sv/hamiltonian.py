import torch
from emu_sv.state_vector import StateVector


class RydbergHamiltonian:
    """Represents the Rydberg Hamiltonian for a system of interacting qubits
    driven by laser fields, including detuning, phase, and interaction terms.

    The Hamiltonian is defined as:

        H = ∑ⱼ (Ωⱼ/2)[cos(ϕⱼ) σˣⱼ + sin(ϕⱼ) σʸⱼ] - ∑ⱼ Δⱼ nⱼ + ∑_{i>j} Uᵢⱼ nᵢ nⱼ

    where:
        - Ωⱼ is the Rabi frequency on qubit j,
        - Δⱼ is the detuning on qubit j,
        - ϕⱼ is the laser phase on qubit j,
        - Uᵢⱼ is the interaction strength between qubits i and j,
        - nⱼ = |1⟩⟨1| is the number operator on qubit j.

    Attributes:
        omegas (torch.Tensor): vector of Rabi frequencies Ωⱼ / 2 for each qubit.
        deltas (torch.Tensor): vector of detunings Δⱼ for each qubit.
        phis (torch.Tensor): vector of phases ϕⱼ for each qubit.
        interaction_matrix (torch.Tensor): matrix Uᵢⱼ for pairwise interactions.
        device (torch.device): device on which all tensors are allocated.
        diag (torch.Tensor): diagonal contribution to the Hamiltonian (detuning + interactions).
        inds (torch.Tensor): index mapping for σˣ operations.
        nqubits (int): number of qubits in the system.

    Methods:
        __mul__(vec): Applies the Hamiltonian H to a state vector |ψ⟩.
        _apply_sigma_operators_real(): Applies only σˣ terms (ϕⱼ = 0).
        _apply_sigma_operators_complex(): Applies generalized σˣ/σʸ terms (ϕⱼ ≠ 0).
        _create_diagonal(): Computes the diagonal part of H from Δⱼ and Uᵢⱼ.
        expect(state): Computes ⟨ψ|H|ψ⟩ for a given StateVector.
    """

    def __init__(
        self,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        interaction_matrix: torch.Tensor,
        device: torch.device,
    ):
        self.nqubits: int = len(omegas)
        self.omegas: torch.Tensor = omegas / 2.0
        self.deltas: torch.Tensor = deltas
        self.phis: torch.Tensor = phis
        self.interaction_matrix: torch.Tensor = interaction_matrix
        self.device: torch.device = device

        self.diag: torch.Tensor = self._create_diagonal()
        self.inds = torch.tensor([1, 0], device=self.device)  # flips the state, for σˣ
        self.complex = self.phis.any()

    def __mul__(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Apply the `RydbergHamiltonian` to the input state vector, i.e. H*|ψ❭.

        - The diagonal part of the Hamiltonian (Δⱼ and Uᵢⱼ terms) is stored and
        applyed directly as H.diag*|ψ❭.
        - The off-diagonal part (Ωⱼ and ϕⱼ terms) are applied sequentially across
        qubit indices in `self._apply_sigma_operators`.

        Args:
            vec (torch.Tensor): vec (torch.Tensor): the input state vector.

        Returns:
            the resulting state vector.
        """
        # (-∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ)|ψ❭
        result = self.diag * vec
        # ∑ⱼΩⱼ/2[cos(ϕⱼ)σˣⱼ + sin(ϕⱼ)σʸⱼ]|ψ❭
        if self.complex:
            self._apply_sigma_operators_complex(result, vec)
        else:
            self._apply_sigma_operators_real(result, vec)
        return result

    def _apply_sigma_operators_real(
        self, result: torch.Tensor, vec: torch.Tensor
    ) -> None:
        """
        Apply the ∑ⱼ(Ωⱼ/2)σˣⱼ operator to the input vector |ψ❭.

        Args:
            vec (torch.Tensor): the input state vector.

        Returns:
            the resulting state vector.
        """
        dim_to_act = 1
        for n, omega_n in enumerate(self.omegas):
            shape_n = (2**n, 2, 2 ** (self.nqubits - n - 1))
            vec = vec.view(shape_n)
            result = result.view(shape_n)
            result.index_add_(dim_to_act, self.inds, vec, alpha=omega_n)

    def _apply_sigma_operators_complex(
        self, result: torch.Tensor, vec: torch.Tensor
    ) -> None:
        """
        Apply the ∑ⱼΩⱼ/2[cos(ϕⱼ)σˣⱼ + sin(ϕⱼ)σʸⱼ] operator to the input vector |ψ❭.

        Args:
            vec (torch.Tensor): the input state vector.

        Returns:
            the resulting state vector.
        """
        c_omegas = self.omegas * torch.exp(1.0j * self.phis)

        dim_to_act = 1
        for n, c_omega_n in enumerate(c_omegas):
            shape_n = (2**n, 2, 2 ** (self.nqubits - n - 1))
            vec = vec.view(shape_n)
            result = result.view(shape_n)
            result.index_add_(
                dim_to_act, self.inds[0], vec[:, 0, :].unsqueeze(1), alpha=c_omega_n
            )
            result.index_add_(
                dim_to_act,
                self.inds[1],
                vec[:, 1, :].unsqueeze(1),
                alpha=c_omega_n.conj(),
            )

    def _create_diagonal(self) -> torch.Tensor:
        """
        Return the diagonal elements of the Rydberg Hamiltonian matrix

            H.diag = -∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ
        """
        diag = torch.zeros(2**self.nqubits, dtype=torch.complex128, device=self.device)

        for i in range(self.nqubits):
            diag = diag.view(2**i, 2, -1)
            i_fixed = diag[:, 1, :]
            i_fixed -= self.deltas[i]
            for j in range(i + 1, self.nqubits):
                i_fixed = i_fixed.view(2**i, 2 ** (j - i - 1), 2, -1)
                # replacing i_j_fixed by i_fixed breaks the code :)
                i_j_fixed = i_fixed[:, :, 1, :]
                i_j_fixed += self.interaction_matrix[i, j]
        return diag.view(-1)

    def expect(self, state: StateVector) -> torch.Tensor:
        """Return the energy expectation value E=❬ψ|H|ψ❭"""
        assert isinstance(
            state, StateVector
        ), "Currently, only expectation values of StateVectors are supported"
        en = torch.vdot(state.vector, self * state.vector)
        assert torch.allclose(en.imag, torch.zeros_like(en.imag), atol=1e-8)
        return en.real
