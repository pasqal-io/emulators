import torch
from emu_sv.state_vector import StateVector


class RydbergHamiltonian:
    """
    Representation of the Rydberg Hamiltonian with light-matter interaction:

        H = ∑ⱼΩⱼ/2[cos(ϕⱼ)σˣⱼ + sin(ϕⱼ)σʸⱼ] - ∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ

    The Hamiltonian is parameterized by driving strengths or amplitudes Ωⱼ (`omegas`), detuning
    values Δⱼ (`deltas`), phases ϕⱼ (`phis`) and interaction terms Uᵢⱼ (`interaction_matrix`).
    Implements an efficient H*|ψ❭ as custom sparse matrix-vector multiplication.

    Attributes:
        omegas (torch.Tensor): driving strength Ωⱼ for each qubit, scaled by a factor 1/2.
        deltas (torch.Tensor): detuning values Δⱼ for each qubit.
        phis (torch.Tensor): phase values ϕⱼ for each qubit.
        interaction_matrix (torch.Tensor): matrix Uᵢⱼ representing pairwise Rydberg
            interaction strengths between qubits.
        nqubits (int): number of qubits in the system.
        diag (torch.Tensor): diagonal elements of the Hamiltonian,
            calculated based on `deltas` and `interaction_matrix`.
        inds (torch.Tensor): index tensor used for vector manipulations
            in matrix-vector multiplications.

    Methods:
        __mul__(vec): implements matrix-vector multiplication with a state vector.
        _create_diagonal(): constructs the diagonal elements of the Hamiltonian
            based on `deltas` and `interaction_matrix`.
        _apply_sigma_operators_complex(): apply all driving sigma operators,
             with driving strenght `omegas` and phases `phis`.
        _apply_sigma_operators_real(): only applies ∑ⱼ(Ωⱼ/2)σˣⱼ when all phases are zero (ϕⱼ=0).
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

        self._apply_sigma_operators = self._apply_sigma_operators_real
        if self.phis.any():
            self._apply_sigma_operators = self._apply_sigma_operators_complex

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
        self._apply_sigma_operators(result, vec)
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
            vec = vec.reshape(shape_n)
            result = result.reshape(shape_n)
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
        c_omegas = self.omegas * torch.exp(1j * self.phis)

        dim_to_act = 1
        for n, c_omega_n in enumerate(c_omegas):
            shape_n = (2**n, 2, 2 ** (self.nqubits - n - 1))
            vec = vec.reshape(shape_n)
            result = result.reshape(shape_n)
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
            diag = diag.reshape(2**i, 2, -1)
            i_fixed = diag[:, 1, :]
            i_fixed -= self.deltas[i]
            for j in range(i + 1, self.nqubits):
                i_fixed = i_fixed.reshape(2**i, 2 ** (j - i - 1), 2, -1)
                # replacing i_j_fixed by i_fixed breaks the code :)
                i_j_fixed = i_fixed[:, :, 1, :]
                i_j_fixed += self.interaction_matrix[i, j]
        return diag.reshape(-1)

    def expect(self, state: StateVector) -> torch.Tensor:
        """Return the energy expectation value E=❬ψ|H|ψ❭"""
        assert isinstance(
            state, StateVector
        ), "Currently, only expectation values of StateVectors are supported"
        return torch.vdot(state.vector, self * state.vector)
