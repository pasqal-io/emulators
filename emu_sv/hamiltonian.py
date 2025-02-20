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
        __mul__(vec): performs matrix-vector multiplication with a vector.
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
        if self.phis.all():
            self._apply_sigma_operators = self._apply_sigma_operators_complex

    def __mul__(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Performs a custom sparse matrix-vector multiplication between the `RydbergHamiltonian`
        and a vector representing the quantum state.


        Computes the product of `RydbergHamiltonian` object's Hamiltonian
        (represented by its diagonal (𝛿ᵢ and  Uᵢⱼ) and off diagonal (𝛺ᵢ) terms) and the input
        vector `vec`. The result is initially reshaped to a tensor with dimensions corresponding
        to the number of qubits, where interactions and detunigs are applied sequentially
        across qubit indices, scaled by the `omegas` values.
        The final result is reshaped to a 1D tensor.

        Args:
            vec (torch.Tensor): The vector to multiply, with dimensions compatible with the
                                Hamiltonian's representation.

        Returns:
            the resulting state vector.
        """
        # (-∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ)|ψ❭
        diag_result = self.diag * vec
        # ∑ⱼΩⱼ/2[cos(ϕⱼ)σˣⱼ + sin(ϕⱼ)σʸⱼ]|ψ❭
        sigma_result = self._apply_sigma_operators(vec)
        result: torch.Tensor
        result = diag_result + sigma_result

        return result

    def _apply_sigma_operators_real(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Applies the ∑ⱼ(Ωⱼ/2)σˣⱼ operator to the input vector |ψ❭.

        Performs a matrix-vector multiplication between a sum of  𝛺ᵢ 𝜎ᵢˣ operators
        and the input state vector `vec`. For each qubit `i`, the operator
        ∑ᵢ (𝛺ᵢ / 2) 𝜎ᵢˣ  applies the Pauli-X gate 𝜎ᵢˣ to the `i`-th qubit of
        the vector `vec`, scaled by the coefficient 𝛺ᵢ / 2.
        The result is accumulated across all qubits to form the final transformed vector.

        Args:
            vec (torch.Tensor): the input state vector.

        Returns:
            the resulting state vector.
        """
        result = torch.zeros_like(vec)

        dim_to_act = 1
        for n, omega_n in enumerate(self.omegas):
            shape_n = (2**n, 2, 2 ** (self.nqubits - n - 1))
            vec = vec.reshape(shape_n)
            result = result.reshape(shape_n)
            result.index_add_(dim_to_act, self.inds, vec, alpha=omega_n)

        return result.reshape(-1)

    def _apply_sigma_operators_complex(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Applies the 1/2∑ᵢ(𝛺ᵢXᵢ + 𝛺ᵢ*Yᵢ) operator to the input vector |ψ❭

        Args:
            vec (torch.Tensor): the input state vector.

        Returns:
            the resulting state vector.
        """
        c_omegas = self.omegas * torch.exp(1j * self.phis)
        result = torch.zeros_like(vec)

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

        return result.reshape(-1)

    def _create_diagonal(self) -> torch.Tensor:
        """
        Constructs the diagonal elements of the Rydberg Hamiltonian matrix

            -∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ

        This method creates a tensor representing the diagonal terms of the
        Hamiltonian, including contributions from detuning `deltas` 𝛿ᵢ
        and interaction terms `interaction_matrix` Uᵢⱼ. This excludes the `omegas` 𝛺ᵢ.
        Each qubit's detuning value is subtracted from the diagonal, and interaction terms are
        added for qubit pairs to represent their couplings.

        Returns:
            the diagonal elements of the RydbergHamiltonian matrix.
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
        """Returns the expectation value of energy E=❬ψ|H|ψ❭"""
        assert isinstance(
            state, StateVector
        ), "currently, only expectation values of StateVectors are supported"
        return torch.vdot(state.vector, self * state.vector)
