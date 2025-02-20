import torch
from emu_sv.state_vector import StateVector


class RydbergHamiltonian:
    """
        Representation of the Rydberg Hamiltonian with light-matter interaction:

            H = ∑ⱼΩⱼ/2[cos(ϕⱼ)σˣⱼ + sin(ϕⱼ)σʸⱼ] - ∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ

        Implements H*|ψ❭ as sparse matrix-vector multiplication.

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
        _apply_sigma_operators_real(): only applies ∑ⱼ(Ωⱼ/2) σˣⱼ when all phases are zero (ϕⱼ=0).

    Notes:
        The `RydbergHamiltonian` class represents the Rydberg Hamiltonian matrix where the diagonal
        terms are the interaction Uᵢⱼ nᵢ⊗nⱼ and detunining 𝛿ᵢnᵢ and off diagonal term omegas 𝛺ᵢ𝜎ᵢˣ
        for a quantum system, allowing efficient computations such as matrix-vector multiplications.
        The Hamiltonian is parameterized by driving strengths or amplitudes 𝛺ᵢ (`omegas`), detuning
        values 𝛿ᵢ (`deltas`), and  interaction terms Uᵢⱼ (`interaction_matrix`).
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
        self.diag: torch.Tensor = self._create_diagonal().to(device=device)
        self.inds = torch.tensor([1, 0], device=device)  # flips the state, for 𝜎ₓ

        self._apply_sigma_operators = self._apply_sigma_operators_real
        if self.phis.all():
            self._apply_sigma_operators = self._apply_sigma_operators_complex

    def __mul__(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Performs a matrix-vector multiplication between the `RydbergHamiltonian`
        and a vector representing the quantum state.


        Computes the product of `RydbergHamiltonian` object's Hamiltonian
        (represented by its diagonal (𝛿ᵢ and  Uᵢⱼ) and off diagonal (𝛺ᵢ) terms) and the input
        vector `vec`. The result is initially reshaped to a tensor with dimensions corresponding
        to the number of qubits, where interactions and detunigs are applied sequentially
        across qubit indices, scaled by the `omegas` values. The final result is reshaped
        to a 1D tensor.

        Args:
            vec (torch.Tensor): The vector to multiply, with dimensions compatible with the
                                Hamiltonian's representation.

        Returns:
            the resulting state vector.

        """
        vec = vec if len(vec) == self.nqubits else vec.reshape((2,) * self.nqubits)

        # (-∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ)|ψ❭
        diag_result = self.diag * vec
        # ∑ⱼΩⱼ/2[cos(ϕⱼ)σˣⱼ + sin(ϕⱼ)σʸⱼ]|ψ❭
        sigmax_result = self._apply_sigma_operators(vec)
        result: torch.Tensor
        result = diag_result + sigmax_result
        return result.reshape(-1)

    def _apply_sigma_operators_real(self, vec: torch.Tensor) -> torch.Tensor:
        """
            Applies the ∑ᵢ(𝛺ᵢ / 2)𝜎ᵢˣ operator to the input vector |ψ❭.

        Performs a matrix-vector multiplication between a sum of  𝛺ᵢ 𝜎ᵢˣ operators
        and the input state vector `vec`. For each qubit `i`, the operator
        ∑ᵢ (𝛺ᵢ / 2) 𝜎ᵢˣ  applies the Pauli-X gate 𝜎ᵢˣ to the `i`-th qubit of
        the vector `vec`, scaled by the coefficient 𝛺ᵢ / 2.
        The result is accumulated across all qubits to form the final transformed vector.

        Args:
            vec (torch.Tensor): The input state vector, with 1D dimension

        Returns:
            torch.Tensor: The resulting state vector after applying the ∑ᵢ (𝛺ᵢ / 2) * 𝜎ᵢˣ
                          operator, with 1 D dimension
        """
        result = torch.zeros(vec.shape, device=vec.device, dtype=torch.complex128)

        for i, omega in enumerate(self.omegas):
            result.index_add_(i, self.inds, vec, alpha=omega)
        # when phi != 0, you need to do o and o.conj() separately, but this is SLOWER
        # res.index_add_(i, torch.tensor(0), v.select(i,1).unsqueeze(i), alpha=o)
        # res.index_add_(i, torch.tensor(1), v.select(i,0).unsqueeze(i), alpha=o.conj())
        return result

    def _apply_sigma_operators_complex(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Applies the 1/2∑ᵢ(𝛺ᵢXᵢ + 𝛺ᵢ*Yᵢ) operator to the input vector |ψ❭

        Args:
            vec (torch.Tensor): The input state vector, with 1D dimension

        Returns:
            torch.Tensor: The resulting state vector
        """
        result = torch.zeros(vec.shape, device=vec.device, dtype=torch.complex128)
        c_omegas = self.omegas * torch.exp(1j * self.phis)
        for i, c_omega in enumerate(c_omegas):
            result.index_add_(i, self.inds[0], vec, alpha=c_omega)
            result.index_add_(i, self.inds[1], vec, alpha=c_omega.conj())
        return result

    def _create_diagonal(self) -> torch.Tensor:
        """
        Constructs the diagonal elements of the Hamiltonian matrix

        -∑ᵢ𝛿ᵢnᵢ +1/2∑ᵢⱼ Uᵢⱼ nᵢ⊗nⱼ

        This method creates a tensor representing the diagonal terms of the
        Hamiltonian, including contributions from detuning `deltas` 𝛿ᵢ
        and interaction terms `interaction_matrix` Uᵢⱼ. This excludes the `omegas` 𝛺ᵢ.
        Each qubit's detuning value is subtracted from the diagonal, and interaction terms are
        added for qubit pairs to represent their couplings.

        Returns:
            torch.Tensor: A tensor with shape (2,)* number of qubits of the computed
            diagonal elements.
        """
        diag = torch.zeros(
            (2,) * self.nqubits, dtype=torch.complex128, device=self.omegas.device
        )

        for i in range(self.nqubits):
            i_fixed = diag.select(i, 1)
            i_fixed -= self.deltas[i]  # add the delta term for this qubit
            for j in range(i + 1, self.nqubits):
                i_j_fixed = i_fixed.select(j - 1, 1)  # j-1 since i was removed
                i_j_fixed += self.interaction_matrix[i, j]
        return diag

    def expect(self, state: StateVector) -> torch.Tensor:
        """Returns the expectation value of energy E=❬ψ|H|ψ❭"""
        assert isinstance(
            state, StateVector
        ), "currently, only expectation values of StateVectors are supported"
        return torch.vdot(state.vector, self * state.vector)
