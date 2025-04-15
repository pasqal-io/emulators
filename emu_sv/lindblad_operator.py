from typing import List, Union
import torch

# from emu_sv.state_vector import StateVector
dtype = torch.complex128


class LindbladOperator:
    def __init__(
        self,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        pulser_linblad: torch.Tensor,
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

    def _create_diagonal(self) -> torch.Tensor:
        """
        Return the diagonal elements of the Rydberg Hamiltonian matrix

            H.diag = -∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ
        """
        diag = torch.zeros(2**self.nqubits, dtype=dtype, device=self.device)

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

    def apply_sum_sigma_x_i_to_density(self, rho: torch.Tensor) -> torch.Tensor:
        # Convert rho to a tensor with shape (2, 2, ..., 2, 2, ..., 2)
        # nqubits dimensions for row indices
        full_shape = (2,) * (2 * self.nqubits)
        rho_view = rho.view(full_shape)

        # Prepare an accumulator for the change in rho
        accum = torch.zeros_like(rho_view, dtype=dtype)

        # Define the 'flip' indices for a Pauli X (i.e., [0,1] -> [1,0])
        flip = torch.tensor([1, 0], device=rho.device)

        # Loop over each qubit j:
        for j, omega_j in enumerate(self.omegas):
            # ---multiplication: X_j * rho ---
            # We act on the row index corresponding to qubit j.
            # Create a copy of rho_view where the j-th row index is flipped.
            # To do this we use advanced indexing along dimension j.

            # Build index slices for all row dimensions: for qubit j use flip index,
            # otherwise use colon
            idx_all_rows: List[Union[slice, torch.Tensor]] = [slice(None)] * self.nqubits
            # Replace the j-th index with the flipped index
            # This is a trick to swap rows of the j-th qubit's index:

            idx_all_rows[j] = flip  # type ignore[call-overload]
            # For the left action we leave the column indices untouched:
            idx_all = idx_all_rows + [slice(None)] * self.nqubits

            # This produces the piece corresponding to X_j * rho.
            h_rho_term = rho_view[tuple(idx_all)]

            accum += (omega_j) * (h_rho_term)

        # Update rho_view
        rho_view = accum

        # Return to the original 2^n x 2^n matrix
        rho_new: torch.Tensor = rho_view.view(2**self.nqubits, 2**self.nqubits)
        return rho_new

    def __matmul__(self, densi_matrix: torch.Tensor) -> torch.Tensor:

        densi_matrix_sum_x = self.apply_sum_sigma_x_i_to_density(densi_matrix)

        # (-∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ)|ψ❭
        storage = torch.zeros_like(densi_matrix, dtype=dtype)
        shape = densi_matrix.shape[0]
        for i in range(shape):
            diag_result = self.diag * densi_matrix[:, i]
            storage[:, i] = diag_result
        # Add the diagonal result to the densi_matrix
        densi_matrix = densi_matrix_sum_x + storage
        return densi_matrix
