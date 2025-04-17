import torch
from emu_base.lindblad_operators import compute_noise_from_lindbladians


dtype = torch.complex128


class LindbladOperator:
    def __init__(
        self,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        pulser_linblads: list[torch.Tensor],
        interaction_matrix: torch.Tensor,
        device: torch.device,
    ):
        self.nqubits: int = len(omegas)
        self.omegas: torch.Tensor = omegas / 2.0
        self.deltas: torch.Tensor = deltas
        self.phis: torch.Tensor = phis
        self.interaction_matrix: torch.Tensor = interaction_matrix
        self.pulser_linblads: list[torch.Tensor] = pulser_linblads
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
        # convert rho to a tensor with shape (2,)* (2*nqubits),
        # where the first nqubits dimensions are for the row indices.
        full_shape = (2,) * (2 * self.nqubits)
        rho_view = rho.view(full_shape)

        # create an accumulator tensor with the same shape and data type as rho_view.
        accum = torch.zeros_like(rho_view)

        # loop over each qubit j and apply the Pauli X  along the row index j
        for j, omega_j in enumerate(self.omegas):
            # flip reverses the order along dimension j,  swaps 0 and 1.
            h_rho_term = torch.flip(rho_view, dims=[j])
            # Accumulate the contribution
            accum.add_(omega_j * h_rho_term)  # in place opperation,
            # NOTE: use this index_add for efficiency

        # return the resulting tensor in its original 2^n x 2^n matrix shape.
        return accum.view(2**self.nqubits, 2**self.nqubits)

    def apply_local_operator_to_density_matrix(
        self, density_matrix: torch.Tensor, local_op: torch.Tensor, target_qubit: int
    ) -> torch.Tensor:
        # reshape density matrix to a 2n-way tensor of shape (2,...,2) x (2,...,2)
        rho = density_matrix.view([2] * (2 * self.nqubits))

        # permute so that the target qubit comes first in both bra and ket spaces.
        # determine new ordering for the bra (first n_qubits) and
        # for ket (next n_qubits) indices.
        perm = list(range(self.nqubits))
        perm.remove(target_qubit)
        perm = [target_qubit] + perm  # target qubit now in the first position
        bra_perm = perm
        ket_perm = [p + self.nqubits for p in perm]
        total_perm = bra_perm + ket_perm
        rho = rho.permute(total_perm)
        # NOTE: broadcasting trick

        # reshape to a 2-index tensor: (2, 2^(n-1)* 2* 2^(n-1)) in order to multiply with local_op.
        # the first index is the target qubit, the second index is the rest of the qubits.
        rho = rho.contiguous().view(2, 2 ** (2 * self.nqubits - 1))  # note

        # apply local_op to the bra index.
        # contract local_op (indices 'ab') with the first index of rho ('b') giving new index 'a'.
        # rho = torch.einsum("ab,bijc->aijc", local_op, rho)
        rho = local_op @ rho

        # reshape back to a 2n-way tensor.
        rho = rho.contiguous().view([2] * (2 * self.nqubits))

        # invert the permutation to return the indices to their original order.
        inv_perm = [0] * (2 * self.nqubits)
        for i, p in enumerate(total_perm):
            inv_perm[p] = i
        rho = rho.permute(inv_perm)

        # reshape back to the full density matrix shape [2**n, 2**n].
        return rho.contiguous().view(2**self.nqubits, 2**self.nqubits)

    def __matmul__(self, densi_matrix: torch.Tensor) -> torch.Tensor:
        # Constructing Hₑ = -i H 𝜌 −0.5∑ₖ L^† L 𝜌

        # Applying ∑ᵢ 𝛺 /2 𝜎ₓ terms in H
        densi_matrix_sum_x = self.apply_sum_sigma_x_i_to_density(densi_matrix)

        # -∑ⱼΔⱼnⱼ + ∑ᵢ﹥ⱼUᵢⱼnᵢnⱼ in H
        # NOTE: get rid of the for loop
        # use broadcasting to apply the diagonal term, reshape the diag tensor
        # look at torch bracasting rules
        storage = torch.zeros_like(densi_matrix, dtype=dtype)
        shape = densi_matrix.shape[0]
        for i in range(shape):
            diag_result = self.diag * densi_matrix[:, i]
            storage[:, i] = diag_result
        # add the diagonal result to the densi_matrix
        h_densi_matrix = densi_matrix_sum_x + storage

        # Applying the Lindblad operators -0.5*i*∑ₖ Lₖ^† Lₖ 𝜌
        sum_lindblas = compute_noise_from_lindbladians(
            self.pulser_linblads
        )  # result 2x2 matrix

        storage_linbdlads = torch.zeros_like(densi_matrix, dtype=dtype)
        for qubit in range(self.nqubits):
            pre_storage_lindblads = self.apply_local_operator_to_density_matrix(
                densi_matrix, sum_lindblas, qubit
            )
            storage_linbdlads += pre_storage_lindblads
        # NOTE: add all the lindblad operators first, then
        # NOTE: fucntion that sums all the lindblads copute_noise_from_linbladian
        # check noise.py, -1j at the begging of the functions

        densi_matrix = h_densi_matrix - storage_linbdlads
        return densi_matrix
