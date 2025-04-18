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

    def apply_local_operator_to_density_matrix_to_local_op(
        self,
        density_matrix: torch.Tensor,
        local_op: torch.Tensor,
        target_qubit: int,
        op_conj_T: bool = False,
    ) -> torch.Tensor:
        """Apply a local operator (e.g., A) acting on one qubit to a full density matrix.
        if op_conj_T= True, Lₖ 𝜌 Lₖ^† else Lₖ 𝜌
        Parameters:
            density_matrix: torch.Tensor of shape [2**n, 2**n]
            local_op: torch.Tensor of shape [2, 2] (complex)
            target_qubit: int in [0, n_qubits - 1], the qubit that A acts on
            n_qubits: int, total number of qubits
            op_conj_t: bool, if True apply A† to the ket index (L_k 𝜌 L_k^†)

        Returns:
            Updated density matrix: torch.Tensor of shape [2**n, 2**n]
        """
        # Reshape density matrix to a 2n-way tensor of shape (2,...,2) x (2,...,2)
        rho = density_matrix.view([2] * (self.nqubits * 2))

        # permutes: target qubit comes first in both bra and ket spaces.
        # determine new ordering for the bra (first n_qubits) and for ket (next n_qubits) indices.
        perm = list(range(self.nqubits))
        perm.remove(target_qubit)
        perm = [target_qubit] + perm  # target now in the first position
        bra_perm = perm
        ket_perm = [p + self.nqubits for p in perm]
        total_perm = bra_perm + ket_perm
        rho = rho.permute(total_perm)

        rho = rho.contiguous().view(
            2, 2 ** (self.nqubits - 1), 2, 2 ** (self.nqubits - 1)
        )

        # apply A to the bra index.
        # contract A (indices 'ab') with the first index of rho ('b')
        rho = torch.einsum("ab,bijc->aijc", local_op, rho)

        # if op_conj= True: apply A† to the ket index.
        # Contract along the third index (the original ket index) of rho with A†:
        # resulting in L_k \rho L_k^†
        if op_conj_T:
            rho = torch.einsum("aijc,jd->aidc", rho, local_op.conj().T)
        else:  # if not apply the idenity instead Lk \rho
            ident = torch.eye(2, dtype=dtype)
            rho = torch.einsum("aijc,jd->aidc", rho, ident)

        # reshape back to a 2n-way tensor.
        rho = rho.contiguous().view([2] * (2 * self.nqubits))

        # invert the permutation to return the indices to their original order.
        inv_perm = [0] * (self.nqubits * 2)
        for i, p in enumerate(total_perm):
            inv_perm[p] = i
        rho = rho.permute(inv_perm)

        # reshape back to the full density matrix shape [2**n, 2**n].
        return rho.contiguous().view(2**self.nqubits, 2**self.nqubits)

    def __matmul__(self, densi_matrix: torch.Tensor) -> torch.Tensor:
        # Constructing Hₑ =  H 𝜌 -𝜌  H  +0.5i∑ₖ L^† L 𝜌+0.5i 𝜌 ∑ₖ L^† L

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
            pre_storage_lindblads = (
                self.apply_local_operator_to_density_matrix_to_local_op(
                    densi_matrix, sum_lindblas, qubit
                )
            )
            storage_linbdlads += pre_storage_lindblads

        # i∑ₖ Lₖ 𝜌  Lₖ^†
        storage_LrhoLdag = torch.zeros_like(densi_matrix, dtype=dtype)
        for j in range(len(self.pulser_linblads)):
            for i in range(self.nqubits):
                pre_storage_LrhoLdag = (
                    self.apply_local_operator_to_density_matrix_to_local_op(
                        densi_matrix, self.pulser_linblads[j], i, op_conj_T=True
                    )
                )
                storage_LrhoLdag += pre_storage_LrhoLdag

        # final density matrix
        result = (
            h_densi_matrix
            - h_densi_matrix.conj().T
            + storage_linbdlads  # already has a -0.5i
            - storage_linbdlads.conj().T
        )
        return result  # I expect this to be multiplied by -1.0j
