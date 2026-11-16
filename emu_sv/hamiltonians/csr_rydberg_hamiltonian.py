"""
`CSRRydbergHamiltonian`: the `RydbergHamiltonian` stored as a CSR sparse matrix.

The torch implementation of `RydbergHamiltonian` applies H in N + 1 passes over the
state, each a separate torch call. For small states the cost of those calls
dominates, and a single CSR matrix-vector product is several times faster. The CSR
matrix holds N + 1 entries per row, so for large states it does not fit in the cache
any more, and the torch implementation, which only streams the state, wins: by
default it is only used on CPU, up to `CSR_MAX_QUBITS` qubits (see `SVBackendImpl`).
"""

import functools
from typing import Callable

import torch
from emu_sv.hamiltonians.rydberg_hamiltonian import RydbergHamiltonian

# By default, CPU simulations up to this many qubits use the CSR format.
CSR_MAX_QUBITS = 16


@functools.lru_cache(maxsize=1)
def _structure(
    nqubits: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    The sparsity pattern of H, which only depends on the number of qubits:
    row k has the diagonal entry and one entry per qubit n, in column k with bit n
    flipped. Qubit n is bit N - 1 - n of the index, the first qubit being the most
    significant one.

    Returns the CSR row pointers and column indices (sorted within each row), and
    for each entry where its value is found: `kind` indexes the 2N + 1 coefficients
    built in `CSRRydbergHamiltonian._matrix` (2n + bit n of the row for qubit n,
    2N for the diagonal), and `diag_slot` are the positions of the diagonal entries.
    """
    dim = 2**nqubits
    rows = torch.arange(dim, device=device).unsqueeze(1)
    flips = 1 << torch.arange(nqubits - 1, -1, -1, device=device)
    cols = torch.cat((rows, rows ^ flips), dim=1)
    row_bits = (rows & flips) != 0
    qubit_kind = 2 * torch.arange(nqubits, device=device) + row_bits
    kind = torch.cat((torch.full_like(rows, 2 * nqubits), qubit_kind), dim=1)
    order = cols.argsort(dim=1)
    cols = cols.gather(1, order).view(-1)
    kind = kind.gather(1, order).view(-1)
    diag_slot = (order == 0).view(-1).nonzero().view(-1)
    crow = torch.arange(0, dim * (nqubits + 1) + 1, nqubits + 1, device=device)
    return crow, cols, kind, diag_slot


class CSRRydbergHamiltonian(RydbergHamiltonian):
    """
    `RydbergHamiltonian` applied to complex128 state vectors as a CSR matrix, on
    any device. The matrix is built on first use, once per Hamiltonian (or per
    operator from `make_op`).

    torch does not record sparse products, so H is applied with the torch
    implementation of `RydbergHamiltonian` whenever autograd needs to record
    through it.
    """

    def __init__(
        self,
        omegas: torch.Tensor,
        deltas: torch.Tensor,
        phis: torch.Tensor,
        interaction_matrix: torch.Tensor,
        device: torch.device,
        noise: torch.Tensor | None,
        interaction_diag: torch.Tensor | None = None,
    ):
        super().__init__(
            omegas=omegas,
            deltas=deltas,
            phis=phis,
            interaction_matrix=interaction_matrix,
            device=device,
            noise=noise,
            interaction_diag=interaction_diag,
        )
        self._csr: torch.Tensor | None = None
        self._requires_grad = any(
            t.requires_grad for t in (omegas, deltas, phis, interaction_matrix)
        )

    def _use_csr(self, requires_grad: bool = False) -> bool:
        """
        Whether H can be applied as a CSR matrix to vectors that do (or do not)
        `requires_grad`, i.e. whether autograd does not need to record through H.
        """
        return not (torch.is_grad_enabled() and (requires_grad or self._requires_grad))

    def _matrix(self, factor: complex = 1.0) -> torch.Tensor:
        """factor·H as a CSR matrix."""
        diag = self.diag.detach()
        crow, cols, kind, diag_slot = _structure(self.nqubits, diag.device)
        if self.complex:
            # see _apply_sigma_operators_complex: omegas_c[:, 1] takes |1❭ to |0❭
            row_0, row_1 = self.omegas_c[:, 1], self.omegas_c[:, 0]
        else:
            row_0 = row_1 = self.omegas
        coefficients = torch.cat(
            (
                torch.stack((row_0, row_1), dim=1).view(-1),
                torch.zeros(1, dtype=row_0.dtype, device=row_0.device),
            )
        )
        coefficients = factor * coefficients.detach().to(diag)
        values = torch.take(coefficients, kind)
        values[diag_slot] = factor * diag
        dim = 2**self.nqubits
        return torch.sparse_csr_tensor(
            crow, cols, values, size=(dim, dim), check_invariants=False
        )

    def make_op(self, factor: complex) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Return the operator x ↦ factor·(H x), e.g. -i dt H for `krylov_exp`.

        As for `FusedRydbergHamiltonian.make_op`, whether to use the CSR matrix is
        decided here, once, and the operator must be used in the grad mode it was
        made in. The factor is applied to the matrix.
        """
        if not self._use_csr():
            return super().make_op(factor)
        matrix = self._matrix(factor)
        return lambda x: matrix @ x

    def __mul__(self, vec: torch.Tensor) -> torch.Tensor:
        """H*|ψ❭, as a CSR product unless autograd needs to record through H."""
        if vec.dtype == torch.complex128 and self._use_csr(vec.requires_grad):
            if self._csr is None:
                self._csr = self._matrix()
            return self._csr @ vec
        return super().__mul__(vec)
