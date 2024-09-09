import torch

from emu_mps.math.krylov_exp import DEFAULT_MAX_KRYLOV_DIM, krylov_exp
from emu_mps.mpo import MPO
from emu_mps.mps import MPS
from emu_mps.utils import split_tensor, new_left_bath


def new_right_bath(
    bath: torch.Tensor, state: torch.Tensor, op: torch.Tensor
) -> torch.Tensor:
    bath = torch.tensordot(state, bath, ([2], [2]))
    bath = torch.tensordot(op.to(bath.device), bath, ([2, 3], [1, 3]))
    return torch.tensordot(state.conj(), bath, ([1, 2], [1, 3]))


"""
function to compute the left baths. The three indices in the bath are as follows:
(bond of state conj, bond of operator, bond of state)
The baths have shape
xx-
xx-
xx-
with the index ordering (top, middle, bottom)
bath tensors are put on the device of the factor to the right
"""


def left_baths(state: MPS, op: MPO, final_qubit: int) -> list[torch.Tensor]:
    state_factor = state.factors[0]
    bath = torch.ones(1, 1, 1, device=state_factor.device, dtype=state_factor.dtype)
    baths = [bath]
    for i in range(final_qubit + 1):
        bath = new_left_bath(bath, state.factors[i], op.factors[i])
        bath = bath.to(state.factors[i + 1].device)
        baths.append(bath)
    return baths


"""
function to compute the right baths. The three indices in the bath are as follows:
(bond of state conj, bond of operator, bond of state)
The baths have shape
-xx
-xx
-xx
with the index ordering (top, middle, bottom)
bath tensors are put on the device of the factor to the left
"""


def right_baths(state: MPS, op: MPO, final_qubit: int) -> list[torch.Tensor]:
    state_factor = state.factors[-1]
    bath = torch.ones(1, 1, 1, device=state_factor.device, dtype=state_factor.dtype)
    baths = [bath]
    for i in range(len(state.factors) - 1, final_qubit - 1, -1):
        bath = new_right_bath(bath, state.factors[i], op.factors[i])
        bath = bath.to(state.factors[i - 1].device)
        baths.append(bath)
    return baths


"""
Computes H(psi) where
    x-    -x
    x  ||  x             ||
H = x- xx -x  and psi = -xx-
    x  ||  x
    x-    -x

Expects the two qubit factors of the MPS precontracted,
with one 'fat' physical index of dim 4 and index ordering
(left bond, physical index, right bond):
         ||
      -xxxxxx-
The Hamiltonian should have an index ordering of
(left bond, out, in, right bond).
The baths must have shape as returned by the functions above.
All tensors must be on the same device
"""


def apply_effective_Hamiltonian(
    state: torch.Tensor,
    ham: torch.Tensor,
    left_bath: torch.Tensor,
    right_bath: torch.Tensor,
) -> torch.Tensor:
    # the optimal contraction order depends on the details
    # this order seems to be pretty balanced, but needs to be
    # revisited when use-cases are more well-known
    state = torch.tensordot(left_bath, state, 1)
    state = torch.tensordot(state, ham, ([1, 2], [0, 2]))
    state = torch.tensordot(state, right_bath, ([3, 1], [1, 2]))
    return state


def evolve_tdvp(
    t: float | complex,
    state: MPS,
    hamiltonian: MPO,
    extra_krylov_tolerance: float,
    max_krylov_dim: int = DEFAULT_MAX_KRYLOV_DIM,
    is_hermitian: bool = True,
) -> None:
    """
    Applies second-order, 2-site TDVP to state in-place.
    """
    state.orthogonalize(0)

    t /= 2
    nfactors = len(state.factors)
    assert nfactors > 1, "tdvp is not implemented for 1 site, just use state vector"

    # sweep left-right
    lbs = [
        torch.ones(1, 1, 1, dtype=state.factors[0].dtype, device=state.factors[0].device)
    ]
    rbs = right_baths(state, hamiltonian, 2)
    for i in range(nfactors - 1):
        ls = state.factors[i]
        lss = ls.shape
        rs = state.factors[i + 1].to(ls.device)
        rss = rs.shape
        s = torch.tensordot(ls, rs, dims=1).reshape(lss[0], 4, rss[-1])
        lh = hamiltonian.factors[i]
        lhs = lh.shape
        rh = hamiltonian.factors[i + 1].to(lh.device)
        rb = rbs.pop()

        h = (
            torch.tensordot(lh, rh, dims=1)
            .transpose(2, 3)
            .reshape(lhs[0], 4, 4, -1)
            .to(s.device)
        )

        op = lambda x: t * apply_effective_Hamiltonian(x, h, lbs[i], rb.to(s.device))

        evol = krylov_exp(
            op,
            s,
            exp_tolerance=state.precision * extra_krylov_tolerance,
            norm_tolerance=state.precision * extra_krylov_tolerance,
            max_krylov_dim=max_krylov_dim,
            is_hermitian=is_hermitian,
        ).reshape(lss[0] * 2, 2 * rss[-1])

        l, r = split_tensor(evol, max_error=state.precision, max_rank=state.max_bond_dim)

        state.factors[i] = l.reshape(lss[0], 2, -1)
        state.factors[i + 1] = r.reshape(-1, 2, rss[-1]).to(state.factors[i + 1].device)

        if i == nfactors - 2:
            break

        lbs.append(
            new_left_bath(lbs[i], state.factors[i], lh).to(state.factors[i + 1].device)
        )
        h = hamiltonian.factors[i + 1].to(state.factors[i + 1].device)
        op = lambda x: -t * apply_effective_Hamiltonian(x, h, lbs[i + 1], rb)

        evol = krylov_exp(
            op,
            state.factors[i + 1],
            exp_tolerance=state.precision * extra_krylov_tolerance,
            norm_tolerance=state.precision * extra_krylov_tolerance,
            max_krylov_dim=max_krylov_dim,
            is_hermitian=is_hermitian,
        )
        state.factors[i + 1] = evol

    # sweep right-left
    rbs = [
        torch.ones(1, 1, 1, dtype=state.factors[0].dtype, device=state.factors[-1].device)
    ]
    for i in range(nfactors - 2, -1, -1):
        rs = state.factors[i + 1]
        rss = rs.shape
        ls = state.factors[i].to(rs.device)
        lss = ls.shape
        s = torch.tensordot(ls, rs, dims=1).reshape(lss[0], 4, rss[-1])
        lh = hamiltonian.factors[i]
        rh = hamiltonian.factors[i + 1].to(lh.device)
        lhs = lh.shape
        lb = lbs.pop()

        h = (
            torch.tensordot(lh, rh, dims=1)
            .transpose(2, 3)
            .reshape(lhs[0], 4, 4, -1)
            .to(s.device)
        )

        op = lambda x: t * apply_effective_Hamiltonian(
            x, h, lb.to(s.device), rbs[nfactors - 2 - i]
        )

        evol = krylov_exp(
            op,
            s,
            exp_tolerance=state.precision * extra_krylov_tolerance,
            norm_tolerance=state.precision * extra_krylov_tolerance,
            max_krylov_dim=max_krylov_dim,
            is_hermitian=is_hermitian,
        ).reshape(lss[0] * 2, 2 * rss[-1])

        l, r = split_tensor(
            evol,
            max_error=state.precision,
            max_rank=state.max_bond_dim,
            orth_center_right=False,
        )

        state.factors[i + 1] = r.reshape(-1, 2, rss[-1])
        state.factors[i] = l.reshape(lss[0], 2, -1).to(state.factors[i].device)

        if i == 0:
            break

        rbs.append(
            new_right_bath(rbs[-1], state.factors[i + 1], rh).to(state.factors[i].device)
        )

        h = hamiltonian.factors[i].to(state.factors[i].device)
        op = lambda x: -t * apply_effective_Hamiltonian(x, h, lb, rbs[-1])

        evol = krylov_exp(
            op,
            state.factors[i],
            exp_tolerance=state.precision * extra_krylov_tolerance,
            norm_tolerance=state.precision * extra_krylov_tolerance,
            max_krylov_dim=max_krylov_dim,
            is_hermitian=is_hermitian,
        )

        state.factors[i] = evol.reshape(lss[0], 2, -1)
