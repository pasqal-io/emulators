import torch
from .mps import MPS
from .mpo import MPO
from .config import Config


def new_left_bath(
    bath: torch.Tensor, state: torch.Tensor, op: torch.Tensor
) -> torch.Tensor:
    # this order is more efficient than contracting the op first in general
    bath = torch.tensordot(bath, state.conj(), ([0], [0]))
    bath = torch.tensordot(bath, op, ([0, 2], [0, 1]))
    return torch.tensordot(bath, state, ([0, 2], [0, 1]))


def new_right_bath(
    bath: torch.Tensor, state: torch.Tensor, op: torch.Tensor
) -> torch.Tensor:
    bath = torch.tensordot(state, bath, ([2], [2]))
    bath = torch.tensordot(op, bath, ([2, 3], [1, 3]))
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


"""
Computes exp(tH)psi where t can be complex and
    x-    -x
    x  ||  x             ||
H = x- xx -x  and psi = -xx-
    x  ||  x
    x-    -x
Where the physical indices of state should be combined
into a fat index, and similar for the corresponding indices in H.
State should be normalized!
All inputs must be on the same device.

"""


def krylov_exp(
    t: float | complex,
    state: torch.Tensor,
    ham: torch.Tensor,
    left_bath: torch.Tensor,
    right_bath: torch.Tensor,
) -> torch.Tensor:
    config = Config()
    max_krylov = config.get_krylov_dim()
    exp_tolerance = config.get_krylov_exp_tolerance()
    norm_tolerance = config.get_krylov_norm_tolerance()

    def exponentiate() -> tuple[torch.Tensor, bool]:
        # approximate next iteration by modifying T, and unmodifying
        T[i - 1, i] = 0
        T[i + 1, i] = 1
        exp = torch.linalg.matrix_exp(t * T[: i + 2, : i + 2])
        T[i - 1, i] = T[i, i - 1]
        T[i + 1, i] = 0

        e1 = abs(exp[i, 0])
        e2 = abs(exp[i + 1, 0]) * n
        if e1 > 10 * e2:
            error = e2
        elif e2 > e1:
            error = e1
        else:
            error = (e1 * e2) / (e1 - e2)

        converged = error < exp_tolerance
        return exp[:, 0], converged

    lanczos_vectors = [state]
    T = torch.zeros(max_krylov + 1, max_krylov + 1, dtype=state.dtype)

    # step 0 of the loop
    v = apply_effective_Hamiltonian(state, ham, left_bath, right_bath)
    a = torch.tensordot(v.conj(), state, dims=3)
    n = torch.linalg.vector_norm(v)
    T[0, 0] = a
    v = v - a * state

    for i in range(1, max_krylov):
        # this block should not be executed in step 0
        b = torch.linalg.vector_norm(v)
        if b < norm_tolerance:
            exp = torch.linalg.matrix_exp(t * T[:i, :i])
            weights = exp[:, 0]
            converged = True
            break
        T[i, i - 1] = b
        T[i - 1, i] = b
        state = v / b
        lanczos_vectors.append(state)
        weights, converged = exponentiate()
        if converged:
            break

        v = apply_effective_Hamiltonian(state, ham, left_bath, right_bath)
        a = torch.tensordot(v.conj(), state, dims=3)
        n = torch.linalg.vector_norm(v)
        T[i, i] = a
        v = v - a * state - b * lanczos_vectors[i - 1]

    if not converged:
        raise RecursionError(
            "exponentiation algorithm did not converge to precision in allotted number of steps."
        )

    result = lanczos_vectors[0] * weights[0]
    for i in range(1, len(lanczos_vectors)):
        result += lanczos_vectors[i] * weights[i]
    return result


"""
Applies 2 sweep, 2-site tdvp to state in-place
State should be normalized and in orthogonal gauge with center at qubit 0
Output state is not normalized (because numerical errors or real part in t)
Hamiltonian should be Hermitian!
"""


def tdvp(t: float | complex, state: MPS, Hamiltonian: MPO) -> None:
    cutoff = Config().get_bond_precision()
    max_dim = Config().get_max_bond_dim()
    t /= 2
    nfactors = len(state.factors)
    assert nfactors > 1, "tdvp is not implemented for 1 site, just use state vector"

    # sweep left-right
    lbs = [
        torch.ones(1, 1, 1, dtype=state.factors[0].dtype, device=state.factors[0].device)
    ]
    rbs = right_baths(state, Hamiltonian, 2)
    for i in range(nfactors - 1):
        ls = state.factors[i]
        lss = ls.shape
        rs = state.factors[i + 1].to(ls.device)
        rss = rs.shape
        s = torch.tensordot(ls, rs, dims=1).reshape(lss[0], 4, rss[-1])
        lh = Hamiltonian.factors[i]
        lhs = lh.shape
        rh = Hamiltonian.factors[i + 1].to(ls.device)

        h = torch.tensordot(lh, rh, dims=1).transpose(2, 3).reshape(lhs[0], 4, 4, -1)
        evol = krylov_exp(t, s, h, lbs[i], rbs[-1 - i].to(ls.device)).reshape(
            lss[0] * 2, 2 * rss[-1]
        )

        u, d, v = torch.linalg.svd(evol, full_matrices=False)
        max_bond = min(state._determine_cutoff_index(d, cutoff), max_dim)
        u = u[:, :max_bond]
        d = d[:max_bond]
        v = v[:max_bond, :]

        state.factors[i] = u.reshape(lss[0], 2, -1)
        state.factors[i + 1] = (
            (d.reshape(-1, 1) * v).reshape(-1, 2, rss[-1]).to(state.factors[i + 1].device)
        )
        if i is not nfactors - 2:
            lbs.append(
                new_left_bath(lbs[i], state.factors[i], lh).to(
                    state.factors[i + 1].device
                )
            )
            evol = krylov_exp(
                -t,
                state.factors[i + 1],
                Hamiltonian.factors[i + 1],
                lbs[i + 1],
                rbs[-1 - i],
            )
            state.factors[i + 1] = evol

    # sweep right-left
    rbs = [
        torch.ones(1, 1, 1, dtype=state.factors[0].dtype, device=state.factors[-1].device)
    ]
    for i in range(i, -1, -1):
        rs = state.factors[i + 1]
        rss = rs.shape
        ls = state.factors[i].to(rs.device)
        lss = ls.shape
        s = torch.tensordot(ls, rs, dims=1).reshape(lss[0], 4, rss[-1])
        rh = Hamiltonian.factors[i + 1]
        lh = Hamiltonian.factors[i].to(rs.device)
        lhs = lh.shape

        h = torch.tensordot(lh, rh, dims=1).transpose(2, 3).reshape(lhs[0], 4, 4, -1)
        evol = krylov_exp(t, s, h, lbs[i].to(rs.device), rbs[nfactors - 2 - i]).reshape(
            lss[0] * 2, 2 * rss[-1]
        )

        u, d, v = torch.linalg.svd(evol, full_matrices=False)
        max_bond = min(state._determine_cutoff_index(d, cutoff), max_dim)
        u = u[:, :max_bond]
        d = d[:max_bond]
        v = v[:max_bond, :]

        state.factors[i + 1] = v.reshape(-1, 2, rss[-1])
        state.factors[i] = (u * d).reshape(lss[0], 2, -1).to(state.factors[i].device)
        if i != 0:
            rbs.append(
                new_right_bath(rbs[-1], state.factors[i + 1], rh).to(
                    state.factors[i].device
                )
            )
            evol = krylov_exp(
                -t, state.factors[i], Hamiltonian.factors[i], lbs[i], rbs[-1]
            )
            state.factors[i] = evol.reshape(lss[0], 2, -1)
