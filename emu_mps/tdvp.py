import torch

from emu_base import krylov_exp
from emu_mps import MPS, MPO
from emu_mps.utils import split_tensor
from emu_mps.mps_config import MPSConfig


def new_right_bath(
    bath: torch.Tensor, state: torch.Tensor, op: torch.Tensor
) -> torch.Tensor:
    bath = torch.tensordot(state, bath, ([2], [2]))
    bath = torch.tensordot(op, bath, ([2, 3], [1, 3]))
    bath = torch.tensordot(state.conj(), bath, ([1, 2], [1, 3]))
    return bath


"""
Function to compute the right baths. The three indices in the bath are as follows:
(bond of state conj, bond of operator, bond of state)
The baths have shape
-xx
-xx
-xx
with the index ordering (top, middle, bottom)
"""


def right_baths(
    state: MPS, op: MPO, final_qubit: int, compute_device: torch.device | str
) -> list[torch.Tensor]:
    state_factor = state.factors[-1]
    bath = torch.ones(1, 1, 1, device=compute_device, dtype=state_factor.dtype)
    baths = [bath.to("cpu")]
    for i in range(len(state.factors) - 1, final_qubit - 1, -1):
        bath = new_right_bath(
            bath, state.factors[i].to(compute_device), op.factors[i].to(compute_device)
        )
        baths.append(bath.to("cpu"))
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
The baths must have shape (top, middle, bottom).
All tensors must be on the same device
"""


def apply_effective_Hamiltonian(
    state: torch.Tensor,
    ham: torch.Tensor,
    left_bath: torch.Tensor,
    right_bath: torch.Tensor,
) -> torch.Tensor:
    assert left_bath.ndim == 3 and left_bath.shape[0] == left_bath.shape[2]
    assert right_bath.ndim == 3 and right_bath.shape[0] == right_bath.shape[2]
    assert left_bath.shape[2] == state.shape[0] and right_bath.shape[2] == state.shape[2]
    assert left_bath.shape[1] == ham.shape[0] and right_bath.shape[1] == ham.shape[3]

    # the optimal contraction order depends on the details
    # this order seems to be pretty balanced, but needs to be
    # revisited when use-cases are more well-known
    state = torch.tensordot(left_bath, state, 1)
    state = torch.tensordot(state, ham, ([1, 2], [0, 2]))
    state = torch.tensordot(state, right_bath, ([3, 1], [1, 2]))
    return state


_TIME_CONVERSION_COEFF = 0.001  # Omega and delta are given in rad/ms, dt in ns


def evolve_pair(
    *,
    state_factors: list[torch.Tensor],
    baths: tuple[torch.Tensor, torch.Tensor],
    ham_factors: list[torch.Tensor],
    dt: float,
    orth_center_right: bool,
    is_hermitian: bool,
    config: MPSConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Time evolution of a pair of tensors of a tensor train using baths and truncated SVD.
    All tensors should be on the same device, where computation is performed.
    """
    assert len(state_factors) == 2
    assert len(baths) == 2
    assert len(ham_factors) == 2

    left_state_factor, right_state_factor = state_factors
    left_bath, right_bath = baths
    left_ham_factor, right_ham_factor = ham_factors

    left_bond_dim = left_state_factor.shape[0]
    right_bond_dim = right_state_factor.shape[-1]

    combined_state_factors = torch.tensordot(
        left_state_factor, right_state_factor, dims=1
    ).reshape(left_bond_dim, 4, right_bond_dim)

    combined_hamiltonian_factors = (
        torch.tensordot(left_ham_factor, right_ham_factor, dims=1)
        .transpose(2, 3)
        .reshape(left_ham_factor.shape[0], 4, 4, -1)
    )

    op = (
        lambda x: -_TIME_CONVERSION_COEFF
        * 1j
        * dt
        * apply_effective_Hamiltonian(
            x, combined_hamiltonian_factors, left_bath, right_bath
        )
    )

    evol = krylov_exp(
        op,
        combined_state_factors,
        exp_tolerance=config.precision * config.extra_krylov_tolerance,
        norm_tolerance=config.precision * config.extra_krylov_tolerance,
        max_krylov_dim=config.max_krylov_dim,
        is_hermitian=is_hermitian,
    ).reshape(left_bond_dim * 2, 2 * right_bond_dim)

    l, r = split_tensor(
        evol,
        max_error=config.precision,
        max_rank=config.max_bond_dim,
        orth_center_right=orth_center_right,
    )

    return l.reshape(left_bond_dim, 2, -1), r.reshape(-1, 2, right_bond_dim)


def evolve_single(
    *,
    state_factor: torch.Tensor,
    baths: tuple[torch.Tensor, torch.Tensor],
    ham_factor: torch.Tensor,
    dt: float,
    is_hermitian: bool,
    config: MPSConfig,
) -> torch.Tensor:
    """
    Time evolution of a single tensor of a tensor train using baths.
    """
    assert len(baths) == 2

    left_bath, right_bath = baths

    op = (
        lambda x: -_TIME_CONVERSION_COEFF
        * 1j
        * dt
        * apply_effective_Hamiltonian(
            x,
            ham_factor,
            left_bath,
            right_bath,
        )
    )

    return krylov_exp(
        op,
        state_factor,
        exp_tolerance=config.precision * config.extra_krylov_tolerance,
        norm_tolerance=config.precision * config.extra_krylov_tolerance,
        max_krylov_dim=config.max_krylov_dim,
        is_hermitian=is_hermitian,
    )
