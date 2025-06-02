import torch

from emu_base.math.krylov_energy_min import krylov_energy_minimization
from emu_base.utils import deallocate_tensor
from emu_mps import MPSConfig
from emu_mps.utils import split_tensor
from emu_mps.tdvp import apply_effective_Hamiltonian


def minimize_energy_pair(
    *,
    state_factors: tuple[torch.Tensor],
    baths: tuple[torch.Tensor, torch.Tensor],
    ham_factors: tuple[torch.Tensor],
    orth_center_right: bool,
    is_hermitian: bool,
    config: MPSConfig,
    residual_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Minimizes the state factors (ψ_i, ψ_{i+1}) using the Lanczos/Arnoldi method
    """

    assert len(state_factors) == 2
    assert len(baths) == 2
    assert len(ham_factors) == 2

    left_state_factor, right_state_factor = state_factors
    left_bath, right_bath = baths
    left_ham_factor, right_ham_factor = ham_factors

    left_device = left_state_factor.device
    right_device = right_state_factor.device

    left_bond_dim = left_state_factor.shape[0]
    right_bond_dim = right_state_factor.shape[-1]

    # Computation is done on left_device (arbitrary)

    right_state_factor = right_state_factor.to(left_device)

    combined_state_factors = torch.tensordot(
        left_state_factor, right_state_factor, dims=1
    ).view(left_bond_dim, 4, right_bond_dim)

    deallocate_tensor(left_state_factor)
    deallocate_tensor(right_state_factor)

    left_ham_factor = left_ham_factor.to(left_device)
    right_ham_factor = right_ham_factor.to(left_device)
    right_bath = right_bath.to(left_device)

    combined_hamiltonian_factors = (
        torch.tensordot(left_ham_factor, right_ham_factor, dims=1)
        .transpose(2, 3)
        .contiguous()
        .view(left_ham_factor.shape[0], 4, 4, -1)
    )

    op = lambda x: apply_effective_Hamiltonian(
        x, combined_hamiltonian_factors, left_bath, right_bath
    )

    updated_state, updated_energy = krylov_energy_minimization(
        op,
        combined_state_factors,
        norm_tolerance=config.precision * config.extra_krylov_tolerance,
        residual_tolerance=residual_tolerance,
        max_krylov_dim=config.max_krylov_dim,
        is_hermitian=is_hermitian,
    )
    updated_state = updated_state.view(left_bond_dim * 2, 2 * right_bond_dim)

    l, r = split_tensor(
        updated_state,
        max_error=config.precision,
        max_rank=config.max_bond_dim,
        orth_center_right=orth_center_right,
    )

    return (
        l.view(left_bond_dim, 2, -1),
        r.view(-1, 2, right_bond_dim).to(right_device),
        updated_energy,
    )
