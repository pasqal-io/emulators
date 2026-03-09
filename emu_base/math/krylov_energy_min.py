import torch
from typing import Callable, Tuple

DEFAULT_MAX_KRYLOV_DIM: int = 200


def Ritz_vector(
        coefficients: list[torch.Tensor],
        lanczos_vectors: list[torch.Tensor],
) -> torch.Tensor:
    """
    Ritz vectors are approximations to the eigenvectors of a matrix
    that are obtained using the Arnoldi method
    """
    final_state: torch.Tensor = sum(c * vec for c, vec in zip(coefficients, lanczos_vectors))
    final_state /= final_state.norm()

    return final_state


def _lowest_eigen_pairs(
    T_trunc: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Return the lowest eigenpair of the hermitian matrix T_trunc.
    eigenvalues of T_trunc are Ritz values == roots of the characteristic polynomial
    """
    #assert torch.allclose(T_trunc, T_trunc.mH, rtol=1e-5, atol=1e-8), "not hermitean"
    if not torch.allclose(T_trunc, T_trunc.mH, atol=1e-7):
        m = T_trunc - T_trunc.mH
        n = m.norm()
        print(n)
        
    eig_energy, eig_state = torch.linalg.eigh(T_trunc)
    
    L, V = torch.linalg.eig(T_trunc)
    assert torch.allclose(torch.real(L).min(), eig_energy[0], atol=1e-7), f"{L, '\n', eig_energy}"

    return eig_energy, eig_state


class KrylovEnergyResult:
    def __init__(
        self,
        ground_state: torch.Tensor,
        ground_energy: float,
        converged: bool,
        happy_breakdown: bool,
        iteration_count: int,
    ):
        self.ground_state = ground_state
        self.ground_energy = ground_energy
        self.converged = converged
        self.happy_breakdown = happy_breakdown
        self.iteration_count = iteration_count


def add_next_krylov_vector_old(T_mat, op, lanczos_vectors, norm_tolerance):
    j = len(lanczos_vectors) - 1
    w = op(lanczos_vectors[-1])

    for k in range(max(0, j - 1), j + 1):
        alpha = torch.vdot(lanczos_vectors[k].reshape(-1), w.reshape(-1))
        #alpha = torch.tensordot(lanczos_vectors[k].conj(), w, dims=w.dim())
        T_mat[k, j] = alpha
        w -= alpha * lanczos_vectors[k]

    beta = torch.linalg.norm(w)
    # Happy breakdown
    if beta < norm_tolerance:
        return True


    T_mat[j, j + 1] = beta
    T_mat[j + 1, j] = beta
    w /= beta
    lanczos_vectors.append(w)
    return False


def add_next_krylov_vector(T_mat, op, lanczos_vectors, norm_tolerance):
    """
    One Lanczos step for Hermitian/symmetric operator op.
    Updates tridiagonal T_mat and appends v_{j+1}.
    Returns True if basis grew, False on (happy) breakdown.
    """
    j = len(lanczos_vectors) - 1
    vj = lanczos_vectors[j]

    # w = A v_j
    w = op(vj)

    # subtract beta_{j-1} v_{j-1} using stored beta (not a projection!)
    if j > 0:
        beta_prev = T_mat[j - 1, j]
        w = w - beta_prev * lanczos_vectors[j - 1]

    # alpha_j = <v_j, w>
    alpha = torch.vdot(vj.reshape(-1), w.reshape(-1))  # vdot conjugates first arg
    # If op is Hermitian, alpha should be real (up to numerical noise)
    # alpha = alpha.real

    T_mat[j, j] = alpha
    w = w - alpha * vj

    beta = torch.linalg.norm(w)
    if beta < norm_tolerance:
        return True  # happy breakdown

    T_mat[j, j + 1] = beta
    T_mat[j + 1, j] = beta

    lanczos_vectors.append(w / beta)
    return False



def krylov_energy_minimization_impl(
    op: Callable[[torch.Tensor], torch.Tensor],
    psi_local: torch.Tensor,
    residual_tolerance: float,
    norm_tolerance: float,
    max_krylov_dim: int = DEFAULT_MAX_KRYLOV_DIM,
) -> KrylovEnergyResult:
    """
    Computes the ground state of a Hermitian operator using Lanczos algorithm.
    The Rayleigh quotient ⟨ψ|H|ψ⟩ is minimized over the Krylov subspace.

    The convergence of the results is determined by a residual norm criterion or a happy breakdown.
    """

    device = psi_local.device
    dtype = psi_local.dtype

    initial_norm = psi_local.norm()
    lanczos_vectors = [psi_local / initial_norm]
    T = torch.zeros(max_krylov_dim + 2, max_krylov_dim + 2, dtype=dtype, device=device)

    converged = False
    happy_breakdown = False
    iteration_count = 0


    for j in range(max_krylov_dim):
        happy_breakdown = add_next_krylov_vector(T, op, lanczos_vectors, norm_tolerance)
        iteration_count += 1
        size = len(lanczos_vectors)

        T_truncated = T[:size, :size]
        vals, vecs = _lowest_eigen_pairs(T_truncated)
        # residual norm convergence check
        final_state = Ritz_vector(vecs[:, 0], lanczos_vectors)
        ground_energy = vals[0]
        if happy_breakdown:
            break
        
        residual = op(final_state) - vals[0] * final_state
        residual_norm = torch.linalg.norm(residual.reshape(-1))
        if residual_norm < residual_tolerance:
            converged = True
            break

        #if happy_breakdown and not converged:
            


        # Restart
        trunc = 10
        if size > 2*trunc:
            vals = vals[:trunc]
            vecs = vecs[:, :trunc]

            new_lanczos_vectors = []
            for k in range(trunc):
                lcsz_vec = Ritz_vector(vecs[:,k], lanczos_vectors)
                lcsz_vec /= lcsz_vec.norm()
                new_lanczos_vectors.append(lcsz_vec)
            lanczos_vectors = new_lanczos_vectors
            
            T = torch.zeros(max_krylov_dim + 2, max_krylov_dim + 2, dtype=dtype, device=device)
            for k in range(len(vals)):
                T[k,k] = vals[k]



        
    return KrylovEnergyResult(
        ground_state=final_state,
        ground_energy=ground_energy.item(),
        converged=converged,
        happy_breakdown=happy_breakdown,
        iteration_count=iteration_count,
    )


def krylov_energy_minimization(
    op: Callable[[torch.Tensor], torch.Tensor],
    v: torch.Tensor,
    norm_tolerance: float,
    residual_tolerance: float,
    max_krylov_dim: int = DEFAULT_MAX_KRYLOV_DIM,
) -> Tuple[torch.Tensor, float]:

    result = krylov_energy_minimization_impl(
        op=op,
        psi_local=v,
        residual_tolerance=residual_tolerance,
        norm_tolerance=norm_tolerance,
        max_krylov_dim=max_krylov_dim,
    )

    if not result.converged and not result.happy_breakdown:
        raise RecursionError(
            f"Krylov ground state solver did not converge within allotted {max_krylov_dim} iterations."
        )

    return result.ground_state, result.ground_energy
