from typing import Callable

import torch
from pyparsing import Any


def index_to_bitstring(nqubits: int, index: int) -> str:
    """
    Convert an integer index into its corresponding bitstring representation.
    """

    msg = f"index {index} can not exceed Hilbert space size d**{nqubits}"
    assert index < 2**nqubits, msg
    return format(index, f"0{nqubits}b")


def sample_indices(
    probabilities: torch.Tensor,
    num_samples: int,
    max_categories: int = 2**24,
) -> torch.Tensor:
    """
    Draw indices with replacement from a 1D tensor of non-negative,
    not necessarily normalized, probabilities.

    torch.multinomial cannot handle more than 2**24 categories, so larger
    tensors (25 qubits and more) are sampled by inverting the cumulative
    distribution, computed in float64.

    Args:
        probabilities: the weight of each index
        num_samples: how many indices to draw
        max_categories: largest size for which torch.multinomial is used

    Returns:
        the drawn indices, on the device of probabilities
    """
    if probabilities.numel() <= max_categories:
        return torch.multinomial(probabilities, num_samples, replacement=True)

    cdf = torch.cumsum(probabilities, dim=0, dtype=torch.float64)
    total = cdf[-1].item()
    if not total > 0:
        # same exception and message as torch.multinomial
        raise RuntimeError("invalid multinomial distribution (sum of probabilities <= 0)")
    cdf /= total  # equals 1.0 from the last non-zero probability on
    draws = torch.rand(num_samples, dtype=torch.float64, device=cdf.device)
    indices = torch.searchsorted(cdf, draws, right=True)
    return indices.clamp_(max=probabilities.numel() - 1)


def choose(
    state_vector_version: Callable,
    density_matrix_version: Callable,
) -> Callable:
    """Returns the observable result function that chooses the correct
    implementation based on the type of state (StateVector or DensityMatrix).
    """
    from emu_sv.state_vector import StateVector
    from emu_sv.density_matrix_state import DensityMatrix

    def result(self: Any, *, state: StateVector | DensityMatrix, **kwargs: Any) -> Any:
        if isinstance(state, StateVector):
            return state_vector_version(self, state=state, **kwargs)
        elif isinstance(state, DensityMatrix):
            return density_matrix_version(self, state=state, **kwargs)
        else:
            raise TypeError(f"Unsupported state: {type(state).__name__}")

    return result
