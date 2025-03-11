import math
import torch
from emu_base import State, DEVICE_COUNT
dtype = torch.complex128


class DensityMatrix(State):
    """Represents a density matrix in a computational basis.
    """

    def __init__(
        self,
        vector: torch.Tensor,
        *,
        gpu: bool = True,
    ):
        # NOTE: this accepts also zero vectors.

        assert math.log2(
            len(vector)
        ).is_integer(), "The number of elements in the vector should be power of 2"

        device = "cuda" if gpu and DEVICE_COUNT > 0 else "cpu"
        self.vector = vector.to(dtype=dtype, device=device)