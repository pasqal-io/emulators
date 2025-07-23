from typing import Callable

from pyparsing import Any


def index_to_bitstring(nqubits: int, index: int) -> str:
    """
    Convert an integer index into its corresponding bitstring representation.
    """

    msg = f"index {index} can not exceed Hilbert space size d**{nqubits}"
    assert index < 2**nqubits, msg
    return format(index, f"0{nqubits}b")


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
