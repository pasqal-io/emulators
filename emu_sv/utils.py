import math


def index_to_bitstring(shape_vector: int, index: int) -> str:
    """
    Convert an integer index into its corresponding bitstring representation.
    """
    nqubits = int(math.log2(shape_vector))
    return format(index, f"0{nqubits}b")
