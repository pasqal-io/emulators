def index_to_bitstring(nqubits: int, index: int) -> str:
    """
    Convert an integer index into its corresponding bitstring representation.
    """

    msg = f"index {index} can not exceed Hilbert space size d**{nqubits}"
    assert index < 2**nqubits, msg
    return format(index, f"0{nqubits}b")
