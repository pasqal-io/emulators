import random


def pick_well_prepared_qubits(eta: float, n: int) -> list[bool]:
    """
    Randomly pick n booleans such that ℙ(False) = eta.
    """

    return [random.random() > eta for _ in range(n)]
