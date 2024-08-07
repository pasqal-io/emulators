import torch
import random


def compute_noise_from_lindbladians(lindbladians: list[torch.Tensor]) -> torch.Tensor:
    """
    Compute the single-qubit Hamiltonian noise term -0.5i∑L†L from all the given lindbladians.
    """

    assert all(
        lindbladian.shape == (2, 2) for lindbladian in lindbladians
    ), "Only single-qubit lindblad operators are supported"

    return (
        -1j
        / 2.0
        * sum(
            (lindbladian.T.conj() @ lindbladian for lindbladian in lindbladians),
            start=torch.zeros(2, 2, dtype=torch.complex128),
        )
    )


def pick_well_prepared_qubits(eta: float, n: int) -> list[bool]:
    """
    Randomly pick n booleans such that ℙ(False) = eta.
    """

    return [random.random() > eta for _ in range(n)]
