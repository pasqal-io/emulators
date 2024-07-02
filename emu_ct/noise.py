import torch


def compute_noise_from_lindbladians(lindbladians: list[torch.Tensor]) -> torch.Tensor:
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
