import torch


def expect_batch(
    state_vector: torch.Tensor, single_qubit_operators: torch.Tensor, n_qubits: int
) -> torch.Tensor:
    """
    Computes expectation values for each qubit and each single qubit operator in
    the batched input tensor.
    Returns a tensor T such that T[q, i] is the expectation value for qubit #q
    and operator single_qubit_operators[i].
    """
    return torch.stack(
        [
            torch.vmap(torch.trace)(
                single_qubit_operators
                @ torch.tensordot(
                    state_vector.view(2**i, 2, -1),
                    state_vector.view(2**i, 2, -1).conj(),
                    dims=([0, 2], [0, 2]),
                )
            )
            for i in range(n_qubits)
        ]
    )


def apply(
    state_vector: torch.Tensor, qubit_index: int, single_qubit_operator: torch.Tensor
) -> torch.Tensor:
    """
    Apply given single qubit operator to qubit qubit_index.
    """
    return (
        single_qubit_operator.to(state_vector.device)
        @ state_vector.view(2**qubit_index, 2, -1)
    ).view(-1)
