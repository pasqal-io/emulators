from typing import List, Optional
import torch
import random
from collections import Counter

DEVICE_COUNT = torch.cuda.device_count()


def new_left_bath(
    bath: torch.Tensor, state: torch.Tensor, op: torch.Tensor
) -> torch.Tensor:
    # this order is more efficient than contracting the op first in general
    bath = torch.tensordot(bath, state.conj(), ([0], [0]))
    bath = torch.tensordot(bath, op.to(bath.device), ([0, 2], [0, 1]))
    return torch.tensordot(bath, state, ([0, 2], [0, 1]))


def _determine_cutoff_index(d: torch.Tensor, max_error: float) -> int:
    assert max_error > 0
    squared_max_error = max_error * max_error
    acc = 0
    for i in range(d.shape[0]):
        acc += d[i]
        if acc > squared_max_error:
            return i
    return 0  # type: ignore[no-any-return]


def split_tensor(
    m: torch.Tensor,
    max_error: float = 1e-5,
    max_rank: int = 1024,
    orth_center_right: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes a low-rank approximation split of m using the Eckart-Young-Mirsky theorem.
    """
    assert m.ndim == 2

    if orth_center_right:
        d, q = torch.linalg.eigh(m @ m.T.conj())
        max_bond = max(
            _determine_cutoff_index(d, max_error),
            d.shape[0] - max_rank,
        )
        left = q[:, max_bond:]
        right = q.T.conj() @ m
        right = right[max_bond:, :]
    else:
        d, q = torch.linalg.eigh(m.T.conj() @ m)
        max_bond = max(
            _determine_cutoff_index(d, max_error),
            d.shape[0] - max_rank,
        )
        right = q.T.conj()[max_bond:, :]
        left = m @ q
        left = left[:, max_bond:]

    return left, right


def truncate_impl(
    factors: list[torch.tensor],
    max_error: float = 1e-5,
    max_rank: int = 1024,
) -> None:
    """
    Eigenvalues-based truncation of a matrix product.
    An in-place operation.

    Note:
        Sweeps from right to left.
        Requires the matrix product to be orthogonalized on the last element.
        At each step moves the orthogonal center to the left while truncating.
    """
    for i in range(len(factors) - 1, 0, -1):
        factor_shape = factors[i].shape

        l, r = split_tensor(
            factors[i].reshape(factor_shape[0], -1),
            max_error=max_error,
            max_rank=max_rank,
            orth_center_right=False,
        )

        factors[i] = r.reshape(-1, *factor_shape[1:])
        factors[i - 1] = torch.tensordot(
            factors[i - 1], l.to(factors[i - 1].device), dims=1
        )


def assign_devices(tensors: List[torch.Tensor], num_gpus_to_use: int) -> None:
    """
    Evenly distributes each tensor in the list to a device.
    If num_gpus_to_use is 0, then all tensors go to CPU.
    """
    num_gpus_to_use = min(len(tensors), num_gpus_to_use)

    if num_gpus_to_use <= 0:
        for i in range(len(tensors)):
            tensors[i] = tensors[i].to("cpu")
        return

    tensors_per_device = len(tensors) // num_gpus_to_use

    if len(tensors) % num_gpus_to_use != 0:
        tensors_per_device += 1

    for i in range(len(tensors)):
        tensors[i] = tensors[i].to(f"cuda:{i // tensors_per_device}")


def extended_mps_factors(
    mps_factors: list[torch.Tensor], where: list[bool]
) -> list[torch.Tensor]:
    """
    Given a valid list of MPS factors, accounting for qubits marked as `True` in `where`,
    fills the `False` positions with new qubits in the |0> state.
    """
    assert len(mps_factors) == sum(1 for b in where if b)

    bond_dimension = 1
    factor_index = 0
    result = []
    for is_factor in where:
        assert 0 <= factor_index <= len(mps_factors)

        if is_factor:
            result.append(mps_factors[factor_index])
            bond_dimension = mps_factors[factor_index].shape[2]
            factor_index += 1
        elif factor_index == len(mps_factors):
            factor = torch.zeros(bond_dimension, 2, 1, dtype=torch.complex128)
            factor[:, 0, :] = torch.eye(bond_dimension, 1)
            bond_dimension = 1
            result.append(factor)
        else:
            factor = torch.zeros(
                bond_dimension, 2, bond_dimension, dtype=torch.complex128
            )
            factor[:, 0, :] = torch.eye(bond_dimension, bond_dimension)
            result.append(factor)
    return result


def extended_mpo_factors(
    mpo_factors: list[torch.Tensor], where: list[bool]
) -> list[torch.Tensor]:
    """
    Given a valid list of MPO factors, accounting for qubits marked as `True` in `where`,
    fills the `False` positions with new MPO identity factors.
    """
    assert len(mpo_factors) == sum(1 for b in where if b)

    bond_dimension = 1
    factor_index = 0
    result = []
    for is_factor in where:
        assert 0 <= factor_index <= len(mpo_factors)

        if is_factor:
            result.append(mpo_factors[factor_index])
            bond_dimension = mpo_factors[factor_index].shape[3]
            factor_index += 1
        elif factor_index == len(mpo_factors):
            factor = torch.zeros(bond_dimension, 2, 2, 1, dtype=torch.complex128)
            factor[:, 0, 0, :] = torch.eye(bond_dimension, 1)
            factor[:, 1, 1, :] = torch.eye(bond_dimension, 1)
            bond_dimension = 1
            result.append(factor)
        else:
            factor = torch.zeros(
                bond_dimension, 2, 2, bond_dimension, dtype=torch.complex128
            )
            factor[:, 0, 0, :] = torch.eye(bond_dimension, bond_dimension)
            factor[:, 1, 1, :] = torch.eye(bond_dimension, bond_dimension)
            result.append(factor)
    return result


def get_extended_site_index(
    where: list[bool], desired_index: Optional[int]
) -> Optional[int]:
    """
    Returns the index in `where` that has `desired_index` preceding True elements.

    This function is used to find the index of the orthogonality center in an MPS obtained
    with `extended_mps_factors` in the presence of dark qubits:
        `where` is the mask specifying whether qubits are well-prepared.
        `desired_index` is the index of the orthogonality center of the MPS without dark qubits.
        The return value is then the index of the orthogonality center
        in the full MPS with added dark qubits.
    """

    if desired_index is None:
        return None

    index = -1
    for extended_index, boolean_value in enumerate(where):
        if boolean_value:
            index += 1
            if index == desired_index:
                return extended_index

    raise ValueError(f"Index {desired_index} does not exist")


def readout_with_error(c: str, *, p_false_pos: float, p_false_neg: float) -> str:
    # p_false_pos = false positive, p_false_neg = false negative
    r = random.random()
    if c == "0" and r < p_false_pos:
        return "1"

    if c == "1" and r < p_false_neg:
        return "0"

    return c


def apply_measurement_errors(
    bitstrings: Counter[str], *, p_false_pos: float, p_false_neg: float
) -> Counter[str]:
    """
    Given a bag of sampled bitstrings, returns another bag of bitstrings
    sampled with readout/measurement errors.

        p_false_pos: probability of false positive
        p_false_neg: probability of false negative
    """

    result: Counter[str] = Counter()
    for bitstring, count in bitstrings.items():
        for _ in range(count):
            bitstring_with_error = "".join(
                readout_with_error(c, p_false_pos=p_false_pos, p_false_neg=p_false_neg)
                for c in bitstring
            )

            result[bitstring_with_error] += 1

    return result


n_operator = torch.tensor(
    [
        [0, 0],
        [0, 1],
    ],
    dtype=torch.complex128,
)


def tensor_trace(tensor: torch.Tensor, dim1: int, dim2: int) -> torch.Tensor:
    """
    Contract two legs of a single tensor.
    """
    assert tensor.shape[dim1] == tensor.shape[dim2], "dimensions should match"
    return tensor.diagonal(offset=0, dim1=dim1, dim2=dim2).sum(-1)
