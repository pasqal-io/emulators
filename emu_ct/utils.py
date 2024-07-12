from typing import List

import torch


DEVICE_COUNT = torch.cuda.device_count()


def _determine_cutoff_index(d: torch.Tensor, max_error: float) -> int:
    assert max_error > 0
    squared_max_error = max_error * max_error
    acc = 0
    for i in range(d.shape[0]):
        acc += d[i]
        if acc > squared_max_error:
            return i
    return 0  # type: ignore[no-any-return]


"""
Computes a low-rank approximation svd of m using the Eckart-Young-Mirsky theorem.
"""


def split_tensor(
    m: torch.Tensor,
    max_error: float = 1e-5,
    max_rank: int = 1024,
    orth_center_right: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
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


"""
Evenly distributes each tensor in the list to a device.
If num_devices_to_use is 0, then all tensors go to CPU.
"""


def assign_devices(tensors: List[torch.Tensor], num_devices_to_use: int) -> None:
    num_devices_to_use = min(len(tensors), num_devices_to_use)

    if num_devices_to_use <= 0:
        for i in range(len(tensors)):
            tensors[i] = tensors[i].to("cpu")
        return

    tensors_per_device = len(tensors) // num_devices_to_use

    if len(tensors) % num_devices_to_use != 0:
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
