import torch

from typing import List, Optional, Union


def ghz_state_factors(
    nqubits: int,
    dtype: torch.dtype = torch.complex128,
    device: Optional[Union[str, torch.device]] = None,
) -> List[torch.Tensor]:
    assert nqubits >= 2
    core_1 = (
        1
        / torch.sqrt(torch.tensor([2.0], device=device, dtype=dtype))
        * torch.tensor(
            [
                [
                    [1.0 + 0.0j, 0.00 + 0.0j],
                    [
                        0.00 + 0.0j,
                        1.0 + 0.0j,
                    ],
                ]
            ],
            dtype=dtype,
            device=device,
        )
    )
    core_mid = torch.tensor(
        [
            [[1.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 0.0 + 0.0j]],
            [[0.0 + 0.0j, 0.0 + 0.0j], [0.0 + 0.0j, 1.0 + 0.0j]],
        ],
        dtype=dtype,
        device=device,
    )
    # similar to core_mid, except no bond to the right
    core3 = torch.tensor(
        [[[1.0 + 0.0j], [0.0 + 0.0j]], [[0.0 + 0.0j], [1.0 + 0.0j]]],
        dtype=dtype,
        device=device,
    )

    cores = [core_1]
    for _ in range(nqubits - 2):
        cores.append(core_mid)
    cores.append(core3)
    return cores
