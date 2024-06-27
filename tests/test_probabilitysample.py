import torch
from emu_ct import MPS

from .utils_testing import ghz_state_factors


seed = 1337  # any number will do
device = "cpu"
dtype = torch.complex128


def test_sampling_ghz5_mps():
    device = "cpu"
    torch.manual_seed(seed)
    num_qubits = 5
    shots = 1000
    ghz_mps = MPS(ghz_state_factors(num_qubits, device=device))
    ghz_mps.truncate()
    bitstring = ghz_mps.sample(shots)

    assert bitstring.get("11111") == 505
    assert bitstring.get("00000") == 495


def test_not_orthogonalized_state():
    torch.manual_seed(seed)
    device = "cpu"
    shots = 1000
    # right orthogonalized mps
    l_factor1 = torch.tensor([[[1, 0], [0, 1j]]], dtype=torch.complex128, device=device)
    l_factor2 = torch.tensor(
        [[[1, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=torch.complex128, device=device
    )
    # the orthogonality center is at the end
    l_factor3 = (
        1
        / torch.sqrt(torch.tensor(2))
        * torch.tensor([[[1], [0]], [[0], [1]]], dtype=torch.complex128, device=device)
    )
    bell = MPS([l_factor1, l_factor2, l_factor3])
    bell.truncate()
    bitstring = bell.sample(shots)
    assert bitstring.get("111") == 489
    assert bitstring.get("000") == 511
