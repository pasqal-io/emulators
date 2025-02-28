import torch
import math
from emu_sv.state_vector import StateVector, inner

pi = torch.tensor(math.pi)

seed = 1337
dtype = torch.complex128
device = "cpu"
# device= "cuda"


def test_inner_algebra_sample():

    factor = torch.sqrt(torch.tensor(2.0))
    state1 = StateVector(
        torch.tensor(
            [1.0 / factor, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / factor], dtype=dtype
        )
    )

    state2 = StateVector(
        torch.tensor(
            [0, 1.0 / factor, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / factor], dtype=dtype
        )
    )

    inner_prod = inner(state1, state2)

    expected = torch.tensor(0.5, dtype=dtype)
    assert torch.allclose(inner_prod, expected)

    add_result = state1 + 2 * torch.exp(pi * 1.0j) * state2

    add_expected = torch.tensor(
        [1.0 / factor, -factor, 0.0, 0.0, 0.0, 0.0, 0.0, -1 / factor], dtype=dtype
    )

    assert torch.allclose(add_result.vector.cpu(), add_expected, rtol=0, atol=1e-6)

    torch.manual_seed(seed)
    sampling1 = state1.sample(1000)
    sampling2 = state2.sample(1000)

    assert sampling1["111"] == 485
    assert sampling1["001"] == 0
    assert sampling1["000"] == 515

    assert sampling2["111"] == 499
    assert sampling2["001"] == 501
    assert sampling2["000"] == 0

    sampling_sum = add_result.sample(1000)

    results = [0] * 8
    results[0] = 157
    results[1] = 654
    results[-1] = 189

    for i in range(8):
        assert sampling_sum["{0:03b}".format(i)] == results[i]


def test_from_string():
    torch.manual_seed(seed)

    basis = ("r", "g")
    state = {"rr": 1.0, "gg": 1.0}
    nqubits = 2

    from_string = StateVector.from_state_string(
        basis=basis, nqubits=nqubits, strings=state
    )

    sampling = from_string.sample(1000)

    values = from_string.vector

    assert torch.allclose(values[0], torch.tensor(0.7071 + 0.0j, dtype=dtype))
    assert torch.allclose(values[1], torch.tensor(0.0 + 0.0j, dtype=dtype))
    assert torch.allclose(values[2], torch.tensor(0.0 + 0.0j, dtype=dtype))
    assert torch.allclose(values[3], torch.tensor(0.7071 + 0.0j, dtype=dtype))

    assert sampling["00"] == 515
    assert sampling["11"] == 485
