import pytest
import torch
import math
from emu_sv.state_vector import StateVector, inner

pi = torch.tensor(math.pi)
factor = 1.0 / torch.sqrt(torch.tensor(2.0))

seed = 1337
dtype = torch.complex128
device = "cpu"
# device= "cuda"


def test_creating_state() -> None:
    single_qubit_state = StateVector(torch.tensor([factor] * 2, dtype=dtype))
    assert single_qubit_state.n_qudits == 1
    assert math.isclose(1.0, single_qubit_state.norm(), rel_tol=1e-5)

    state_5qubits_rnd = 2 * StateVector(torch.rand(2**5))  # factor 2 to have norm > 1
    assert state_5qubits_rnd.n_qudits == 5
    assert state_5qubits_rnd.norm() > 1.0
    state_5qubits_rnd._normalize()
    assert math.isclose(1.0, state_5qubits_rnd.norm(), rel_tol=1e-5)


def test_inner_and_overlap() -> None:
    tensor1 = torch.tensor([factor, 0, 0, 0, 0, 0, 0, factor], dtype=dtype)
    tensor2 = torch.tensor([0, factor, 0, 0, 0, 0, 0, factor], dtype=dtype)

    state1 = StateVector(tensor1)
    state2 = StateVector(tensor2)

    inner_prod = inner(state1, state2)  # testing inner
    ovrlp = state1.overlap(state2)  # testing overlap

    expected = torch.dot(tensor1, tensor2)

    assert math.isclose(abs(inner_prod - expected), 0)
    assert math.isclose(abs(ovrlp - expected), 0)


def test_norm() -> None:
    nqubits = 5
    rnd_tensor = torch.rand(2**nqubits)
    state = StateVector(rnd_tensor)

    nrm_expected = torch.linalg.norm(rnd_tensor).item()
    nrm_state = state.norm()
    assert math.isclose(nrm_state, nrm_expected, rel_tol=1e-5)


def test_rmul_add() -> None:
    tensor = torch.tensor([factor, 0, 0, 0, 0, 0, 0, factor], dtype=dtype)
    coeff = 5
    state1 = StateVector(coeff * tensor)
    state2 = coeff * StateVector(tensor)  # test __rmul__
    assert math.isclose(state1.norm(), state2.norm(), rel_tol=1e-5)

    state2 = state1 + state1  # test __rmul__
    assert math.isclose((2 * state1).norm(), state2.norm(), rel_tol=1e-5)


def test_index_to_bitstring() -> None:
    nqubits = 3
    tensor = torch.tensor([1] * 2**nqubits, dtype=dtype)
    state = StateVector(tensor)
    state._normalize()
    state._index_to_bitstring(0)

    assert "000" == state._index_to_bitstring(0)
    assert "001" == state._index_to_bitstring(1)
    assert "010" == state._index_to_bitstring(2)
    assert "111" == state._index_to_bitstring(7)

    indx = 8  # 8 is above Hilbert space of 3 qubits
    with pytest.raises(AssertionError) as msg:
        state._index_to_bitstring(indx)
    assert (
        str(msg.value) == f"index {indx} can not exceed Hilbert space size d**{nqubits}"
    )


def test_inner_algebra_sample() -> None:
    tensor1 = torch.tensor([factor, 0, 0, 0, 0, 0, 0, factor], dtype=dtype)
    tensor2 = torch.tensor([0, factor, 0, 0, 0, 0, 0, factor], dtype=dtype)

    state1 = StateVector(tensor1)
    state2 = StateVector(tensor2)

    inner_prod = inner(state1, state2)  # testing inner
    ovrlp = state1.overlap(state2)  # testing overlap

    expected = torch.dot(tensor1, tensor2)

    assert math.isclose(abs(inner_prod - expected), 0)
    assert math.isclose(abs(ovrlp - expected), 0)

    add_result = state1 + (-2) * state2

    tensor_expected = [factor, -1 / factor, 0, 0, 0, 0, 0, -factor]
    add_expected = torch.tensor(tensor_expected, dtype=dtype)

    assert torch.allclose(add_result.vector.cpu(), add_expected, rtol=0, atol=1e-6)

    torch.manual_seed(seed)
    sampling1 = StateVector(state1.vector, gpu=False).sample(num_shots=1000)
    sampling2 = StateVector(state2.vector, gpu=False).sample(num_shots=1000)

    assert sampling1["111"] == 485
    assert sampling1["001"] == 0
    assert sampling1["000"] == 515

    assert sampling2["111"] == 499
    assert sampling2["001"] == 501
    assert sampling2["000"] == 0

    sampling_sum = StateVector(add_result.vector, gpu=False).sample(num_shots=1000)

    results = [0] * 8
    results[0] = 157
    results[1] = 654
    results[-1] = 189

    for i in range(8):
        assert sampling_sum["{0:03b}".format(i)] == results[i]


def test_from_string() -> None:
    torch.manual_seed(seed)

    basis = ("r", "g")
    state = {"rr": 1.0, "gg": 1.0}

    from_string = StateVector.from_state_amplitudes(
        eigenstates=basis,
        amplitudes=state,
    )

    sampling = StateVector(from_string.vector, gpu=False).sample(num_shots=1000)

    values = from_string.vector

    assert torch.allclose(values[0], torch.tensor(0.7071 + 0.0j, dtype=dtype))
    assert torch.allclose(values[1], torch.tensor(0.0 + 0.0j, dtype=dtype))
    assert torch.allclose(values[2], torch.tensor(0.0 + 0.0j, dtype=dtype))
    assert torch.allclose(values[3], torch.tensor(0.7071 + 0.0j, dtype=dtype))

    assert sampling["00"] == 515
    assert sampling["11"] == 485
