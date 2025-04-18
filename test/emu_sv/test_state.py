import pytest
import torch
import math
from emu_sv import StateVector, inner
from emu_sv.utils import index_to_bitstring

pi = torch.tensor(math.pi)
factor = 1.0 / torch.sqrt(torch.tensor(2.0))

seed = 1337
dtype = torch.complex128
device = "cpu"
# device= "cuda"

gpu = False if device == "cpu" else True


def test_creating_state() -> None:
    # test constructor
    single_qubit_state = StateVector(torch.tensor([factor] * 2, dtype=dtype))
    assert single_qubit_state.n_qudits == 1
    assert math.isclose(1.0, single_qubit_state.norm(), rel_tol=1e-5)

    # test _normalize()
    state_5qubits_rnd = 2 * StateVector(torch.rand(2**5))  # factor 2 to have norm > 1
    assert state_5qubits_rnd.n_qudits == 5
    assert state_5qubits_rnd.norm() > 1.0
    state_5qubits_rnd._normalize()
    assert math.isclose(1.0, state_5qubits_rnd.norm(), rel_tol=1e-5)

    # test make()
    nqubits = 3
    state = StateVector.make(num_sites=nqubits)  # create |00..0>
    zero_state_tensor = torch.tensor([0] * 2**nqubits, dtype=dtype)
    zero_state_tensor[0] = 1
    zero_state = StateVector(zero_state_tensor)
    assert state.overlap(zero_state) == 1.0

    # test zero()
    nqubits = 3
    state = StateVector.zero(num_sites=nqubits, gpu=False)  # create |00..0>
    tensor = state.vector
    expected = torch.tensor([0] * 2**nqubits, dtype=dtype)
    assert torch.allclose(expected, tensor)


def test_inner_and_overlap() -> None:
    tensor1 = torch.tensor([factor, 0, 0, 0, 0, 0, 0, factor], dtype=dtype)
    tensor2 = torch.tensor([0, factor, 0, 0, 0, 0, 0, factor], dtype=dtype)

    state1 = StateVector(tensor1)
    state2 = StateVector(tensor2)

    inner_prod = inner(state1, state2)  # testing inner
    ovrlp = state1.overlap(state2)  # testing overlap

    expected = torch.dot(tensor1, tensor2)

    assert torch.allclose(inner_prod, expected)
    assert torch.allclose(ovrlp, expected)


def test_norm() -> None:
    nqubits = 5
    rnd_tensor = torch.rand((2**nqubits), dtype=dtype)
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

    assert "000" == index_to_bitstring(state.n_qudits, 0)
    assert "001" == index_to_bitstring(state.n_qudits, 1)
    assert "010" == index_to_bitstring(state.n_qudits, 2)
    assert "111" == index_to_bitstring(state.n_qudits, 7)

    indx = 8  # 8 is above Hilbert space of 3 qubits
    with pytest.raises(AssertionError) as msg:
        index_to_bitstring(state.n_qudits, indx)
    assert (
        str(msg.value) == f"index {indx} can not exceed Hilbert space size d**{nqubits}"
    )


def test_sample() -> None:

    torch.manual_seed(seed)

    tensor = torch.tensor([factor, 0, 0, 0, 0, 0, 0, factor], dtype=dtype)
    state = StateVector(tensor, gpu=False)
    sampling = state.sample(num_shots=1000)

    assert sampling["111"] == 485
    assert sampling["001"] == 0
    assert sampling["000"] == 515


def test_from_amplitudes() -> None:
    state = StateVector.from_state_amplitudes(
        eigenstates=("r", "g"),
        amplitudes={"rr": 1.0, "gg": 1.0},
    )
    expected_state = StateVector(torch.tensor([factor, 0, 0, factor], dtype=dtype))

    result = state.overlap(expected_state)
    assert math.isclose(result.real, 1.0, rel_tol=1e-5)
    assert math.isclose(result.imag, 0.0, rel_tol=1e-5)


@pytest.mark.parametrize(
    "eig, ampl",
    [
        (("r", "g"), {4 * "r": 1.0}),
        (("r", "g"), {4 * "g": 1.0}),
    ],
)
def test_to_abstr_repr(eig, ampl) -> None:
    initial_state = StateVector.from_state_amplitudes(eigenstates=eig, amplitudes=ampl)
    abstr = initial_state._to_abstract_repr()

    assert ampl == abstr["amplitudes"]
    assert eig == abstr["eigenstates"]


# @pytest.mark.parametrize(
#    "eig, ampl",
#    [
#        (("0", "1"), {4 * "r": 1.0}),
#        (("r", "g"), {4 * "1": 1.0}),
#    ],
# )
# def test_constructor(eig, ampl) -> None:
#    with pytest.raises(ValueError):
#        MPS.from_state_amplitudes(eigenstates=eig, amplitudes=ampl)
