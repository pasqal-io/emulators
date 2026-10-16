import pytest
import torch
import math
from emu_sv import StateVector, inner
from emu_sv.utils import index_to_bitstring, sample_indices

pi = torch.tensor(math.pi)
factor = 1.0 / torch.sqrt(torch.tensor(2.0))

seed = 1337
dtype = torch.complex128
device = "cuda" if torch.cuda.is_available() else "cpu"

gpu = False if device == "cpu" else True


def test_creating_state() -> None:
    # test constructor
    single_qubit_state = StateVector(torch.tensor([factor] * 2, dtype=dtype), gpu=gpu)
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
    zero_state_tensor = torch.tensor([0] * 2**nqubits, dtype=dtype, device=device)
    zero_state_tensor[0] = 1
    zero_state = StateVector(zero_state_tensor)
    assert state.overlap(zero_state) == 1.0

    # test zero()
    nqubits = 3
    state = StateVector.zero(num_sites=nqubits, gpu=gpu)  # create |00..0>
    tensor = state.data
    expected = torch.tensor([0] * 2**nqubits, dtype=dtype, device=device)
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
    assert torch.allclose(ovrlp, torch.abs(expected) ** 2)


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


@pytest.mark.usefixtures("restore_rng_state")
def test_sample() -> None:

    torch.manual_seed(seed)

    tensor = torch.tensor([factor, 0, 0, 0, 0, 0, 0, factor], dtype=dtype)
    state = StateVector(tensor, gpu=False)  # rng is different on GPU
    sampling = state.sample(num_shots=1000)

    assert sampling["111"] == 485
    assert sampling["001"] == 0
    assert sampling["000"] == 515


@pytest.mark.usefixtures("restore_rng_state")
def test_sample_indices_uses_multinomial_up_to_max_categories() -> None:
    probabilities = torch.rand(2**5, dtype=torch.float64, device=device)

    torch.manual_seed(seed)
    expected = torch.multinomial(probabilities, 1000, replacement=True)
    torch.manual_seed(seed)
    outcomes = sample_indices(probabilities, 1000)

    assert torch.equal(outcomes, expected)


@pytest.mark.usefixtures("restore_rng_state")
def test_sample_indices_beyond_max_categories() -> None:
    # max_categories=1 forces the inverse-CDF path used beyond 2**24 categories
    torch.manual_seed(seed)
    num_samples = 100_000
    weights = torch.tensor([1.0, 0.0, 3.0, 6.0, 0.0], dtype=torch.float64)
    probabilities = 2.5 * weights.to(device)  # not normalized

    outcomes = sample_indices(probabilities, num_samples, max_categories=1)

    assert outcomes.device == probabilities.device
    counts = torch.bincount(outcomes.cpu(), minlength=5).to(torch.float64)
    assert counts[1] == 0 and counts[4] == 0
    expected = num_samples * weights / weights.sum()
    nonzero = weights > 0
    chi2 = (((counts - expected)[nonzero]) ** 2 / expected[nonzero]).sum()
    assert chi2 < 18.4  # 99.99% quantile of chi2 with 2 degrees of freedom


@pytest.mark.usefixtures("restore_rng_state")
@pytest.mark.parametrize("index", [0, 7])
def test_sample_indices_beyond_max_categories_single_outcome(index: int) -> None:
    probabilities = torch.zeros(8, dtype=torch.float64, device=device)
    probabilities[index] = 0.3

    outcomes = sample_indices(probabilities, 1000, max_categories=1)

    assert torch.all(outcomes == index)


def test_sample_indices_beyond_max_categories_zero_probabilities() -> None:
    probabilities = torch.zeros(8, dtype=torch.float64, device=device)
    with pytest.raises(RuntimeError, match="sum of probabilities <= 0"):
        sample_indices(probabilities, 10, max_categories=1)


def test_from_amplitudes() -> None:
    state = StateVector.from_state_amplitudes(
        eigenstates=("r", "g"),
        amplitudes={"rr": 1.0, "gg": 1.0},
    )
    expected_state = StateVector(torch.tensor([factor, 0, 0, factor], dtype=dtype))

    result = state.overlap(expected_state)
    assert math.isclose(result, 1.0, rel_tol=1e-5)


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


def test_constructor() -> None:
    with pytest.raises(ValueError):
        eig = ("r", "g")
        ampl = {4 * "1": 1.0}
        StateVector.from_state_amplitudes(eigenstates=eig, amplitudes=ampl)

    with pytest.raises(NotImplementedError):
        eig = ("0", "1")
        ampl = {4 * "1": 1.0}
        StateVector.from_state_amplitudes(eigenstates=eig, amplitudes=ampl)
