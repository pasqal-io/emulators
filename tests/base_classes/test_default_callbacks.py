from collections import Counter
from unittest.mock import MagicMock, Mock
from pytest import approx
import torch

from emu_mps import (
    MPO,
    MPS,
    BitStrings,
    CorrelationMatrix,
    Energy,
    EnergyVariance,
    Expectation,
    Fidelity,
    MPSConfig,
    QubitDensity,
    StateResult,
)
from emu_mps.base_classes.results import Results


def test_state_result():
    callback = StateResult(times=[10])
    result = Results()
    config = None
    nqubits = 5
    state = MPS(nqubits)

    norm2 = state.inner(state)
    H = None
    callback(config, 5, state, H, result)
    assert len(result._results.keys()) == 0
    callback(config, 10, state, H, result)

    output = result[callback.name()][10]

    # same mathematical state
    assert output.inner(output) == norm2
    assert output.inner(state) == norm2
    # but a copy
    assert state is not output
    for i in range(nqubits):
        assert state.factors[i] is not output.factors[i]


def test_bit_strings():
    callback = BitStrings(times=[10], num_shots=1000)
    result = Results()
    mock_noise = MagicMock()
    mock_noise.p_false_pos = 0.1
    mock_noise.p_false_neg = 0.3
    mock_noise.noise_types = ["SPAM"]
    config = MPSConfig(noise_model=mock_noise)
    state = MagicMock()
    state.sample = Mock(return_value=Counter({"a": 4, "b": 2}))

    H = None
    callback(config, 5, state, H, result)
    assert len(result._results.keys()) == 0
    callback(config, 10, state, H, result)
    output = result[callback.name()][10]
    print(output)
    assert output["a"] == 4
    state.sample.assert_called_with(1000, 0.1, 0.3)


def test_qubit_density():
    nqubits = 5
    callback = QubitDensity(times=[10], basis={"r", "g"}, nqubits=nqubits)
    result = Results()
    config = None
    state = MPS(
        [torch.tensor([0, 2], dtype=torch.complex128).reshape(1, 2, 1)] * nqubits
    ) + MPS([torch.tensor([3, 0], dtype=torch.complex128).reshape(1, 2, 1)] * nqubits)
    H = MPO
    callback(config, 5, state, H, result)
    assert len(result._results.keys()) == 0
    callback(config, 10, state, H, result)

    output = result[callback.name()][10]
    assert output == [4**nqubits] * nqubits


def test_correlation_matrix():
    nqubits = 5
    callback = CorrelationMatrix(times=[10], basis={"r", "g"}, nqubits=nqubits)
    result = Results()
    config = None
    state = MPS(
        [torch.tensor([0, 2], dtype=torch.complex128).reshape(1, 2, 1)] * nqubits
    ) + MPS([torch.tensor([3, 0], dtype=torch.complex128).reshape(1, 2, 1)] * nqubits)
    H = MPO
    callback(config, 5, state, H, result)
    assert len(result._results.keys()) == 0
    callback(config, 10, state, H, result)

    output = result[callback.name()][10]
    assert len(output) == nqubits
    for vector in output:
        assert vector == [4**nqubits] * nqubits


def test_expectation():
    nqubits = 5

    basis = {"r", "g"}
    x = {"sigma_rg": 1.0, "sigma_gr": 1.0}
    xs = [(x, [i for i in range(nqubits)])]
    op = MPO.from_operator_string(basis, nqubits, [(1.0, xs)])

    callback = Expectation(times=[10], operator=op)
    result = Results()
    config = None

    state = MPS(
        [torch.tensor([0, 2], dtype=torch.complex128).reshape(1, 2, 1)] * nqubits
    ) + MPS([torch.tensor([3, 0], dtype=torch.complex128).reshape(1, 2, 1)] * nqubits)

    H = None
    callback(config, 5, state, H, result)
    assert len(result._results.keys()) == 0
    callback(config, 10, state, H, result)

    output = result[callback.name()][10]
    assert output == approx(2 * (6**nqubits))


def test_fidelity():
    nqubits = 5

    fid_state = MPS(
        [torch.tensor([1, 0], dtype=torch.complex128).reshape(1, 2, 1)] * nqubits
    )
    callback = Fidelity(times=[10], state=fid_state)

    result = Results()
    config = None

    state = MPS(
        [torch.tensor([0, 2], dtype=torch.complex128).reshape(1, 2, 1)] * nqubits
    ) + MPS([torch.tensor([3, 0], dtype=torch.complex128).reshape(1, 2, 1)] * nqubits)

    H = None
    callback(config, 5, state, H, result)
    assert len(result._results.keys()) == 0
    callback(config, 10, state, H, result)

    output = result[callback.name()][10]
    assert output == 3**nqubits


def test_energy():
    nqubits = 5

    basis = {"r", "g"}
    x = {"sigma_rg": 1.0, "sigma_gr": 1.0}
    xs = [(x, [i for i in range(nqubits)])]
    H = MPO.from_operator_string(basis, nqubits, [(1.0, xs)])

    callback = Energy(times=[10])
    result = Results()
    config = None

    state = MPS(
        [torch.tensor([0, 2], dtype=torch.complex128).reshape(1, 2, 1)] * nqubits
    ) + MPS([torch.tensor([3, 0], dtype=torch.complex128).reshape(1, 2, 1)] * nqubits)

    callback(config, 5, state, H, result)
    assert len(result._results.keys()) == 0
    callback(config, 10, state, H, result)

    output = result[callback.name()][10]
    assert output == approx(2 * (6**nqubits))


def test_energy_variance():
    nqubits = 5

    basis = {"r", "g"}
    x = {"sigma_rg": 1.0, "sigma_gr": 1.0}
    xs = [(x, [i for i in range(nqubits)])]
    H = MPO.from_operator_string(basis, nqubits, [(1.0, xs)])

    callback = EnergyVariance(times=[10])
    result = Results()
    config = None

    state = MPS(
        [torch.tensor([0, 2], dtype=torch.complex128).reshape(1, 2, 1)] * nqubits
    ) + MPS([torch.tensor([3, 0], dtype=torch.complex128).reshape(1, 2, 1)] * nqubits)

    callback(config, 5, state, H, result)
    assert len(result._results.keys()) == 0
    callback(config, 10, state, H, result)

    output = result[callback.name()][10]
    assert output == approx(state.inner(state) - 4 * 6 ** (2 * nqubits))
