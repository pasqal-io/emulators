from copy import deepcopy
from typing import Any

from emu_base.base_classes.callback import Callback
from emu_base.base_classes.config import BackendConfig
from emu_base.base_classes.operator import Operator
from emu_base.base_classes.state import State


class StateResult(Callback):
    """
    Store the quantum state in whatever format the backend provides

    Args:
        evaluation_times: the times at which to store the state
    """

    def __init__(self, evaluation_times: set[int]):
        super().__init__(evaluation_times)

    def name(self) -> str:
        return "state"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        return deepcopy(state)


class BitStrings(Callback):
    """
    Store bitstrings sampled from the current state. Error rates are taken from the config
    passed to the run method of the backend. The bitstrings are stored as a Counter[str].

    Args:
        evaluation_times: the times at which to sample bitstrings
        num_shots: how many bitstrings to sample each time this observable is computed
    """

    def __init__(self, evaluation_times: set[int], num_shots: int = 1000):
        super().__init__(evaluation_times)
        self.num_shots = num_shots

    def name(self) -> str:
        return "bitstrings"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        p_false_pos = (
            0.0 if config.noise_model is None else config.noise_model.p_false_pos
        )
        p_false_neg = (
            0.0 if config.noise_model is None else config.noise_model.p_false_neg
        )

        return state.sample(self.num_shots, p_false_pos, p_false_neg)


_fidelity_counter = -1


class Fidelity(Callback):
    """
    Store the inner product of the given state |ψ>with the state |φ(t)> obtained by time evolution,
    i.e. <ψ|φ(t)>.

    Args:
        evaluation_times: the times at which to compute the fidelity
        state: the state |ψ>. Note that this must be of appropriate type for the backend
    """

    def __init__(self, evaluation_times: set[int], state: State):
        super().__init__(evaluation_times)
        global _fidelity_counter
        _fidelity_counter += 1
        self.index = _fidelity_counter
        self.state = state

    def name(self) -> str:
        return f"fidelity_{self.index}"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        return self.state.inner(state)


_expectation_counter = -1


class Expectation(Callback):
    """
    Store the expectation of the given operator on the current state (i.e. <φ(t)|operator|φ(t)>).

    Args:
        evaluation_times: the times at which to compute the expectation
        operator: the operator to measure. Must be of appropriate type for the backend.
    """

    def __init__(self, evaluation_times: set[int], operator: Operator):
        super().__init__(evaluation_times)
        global _expectation_counter
        _expectation_counter += 1
        self.index = _expectation_counter
        self.operator = operator

    def name(self) -> str:
        return f"expectation_{self.index}"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        return self.operator.expect(state)


class CorrelationMatrix(Callback):
    """
    Store the correlation matrix for the current state.
    Requires specification of the basis used in the emulation
    https://pulser.readthedocs.io/en/stable/conventions.html
    It currently only supports the rydberg basis ('r','g').
    The diagonal of this matrix is the QubitDensity. The correlation matrix
    is stored as a list of lists.

    Args:
        evaluation_times: the times at which to compute the correlation matrix
        basis: the basis used by the sequence
        nqubits: the number of qubits in the Register
    """

    def __init__(self, evaluation_times: set[int], basis: tuple[str, ...], nqubits: int):
        super().__init__(evaluation_times)
        self.operators: list[list[Operator]] | None = None
        assert set(basis) == {
            "r",
            "g",
        }, "Correlation matrix is only defined on rydberg-ground"
        self.basis = basis
        self.nqubits = nqubits

    def name(self) -> str:
        return "correlation_matrix"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        if hasattr(state, "get_correlation_matrix") and callable(
            state.get_correlation_matrix
        ):
            return state.get_correlation_matrix()

        if self.operators is None or not isinstance(self.operators[0], type(H)):
            self.operators = [
                [
                    H.from_operator_string(
                        self.basis,
                        self.nqubits,
                        [(1.0, [({"sigma_rr": 1.0}, list({i, j}))])],
                    )
                    for j in range(self.nqubits)
                ]
                for i in range(self.nqubits)
            ]
        return [[op.expect(state).real for op in ops] for ops in self.operators]


class QubitDensity(Callback):
    """
    Requires specification of the basis used in the emulation
    https://pulser.readthedocs.io/en/stable/conventions.html
    It currently only supports the rydberg basis ('r','g') and
    it computer the probability that each qubit is in the r state.
    The qubit density is stored as a list.

    Args:
        evaluation_times: the times at which to compute the density
        basis: the basis used by the sequence
        nqubits: the number of qubits in the Register
    """

    def __init__(self, evaluation_times: set[int], basis: tuple[str, ...], nqubits: int):
        super().__init__(evaluation_times)
        self.operators: list[Operator] | None = None
        assert set(basis) == {"r", "g"}, "Qubit density is only defined on rydberg-ground"
        self.basis = basis
        self.nqubits = nqubits

    def name(self) -> str:
        return "qubit_density"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        if self.operators is None or not isinstance(self.operators[0], type(H)):
            self.operators = [
                H.from_operator_string(
                    self.basis, self.nqubits, [(1.0, [({"sigma_rr": 1.0}, [i])])]
                )
                for i in range(self.nqubits)
            ]
        return [op.expect(state).real for op in self.operators]


class Energy(Callback):
    """
    Store the expectation value of the current Hamiltonian (i.e. <φ(t)|H(t)|φ(t)>)

    Args:
        evaluation_times: the times at which to compute the expectation
    """

    def __init__(self, evaluation_times: set[int]):
        super().__init__(evaluation_times)

    def name(self) -> str:
        return "energy"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        return state.inner(H * state).real


class EnergyVariance(Callback):
    """
    Store the variance of the current Hamiltonian (i.e. <φ(t)|H(t)^2|φ(t)> - <φ(t)|H(t)|φ(t)>^2)

    Args:
        evaluation_times: the times at which to compute the variance
    """

    def __init__(self, evaluation_times: set[int]):
        super().__init__(evaluation_times)

    def name(self) -> str:
        return "energy_variance"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        h_state = H * state
        return h_state.inner(h_state).real - state.inner(H * state).real ** 2


class SecondMomentOfEnergy(Callback):
    """
    Store the expectation value <φ(t)|H(t)^2|φ(t)>.
    Useful for computing the variance when averaging over many executions of the program.

    Args:
        evaluation_times: the times at which to compute the variance
    """

    def __init__(self, evaluation_times: set[int]):
        super().__init__(evaluation_times)

    def name(self) -> str:
        return "second_moment_of_energy"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        h_state = H * state
        return h_state.inner(h_state).real
