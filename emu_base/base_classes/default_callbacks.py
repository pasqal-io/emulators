from copy import deepcopy
from typing import Any

from emu_base.base_classes.callback import Callback, AggregationType
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

    name = "state"

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

    name = "bitstrings"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        p_false_pos = (
            0.0 if config.noise_model is None else config.noise_model.p_false_pos
        )
        p_false_neg = (
            0.0 if config.noise_model is None else config.noise_model.p_false_neg
        )

        return state.sample(self.num_shots, p_false_pos, p_false_neg)

    default_aggregation_type = AggregationType.BAG_UNION


_fidelity_counter = -1


class Fidelity(Callback):
    """
    Store $<ψ|φ(t)>$ for the given state $|ψ>$,
    and the state $|φ(t)>$ obtained by time evolution.

    Args:
        evaluation_times: the times at which to compute the fidelity
        state: the state |ψ>. Note that this must be of appropriate type for the backend

    Examples:
        >>> state = State.from_state_string(...) #see State API
        >>> fidelity = Fidelity([400], state) #measure fidelity on state at t=400ns
    """

    def __init__(self, evaluation_times: set[int], state: State):
        super().__init__(evaluation_times)
        global _fidelity_counter
        _fidelity_counter += 1
        self.index = _fidelity_counter
        self.state = state

    @property
    def name(self) -> str:
        return f"fidelity_{self.index}"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        return self.state.inner(state)


_expectation_counter = -1


class Expectation(Callback):
    """
    Store the expectation of the given operator on the current state
    (i.e. $\\langle φ(t)|\\mathrm{operator}|φ(t)\\rangle$).

    Args:
        evaluation_times: the times at which to compute the expectation
        operator: the operator to measure. Must be of appropriate type for the backend.

    Examples:
        >>> op = Operator.from_operator_string(...) #see Operator API
        >>> expectation = Expectation([400], op) #measure the expecation of op at t=400ns
    """

    def __init__(self, evaluation_times: set[int], operator: Operator):
        super().__init__(evaluation_times)
        global _expectation_counter
        _expectation_counter += 1
        self.index = _expectation_counter
        self.operator = operator

    @property
    def name(self) -> str:
        return f"expectation_{self.index}"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        return self.operator.expect(state)

    default_aggregation_type = AggregationType.MEAN


class CorrelationMatrix(Callback):
    """
    Store the correlation matrix for the current state.
    Requires specification of the basis used in the emulation
    https://pulser.readthedocs.io/en/stable/conventions.html
    It currently supports
    - the rydberg basis ('r','g')
    - the xy basis ('0', '1')
    and returns

    `[[<φ(t)|n_i n_j|φ(t)> for j in qubits] for i in qubits]`

    n_i being the operator that projects qubit i onto the state that measures as 1.
    The diagonal of this matrix is the QubitDensity. The correlation matrix
    is stored as a list of lists.

    Args:
        evaluation_times: the times at which to compute the correlation matrix
        basis: the basis used by the sequence
        nqubits: the number of qubits in the Register

    Notes:
        See the API for `Operator.from_operator_string` for an example of what to do with
        basis and nqubits.
    """

    def __init__(self, evaluation_times: set[int], basis: tuple[str, ...], nqubits: int):
        super().__init__(evaluation_times)
        self.operators: list[list[Operator]] | None = None
        self.basis = set(basis)
        if self.basis == {"r", "g"}:
            self.op_string = "rr"
        elif self.basis == {"0", "1"}:
            self.op_string = "11"
        else:
            raise ValueError("Unsupported basis provided")
        self.nqubits = nqubits

    name = "correlation_matrix"

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
                        [(1.0, [({self.op_string: 1.0}, list({i, j}))])],
                    )
                    for j in range(self.nqubits)
                ]
                for i in range(self.nqubits)
            ]
        return [[op.expect(state).real for op in ops] for ops in self.operators]

    default_aggregation_type = AggregationType.MEAN


class QubitDensity(Callback):
    """
    Requires specification of the basis used in the emulation
    https://pulser.readthedocs.io/en/stable/conventions.html
    It currently supports
    - the rydberg basis ('r','g')
    - the xy basis ('0', '1')
    and returns

    `[<φ(t)|n_i|φ(t)> for i in qubits]`

    n_i being the operator that projects qubit i onto the state that measures as 1.
    The qubit density is stored as a list.

    Args:
        evaluation_times: the times at which to compute the density
        basis: the basis used by the sequence
        nqubits: the number of qubits in the Register

    Notes:
        See the API for `State.from_state_string` for an example of what to do with
        basis and nqubits.
    """

    def __init__(self, evaluation_times: set[int], basis: tuple[str, ...], nqubits: int):
        super().__init__(evaluation_times)
        self.operators: list[Operator] | None = None
        self.basis = set(basis)
        if self.basis == {"r", "g"}:
            self.op_string = "rr"
        elif self.basis == {"0", "1"}:
            self.op_string = "11"
        else:
            raise ValueError("Unsupported basis provided")
        self.nqubits = nqubits

    name = "qubit_density"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        if self.operators is None or not isinstance(self.operators[0], type(H)):
            self.operators = [
                H.from_operator_string(
                    self.basis, self.nqubits, [(1.0, [({self.op_string: 1.0}, [i])])]
                )
                for i in range(self.nqubits)
            ]
        return [op.expect(state).real for op in self.operators]

    default_aggregation_type = AggregationType.MEAN


class Energy(Callback):
    """
    Store the expectation value of the current Hamiltonian
    (i.e. $\\langle φ(t)|H(t)|φ(t) \\rangle$)

    Args:
        evaluation_times: the times at which to compute the expectation
    """

    def __init__(self, evaluation_times: set[int]):
        super().__init__(evaluation_times)

    name = "energy"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        return H.expect(state).real

    default_aggregation_type = AggregationType.MEAN


class EnergyVariance(Callback):
    """
    Store the variance of the current Hamiltonian
    (i.e. $\\langle φ(t)|H(t)^2|φ(t)\\rangle - \\langle φ(t)|H(t)|φ(t)\\rangle^2$)

    Args:
        evaluation_times: the times at which to compute the variance
    """

    def __init__(self, evaluation_times: set[int]):
        super().__init__(evaluation_times)

    name = "energy_variance"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        h_squared = H @ H
        return h_squared.expect(state).real - H.expect(state).real ** 2

    # Explicitely setting this to None out of safety: in the case of MonteCarlo,
    # the aggregated variance cannot be computed from this callback.
    # Instead, one first need to average Energy and SecondMomentOfEnergy,
    # and then compute the variance with the formula:
    # AggregatedEnergyVariance = AveragedSecondMomentOfEnergy - AveragedEnergy**2
    default_aggregation_type = None


class SecondMomentOfEnergy(Callback):
    """
    Store the expectation value $\\langle φ(t)|H(t)^2|φ(t)\\rangle$.
    Useful for computing the variance when averaging over many executions of the program.

    Args:
        evaluation_times: the times at which to compute the variance
    """

    def __init__(self, evaluation_times: set[int]):
        super().__init__(evaluation_times)

    name = "second_moment_of_energy"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        h_squared = H @ H
        return h_squared.expect(state).real

    default_aggregation_type = AggregationType.MEAN
