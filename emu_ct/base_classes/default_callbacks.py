from emu_ct.base_classes.callback import Callback
from typing import Any
from pulser.register.base_register import QubitId
from emu_ct.base_classes.config import BackendConfig
from emu_ct.base_classes.state import State
from emu_ct.base_classes.operator import Operator, OperatorString
from emu_ct.base_classes.common import apply_measurement_errors
from copy import deepcopy


class StateResult(Callback):
    def __init__(self, times: set[int]):
        super().__init__(times)

    def name(self) -> str:
        return "state"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        return deepcopy(state)


class BitStrings(Callback):
    def __init__(self, times: set[int], num_shots: int = 1000):
        super().__init__(times)
        self.num_shots = num_shots

    def name(self) -> str:
        return "bitstrings"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        bitstrings = state.sample(self.num_shots)

        if (
            config.noise_model is not None
            and "SPAM" in config.noise_model.noise_types
            and (
                config.noise_model.p_false_pos > 0.0
                or config.noise_model.p_false_neg > 0.0
            )
        ):
            bitstrings = apply_measurement_errors(
                bitstrings,
                p_false_pos=config.noise_model.p_false_pos,
                p_false_neg=config.noise_model.p_false_neg,
            )
        return bitstrings


_fidelity_counter = -1


class Fidelity(Callback):
    def __init__(self, times: set[int], state: State):
        super().__init__(times)
        global _fidelity_counter
        _fidelity_counter += 1
        self.index = _fidelity_counter
        self.state = state

    def name(self) -> str:
        return f"fidelity_{self.index}"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        return self.state.inner(state)


class CorrelationMatrix(Callback):
    def __init__(self, times: set[int], basis: tuple[str, ...], qubits: list[QubitId]):
        super().__init__(times)
        self.operators: list[list[Operator]] | None = None
        assert set(basis) == {
            "r",
            "g",
        }, "Correlation matrix is only defined on rydberg-ground"
        self.basis = basis
        self.qubits = qubits

    def name(self) -> str:
        return "correlation_matrix"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        if self.operators is None or not isinstance(self.operators[0], type(H)):
            self.operators = [
                [
                    H.from_operator_string(
                        self.basis,
                        self.qubits,
                        [(OperatorString([1.0], ["sigma_rr"]), {i, j})],
                    )
                    for j in self.qubits
                ]
                for i in self.qubits
            ]
        return [[state.inner(op * state).real for op in ops] for ops in self.operators]


class QubitDensity(Callback):
    def __init__(self, times: set[int], basis: tuple[str, ...], qubits: list[QubitId]):
        super().__init__(times)
        self.operators: list[Operator] | None = None
        assert set(basis) == {"r", "g"}, "Qubit density is only defined on rydberg-ground"
        self.basis = basis
        self.qubits = qubits

    def name(self) -> str:
        return "qubit_density"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        if self.operators is None or not isinstance(self.operators[0], type(H)):
            self.operators = [
                H.from_operator_string(
                    self.basis, self.qubits, [(OperatorString([1.0], ["sigma_rr"]), {i})]
                )
                for i in self.qubits
            ]
        return [state.inner(op * state).real for op in self.operators]


class Energy(Callback):
    def __init__(self, times: set[int]):
        super().__init__(times)

    def name(self) -> str:
        return "energy"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        return state.inner(H * state).real


class EnergyVariance(Callback):
    def __init__(self, times: set[int]):
        super().__init__(times)

    def name(self) -> str:
        return "energy_variance"

    def apply(self, config: BackendConfig, t: int, state: State, H: Operator) -> Any:
        h_state = H * state
        return h_state.inner(h_state).real - state.inner(H * state).real ** 2
