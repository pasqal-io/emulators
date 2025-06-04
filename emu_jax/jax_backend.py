import math
from typing import Mapping, Sequence, Type, TypeVar, Counter
import torch

from pulser.backend import EmulatorBackend
from pulser.backend import Results

from emu_base import PulserData
from emu_sv import SVConfig
from pulser.backend import State
from pulser.backend.state import Eigenstate

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


_TIME_CONVERSION_COEFF = 0.001  # Omega and delta are given in rad/μs, dt in ns


JaxStateVectorType = TypeVar("JaxStateVectorType", bound="JaxStateVector")


class JaxStateVector(State[complex, jax.Array]):
    def __init__(self, qubit_count: int):
        self.qubit_count = qubit_count
        self.state = jnp.zeros(1 << qubit_count, dtype=complex)
        self.state = self.state.at[0].set(1.0)


    @classmethod
    def _from_state_amplitudes(
        cls: Type[JaxStateVectorType],
        *,
        eigenstates: Sequence[Eigenstate],
        amplitudes: Mapping[str, complex],
    ) -> tuple[JaxStateVectorType, Mapping[str, complex]]:
        pass


    @property
    def n_qudits(self) -> int:
        return self.qubit_count

    def sample(
        self,
        *,
        num_shots: int = 1000,
        one_state: Eigenstate | None = None,
        p_false_pos: float = 0.0,
        p_false_neg: float = 0.0,
    ) -> Counter[str]:
        raise NotImplementedError()

    def overlap(self, other: JaxStateVectorType, /) -> jax.Array:
        return jnp.dot(self.state, other)


exp_tolerance = 1e-7


def get_initial_diag_term(interaction_matrix, index):
    qubit_count = interaction_matrix.shape[0]

    def loop_body(params):
        diag_element, current_qubit, current = params

        def inner_loop_body(params):
            result, secondary_qubit, secondary = params
            return jnp.where(secondary & 1, result + interaction_matrix[current_qubit, secondary_qubit],
                             result), secondary_qubit - 1, secondary >> 1

        inner_result, _, _ = jax.lax.while_loop(lambda params: params[2] != 0, inner_loop_body, (0, current_qubit - 1, current >> 1))
        return jnp.where(current & 1, diag_element + inner_result, diag_element), current_qubit - 1, current >> 1

    result, _, _ = jax.lax.while_loop(lambda params: params[2] != 0, loop_body, (0, qubit_count - 1, index))

    return result


vectorized_get_initial_diag_term = jax.vmap(get_initial_diag_term, (None, 0), 0)


def apply_ham_single(omegas, deltas, phis, initial_diag, v, index, value):
    qubit_count = omegas.shape[0]

    def loop_body(params):
        diag_element, current_qubit, current = params
        return jnp.where(current & 1, diag_element - deltas[current_qubit], diag_element), current_qubit - 1, current >> 1

    diag_element, _, _ = jax.lax.while_loop(lambda params: params[2] != 0, loop_body, (initial_diag[index], qubit_count - 1, index))

    result = diag_element * value

    for qubit in range(qubit_count):
        flipped = index ^ (1 << (qubit_count - qubit - 1))
        result += omegas[qubit] * v[flipped]

    return result


vectorized_apply_ham_single = jax.vmap(apply_ham_single, (None, None, None, None, None, 0, 0), 0)


def apply_ham(omegas, deltas, phis, initial_diag, v):
    # assert not jnp.any(phis)  # Not implemented
    return vectorized_apply_ham_single(omegas, deltas, phis, initial_diag, v, jnp.arange(v.shape[0]), v)


def exponentiate(op, v):
    def loop_body(params):
        index, accumulator, ex, err = params
        ex = op(ex) / index

        return index + 1, accumulator + ex, ex, jnp.linalg.norm(ex)

    _, result, _, _ = jax.lax.while_loop(lambda x: x[3] > exp_tolerance, loop_body, (1, v, v, 999.))

    return result


def evolve(dt, omegas, deltas, phis, initial_diag, v):
    return exponentiate(lambda x: -1j * dt * apply_ham(omegas, deltas, phis, initial_diag, x), v)


class JaxSVBackend(EmulatorBackend):
    """
    A backend for emulating Pulser sequences using state vectors and sparse matrices.
    Noisy simulation is supported by solving the Lindblad equation and using effective
    noise channel or jump operators
    """

    default_config = SVConfig()

    def run(self) -> Results:
        """
        Emulates the given sequence.

        Returns:
            the simulation results
        """
        assert self._config.initial_state is None

        pulser_data = PulserData(sequence=self._sequence, config=self._config, dt=self._config.dt)

        results = Results(atom_order=(), total_duration=pulser_data.target_times[-1])

        pulser_data.omega = jnp.array(pulser_data.omega) / 2
        pulser_data.delta = jnp.array(pulser_data.delta)
        pulser_data.phi = jnp.array(pulser_data.phi)

        step_count = pulser_data.omega.shape[0]
        qubit_count = pulser_data.omega.shape[1]

        state = JaxStateVector(qubit_count)
        interaction_matrix = pulser_data.full_interaction_matrix.to(torch.complex128)

        initial_diag = vectorized_get_initial_diag_term(jnp.array(interaction_matrix), jnp.arange(1 << qubit_count))

        for step in range(step_count):
            print(step)
            dt = pulser_data.target_times[step + 1] - pulser_data.target_times[step]
            converted_dt = dt * _TIME_CONVERSION_COEFF
            omegas = pulser_data.omega[step]
            deltas = pulser_data.delta[step]
            phis = pulser_data.phi[step]

            state.state = evolve(
                converted_dt, omegas, deltas, phis, initial_diag, state.state
            )

            # print(state.state[:10])
            energy = jnp.vdot(state.state, apply_ham(omegas, deltas, phis, initial_diag, state.state))

            print(energy)

        return results