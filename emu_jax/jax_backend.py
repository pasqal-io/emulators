import math
from typing import Mapping, Sequence, Type, TypeVar, Counter
import torch
from functools import partial

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
    def __init__(self, qubit_count: int, data: jax.Array):
        self.qubit_count = qubit_count
        self.data = data

    @staticmethod
    def make(qubit_count: int):
        data = jnp.zeros(1 << qubit_count, dtype=complex)
        data = data.at[0].set(1.0)
        return JaxStateVector(qubit_count, data)

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
        return jnp.vdot(self.data, other.data)


class JaxRydbergHamiltonian:
    def __init__(
        self,
        omegas,
        deltas,
        phis,
        initial_diag,
    ):
        self.qubit_count: int = omegas.shape[0]
        self.omegas = omegas / 2.0
        self.deltas = deltas
        self.phis = phis
        self.initial_diag = initial_diag

    def expect(self, state: JaxStateVector):
        return jnp.vdot(state.data, self._apply_to(state.data))

    def apply_to(self, state: JaxStateVector):
        return JaxStateVector(qubit_count=self.qubit_count, data=self._apply_to(state.data))

    def _apply_to(self, data: jax.Array):
        return apply_ham(self.omegas, self.deltas, self.phis, self.initial_diag, data)


jax.tree_util.register_pytree_node(
    JaxRydbergHamiltonian,
    lambda ham: ((ham.omegas * 2, ham.deltas, ham.phis, ham.initial_diag), None),
    lambda aux_data, children: JaxRydbergHamiltonian(*children)
)


@jax.jit
def get_initial_diag_term_inner_loop_body(params):
    result, interaction_matrix, current_qubit, secondary_qubit, secondary = params
    return jnp.where(secondary & 1, result + interaction_matrix[current_qubit, secondary_qubit],
                     result), interaction_matrix, current_qubit, secondary_qubit - 1, secondary >> 1


@jax.jit
def get_initial_diag_term_loop_body(params):
    diag_element, interaction_matrix, current_qubit, current = params

    inner_result, *_ = jax.lax.while_loop(lambda params: params[4] != 0, get_initial_diag_term_inner_loop_body, (0, interaction_matrix, current_qubit, current_qubit - 1, current >> 1))
    return jnp.where(current & 1, diag_element + inner_result, diag_element), interaction_matrix, current_qubit - 1, current >> 1


@jax.jit
def get_initial_diag_term(interaction_matrix, index):
    qubit_count = interaction_matrix.shape[0]

    result, *_ = jax.lax.while_loop(lambda params: params[3] != 0, get_initial_diag_term_loop_body, (0., interaction_matrix, qubit_count - 1, index))

    return result


vectorized_get_initial_diag_term = jax.vmap(get_initial_diag_term, (None, 0), 0)


@jax.jit
def deltas_loop_body(params):
    diag_element, deltas, current_qubit, current = params
    return jnp.where(current & 1, diag_element - deltas[current_qubit], diag_element), deltas, current_qubit - 1, current >> 1


@jax.jit
def apply_ham_single(omegas, deltas, phis, initial_diag, v, index, value):
    qubit_count = omegas.shape[0]

    diag_element, *_ = jax.lax.while_loop(lambda params: params[3] != 0, deltas_loop_body, (initial_diag[index], deltas, qubit_count - 1, index))

    result = diag_element * value

    for qubit in range(qubit_count):
        flipped = index ^ (1 << (qubit_count - qubit - 1))
        result += omegas[qubit] * v[flipped]

    return result


vectorized_apply_ham_single = jax.vmap(apply_ham_single, (None, None, None, None, None, 0, 0), 0)


@jax.jit
def apply_ham(omegas, deltas, phis, initial_diag, v):
    return vectorized_apply_ham_single(omegas, deltas, phis, initial_diag, v, jnp.arange(v.shape[0]), v)


@partial(jax.jit, static_argnames=['j'])
def krylov_step(dt, ham, lanczos_vectors, j, T):
    w = -1j * dt * ham._apply_to(lanczos_vectors[-1])
    n = jnp.linalg.norm(w)

    if j >= 1:
        overlap1 = jax.numpy.vdot(lanczos_vectors[-2], w)
        T = T.at[j-1, j].set(overlap1)
        w = w - overlap1 * lanczos_vectors[-2]

    overlap2 = jax.numpy.vdot(lanczos_vectors[-1], w)
    T = T.at[j, j].set(overlap2)
    w = w - overlap2 * lanczos_vectors[-1]
    remaining_norm = jnp.linalg.norm(w)
    T = T.at[j+1, j].set(remaining_norm)
    lanczos_vectors = lanczos_vectors + (w / remaining_norm,)

    T = T.at[j + 2, j + 1].set(1)

    expdT = jax.scipy.linalg.expm(T[: j + 3, : j + 3])

    # Local truncation error estimation
    err1 = abs(expdT[j + 1, 0])
    err2 = abs(expdT[j + 2, 0] * n)

    err = jnp.where(err1 < err2, err1, err1 * err2 / (err1 - err2))
    coeffs = expdT[:len(lanczos_vectors), 0]

    return err, T, coeffs, lanczos_vectors


def result(coeffs, lanczos_vectors):
    return sum(a * b for a, b in zip(coeffs, lanczos_vectors))


max_krylov_dim = 10  # FIXME, low to make compile times short.


def evolve_krylov_body(j, dt, ham, lanczos_vectors, T, *, krylov_tolerance):
    err, T, coeffs, lanczos_vectors = krylov_step(dt, ham, lanczos_vectors, j, T)

    if j >= max_krylov_dim:
        # Cannot raise exception in jitted function
        return result(coeffs, lanczos_vectors)

    return jax.lax.cond(err < krylov_tolerance, lambda *_: result(coeffs, lanczos_vectors),
                        partial(evolve_krylov_body, j + 1, krylov_tolerance=krylov_tolerance), dt, ham, lanczos_vectors, T)


@jax.jit
def evolve_krylov(dt, ham, v, *, krylov_tolerance):
    initial_norm = jnp.linalg.norm(v)
    T = jnp.zeros((max_krylov_dim + 2, max_krylov_dim + 2), dtype=complex)

    lanczos_vectors = (v / initial_norm,)

    return initial_norm * evolve_krylov_body(0, dt, ham, lanczos_vectors, T, krylov_tolerance=krylov_tolerance)


class JaxSVBackend(EmulatorBackend):
    """
    A backend for emulating Pulser sequences using state vectors and sparse matrices.
    Noisy simulation is supported by solving the Lindblad equation and using effective
    noise channel or jump operators
    """

    default_config = SVConfig()

    def run(self) -> Results:
        assert self._config.initial_state is None
        krylov_tolerance = self._config.krylov_tolerance

        pulser_data = PulserData(sequence=self._sequence, config=self._config, dt=self._config.dt)

        results = Results(atom_order=(), total_duration=pulser_data.target_times[-1])

        pulser_data.omega = jnp.array(pulser_data.omega)
        pulser_data.delta = jnp.array(pulser_data.delta)
        pulser_data.phi = jnp.array(pulser_data.phi)
        assert not jnp.any(pulser_data.phi)

        step_count = pulser_data.omega.shape[0]
        qubit_count = pulser_data.omega.shape[1]

        state = JaxStateVector.make(qubit_count)
        interaction_matrix = pulser_data.full_interaction_matrix.to(torch.complex128)

        initial_diag = vectorized_get_initial_diag_term(jnp.array(interaction_matrix), jnp.arange(1 << qubit_count))

        for step in range(step_count):
            print(step)
            dt = pulser_data.target_times[step + 1] - pulser_data.target_times[step]
            converted_dt = dt * _TIME_CONVERSION_COEFF
            omegas = pulser_data.omega[step]
            deltas = pulser_data.delta[step]
            phis = pulser_data.phi[step]

            ham = JaxRydbergHamiltonian(omegas, deltas, phis, initial_diag)

            state.data = evolve_krylov(
                converted_dt, ham, state.data, krylov_tolerance=krylov_tolerance
            )

            norm_time = pulser_data.target_times[step + 1] / pulser_data.target_times[-1]
            for callback in self._config.observables:
                callback(
                    self._config,
                    norm_time,
                    state,
                    ham,
                    results,
                )

        return results
