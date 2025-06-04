import jax.numpy as jnp
import torch
import numpy as np

from emu_jax import vectorized_get_initial_diag_term, JaxRydbergHamiltonian
from emu_sv.hamiltonian import RydbergHamiltonian


def test_hamiltonian():
    qubit_count = 15
    torch.manual_seed(123)
    interaction_matrix = torch.rand(qubit_count, qubit_count, dtype=torch.complex128)
    interaction_matrix = interaction_matrix + interaction_matrix.T
    omegas = torch.rand(qubit_count, dtype=torch.complex128)
    deltas = torch.rand(qubit_count, dtype=torch.complex128)
    phis = torch.zeros(qubit_count, dtype=torch.complex128)
    state = torch.rand(1 << qubit_count, dtype=torch.complex128)
    state /= state.norm()

    initial_diag = vectorized_get_initial_diag_term(jnp.array(interaction_matrix), jnp.arange(1 << qubit_count))

    jax_hamiltonian = JaxRydbergHamiltonian(jnp.array(omegas), jnp.array(deltas), jnp.array(phis), initial_diag)

    jax_result = torch.from_numpy(np.asarray(jax_hamiltonian._apply_to(jnp.array(state))))

    expected_hamiltonian = RydbergHamiltonian(
        omegas=omegas,
        deltas=deltas,
        phis=phis,
        interaction_matrix=interaction_matrix,
        device=state.device,
    )

    expected_result = expected_hamiltonian * state

    assert torch.allclose(jax_result, expected_result)