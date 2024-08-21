# Explanation of config values

The following config values to EMU-MPS relate to the functioning of the tdvp algorithm used to evolve the quantum state in time, and will be explained in more detail below:

- precision
- max_bond_dim
- max_krylov_dim
- extra_krylov_tolerance

## precision

The 2-site TDVP algorithm used in EMU-MPS works by repeatedly time-evolving two neighbouring qubits in the MPS, and then truncating the result. Truncation is done by applying an [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) to the matrix representing the 2-qubit subsystem.
The singular values give much information about the state. Denote the singular values by $d_i$, and assume they are ordered in decreasing magnitude.
Then the norm of the state will be $\sum_i d_i^2$ and the entanglement entropy between the left and right parts of the state will be $\sum_i d_i \log_2(d_i)$, for example.

The truncation mentioned above functions by throwing away the smallest singular values, until their squared sum exceeds $precision^2$. The result is that the truncation procedure finds the smallest MPS whose distance is less than `precision` away from the original state.

This final interpretation is useful in estimating the error incurred in TDVP through truncation. TDVP sweeps from left two right over neighbouring pairs of qubits, and back. This means that for each timestep, `2*(nqubits-1)` truncations are performed, so by the triangle inequality, TDVP will output a state whose distance is than `2*(nqubits-1)*precision` from the state TDVP would have output without truncation. Note that the truncation errors will not all point in the same direction, so the actual error will likely be closer to `sqrt(2*(nqubits-1))*precision`, similar to the error in a gaussian random walk. It's default value is `1e-5`, meaning that each tdvp step will likely be accurate up to order `1e-4`.

## max_bond_dim

In addition to the above procedure, at each truncation step, no more than `max_bond_dim` singular values are kept. This parameter will impose a hard cap on the memory consumed by the quantum state, at the cost of losing control over the magnitude of the truncation errors.

## max_krylov_dim

Time evolution of each qubit pair is done by the [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm). This algorithm works by iteratively constructing a basis in which the Hamiltonian is close to diagonal. In this basis, the time-evolution of the state can efficiently be approximated by exponentiating a truncated Hamiltonian. The algorithm will never construct more than `max_krylov_dim` basis vectors, limiting the runtime and memory consumption of the time-evolution.

## extra_krylov_tolerance

In addition to the above hard cap on the number of basis vectors, the algorithm will also attempt to estimate the error incurred by computing the matrix exponential using only the current basis vectors. In principle, it is not needed to compute the time-evolution more precisely than `precision` since extra precision will be lost in the truncation. However, in practice it turns out that existing error estimates tend to underestimate the error. `extra_krylov_tolerance` is a fudge factor for how big the desired precision should be compared to `precision`. Its default value is `1e-3`.
