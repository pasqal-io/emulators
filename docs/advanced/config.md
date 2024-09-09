# Explanation of config values

The following config values to EMU-MPS relate to the functioning of the tdvp algorithm used to evolve the quantum state in time, and will be explained in more detail below:

- dt
- precision
- max_bond_dim
- max_krylov_dim
- extra_krylov_tolerance

## dt

Emu-MPS assumes the Hamiltonian is piece-wise constant in time for intervals of `dt`. It then constructs the Hamiltonian by sampling the amplitude, detuning and phase of the pulse midway through the interval, and making a single Hamiltonian. The TDVP algorithm is then used to evolve the state by `dt`. There are two sources of error related to `dt`.

- The discretization of the pulse
- [TDVP](errors.md)

Both sources of error dictate that `dt` shall not be too small, but the functioning of TDVP also dictates that a very small `dt` requires improving the precision, as described in the next section.

## precision

The 2-site TDVP algorithm used in EMU-MPS works by repeatedly time-evolving two neighbouring qubits in the MPS, and then truncating the result. Truncation is done by applying an [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) to the matrix representing the 2-qubit subsystem.
The singular values give much information about the state. Denote the singular values by $d_i$, and assume they are ordered in decreasing magnitude.
Then the norm of the state will be $\sum_i d_i^2$ and the entanglement entropy between the left and right parts of the state will be $\sum_i d_i \log_2(d_i)$, for example.

The truncation mentioned above functions by throwing away the smallest singular values, until their squared sum exceeds $precision^2$. The result is that the truncation procedure finds the smallest MPS whose distance is less than `precision` away from the original state.
As described on [the page of errors in TDVP](errors.md#truncation-of-the-state), the error in TDVP increases with the number of timesteps, so for long sequences or small `dt`, improving the precision might be required.

## max_bond_dim

In addition to the above procedure, at each truncation step, no more than `max_bond_dim` singular values are kept. This parameter will impose a hard cap on the memory consumed by the quantum state, at the cost of losing control over the magnitude of the truncation errors.

## max_krylov_dim

Time evolution of each qubit pair is done by the [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm). This algorithm works by iteratively constructing a basis in which the Hamiltonian is close to diagonal. In this basis, the time-evolution of the state can efficiently be approximated by exponentiating a truncated Hamiltonian. The algorithm will never construct more than `max_krylov_dim` basis vectors, limiting the runtime and memory consumption of the time-evolution.

Note that the number of iterations the Lanczos algorithm needs to converge to the required tolerance depends on the `dt` parameter also present in the config (see the [api specification](../api.md#mpsconfig)). The default value of `max_krylov_dim` should work for most reasonable values of `dt`, so if you get a recursion error out of the Lanczos algorithm, ensure you understand how the [errors in TDVP](errors.md) depend on `dt`.

## extra_krylov_tolerance

In addition to the above hard cap on the number of basis vectors, the algorithm will also attempt to estimate the error incurred by computing the matrix exponential using only the current basis vectors. In principle, it is not needed to compute the time-evolution more precisely than `precision` since extra precision will be lost in the truncation. However, in practice it turns out that existing error estimates tend to underestimate the error. `extra_krylov_tolerance` is a fudge factor for how big the desired precision should be compared to `precision`. Its default value is `1e-3`.
