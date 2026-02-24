# An explanation of the sources of error in TDVP

Emu-mps uses a 2nd order 2-site time-dependent variational principle to compute the time evolution of the qubit registers ([see here](algorithms.md)).
There are four sources of error inherent in this algorithm ([see here](https://tensornetwork.org/mps/algorithms/timeevo/tdvp.html))

A. Effective description of long-range terms in the Hamiltonian (long range approximation)

B. Looping over pairs of qubits (Trotter like error - sweep errors)

C. iterative computation of the 2-site effective evolution (numerical Lanczos(Krylov)error )

D. Truncation of the state (truncation error)

Let us briefly explain how each of these terms introduce errors into the simulation, and let us try to estimate their size.

## Effective description of long-range terms in the Hamiltonian

The Rydberg Hamiltonian is long range, so when evolving 2 neighbouring qubits in one of the TDVP steps, it is necessary to approximate terms coupling these two qubits to far away qubits. Specifically, say we are evolving the pair $(j,j+1)$ and the Hamiltonian contains an interaction term of the form $A_iB_j$ where $i < j-1$, so that this interaction term is not taken into account by any of the other pair evolutions (currently the Rydberg $n_i n_j$ and XY interactions are supported). Then as part of the effective Hamiltonian for the pair, this interaction term shows up as $Tr_{<j}(A_iB_j)$, where $Tr_{<j}$ denotes the partial trace over the left side of the system.
Unless the system is in an eigenstate of $A_i$, this term will only approximate the action of the interaction term, and the error is proportional to the variance $Var(A_i)$.

For example, take the term $\sigma^-_i\sigma^+_n$ from the XY-Hamiltonian, and assume $|\psi> = |1>_i|0>_n\otimes \phi$ where $\phi$ denote the state on the other qubits, that will not impact the result in the example, other than that it must be normalized.
In this case we compute $Tr_{<n}(\sigma^-_i\sigma^+_n) = 0$ because $\sigma^-_i|1>_i \perp |1>_i$, and the interaction term is not taken into account. The above example was chosen to be particularly bad, since $\sigma^-$ is not diagonalizable, and $|\psi>$ was as far from an eigenvector as possible, for other states, the error incurred in the approximation will be smaller. For the Rydberg interaction, which is diagonalizable, the maximum error is smaller. However, this shows that when simulating systems with long-range interactions (2d systems, for example, behave like 1d systems with long-range interactions according to the above reasoning), care should be taken that the interaction terms are properly accounted for by the TDVP scheme.

## Looping over pairs of qubits

Even if the Hamiltonian only has nearest-neighbour interactions, so that the above error is $0$, we still incur an error by repeatedly evolving a 2-site subsystem, rather than the entire system at once. Take for example the interaction term $A_nB_{n+1}$, in the 2-site TDVP scheme, there are 10 time evolution steps that incorporate this interaction term:

- 3 2-site time evolutions evolving either qubit $n$ or $n+1$ during the left-right sweep
- 2 1-site time evolutions evolving either qubit $n$ or $n+1$ during the left-right sweep
- the same 5 time evolutions durig the right-left sweep

Similar to how for trotterization

$$
e^{-i t (A + B)} = e^{-i t A}e^{-i t B} +O(t^2) = e^{-it B / 2} e^{-i t A} e^{-i t B / 2} + O(t^3)
$$

so also, by sweeping left-right and then right-left, the magnitude of this error reduced is reduced from $O(dt)$ to $O(dt^2)$. The prefactor in the order notation depends on the bond-dimension of the state, becoming smaller as the bond-dimension grows.

## Iterative computation of the 2-site effective evolution

Each 2-site time evolution corresponds to solving a Schroedinger equation for the corresponding subsystem, which is done numerically, and incurs a corresponding numerical error. We solve the Schroedinger equation by using the [Lanczos algorithm](https://en.wikipedia.org/wiki/Lanczos_algorithm) to exponentiate the effective 2-site Hamiltonian directly. This algorithm computes the vector $e^{i t H}\psi$ by iteratively constructing the vectors $\{\psi, H\psi,..., H^n\psi\}$ and exponentiating $H$ on this subspace. The algorithm aboards the iterations when the estimated precision for $e^{i t H}\psi$ has been obtained (this precision can be set via the [config](config.md)), but experience teaches that the error is underestimated. When choosing the precision high enough, this error is negligible compared the others in described here.

## Truncation of the state

After each 2-site evolution, an SVD is applied to split the vector for the 2-site subsystem back into 2 tensors for the MPS. The behaviour of this truncation is identical to that of general MPS truncation ([see here](mps/index.md)).

As explained there, each truncation finds the smallest MPS whose norm-distance is less than the precision from the original MPS. TDVP sweeps from left two right over neighbouring pairs of qubits, and back. This means that for each timestep, `2*(nqubits-1)` truncations are performed, so by the triangle inequality, TDVP will output a state whose distance is less than `2*(nqubits-1)*precision` from the state TDVP would have output without truncation. Note that the truncation errors will not all point in the same direction, so the actual error will likely be closer to `sqrt(2*(nqubits-1))*precision`, similar to the error in a gaussian random walk. The default precision is `1e-5`, meaning that each tdvp step will likely be accurate up to order `1e-4` assuming no more than order `1e2` qubits.

Similarly, when performing multiple TDVP steps, the maximum possible error scales linearly in the number of steps, but the error is more likely to scale as the square root of the number of time steps. Notice that there is a tradeoff when decreasing the value of $dt$ between the truncation error and the other errors in this list. Decreasing $dt$ means applying more truncations, which means a bigger expected error. Additionally, when $|e^{- i t H}\psi - \psi| \approx precision$ TDVP becomes meaningless, because each time evolution step is accompanied by a truncation that perturbs the state at least as much.

When in doubt about the convergence of the algorithm, try to improve the precision of both truncation and the Lanczos algorithm, and also make sure that `max_bond_dim` does not truncate the state too agressively. This can be done by tweaking these parameters, and checking whether output observables like the correlation matrix and energy variance change significantly. The effective 2-site Hamiltonian used to evolve each subsystem is constructed in such a way all powers of the Hamiltonian are constants of the motion. This means that any change in the moments of $H(t)$ (and specifically the expectation and variance of the energy) due to the tdvp step at time $t$ is due to truncation, or the precision in the Lanczos algorithm. Regarding the correlation matrix, long-range entanglement, as signified by elements of the correlation matrix far from the diagonal, contributes strongly to the bond dimension of the MPS, so the parts of the wave function creating such entanglement are likely to be truncated away when truncation is performed too agressively. As a corrolary, observables which do not strongly depend on off-diagonal elements of the correlation matrix are less sensitive to truncation. After these considerations, when still in doubt, try to reduce $dt$. When still in doubt, question whether TDVP correctly takes into account long-range interactions.

## Errors and how to control them using MPSConfig

Each one of the above errors can be controlled (or make it worst ) using `MPSConfig`

| Config parameter         | What it controls                                                                                                                            | Which error sources are concerned                                                                          |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `dt`                     | Time step size for evolution                                                                                                                | **B** (time discretization / sweep error) and **A** indirectly (smaller dt means more steps then more approximations)  |
| `precision`              | How “aggressive” the truncation is: the sum of squared dropped singular values $\leq$ `precision²`       | **D** (truncation) directly and **C** indirectly ( since Krylov tolerance is set relative to it)          |
| `max_bond_dim`           | Hard cap on how many singular values are retained                                                                                           | **D** (if bond dim is too small, you’ll drop larger singular values)                                              |
| `extra_krylov_tolerance` | Multiplier on `precision` to set Krylov convergence tolerance: tolerance $\approx$ `precision * extra_krylov_tolerance` | **C** (numerical error in the Krylov / Lanczos evolution)                                                         |

## Conceptual diagram of errors sources in 2-site TDVP

The following flow diagram, shows the evolution cycle for one time step and highlights where the four error sources come in and what varaibles in the config can control them

<div align="center">
  <img src="../mps/images/emu_mps_diagram_errors.svg" width="80%" alt="TDVP Error Sources Diagram">
</div>
