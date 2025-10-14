# Summary of the available algorithms

## Time Dependent Variational Principle (TDVP)
Emu-mps uses a second order 2-site TDVP to compute the time-evolution of the system ([see here for details](https://tensornetwork.org/mps/algorithms/timeevo/tdvp.html)).
Briefly, the algorithm repeatedly computes the time-evolution for 2 neighbouring qubits while truncating the resulting MPS to keep the state small. It does this by

- evolving qubit 1 and 2 forwards in time by $dt/2$
- evolving qubit 2 backwards by $dt/2$
- evolving qubit 2 and 3 forwards in time by $dt/2$

...

- evolving qubit $n-1$ and $n$ forward in time by $dt$
- evolving qubit $n-1$ backwards in time by $dt/2$
- evolving qubit $n-2$ and $n-1$ forward in time by $dt/2$

...

- evolving qubit 1 and 2 forwards in time by $dt/2$

The fact that we sweep left-right and the right-left with timesteps of $dt/2$ makes this a second-order TDVP.

## Density Matrix Renormalization Group (DMRG)
`DMRG` is a powerful variational method for finding the ground state and the first few excited states of strongly correlated $1$D and $2$D quantum many-body systems. More precisely, DMRG finds an MPS representation of the low-energy eigenstates by variationally optimizing MPS tensors to minimize the energy of the system. For a more detailed description of the algorithm, please refer to [[1]](https://tensornetwork.org/mps/algorithms/dmrg/).

### How DMRG can be useful in the emulators context
In the `emu-mps` context, we use DMRG to replicate the dynamics of an adiabatically long Pulser sequence, thereby computing the ground state of the time-dependent Hamiltonian along the pulse. This can be done because according to the adiabatic theorem, a sufficiently slow (adiabatic) drive keeps the system close to the ground state of the time-dependent Hamiltonian, therefore sweeping DMRG over the instantaneous Hamiltonian at successive times provides an efficient approximation to the adiabatic evolution. For a more detailed description of the adiabatic theorem please refer to [[2]](https://en.wikipedia.org/wiki/Adiabatic_theorem) and [[3]](https://arxiv.org/pdf/2406.12392).

### The algorithm
Emu-mps implements a `two-site DMRG`. The algorithm repeatedly optimizes two neighbouring MPS tensors, while controlling the `bond dimension` by truncating the singular values up to some precision configured by the user. To be more precise:

- At each two-site step the MPS tensors are contracted over the common index, and the effective Hamiltonian for that two-site block is minimized using the `Lanczos eigensolver`.
- The optimized block is split back into two tensors via `SVD`. This is where the truncation procedure takes place.
- The algorithm performs `left-to-right` and `right-to-left` sweeps until the ground-state energy converges.

#### Sweeping mechanism

- minimize qubits $1$ and $2$ (left --> right)
- minimize qubits $2$ and $3$ (left --> right)
- ...
- minimize qubits $l-1$ and $l$ (left --> right)
- minimize qubits $l$ and $l-1$ (right --> left)
- ...
- minimize qubits $3$ and $2$ (right --> left)
- minimize qubits $2$ and $1$ (right --> left)

    ------ End of one full sweep ------

At the end of the sweep (back and forth), check energy convergence:

- If converged, move to the next time-step.
- If not converged, restart the sweeping procedure.
