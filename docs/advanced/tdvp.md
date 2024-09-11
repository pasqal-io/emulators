# Summary of the TDVP algorithm

EMU-MPS uses a second order 2-site TDVP to compute the time-evolution of the system ([see here for details](https://tensornetwork.org/mps/algorithms/timeevo/tdvp.html)).
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
