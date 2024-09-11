# Estimating the memory consumption of a simulation

The presence of the `max_bond_dim` and `max_krylov_dim` [config](config.md) parameter means an upper bound on memory consumption can be computed. By limiting the `max_bond_dim` of a simulation, it can be guaranteed to run for arbitrary times for an arbitrary number of qubits. Of course, the sources of error described on the [error page](errors.md) imply that limiting the memory consumption of the program will negatively impact the quality of the results once a certain threshold is exceeded. The page in [this link](convergence.md) outlines a case study to determine whether emulation results are accurate. This page will outline how to estimate the memory consumption of a simulation, given `max_bond_dim`, `max_krylov_dim` and $N$, the latter being the number of qubits to be simulated.

There are four contributions to the peak memory consumption of EMU-MPS

- the state
- the baths
- the krylov space
- temporary tensors


## Contribution from the state

The quantum state is stored in MPS format ([see here](mps/index.md)). At worst, the bond dimensions of the tensors in an MPS grow exponentially inwards as

$$
2,4,8,16...,16,8,4,2
$$

in which case an MPS will take __more__ memory than a state vector. When `max_bond_dim < 2^{nqubits/2}` the bond dimensions in the center all cap at `max_bond_dim`. Let $d$ denote the value of `max_bond_dim`. Since each tensor in the MPS has 2 bonds of size at most $d$, and a physical index of size $p=2$, where each element in the tensor takes $s=16$ bytes (2 8-byte floats to store a complex number), the memory consumption of the state

$$
|\psi| < spNd^2 = 32Nd^2
$$

Note that this is a strict over-estimation because the outer bonds in the MPS will be much smaller than $d$.

## Contribution from the baths

For TDVP, for each qubit a bath tensor is stored which has 3 indices whose size depends on the state that is evolved, and on the Hamiltonian. The bath tensors are used to compute an effective interaction between the 2-qubit subsystem being evolved, and the rest of the system ([see here](tdvp.md)). The computation is slightly involved, but the result is as follows.

$$
|bath| < sd^2N[N+10]/4 = 4d^2N(N+10)
$$

Note that the baths take up more memory than the state, always, and potentially much more. Furthermore, just as for the state this is a strict over-estimation, because it assumes all the bonds in the state are of size $d$.

## Contribution from the Krylov space

The remainder of the memory consumption is to compute the time-evolution of qubit pairs in TDVP. This is done by contracting 2 tensors from the MPS together into a single 2-qubit tensor, and time-evolving it by applying an effective Hamiltonian constructed from the baths and the Hamiltonian MPO. Each 2-qubit tensor has a size bounded by $sp^2d^2$, so the memory of the Krylov vectors used in the Lanczos algorithm is obeys

$$
|krylov| \leq ksp^2d^2 = 64*k*d^2
$$

where $k$ is the value of `max_krylov_dim`. Recall that the default value of $k = 100$ and if the Lanczos algorithm requires more Krylov vectors to converge to the tolerance, it will error, rather than exceed the above bound.

## Contribution from temporary tensors

Finally, to compute the above Krylov vectors, the effective two-site Hamiltonian has to be applied to the previous Krylov vector to obtain the next one. The resulting tensor network contraction cannot be done in-place, so it has to store two intermediate results that get very large. The intermediate results takes the most memory at the center qubit, where the bond dimension of the Hamiltonian becomes $h$, where

$$
|intermediate| = 2*shp^2d^2 = 128hd^2
$$

It should be noted that the value of $h$ cited above assumes that all qubits in the system interact via a two-body term, which is technically true for the Rydberg interaction. When some of these interaction terms can be neglected, the value of $h$ can be reduced, leading to significant memory savings in $|intermediate|$ and $|bath|$. These optimizations have yet to be performed.

Putting all of this together, for the total memory consumption $m$ of the program, we can write the following bound:

$$
 m = |\psi| + |bath| + |krylov| + |intermediate| < 32Nd^2 + 4d^2N(N+10) + 64*k*d^2 + 64(N+4)d^2 = 4d^2[N(N+34) + 16k + 64]
$$

Note that this estimate is pessimistic, since not all $k$ krylov vectors are likely to be needed, and not all tensors in $\psi$ and the baths have the maximum bond dimension $d$. On the other hand, the estimate for $|intermediate|$ is likely to be accurate, since the bond dimension of $d$ is probably attained at the center qubit.

For example, the results from the [case study](convergence.md) were obtained using $N=49$ and $d=1600$ on 2 gpu's. Taking the above formula, and halving the contributions from $\psi$ and $|bath|$ since they are split evenly on the gpu's, we reproduce the memory consumption of the program for $k=13$. Notice that the actual number of Krylov vectors required to reach convergence is likely closer to around $30$, but here we underestimate it, since the contributions of $\psi$ and $|bath|$ are over-estimated.
