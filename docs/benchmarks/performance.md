# Performance

As outlined above, in the performance benchmarks, we will track several relevant metrics associated with runtime, memory usage and bond dimension:

- __Bond dimension $\chi$:__ the maximum internal link dimension of the MPS representation of the time evolved state ([see here](../advanced/mps/index.md#bond-dimension)).
- __State size $|\psi|$:__ memory footprint of the state (in MB).
- __RSS:__ peak memory allocated by the emulation.
- $\Delta t$: CPU/GPU time to complete a time step.

We will give information about these metrics for various values of __N__, the qubit number, to give an idea of how performance scales.

## Adiabatic sequence

We run an adiabatic sequence to make an antiferromagnetic (AFM) state, as taken from [Pulser documentation](https://pulser.readthedocs.io/en/stable/tutorials/afm_prep.html), alongside the biggest register:



Performance metrics, for the defined sequence and for the biggest register are shown below, in the left column of the figures, for CPU and GPU workloads.
From the plots it is easy to understand that all the metrics heavily correlate with each other. Specifically a higher bond dimension will translate to higher memory footprint and longer runtimes ([see here](../advanced/mps/index.md)).

<img src="../benchmark_plots/adiabatic_afm_state_cpu.png"  width="49.7%">
<img src="../benchmark_plots/adiabatic_afm_state_gpu.png"  width="49.7%">

In the right column (both CPU and GPU figure), we explore the available register size. Simply increasing the number of atoms by $N=N_x\times N_y$, and extracting the maximum metric and the total runtime for each run, the user can get a feeling on how much memory and time a specific sequence is going to take to emulate. Note that all qubit numbers which are not a square show up twice, since the rectangles making up this qubit number can be oriented two ways. The reasons why the orientation matters is explained by the results in the benchmark on [qubit shuffling](#qubit-shuffling). Note that it's possible to simulate larger systems than done in this benchmark. For example, by tuning the config parameters, it's possible to accurately simulate the above pulse for a 7x7 grid ([see here](../advanced/convergence/index.md)).

## Quench

Here, we explore performance in the very same way as before, but for the quench sequence:



The overall metrics, as before, both for a single run (left) and for multiple runs varying the register size (right, $N=N_x\times N_y$) are presented below:

<img src="../benchmark_plots/quench_performance_cpu.png"  width="49.7%">
<img src="../benchmark_plots/quench_performance_gpu.png"  width="49.7%">

As expected, the quench requires significantly more memory to run than the adiabatic sequence ([see here](../advanced/hamiltonian.md)).

## Qubit shuffling

A seemingly innocuous operation like reordering the register labels can actually affect the performance, as a consequence of the MPS representation ([see here](../advanced/mps/index.md)). In simple terms, the additional memory cost, and thus performance decrease, comes from representing two strongly interacting atoms in two far apart tensors in the MPS, since all the intermediate tensors in the chain have to somehow pass that information between them.

To be more quantitative, in the following benchmark case, we run the same AFM sequence from before, but shuffling the qubit labeling order.

The unshuffled register ordering is that given by `Register.rectangle` as used in the above two sequences. For the 3x3 grid used in this benchmark, that means a register ordering of

<table>
 <tr><td>1 </td><td> 2 </td><td> 3 </td></tr>
 <tr><td>4 </td><td> 5 </td><td> 6 </td></tr>
 <tr><td>7 </td><td> 8 </td><td> 9 </td></tr>
</table>
 Compare this with the shuffled register, which was constructed to put qubits that are close in physical space far away in index space
<table>
 <tr><td> 2 </td><td> 7 </td><td> 4 </td></tr>
 <tr><td> 5 </td><td> 1 </td><td> 9 </td></tr>
 <tr><td> 8 </td><td> 3 </td><td> 6 </td></tr>
</table>
<img src="../benchmark_plots/qubit_shuffling_cpu.png"  width="49.7%">

The left column of the image shows no accuracy degradation from the qubit shuffling, returning equivalent observables. That is expected since both runs were able to converge to the desired precision.

However, performance metrics (allocations and runtime) of the shuffled case significantly worsen, because shuffling the qubits introduces artificial long-range entanglement into the system, increasing the bond dimension. This larger bond dimension means the matrices involved in the computations are bigger, requiring more memory and compute time.

In the future we plan to apply register ordering strategies by default, but for the moment, the take-home message is that a good register embedding is important. Ideally, one should keep strongly interactive pairs or atoms the closest possible when enumerating them in the register.
