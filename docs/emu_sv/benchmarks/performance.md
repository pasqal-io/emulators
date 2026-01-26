# Performance

Here, as anticipated in the [introduction](../benchmarks/index.md) page of the benchmarks, we will track the runtime of emu-sv by comparing runs on GPU vs CPU, and CPU vs pulser-simulation (which is cpu only).

## Adiabatic sequence

We run an adiabatic sequence to make an antiferromagnetic (AFM) state, as taken from one of our [tutorials](../notebooks/getting_started), for a line of atoms. In contrast to emu-mps, the performance of emu-sv does not depend on the type of sequence, other than through its duration, so there is no real benefit to showing results for the adiabatic and the quench sequence separately.

First, let us compare emu-sv with pulser:

<div style="text-align:center;">
<img src="../benchmark_plots/emu_sv_pulser_runtimes.png"  width="49.7%">
</div>


Runtimes are shown for two different values of `dt` which shows that halving `dt` almost doubles the runtime. Halving `dt` doubles the number of timesteps taken by the program, but the algorithm for computing a single timestep converges faster, leading to a sublinear scaling of the runtime in terms of the number of timesteps. For larger values of `dt` emu-sv can be seen to outperform pulser for 8 qubits and up on the cluster, while for the smallest possible `dt = 1.0`, it will be faster starting at `9` qubits.

Next, let us compare runs of emu-sv between CPU and GPU:

<div style="text-align:center;">
<img src="../benchmark_plots/emu_sv_runtimes.png"  width="49.7%">
</div>

There is a marginal runtime difference between CPU and GPU for smaller qubit numbers, which is mostly coincidental, since neither hardware is saturated with computations yet, and exponential scaling of the runtime has not yet set in. When comparing the runtimes between 19 and 20 qubits, they can be seen to roughly double for both CPU and GPU, as expected by exponential scaling. This shows that for larger qubit numbers, the GPU is about 4 times faster than CPU. Contrast this with the benchmarks of emu-mps, which show a relative factor of about 20 for larger bond-dimensions, a number much closer to the theoretical ratio of computational power. This suggests there are improvements to be made in the performance of emu-sv on gpu at least.
