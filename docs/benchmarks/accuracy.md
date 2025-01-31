# Accuracy

Here we discuss the emulator accuracy, as compared to Pulser state vector solver [backend](https://pulser.readthedocs.io/en/stable/tutorials/simulating.html), but in the future we might directly compare with QPU results.
Accuracy, here, specifically refers to observables:

- __Energy:__ $E = \langle\psi|H|\psi\rangle$
- __Energy variance:__ $\Delta E = \langle\psi|H^2|\psi\rangle-E^2$
- __Magnetization:__ $\langle P_{0}^j\rangle$ where $P_{0}^j$ projects qubit $j$ onto the $|0>$ state

The emulated sequences are going to be the same as before, an adiabatic and a quench. In both cases, 9 qubits arrayed in a 3x3 grid are used, so that the results can also be simulated in Pulser. We will check accuracy against two main tunable parameters in emu-mps:

- `precision`<sup>[[1]](../advanced/errors.md#truncation-of-the-state)</sup>: at each step, throw away components of the state whose sum weighs less that the specified precision.
- `dt`: sampling time step of the sequence.

The goal is to show that for qubit numbers accessible to Pulser, the results are identical up to good precision.

<img src="../benchmark_plots/afm_state_fidelity.png"  width="49.7%">
<img src="../benchmark_plots/quench_fidelity.png"  width="49.7%">

Both sequences are emulated multiple times by varying both the precision and time step. Notice that any deviations from Pulser for the adiabatic sequence are impossible to detect at the scale of the graph for a wide range of emulation parameters. For larger qubit numbers, such as the 7x7 grid, the question of convergence is much subtler ([see here](../advanced/convergence.md)). Rather, what is interesting there, is that even for a 2d system, emu-mps correctly treats the Rydberg interaction, regardless of the [effective description of long-range interaction terms](../advanced/errors.md#effective-description-of-long-range-terms-in-the-hamiltonian) that emu-mps uses.

For the quench sequence, agreement with Pulser is still good for all shown parameter combinations, with the possible exception of the yellow curve, which has a deviation of 1%. For the quench sequence, the energy and energy variance are conserved quantities, meaning that all variation therein come from errors. Even though the relative errors are small, it's instructive to analyze the sources of these errors. For example, we see that emu-mps incurs the biggest error at the start of the emulation, when the bond dimension is still small (the bond dimension starts at 1, and increases from there). For a time-constant Hamiltonian, all deviations in the mean and variance of the energy come from truncation, and as expected, improving the precision reduces the error in the energy variance ([see here](../advanced/errors.md)). Finally, as explained in error sources in TDVP ([see here](../advanced/errors.md#truncation-of-the-state)), we see that reducing $dt$ below a threshold (somewhere in the range of 1-5) causes a quick growth of the truncation errors, which requires improving the precision.

The errors incurred by emu-mps can be contrasted with Pulser, which uses a generic ODE solver backend that does not take into account constants of the motion. Both the mean and variance of the energy exhibit a deviation from their initial value that is linear in the number of time-steps taken by the solver.
