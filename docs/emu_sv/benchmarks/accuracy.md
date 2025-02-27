# Accuracy

Here we discuss the emulator accuracy, as compared to Pulser state vector solver [backend](https://pulser.readthedocs.io/en/stable/tutorials/simulating.html), but in the future we might directly compare with QPU results.
Accuracy, here, specifically refers to observables:

- __Energy:__ $E = \langle\psi|H|\psi\rangle$
- __Energy variance:__ $\Delta E = \langle\psi|H^2|\psi\rangle-E^2$
- __Magnetization:__ $\langle P_{0}^j\rangle$ where $P_{0}^j$ projects qubit $j$ onto the $|0\rangle$ state

The emulated sequences are going to be the same as before, an adiabatic and a quench. In both cases, 9 qubits arrayed in a 3x3 grid are used, so that the results can also be simulated in Pulser. We will check accuracy against two main tunable parameters in emu-sv:

- `krylov_tolerance`: the convergence tolerance when exponentiating the Hamiltonian in each time-step.
- `dt`: sampling time step of the sequence.

The goal is to show that for qubit numbers accessible to Pulser, the results are identical up to good precision.

<img src="../benchmark_plots/emu_sv_adiabatic_afm_state.png"  width="49.7%">
<img src="../benchmark_plots/emu_sv_quench_fidelity.png"  width="49.7%">

It is worth noting that for the quench sequence (where the Hamiltonian is constant in time), emu-sv is more accurate. The evolution method used by emu-sv, evolving the system by assuming the Hamiltonian is constant for intervals of length `dt` is exact in this case. The `krylov_tolerance` is an estimate of the error incurred in each time-step, which experimentally overestimates the error for values < `1e-8`, and the deviation with pulser is an order of magnitude larger. This is consistent with the accuracy settings used by pulser-simulation internally.

This is interesting to contrast with the results for the adiabatic sequence, where the emu-sv solver shows deviations from pulser-simulation due to the assumption of piecewise-constant Hamiltonian. The difference between pulser-simulation and emu-sv decreases with decreasing `dt`, as the step-function used by emu-sv more closely resembles the qubic interpolation done by pulser-simulation. Which emulator should be considered closer to the truth depends on what is desired. When hardware modulation is taken into account, response times in the QPU hardware make the actual time-dependent Hamiltonian smooth as a function of time, and the qubic interpolation in pulser-simulation is closer to the truth. When ignoring hardware modulation, one assumes a model of the hardware where the laser is piecewise constant, and switches discretely at the device resolution. This implies emu-sv models the abstract ideal device more closely. That said, agreement between pulser-simulation and emu-sv is likely to be good enough for most applications.
