# Accuracy

Here we discuss the emulator accuracy, as compared to Pulser state vector solver [backend](https://pulser.readthedocs.io/en/v1.6.0/apidoc/_autosummary/pulser_simulation.QutipBackend.html#), but in the future we might directly compare with QPU results.
Accuracy, here, specifically refers to observables:

- __Energy:__ $E = \langle\psi|H|\psi\rangle$
- __Energy variance:__ $\Delta E = \langle\psi|H^2|\psi\rangle-E^2$
- __Magnetization:__ $\langle P_{0}^j\rangle$ where $P_{0}^j$ projects qubit $j$ onto the $|0\rangle$ state

The emulated sequences are going to be the same as before, an adiabatic and a quench. In both cases, 9 qubits arrayed in a 3x3 grid are used, so that the results can also be simulated in Pulser. We will check accuracy against two main tunable parameters in emu-sv:

- `krylov_tolerance`: the convergence tolerance when exponentiating the Hamiltonian in each time-step.
- `dt`: sampling time step of the sequence.

The goal is to show that for qubit numbers accessible to Pulser, the results are identical up to good precision.

<img src="../benchmark_plots/emu_sv_quench_fidelity.png"  width="99.5%">

For the quench sequence (where the Hamiltonian is constant in time), emu-sv shows better accuracy than Pulser. The evolution method used by emu-sv, which evolves the system by assuming the Hamiltonian is constant over intervals of length $dt$, is exact in this case. The `krylov_tolerance` configuration parameter is an estimate of the error incurred in each time-step. Experimentally, it overestimates the error for values < `1e-8`, which is the default value used in the above. As a consequence, the deviation from pulser shown in the graph is dominated by the numerical error in pulser when doing the simulation. This is consistent with the accuracy settings used by pulser-simulation internally.

<img src="../benchmark_plots/emu_sv_adiabatic_afm_state.png"  width="99.5%">

This is interesting to contrast with the results for the adiabatic sequence, where the emu-sv solver shows deviations from pulser-simulation due to the assumption of piecewise-constant Hamiltonian. This assumption means the numerical error scales as $dt^2$. On the surface, this is quite bad, but since the prefactors scale with the time-dependence of $H(t)$, the error is of acceptable levels in the case of an adiabatic sequence. The graph shows the difference between pulser-simulation and emu-sv is reduced by a factor $4$ when $dt$ is halved, as the step-function used by emu-sv more closely resembles the qubic interpolation done by pulser-simulation. Which emulator should be considered closer to the truth depends on what is desired. When hardware modulation is taken into account, response times in the QPU hardware make the actual time-dependent Hamiltonian smooth as a function of time, and the qubic interpolation in pulser-simulation is closer to the truth. When ignoring hardware modulation, one assumes a model of the hardware where the laser is piecewise constant, and switches discretely at the device resolution. This implies emu-sv models the abstract ideal device more closely. That said, agreement between pulser-simulation and emu-sv is likely to be good enough for most applications.
