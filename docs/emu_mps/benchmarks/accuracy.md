# Accuracy

Here we discuss the emulator accuracy, as compared to Pulser state vector solver [backend](https://pulser.readthedocs.io/en/stable/tutorials/simulating.html), but in the future we might directly compare with QPU results.
Accuracy, here, specifically refers to observables:

- __Energy:__ $E = \langle\psi|H|\psi\rangle$
- __Energy variance:__ $\Delta E = \langle\psi|H^2|\psi\rangle-E^2$
- __Magnetization:__ $\langle P_{0}^j\rangle$ where $P_{0}^j$ projects qubit $j$ onto the $|0\rangle$ state

The emulated sequences are going to be the same as before, an adiabatic and a quench. In both cases, 9 qubits arrayed in a 3x3 grid are used, so that the results can also be simulated in Pulser. We will check accuracy against two main tunable parameters in emu-mps:

- `precision`<sup>[[1]](../advanced/errors.md#truncation-of-the-state)</sup>: at each step, throw away components of the state whose sum weighs less that the specified precision.
- `dt`: sampling time step of the sequence.

The goal is to show that for qubit numbers accessible to Pulser, the results are identical up to good precision for both the quench and the adiabatic sequence. Both sequences are emulated multiple times by varying both the precision and time step.

<img src="../benchmark_plots/quench_fidelity.png"  width="99.5%">

For the quench sequence, agreement with Pulser is still good for all shown parameter combinations, with the possible exception of the yellow curve, which has a deviation of $0.5\%$ in the energy variance.
Let us first consider the energy and its variance, which for the quench sequence are conserved quantities, meaning that all variation therein come from errors. For example, we see that emu-mps incurs the biggest error in the energy at the start of the emulation, when the bond dimension is still small (the bond dimension starts at 1, and increases gradually during time evolution). For a time-independent Hamiltonian, all deviations in the mean and variance of the energy originate from the truncation mechanism, and as expected, improving the precision reduces the error in the energy variance ([see here](../advanced/errors.md)). It will also reduce the error in the energy, but the generic ODE solver used by pulser-simulation does not take into account energy as a constant of the motion, meaning that the energy computed by pulser shows a linear deviation from the initial value that dominates the graph.
Finally, let us consider the qubit occupation. As explained in error sources in TDVP ([see here](../advanced/errors.md#truncation-of-the-state)), we see that reducing $dt$ below a threshold (somewhere in the range of 1-5) causes a quick growth of the truncation errors, which requires improving the precision. The size of the truncation error can be estimated by comparing deviations from pulser with those observer by [emu-sv](../../emu_sv/benchmarks/accuracy.md#accuracy), by which means we see that for `precision=1e-5` all deviations are dominated by truncation errors, which are especially severe for the yellow curve, where the most truncations occur.

<img src="../benchmark_plots/afm_state_fidelity.png"  width="99.5%">

 By comparing the deviations from pulser of emu-mps with those of [emu-sv](../../emu_sv/benchmarks/accuracy.md#accuracy), we see that for `precision=1e-8` the deviations from pulser are dominated by the discretization scheme used by emu-mps. For `precision=1e-5`, we see that the truncation errors are of a similar magnitude as the discretization errors. This finding motivates the default configuration values of `dt=10` and `precision=1e-5` for emu-mps. For larger qubit numbers, such as the 7x7 grid, the question of convergence is much subtler ([see here](../advanced/convergence.md)). Rather, what is interesting there, is that even for a 2D system, emu-mps correctly treats the Rydberg interaction, regardless of the [effective description of long-range interaction terms](../advanced/errors.md#effective-description-of-long-range-terms-in-the-hamiltonian) that emu-mps uses.



## effect of qubit ordering

On the performance benchmarks page, we show how a good qubit ordering can improve [performance](performance.md#qubit-shuffling). Here we will show that a good qubit ordering also improves the accuracy of emu-mps significantly. For the purposes of the demonstration, we use a custom 12-qubit pulse as follows:

```python
mock_device = AnalogDevice
duration = 6000
amplitude_maximum = np.pi
delta = np.pi
reg = pulser.register.Register.rectangle(3, 4, spacing=5)
seq = Sequence(reg, mock_device)
seq.declare_channel("ryd_glob", "rydberg_global")
rise_duration = duration / 3
fall_duration = duration / 3
sweep_duration = duration - rise_duration - fall_duration
rise = pulser.Pulse.ConstantDetuning(
    RampWaveform(rise_duration, 0.0, amplitude_maximum), -delta, 0.0
)
sweep = pulser.Pulse.ConstantAmplitude(
    amplitude_maximum, RampWaveform(sweep_duration, -delta, delta), 0.0
)
fall = pulser.Pulse.ConstantDetuning(
    RampWaveform(fall_duration, amplitude_maximum, 0.0), delta, 0.0
)
amp = CompositeWaveform(rise.amplitude, sweep.amplitude, fall.amplitude)
det = CompositeWaveform(rise.detuning, sweep.detuning, fall.detuning)
pulse = pulser.Pulse(amp, det, 0)
seq.add(
    pulse,
    "ryd_glob",
    protocol="no-delay",
)
```

The register spacing is immaterial because we run the sequence twice with a custom interaction matrix. We will plot the difference between the two corralation matrices at the end of the sequence for various parameters. The two interaction matrices contain only `0` and `1`, where the ones are between qubit pairs

 `[(6, 7), (8, 9), (10, 11), (7, 0), (7, 3), (9, 1), (9, 5), (11, 3), (11, 5), (6, 1), (6, 2), (8, 0), (8, 4), (10, 2), (10, 4)]`

 and

 `[(6, 7), (8, 9), (10, 11), (7, 1), (7, 3), (9, 1), (9, 5), (11, 3), (11, 5), (6, 0), (6, 2), (8, 0), (8, 4), (10, 2), (10, 4)]`

 respectively. As can be seen, only two of the interaction terms are different `(6,1) -> (6,0)` and `(7,0) -> (7,1)`, causing the correlation matrices to be extremely similar, requiring good accuracy for the simulation. Furthermore, since the two differing terms are "long range", these form a good stress test for emu-mps, which uses an effective description of such long-range terms. The results are as follows:

<div style="text-align:center;">
<img src="../benchmark_plots/sv_optimatrix_fidelity.png"  width="90%">
</div>

Emu-sv is used as a source of truth. The most salient feature is that the shown difference is largest on the  `(6,1)`, `(6,0)`, `(7,0)` and `(7,1)` terms which are precisely the terms in the interaction matrix that have been changed. The checkerboard pattern is explained because while one interaction term is added, the other is removed, causing opposite signs in the difference. Then, to subleading order, you can see repeats of this effect as the changed interaction matrix causes further differences in the correlation structure. It can be seen that for a precision of `1e-7` emu-mps is not able to capture the differences in correlation at all without reordering: the difference between the two correlation matrices is essentially zero (see top right in the figure). As explained above, terms in the interaction matrix far from the diagonal are difficult to capture for emu-mps. Notice that qubit reordering alleviates this problem, and although agreement with emu-sv is not exact, the fundamental structure of the problem is visible. The same is true for a precision `1e-6` but the errors in emu-mps will be somewhat larger. Setting the precision to `1e-8` causes emu-mps to capture the long-range correlations more accurately, even without qubit reordering. It should be noted that in this case, qubit reordering still has positive effects. Firstly, the bond dimension required to accurately describe the quantum state will be lower, decreasing the runtime. Secondly, the results with qubit ordering are much more stable than those without. For example, when running the simulation without qubit ordering the results are hardware dependent: there is a noise of a similar magnitude as for precision `1e-7` without reordering, which just happens to be negligible on the AMD EPYC 7742 where this graph was generated. This problem vanishes when qubit reordering is used, and demonstrates the fundamental instability of TDVP in the presence of long-range interactions.
