# EMU-MPS benchmarks

Users should expect _Emu-MPS_ to emulate the QPU for<sup>[[1]](#performance)</sup>
- 2d systems up to 30 atoms for quenches and 50 adiabatic sequences
- Realistic sequences (~μs)

on Pasqal's DGX-cluster.
Although Emu-MPS should run on most common hardware configurations (with or without gpu), for best performance on heavy workloads we recommend using a cluster GPU (NVIDIA A100).
The emulator is mostly limited by the available memory (40 GB on an NVIDIA A100), as it limits the maximum number of qubits and the pulse duration that can be simulated. It is an ongoing effort to improve performance by making _Emu-MPS_ distribute work to either optimize for runtime or memory profile.

---

Benchmark efforts, as documented here, are meant to provide insights for _Emu-MPS_ users about

- **Performance**: runtime, memory usage, bond dimension<sup>[[2]](#bond-dimension)</sup> as a function of qubit number
- **Accuracy**: different precision levels as compared to state vector solvers

given a set of meaningful sequences of interest (quench, adiabatic and use-case sequences) that we are going to introduce case by case. Finally, we will only focus on 2d atomic registers as they represent the most numerically challenging and interesting case to study.


- ### I. [Basic sequences](#i-basic-sequences)
    - [Performance](#ia-performance)
    - [Accuracy](#ib-accuracy)
- ### II. [Noise](#ii-noise) (coming soon)
- ### III. [Use cases](#iii-use-cases) (coming soon)

Feedbacks are more than welcome! If you are interested in the performance of any sequence, please, do not hesitate to get in touch with the Emulation team.



# I. Basic sequences
A first class of benchmarks covers the minimal meaningful pulse sequences that can be realistically run on the QPU:
- __Adiabatic time evolution:__ the evolution is _slow enough_ to guarantee that the evolved state is always an equilibrium state.
- __Quench:__ The system is strongly driven out of equilibrium

Both are interesting and they complement each other. Quenches, in particular, are typically computationally harder to emulate. For more details, please, have a look at insight [[3]](#qpu-hamiltonian).

### I.a Performance

As outlined above, in the performance benchmarks, we will track several relevant metrics associated with runtime, memory usage and bond dimension:

- __Bond dimension $\chi$:__ the maximum internal link dimension of the MPS representation of the time evolved state<sup>[[2]](#bond-dimension)</sup>.
- __State size $|\psi|$:__ memory footprint of the state (in MB).
- __RSS:__ peak memory allocated by the emulation.
- $\Delta t$: CPU/GPU time to complete a time step.
- __N__: the number of qubits

#### Adiabatic sequence

We run an adiabatic sequence to make an antiferromagnetic (AFM) state, as taken from [Pulser documentation](https://pulser.readthedocs.io/en/stable/tutorials/afm_prep.html), alongside the biggest register:

```python
# from https://pulser.readthedocs.io/en/stable/tutorials/afm_prep.html
# parameters in rad/µs and ns
Omega_max = 2.0 * 2 * np.pi
U = Omega_max / 2.0
delta_0 = -6 * U
delta_f = 2 * U
t_rise = 500
t_fall = 1000
t_sweep = (delta_f - delta_0) / (2 * np.pi * 10) * 3000
R_interatomic = MockDevice.rydberg_blockade_radius(U)
reg = Register.rectangle(rows, columns, R_interatomic, prefix="q")
if perm_map:
    reg_coords = reg._coords
    reg = Register.from_coordinates([reg_coords[i] for i in perm_map])
rise = Pulse.ConstantDetuning(RampWaveform(t_rise, 0.0, Omega_max), delta_0, 0.0)
sweep = Pulse.ConstantAmplitude(
    Omega_max, RampWaveform(t_sweep, delta_0, delta_f), 0.0
)
fall = Pulse.ConstantDetuning(RampWaveform(t_fall, Omega_max, 0.0), delta_f, 0.0)
seq = Sequence(reg, MockDevice)
seq.declare_channel("ising", "rydberg_global")
seq.add(rise, "ising")
seq.add(sweep, "ising")
seq.add(fall, "ising")
```

Performance metrics, for the defined sequence and for the biggest register are shown below, in the left column of the figures, for CPU and GPU workloads.
From the plots it is easy to understand that all the metrics heavily correlate with each other. Specifically a higher bond dimension will translate in higher memory footprint and runtimes<sup>[[2]](#bond-dimension)</sup>.

<img src="./benchmark_plots/adiabatic_afm_state_cpu.png"  width="49.7%">
<img src="./benchmark_plots/adiabatic_afm_state_gpu.png"  width="49.7%">

In the right column (both CPU and GPU figure), we explore the available register size. Simply increasing the number of atoms by $N=N_x\times N_y$, and extracting the maximum metric and the total runtime for each run, the user can get a feeling on how much memory and time a specific sequence is going to take to emulate.

#### Quench

Here, we explore performance in the very same way as before, but for the quench sequence:

```python
hx = 1.5  # hx/J_max
hz = 0  # hz/J_max
t = 1.5  # t/J_max
# Set up pulser simulations
R = 7  # microm
reg = Register.rectangle(nx, ny, R, prefix="q")
# Conversion from Rydberg Hamiltonian to Ising model
U = AnalogDevice.interaction_coeff / R**6  # U_ij
NN_coeff = U / 4
omega = 2 * hx * NN_coeff
delta = -2 * hz * NN_coeff + 2 * U
T = np.round(1000 * t / NN_coeff)
seq = Sequence(reg, MockDevice) #circumvent the register spacing constraints
seq.declare_channel("ising", "rydberg_global")
# Add the main pulse to the pulse sequence
simple_pulse = Pulse.ConstantPulse(T, omega, delta, 0)
seq.add(simple_pulse, "ising")
```

The overall metrics, as before, both for a single run (left) and for multiple runs varying the register size (right, $N=N_x\times N_y$) are presented below:

<img src="./benchmark_plots/quench_performance_cpu.png"  width="49.7%">
<img src="./benchmark_plots/quench_performance_gpu.png"  width="49.7%">

As expected<sup>[[3]](#qpu-hamiltonian)</sup>, the quench requires significantly more memory to run compared to the adiabatic sequence.

#### Qubit shuffling

A seemingly innocuous operation like reordering the register labels can actually affect the performance, as a consequence of the MPS representation<sup>[[2]](#bond-dimension)</sup>. In simple terms, the additional memory cost, and thus performance, comes from representing for example two strongly interacting atoms, in two far apart tensors in the MPS, since all the intermediate tensors in the chain have to somehow pass that information between them.

To be more quantitative, in the following benchmark case, we run the same AFM sequence from before, but shuffling the qubit labeling order.

The unshuffled register ordering is that given by `Register.rectangle` as used in the above two sequences. For the 3x3 grid used in this benchmark, that means a register ordering of

 1  2  3

 4  5  6

 7  8  9

 Compare this with the shuffled register, which was constructed to put qubits that are close in physical space far away in index space

 2  7  4

 5  1  9

 8  3  6

<img src="./benchmark_plots/qubit_shuffling_cpu.png"  width="49.7%">

[TODO: fix this the black bars in the plot]

The left column of the image shows no accuracy degradation from the qubit shuffling, returning equivalent observables. That is expected since both runs were able to converge to the desired precision.

However, performance metrics (allocations and runtime) of the shuffled case significantly worsen, because shuffling the qubits introduces artificial long-range entanglement into the system, increasing the bond dimension. This larger bond dimension means the matrices involved in the computations are bigger, requiring more memory and compute time.

In the future we plan to apply register ordering strategies by default, but for the moment, the take-home message is that a good register embedding is important. Ideally, one should keep strongly interactive pairs or atoms the closest possible when enumerating them in the register.

### I.b Accuracy

Here we discuss the emulator accuracy, as compared to Pulser state vector solver backend (qutip), but in the future might directly compare QPU results.
Accuracy, here, specifically refer to observables:

- __Energy:__ $E = \langle\psi|H|\psi\rangle$
- __Energy variance:__ $\Delta E = \langle\psi|H^2|\psi\rangle-E^2$
- __Magnetization:__ $P_{r}^j = |\langle r^j|\psi\rangle|^2$

The emulated sequences are going to be the same as before, an adiabatic and a quench. We will check accuracy against two main tunable parameters in _Emu-MPS_:

- `precision`<sup>[[4]](#timestep-size-and-precision)</sup>: at each step, throw away components of the state whose sum weighs less that the specified precision.
- time step `dt`: sampling time of the sequence.

<img src="./benchmark_plots/afm_state_fidelity.png"  width="49.7%">
<img src="./benchmark_plots/quench_fidelity.png"  width="49.7%">

Both sequences are emulated multiple times by varying the both precision and time step. Looking at the results for the quench sequence, we see that Emu-MPS incurs the biggest error at the start of the emulation, when the bond dimension is still small (the bond dimension starts at 1, and increases from there). As mentioned in the discussion on [error sources in TDVP](../errors.md), for a time-constant Hamiltonian, all deviations in the mean and variance of the energy come from truncation, and as expected, improving the precision reduces the error in the energy variance. Finally, as explained in [error sources in TDVP](../errors.md#truncation-of-the-state), we see that reducing $dt$ below a threshold (somewhere in the range of 1-5) causes a quick growth of the truncation errors, which requires improving the precision.

This bevaviour can be contrasted with pulser, which uses a generic ODE solver backend that does not take into account constants of the motion. Both the mean and variance of the energy exhibit a deviation from their initial value that is linear in the number of time-steps taken by the solver.

Similar considerations likely hold for the adiabatic AFM sequence too, but the errors would be dwarfed by the variations in the energy due to the time-dependence of the Hamiltonian. Rather, what is interesting there, is that even for a 2d system, Emu-MPS correctly treats the Rydberg interaction, regardless of the [effective description of long-range interaction terms](../errors.md#effective-description-of-long-range-terms-in-the-hamiltonian) that Emu-MPS uses.

[TODO: For a more in depth discussion change the plots to have the observables on the left column and difference respect to Pulser state vector on right column]

# II. Noise
Coming soon...

# III. Use cases
Coming soon...

# Insights
## Performance

<b>a. Matrix product representation</b>

As opposed to state vector solvers (of Master/Schrödinger equation), tensor network based approaches use adaptive data structures, which in the case of _Emu-MPS_ are called [matrix product state/operators (MPS/MPO)](http://tensornetwork.org/mps/). In many relevant use cases, this makes representation more memory-efficient, which allows pushing for higher number of qubits compared to state vector solvers. However, it has the drawback to make the cost of the simulation less predictable since there is no _a priori_ method to know how much information is going to be relevant at the next step of the solver.

The take-home message is that a reasonable way to assess _Emu-MPS_ performance is by __benchmarking relevant and meaningful sequences/use-cases__.

----

<b>b. QPU hardware</b>

Different devices can have different $C_6$ coefficients and support for different maximum driving amplitudes $\Omega$ [ref to Hamiltonian here].
Intuitively, under stronger interactions (rydberg-rydberg and laser-rydberg),
bond dimension will grow more quickly<sup>[[2]](#bond-dimension)</sup>, thus affecting performance of our tensor network based emulator.
For a list of the available devices and their specifications, please refer to [Pulser documentation](https://pulser.readthedocs.io/en/stable/tutorials/virtual_devices.html).

----

<b>c. CPU/GPU hardware</b>

_Emu-MPS_ is built on top of [pytorch](https://pytorch.org/). Thus, it can run on most available CPUs and GPUs, from a laptop to a cluster. The presented benchmarks are run on an NVIDIA DGX cluster node, requesting the following resources

- GPU: 1 NVIDIA A100 (40 GB)
- CPU: 16 cores on AMD EPYC 7742

Of course, performance will vary depending on the hardware.
For this reason, if at any point of your work, performance becomes critical, we always recommend to use Pasqal's DGX cluster.
If you intend to run _Emu-MPS_ on your laptop, for example, please be aware that the suggestion to use a GPU for heavier workloads might not be valid.
In such case it is always good to check performance on a couple of runs, changing the _Emu-MPS_ config default values as documented in the [API](../api.md#mpsconfig).
In particular `num_devices_to_use = 0` will run the emulation on CPU, while `num_devices_to_use ≥ 1` on GPU/s.


## Bond dimension
Please, have a look at [http://tensornetwork.org/mps/](http://tensornetwork.org/mps/) for a more general introduction to matrix product states.

The MPS is the best understood factorization of an arbitrary tensor, for which many efficient algorithms have been developed. For a quick understanding, in tensor diagram notation, let's consider the wavefunction of $N$ qubits:

<img src="./images/mps_bond_dimension.png" width="50%" style="background-color:white;">

Alternatively, the MPS of the state can be expressed in traditional notation as

$$
|s_1 s_2\dots s_N\rangle = \sum_{\{\alpha\}}A^{s_1}_{\alpha_1}A^{s_2}_{\alpha_1\alpha_2}\dots A^{s_N}_{\alpha_N}
$$

The state is therefore is represented as a product of tensors. The contracted (or summed over) indices $\{\alpha\}$ are called __bond indices__ and their dimension (the bond dimension) can vary from bond to bond.

__The bond dimension required to perfectly represent a state depends on its entanglement (roughly, how much quantum information is stored in it). Up to this limit, a higher bond dimension will mean that the state is represented more faithfully. However, a higher bond dimension also implies that size of the state will be bigger, thus making the emulation more expensive.__

As a consequence, the real power of the MPS representation is that the bond dimension, $\chi= dim(\alpha)$, gives us an additional knob to control how much relevant information about the state we want to capture, potentially making it a more efficient representation compared to the state vector one.

The most physically-relevant way to do it in _Emu-MPS_ is by specifying the `precision` argument during the time evolution. Doing so, at each step, will throw away components of the state whose sum weighs less that the specified precision, achieving a smaller bond dimension and therefore reducing the memory footprint of the state.

As an additional feature, _Emu-MPS_ also allows to conveniently fix the maximum bond dimension allowed, by specifying the `max_bond_dim` argument. Intuitively, the truncation algorithm will select the `max_bond_dim` most relevant components of the state. The drawback is that the error cannot be estimated anymore a priori.


## QPU Hamiltonian
In all cases we will refer to $H$ as the rydberg-rydberg Hamiltonian that can be implemented on Pasqal's hardware,

$$
H = -\sum_j\Delta_jn_j \ + \ \sum_j\Omega_j\sigma^x_j \ + \ H_{rr}
$$

where the interaction Hamiltonian reads

$$
H_{rr} = \sum_{i>j}\frac{C_{6}}{r_{ij}^{6}} n_{i}n_{j}
$$

Values of $\Omega_j$ and $\Delta_j$ respectively represent the amplitude and the detuning the driving field applied to the qubit $j$. Avoiding technical details we will refer to eigenstates of $H$ (and in particular to the ground state) as equilibrium states.

We then explore two time evolution protocols:

- __Adiabatic evolution:__ Here at each time step, the evolution of the driving $\Omega, \Delta$ is _slow enough_ to guarantee that the evolved state is still an equilibrium state of $H$.
- __Quench:__ One of the most fundamental protocols to drive a system out of equilibrium, it is realized here as follows: at time $t=0$ the system is prepared in the ground state $|\psi_0\rangle$ of $H_0$. The driving field is then suddenly turned on ($\Omega\neq0$) and the system is evolved for $t > 0$, as $|\psi\rangle=e^{-iHt}|\psi_0\rangle$.

As anticipated, they typically complement each other.
Since the matrix product state approach in _Emu-MPS_ strives to minimize the stored information, keeping track of a single equilibrium state in adiabatic time evolution is typically easier. While this single state can be a complicated object itself, quenches, driving the system out of equilibrium, involves taking into account multiple excited states, thus (again, typically as a rule of thumb), computationally harder to emulate.
