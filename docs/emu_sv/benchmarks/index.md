# emu-sv benchmarks

All the benchmarks are run on a single NVIDIA A100 GPU of Pasqal's DGX-cluster and for best performance on heavy workloads we recommend using a similar setup.
There, users should expect emu-sv to emulate up to > 25 qubits. However, at these qubit numbers, the exponential scaling in runtime and performance scale exponentially and emu-mps is likely to be faster.

## Use-case benchmarks

Benchmark efforts, documented here, are meant to provide insights for emu-sv users about

- **Performance**: runtime
- **Accuracy**: different precision levels as compared to pulser-simulation

given a set of meaningful sequences of interest (quench, adiabatic and use-case sequences) that we are going to introduce case by case. Finally, we will only focus on 2D atomic registers as they represent the most numerically challenging and interesting case to study.

The benchmarks are ordered in subpages by general topic.

- [Accuracy](./accuracy.md)
- [Performance](./performance.md)

The accuracy benchmarks compare results between emulators to create confidence in the results emu-sv generates. The performance benchmarks exist to exhibit the runtime characteristics of emu-sv. Based on these, the reader should get a feel for what kind of parameters would be required to be able to run a given sequence in a given time.

## Sequences used

- __Adiabatic evolution:__ Here at each time step, the evolution of the driving $\Omega, \Delta$ is _slow enough_ to guarantee that the evolved state is still an equilibrium state of $H$. Note that the adiabaticity of a sequence is dependent on the energy gaps in the Hamiltonian, and since these gaps decrease with qubit number, most sequences are only adiabatic up to a given qubit number.

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

- __Quench:__ One of the most fundamental protocols to drive a system out of equilibrium, it is realized here as follows: at time $t=0$ the system is prepared in the ground state $|\psi_0\rangle$ of $H_0$. The driving field is then suddenly turned on ($\Omega\neq0$) and the system is evolved for $t > 0$, as $|\psi\rangle=e^{-iHt}|\psi_0\rangle$.

```python
hx = 1.5  # hx/J_max
hz = 0  # hz/J_max
t = 1.5  # t/J_max
# Set up Pulser simulations
R = 7  # μm
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

For emu-sv the relevant difference between the two is that the adiabatic sequence is (weakly) time-dependent, and the quench is not time-dependent at all. This has an effect on the accuracy with which they're simulated (see the [accuracy page](./accuracy.md)).

## CPU/GPU hardware

Emu-sv is built on top of [pytorch](https://pytorch.org/). Thus, it can run on most available CPUs and GPUs, from a laptop to a cluster. The presented benchmarks are run on an NVIDIA DGX cluster node, requesting the following resources

- GPU: 1 NVIDIA A100 (40 GB)
- CPU: a benchmark-dependent number of cores on an AMD EPYC 7742

Of course, performance will vary depending on the hardware.
For this reason, if at any point of your work, performance becomes critical, we always recommend to use Pasqal's DGX cluster.
If you intend to run emu-sv on your laptop, for example, please be aware that the suggestion to use a GPU for heavier workloads might not be valid.
In such case it is always good to check performance on a couple of runs, changing the emu-sv config default values as documented in the [API](../api.md#svconfig).
In particular `gpu = False` will run the emulation on CPU, while `gpu = True` will run it on GPU.
