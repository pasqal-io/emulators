# Noise Implementation in emu-mps
To faithfully emulate the Pasqal QPU using emu-mps, we need to include noise effects, as these effects cannot be neglected in a real quantum system—they significantly impact the performance and fidelity of the QPU.

In open quantum many-body systems, noise is typically expressed in terms of **mixed states** and **noise channels** using a **density matrix representation**. Similar to a state-vector emulator, emu-mps **only handles pure states**. Therefore, we implement noise using a higher order Monte Carlo method ([see here](https://www.phys.ens.psl.eu/~dalibard/publi3/osa_93.pdf)), where we evolve the system using an **effective Hamiltonian** and then apply a quantum jump at certain times. This method is probabilistic in the sense that it approximates the simulation of a mixed state using many non-deterministic pure state simulations.

## Noise Types

Our implementation supports different types of noise:

- **relaxation**: due to a decay from the Rydberg to the ground state. It is parameterized by **relaxation_rate**.
- **dephasing**: due to a random phase flip (Z). It is parameterized by **dephazing_rate**.
- **depolarizing**: used as a tool to test the system under a uniform combination of phase flip (Z) and bit flip (X) errors. It is parameterized by **depolarizing_rate**.
- **eff_noise**: general effective noise channel defined by the set of collapse operators **eff_noise_opers** and their corresponding rates **eff_noise_rates**.
- **SPAM errors**: parameterized by **state_prep_error**, **p_false_pos** and **p_false_neg**.

 Users can refer to the [Pulser documentation](https://pulser.readthedocs.io/en/stable/tutorials/noisy_sim.html) for a detailed overview of the different noise models currently available. Currently, emu-mps does not support the **leakage**.

## Effective Hamiltonian
The non-hermitian **effective Hamiltonian** used in noisy emu-mps simulations includes both the physical Hamiltonian $H_{physical}$, which governs the noiseless evolution of the system, and a term representing noise:

$$
H_{\text{eff}} = H_{\text{physical}} \ - \ \frac{i}{2} \sum_m L^{\dagger}_m L_m.
$$

where:
- $H_{physical}$ is the Hamiltonian of the noiseless system.
- The second term is a sum over the **Lindblad operators** (`L`), which represent different types of noise.

## Time Evolution Mechanism

The system undergoes deterministic time evolution from time $t$ to $t + \delta t$ using TDVP ([see here](algorithms.md)) with the effective Hamiltonian. At the end of each evolution step, the norm of the evolved quantum state $\vert \psi (t + \delta t)\rangle$ is compared to a collapse threshold, which is a random number between $0$ and $1$:

- **If the square of the norm of the evolved state is greater than the random number**, the system successfully evolves under the effective Hamiltonian $H_{\text{eff}}$ to time $t + \delta t$, and proceeds to the next time step.
- **If the square of the norm of the evolved state is less than the random number**, a **quantum jump** occurs. This can be understood as a simulation of a noise event (e.g. spontaneous emission, dephasing, etc.).

---

# **WARNING:**

**It is important to note that the norm of the state also decreases due to truncation effects. Therefore, it is recommended to choose an appropriate precision when performing Monte Carlo simulations. Additionally, computing the collapse times may become unreliable when the maximum bond dimension chosen by the user is reached, as truncation errors can become difficult to control.**

---

### Locating the Quantum Jump

To determine when a quantum jump occurs between time $t$ and $t + \delta t$, the TDVP is applied multiple times (both forward and backward in time) to approximate the collapse time. During this process, the norm of the evolved state is checked and compared to the collapse threshold. This evolution procedure is repeated until the norm converges to the collapse threshold.

### Applying the Quantum Jump

Once the time of collapse is located, a Lindblad operator is randomly applied to a qubit (based on the collapse weight), then the evolution of the normalized state is continued to complete the time step of size $\delta t$.

Upon completion of the current time step, the time evolution continues with the next step.

## Physical Interpretation: Spontaneous Emission in the Two-Level System

To better understand how the quantum jump process describes a physical event (noise) which occurs during time evolution, let us consider a two-level system initially in the state $\vert \psi(t)\rangle = \alpha\vert g\rangle + \beta \vert e\rangle$, where $\alpha$ and $\beta$ are complex coefficients. By setting both the amplitude and detuning to zero, $\Omega = \delta = 0$, we can then ask what happens in a single step depending on whether a **spontaneously emitted photon** occurs or not. For this, we examine a single quantum jump operator $L = \sqrt{\Gamma}\vert g\rangle\langle e\vert$, and an effective Hamiltonian $H_{eff} = -i(\Gamma/2)\vert e\rangle\langle e\vert$.

If a quantum jump occurs in a time step $\delta t$, then the state following the jump becomes

\[
\vert \psi(t + \delta t)\rangle = \frac{L \vert \psi(t) \rangle}{\| L \vert \psi(t) \rangle \|} = \vert g \rangle.
\]

In other words, when a spontaneous emission event occurs, the evolved state of the system collapses onto the ground state $\vert g\rangle$. This demonstrates how noisy events, like spontaneous emission, can alter the dynamics of a quantum system, even in the absence of direct observation of emitted photons.
