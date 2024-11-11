# Noise

Here, we analyze the time evolution of a quantum state using the adiabatic sequence <sup>[[1]](./index.md#sequences-used)</sup> under the influence of **depolarizing noise**. Typically, quantum systems are affected by interactions with their surrounding environment, making them **open** systems. To model the dynamics of such noisy quantum systems, one typically solves the **Lindblad "Master" equation**, which governs the time evolution of the density matrix, $\rho$.

The following plot illustrates the time evolution of the initial state under the adiabatic $2\text{D}$ sequence. This is done by tracking the evolution of the **energy** (top left), **variance** (bottom left), and **magnetization** (top and bottom right) **in the presence of depolarizing noise** for a $(2\times2)$ qubit register ($4$ qubits). Specifically, we compare results from two different methods:

- **Pulser**: explicitly solves the Lindblad Master equation, obtaining full information about the noise in the system through its probability distribution in phase space, as given by the density matrix

- **EMU-MPS**: uses a **Monte Carlo (MC)** method to probe the noise by obtaining sample statistics from its underlying probability distribution (see [noise.md](../advanced/noise.md) for further details).

The goal of this study is to demonstrate that the results obtained using the Monte Carlo method implemented in EMU-MPS are qualitatively similar to those found by solving the Lindblad master equation in Pulser.

<img src="../benchmark_plots/afm_state_fidelity_with_noise.png"  width="49.7%">

The key advantage of the Monte Carlo method, if the Hilbert space of $N$ qubits has dimension $dim(H) = d^N$​, then propagating the density matrix using the Lindblad equation requires handling an object of size $[dim(H)]^2$​. In contrast, the stochastic sampling of states with EMU-MPS involves the propagation of state vectors of size $dim(H)$ only. This drastically reduces the computational cost, especially when the number of qubits is large.

In this study, we consider two different depolarization noise rates: **$0.2$** and **$0.5$**. These represent different levels of interaction with the environment, with $0.5$ introducing stronger noise effects than $0.2$. For the EMU-MPS simulations, the following parameters are used:

- **Monte Carlo runs**: 100

- **Precision**: $10^{-6}$, which is better than the default value ($10^{-5}$), as recommended in the [**warning** found here](../advanced/noise.md).

Since the Monte Carlo method in EMU-MPS relies on stochastic sampling, the number of Monte Carlo runs chosen by the user determines the accuracy of the simulation. Each data point (e.g., in the energy plot) in the EMU-MPS results represents the **statistical average observable value across all Monte Carlo runs at a given time $t$**. The plots demonstrate that with $100$ Monte Carlo runs, EMU-MPS already yields qualitative agreement with Pulser. We expect that, increasing the number of Monte Carlo runs should smoothen the EMU-MPS curves further, leading to even closer agreement with the Pulser method.

The overall energy of the system initially rises due to the presence of depolarizing noise, which introduces interactions between the system and the environment. This interaction reduces even further the strength of the spin correlations in the paramagnetic state. The system with higher noise rate ($0.5$) experiences stronger interaction effects, leading to a more pronounced increase in both the energy and energy fluctuations $\Delta E$. However, as the system continues to evolve, it begins to move toward an antiferromagnetic (AFM) correlated state, causing the energy and fluctuations to decrease. During this middle phase, the spin correlations become more meaningful, and the state grows gradually ordered. Eventually, the system undergoes a quantum phase transition at $t \approx 3000$ ns, moving from a paramagnetic state, where the spins are randomly aligned, to a true AFM state with well-defined spin ordering. This transition is reflected in the further reduction of energy fluctuations as the state becomes antiferromagnetic.
