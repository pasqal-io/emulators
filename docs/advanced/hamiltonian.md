# QPU Hamiltonian
In all cases we will refer to $H$ as the rydberg-rydberg Hamiltonian that can be implemented on Pasqal's hardware,

$$
H = -\sum_j\Delta_jn_j \ + \ \sum_j\Omega_j\sigma^x_j \ + \ H_{i}
$$

where $H_i$ is the interaction term in the Hamiltonian.
Values of $\Omega_j$ and $\Delta_j$ respectively represent the amplitude and the detuning of the driving field applied to the qubit $j$. Avoiding technical details we will refer to eigenstates of $H$ (and in particular to the ground state) as equilibrium states.

In the QPU, various kinds of interaction terms can be generated, and EMU-MPS supports the Rydberg interaction term and the XY interaction.

The Rydberg interaction reads

$$
H_{rr} = \sum_{i>j} U_{ij} n_{i}n_{j}
$$

where

$$
U_{ij} = \frac{C_{6}}{r_{ij}^{6}},
$$

and the XY interaction reads

$$
H_{xy} = \sum_{i>j} U_{ij} (\sigma^+_{i}\sigma^-_{j} + h.c.)
$$

where

$$
U_{ij} = \frac{C_{3}(1-3 \cos^2(\theta_{ij}))}{r_{ij}^{3}},
$$

In these formulas, $r_{ij}$ represents the distance between qubits $i$ and $j$, and $\theta_{ij}$ represents a configurable angle ([see here](https://pulser.readthedocs.io/en/stable/tutorials/xy_spin_chain.html)).
Currently, Pasqal quantum devices only support Rydberg interactions, and different devices have different $C_6$ coefficients and support for different maximum driving amplitudes $\Omega$.
Intuitively, under stronger interactions (rydberg-rydberg and laser-rydberg),
bond dimension will grow more quickly ([see here](mps/index.md)), thus affecting performance of our tensor network based emulator.
For a list of the available devices and their specifications, please refer to the Pulser documentation ([see here](https://pulser.readthedocs.io/en/stable/tutorials/virtual_devices.html)).
