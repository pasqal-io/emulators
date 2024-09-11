# QPU Hamiltonian
In all cases we will refer to $H$ as the rydberg-rydberg Hamiltonian that can be implemented on Pasqal's hardware,

$$
H = -\sum_j\Delta_jn_j \ + \ \sum_j\Omega_j\sigma^x_j \ + \ H_{i}
$$

where $H_i$ is the interaction term in the Hamiltonian. In the QPU, various kinds of interaction terms can be generated,
such as a Rydberg interaction term and an XY interaction.

Currently, emu-mps only supports the Rydberg interaction

$$
H_{rr} = \sum_{i>j}\frac{C_{6}}{r_{ij}^{6}} n_{i}n_{j}
$$

Values of $\Omega_j$ and $\Delta_j$ respectively represent the amplitude and the detuning of the driving field applied to the qubit $j$. Avoiding technical details we will refer to eigenstates of $H$ (and in particular to the ground state) as equilibrium states.

Different devices can have different $C_6$ coefficients and support for different maximum driving amplitudes $\Omega$.
Intuitively, under stronger interactions (rydberg-rydberg and laser-rydberg),
bond dimension will grow more quickly ([see here](mps/index.md)), thus affecting performance of our tensor network based emulator.
For a list of the available devices and their specifications, please refer to the Pulser documentation ([see here](https://pulser.readthedocs.io/en/stable/tutorials/virtual_devices.html)).
