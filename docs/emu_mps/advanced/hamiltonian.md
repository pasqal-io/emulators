# QPU Hamiltonian

We write the system Hamiltonian $H$ in the form

$$
H = -\sum_j \Delta_j n_j \;+\; \sum_j \Omega_j \sigma^x_j \;+\; H_{\text{int}}
$$

where:

- $\Delta_j$ is the detuning applied to qubit $j$ (energy offset).
- $\Omega_j$ is the drive amplitude on qubit $j$ (couples the two levels).
- $n_j, \sigma^x_j$ are the usual number and Pauli-X operators on site $j$.
- $H_{\text{int}}$ collects all pairwise interaction terms.

Think of $\Omega_j$ and $\Delta_j$ as the pulse parameters that control each qubit. Eigenstates of $H$ (in particular the ground state) are sometimes called equilibrium states.

emu-mps supports two types of pairwise interactions below. Pasqal QPUs currently exposes only the Rydberg interaction, but the emulator can handle both Rydberg and XY interactions.

## Rydberg interaction

The Rydberg term is

$$
H_{rr} = \sum_{i>j} U_{ij}\, n_i n_j
\qquad\text{with}\qquad
U_{ij} = \frac{C_6}{r_{ij}^6},
$$

where

- $r_{ij}$ is the distance between qubits $i$ and $j$,
- $C_6$ is the device-dependent van der Waals coefficient.

This term penalizes having two nearby atoms both excited to the Rydberg state; it decays quickly with distance $(\varpropto 1/r^6)$.

## XY interaction

The XY term (spin-exchange) is

$$
H_{xy} = \sum_{i>j} U_{ij}\,(\sigma^+_i \sigma^-_j + \sigma^-_i \sigma^+_j)
\qquad\text{with}\qquad
U_{ij} = \frac{C_3(1 - 3\cos^2\theta_{ij})}{r_{ij}^3},
$$

where

- $C_3$ is a coupling constant,
- $\theta_{ij}$ is the angle between the magnetic field and vector of the two atoms, see the Pulser XY [tutorial](https://pulser.readthedocs.io/en/stable/tutorials/xy_spin_chain.html).

The XY interaction mediates excitation hopping between sites and decays more slowly $(\varpropto 1/r^3)$ and can be anisotropic because of the angular factor.

## Practical notes

- $r_{ij}$ and $\theta_{ij}$: distances and angles are determined by the register coordinates and the magnetic field; see Pulser XY [tutorial](https://pulser.readthedocs.io/en/stable/tutorials/xy_spin_chain.html) for details.

- Device differences: different devices use different $C_6$ (and $C_3$) values and support different maximum $\Omega$. Check device specs in the Pulser devices and virtual devices [tutorial](https://pulser.readthedocs.io/en/stable/tutorials/virtual_devices.html).

- Performance impact: stronger interactions and stronger drives tend to increase entanglement, which raises the required MPS bond dimension and increases memory/CPU cost. See the MPS performance notes for details ([mps/index.md](mps/index.md)).

- When modeling experiments: use the device parameters ($C_6$, geometry, max $\Omega$) that match your target hardware for realistic simulations.

For more device details and examples, see the Pulser documentation: <https://pulser.readthedocs.io/en/stable/tutorials/virtual_devices.html>
