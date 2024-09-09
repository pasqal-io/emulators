# Estimating convergence of the simulation, a case study

As discussed on the page regarding [memory consumption](memory.md), by limiting the bond-dimension of the state, systems of arbitrary size can be simulated. However, care must be taken that the results obtained are accurate. By setting `max_bond_dim = 1600` `extra_krylov_tolerance=1e-5` and `precision=1e-6`, we were able to run the adiabatic pulse from the [benchmarks](../benchmarks/index.md#adiabatic-sequence) for a 7x7 grid, but were the results accurate? The obtained qubit density was

<table>
  <tr><td>0.87 </td><td> 0.47 </td><td> 0.62 </td><td> 0.56 </td><td> 0.62 </td><td> 0.47 </td><td> 0.87 </td></tr>
  <tr><td>0.47 </td><td> 0.43 </td><td> 0.41 </td><td> 0.42 </td><td> 0.41 </td><td> 0.43 </td><td> 0.47 </td></tr>
  <tr><td>0.62 </td><td> 0.41 </td><td> 0.47 </td><td> 0.45 </td><td> 0.47 </td><td> 0.41 </td><td> 0.62 </td></tr>
  <tr><td>0.56 </td><td> 0.42 </td><td> 0.45 </td><td> 0.45 </td><td> 0.45 </td><td> 0.42 </td><td> 0.56 </td></tr>
  <tr><td>0.62 </td><td> 0.41 </td><td> 0.47 </td><td> 0.45 </td><td> 0.47 </td><td> 0.41 </td><td> 0.62 </td></tr>
  <tr><td>0.47 </td><td> 0.43 </td><td> 0.41 </td><td> 0.42 </td><td> 0.41 </td><td> 0.43 </td><td> 0.47 </td></tr>
  <tr><td>0.87 </td><td> 0.47 </td><td> 0.62 </td><td> 0.56 </td><td> 0.62 </td><td> 0.47 </td><td> 0.87 </td></tr>
</table>

## trustworthiness

First let's discuss whether these results are trustworthy, and then how the values of `max_bond_dim` and `precision` were obtained iteratively.

As described in the page on [sources of error](errors.md#truncation-of-the-state), for verifying the validity of the results, users should vary the parameters of the emulator until the results become insensitive to their variation. This is what was done to obtain the above results, and then the main question is whether long-range interactions are correctly taken into account by the TDVP algorithm. It is worth noting that TDVP is a 1-d algorithm, so that qubits that are close in physical space are not necessarily close in the MPS layout. Specifically, the register ordering we used in the MPS simply concatenates all the colums of the above grid, and uses the resulting linear ordering, so while qubit $(1,1)$ and $(1,2)$ are neighbours in the MPS, qubi $(1,1)$ and $(2,1)$ are not, and this creates an artifical long-range interaction, which TDVP handles through an effective description. It can be seen that this is correctly done because the qubit densities shown above obey the symmetries of the grid, they are reflection symmetric along the horizontal and vertical axis, as well as rotation symmetric around right angles (and because these tranformations form a group, reflection symmetry around the diagonals follows, etc.).

It is noteworthy that for smaller systems the given pulse creates reasonable anti-ferromagnetic structure in the resulting state, but here it's only weakly present. Since the energy-gap decreases with qubit number, at some qubit number, the pulse is not adiabatic enough to create good anti-ferromagnetic structure anymore. Since the results seem trustworthy due to the above reasoning, that limit seems to have been exceeded, and a longer pulse is required to create good antiferromagnetism.

## exploring the parameter space

Since the starting point of a parameter search is always a bit arbitrary, we started by seeing with what `max_bond_dim` we could still simulate the pulse, for the default precision, on 2 gpus. This turns out to have been 1800. Then the first point of investigation was how much this result depends on bond dimension. Only in the below equations, let $\psi_d$ denote the state obtained for `max_bond_dim=d`, then we found

$$
\langle\psi_{1800}, \psi_{1700}\rangle = 0.999
$$

$$
\langle\psi_{1800}, \psi_{1600}\rangle = 0.999
$$

These inner products mean the output state hardly changes, and sure enough, various observables, such as the fidelity on the antiferromagnetic state and the qubit density are also basically constant.
We could probably have reduced `max_bond_dim` even further without much consequence, but `max_bond_dim=1600` was sufficient to run simulations with much better precisions, so we did not investigate further. Recall that by default `precision=1e-5`. We now ran simulations, all with `max_bond_dim=1600` but `precision` one of `[1e-7,2e-7,5e-7,1e-6,2e-6,5e-6]` and `extra_krylov_tolerance=1e-5`. The value of `extra_krylov_tolerance` was chosen so that for `precision=1e-7`, the precision in the Lanczos algorithm was the same as the default in ITensors. Only in the below equations, let $\psi_p$ denote the state obtained for `precision=p`, then we found

$$
\langle\psi_{1e-7}, \psi_{1e-5}\rangle = 0.937
$$

$$
\langle\psi_{2e-7}, \psi_{1e-5}\rangle = 0.937
$$

$$
\langle\psi_{5e-7}, \psi_{1e-5}\rangle = 0.937
$$

$$
\langle\psi_{1e-6}, \psi_{1e-5}\rangle = 0.938
$$

$$
\langle\psi_{2e-6}, \psi_{1e-5}\rangle = 0.912
$$

$$
\langle\psi_{5e-6}, \psi_{1e-5}\rangle = 0.898
$$

Notice that this list of inner products stabilizes up to 3 digits for precisions better than `1e-6`. Assuming a Hermitian operator $A$ and a normalized state, we can write

$$
\langle\psi |A | \psi \rangle - \langle \phi | A | \phi \rangle \leq 4 \| A \| (1 - abs(\langle \psi, \phi \rangle))
$$

Since $\|A\| = 1$ if $A$ is a Pauli string operator, we can specifically expect the qubit density to be accurate up to the second digit for `precision < 1e-6`. This is why the qubit density was only printed for two digits at the start of this page, and from the numerics, we also find the above observation holds. The above also shows that for `precision > 1e-6` we can expect variations in the qubit density in the second digit, which we also observe in the numerics. Specifically, for `precision > 1e-6`, the reflection and rotation symmetries of the qubit densities only hold up the first digit. Since the qubit density at the center of the grid only differs in the second digit, precisions larger that `1e-6` were not deemed sufficient to judge whether the pulse was accurately simulated.

Since the results obtained for `max_bond_dim = 1600` `extra_krylov_tolerance=1e-5` and `precision=1e-6` are accurate enough for the purposes of this discussion, we did not investigate the impact of varying `dt`.
