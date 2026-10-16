<div align="center">
  <img src="docs/logos/LogoTaglineSoftGreen.svg">

  # OptiMatrix
  ![Code Coverage](https://img.shields.io/badge/Coverage-100%25-brightgreen.svg)

</div>


OptiMatrix is a Python package designed to improve emu-mps simulations in terms of memory and therefore performance.
It achieves this by improving qubit ordering, arranging them to closely resemble a 1D system. This approach ensures that the system is represented as a 1D structure with minimal long-range interactions.

Effectively, it reduces the bandwidth of an interaction matrix $A_{ij}$ in the Hamiltonian
$$
H = \sum A_{ij} n^z_i n^z_j + ...,
$$
which perfectly fits to the description of the Rydberg or Ising-like systems.

Methods used in the package are based on the Cuthill-McKee algorithm and can be further improved with integer linear programming techniques, see
[arxiv paper](https://arxiv.org/abs/2404.15165).


## Examples

 - 1D open chain. Randomly shuffled qubits got sorted.\
 Before $\to$ after.
<div style="display: flex; justify-content: center;">
  <img src="./docs/images_from_notebooks/1D_before.png" alt="Image 1" style="width: 45%; margin-right: 10px;">
  <img src="./docs/images_from_notebooks/1D_after.png" alt="Image 2" style="width: 45%;">
</div>

 - 1D periodic chain. Periodic 1D nearest neighbor interacting chain transforms into *open* chain with next nearest neighbor interactions.\
 Before $\to$ after
<div style="display: flex; justify-content: center;">
  <img src="./docs/images_from_notebooks/1D_PBC_before.png" alt="Image 1" style="width: 45%; margin-right: 10px;">
  <img src="./docs/images_from_notebooks/1D_PBC_after.png" alt="Image 2" style="width: 45%;">
</div>

 - 2D system. The classical zig-zag line is optimised such that the long interactions in quasi 1D are shorter.\
 Before $\to$ after
<div style="display: flex; justify-content: center;">
  <img src="./docs/images_from_notebooks/2D_before.png" alt="Image 1" style="width: 45%; margin-right: 10px;">
  <img src="./docs/images_from_notebooks/2D_after.png" alt="Image 2" style="width: 45%;">
</div>

The interaction matrix $A_{ij}$ in the Hamiltonian $$ H = \sum A_{ij} n^z_i n^z_j $$ transforms as:


<div style="text-align: center;">
  <img src="docs/images/band.jpeg" alt="Bandwidth" width="90%">
</div>
