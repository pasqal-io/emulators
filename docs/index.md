# Welcome to EMU-MPS
**EMU-MPS** is a [Pulser](https://github.com/pasqal-io/Pulser) backend, designed to **EMU**late the dynamics of programmable arrays of neutral atoms, with matrix product states (**MPS**). MPSs are a way of encoding quantum states such that the memory required to represent the wavefunction depends on its entanglement, roughly, how much quantum information is stored in it. For product states, an MPS only takes `d*N` numbers to store the state for `N` `d`-level qudits, while for maximally entangled states, it'll take a multiple of the memory in a state vector. For systems of interest, MPSs are expected to be more efficient than state-vectors, allowing the user to simulate more qubits. For more information, see [Tensor Network](https://tensornetwork.org/). EMU-MPS is built on [PyTorch](https://pytorch.org/), and in the future we intend to make it differentiable.

## Setup

You can install from source, or download the package from the private pypi registry that pasqal maintains in gitlab.
For developers, we recommend installing from source, for users we recommend installing from the registry.

**Warning:** installing emu-mps will update pulser-core

We always recommend using a virtual environment.

<details>
  <summary>Click me to see how it is done</summary>

  Creating a virtual environment using python:

  ```
  python -m venv .venv
  ```

  Or

  ```
  python -m venv /path/to/new/virtual/environment
  ```

  Replace `/path/to/new/virtual/environment` with your desired directory path.

  Then activate the environment On linux or MacOS

  ```
  source /path/to/new/virtual/environment/bin/activate
  ```

  While on Windows it's

  ```
  C:\> /path/to/new/virtual/environment/Scripts/activate
  ```

  Remember to replace `/path/to/new/virtual/environment` with the actual path to your virtual environment. Once the environment is activated, you can clone emu_mps and install it using

</details>

### installing from the registry

When pip is configured to know about the pasqal registry, EMU-MPS installs as

```bash
pip install emu-mps
```
When pip is not already configured, the easiest way to do so, is to add a file
`~/.config/pip/pip.conf`

containing the following:

```
[global]
extra-index-url=https://<USERNAME>:<PASSWORD>@gitlab.pasqal.com/api/v4/projects/190/packages/pypi/simple
```

For the username and password required for the above url, please contact a member of emulation team or qlibs.


It is also possible to add the `extra-index-url` to the `pip install` command directly, if you somehow don't want to create a `pip.conf` file.

### installing from source
git clone this [repository ](https://gitlab.pasqal.com/emulation/rydberg-atoms/emu-ct) or download


Then, `cd` into the root folder of the repo and type

```bash
pip install -e .
```

<details>
  <summary>Guidelines for developers </summary>
  We recommend using an environment, git clone the repository, then inside the `emu_mps` folder

```bash
pip install -e .
```

  Also, the installation of pytest, nbmake, pre-commit.

  Do not forget to run the unit test suite by simply running `pytest` command.

  Another way can be using hatch.

  #### virtual environment with `hatch`

  ```bash
  python -m pip install hatch
  python -m hatch -v shell
  ```

  When inside the shell with development dependencies, install first the pre-commit hook:
  ```
  pre-commit install
  ```
</details>


## Running a Pulser sequence and getting results

EMU-MPS is meant to run Pulser sequences. Assuming the existence of a Pulser sequence called `seq` (see the [Pulser docs](https://pulser.readthedocs.io/en/stable/tutorials/creating.html)), you can do the following

```python
import emu_mps

#setup the config
dt = 10 #ns, this is the default value
some_integer_factor = 5
times = [some_integer_factor * dt] #every time has to be a multiple of dt
bitstrings = emu_mps.BitStrings(evaluation_times = times, num_shots = 1000)
config = emu_mps.MPSConfig(dt = dt, observables = [bitstrings])

backend = emu_mps.MPSBackend()
results = backend.run(seq, config)

#the results object will contain bitstrings, as per the config
#they can be retrieved via the name property of the observable
bitstring_times = results.get_result_times(bitstrings.name())
strings = results.get_result(bitstrings.name(), bitstring_times[0])
```

In the above, `strings` will be a `Counter[str]` that counts the occurrences of each bitstring that was measured.
Note that the emulation stops at the largest multiple of `dt` which is not larger than the sequence duration.
So if you make a Pulser sequence the duration of which is not a multiple of `dt`, the simulation time will fall slightly short.

## Supported features

The following features are currently supported:

- All Pulser sequences that use only the rydberg channel
- MPS and MPO can be constructed using the abstract Pulser format.
- The following noise types:
    - [SPAM](https://pulser.readthedocs.io/en/stable/tutorials/spam.html)
    - [Monte Carlo quantum jumps](https://pulser.readthedocs.io/en/stable/tutorials/effective_noise.html)
- The following [basis states](https://pulser.readthedocs.io/en/stable/conventions.html) in a sequence:
    - [ground-rydberg](https://pulser.readthedocs.io/en/stable/review.html#programmable-arrays-of-rydberg-atoms)
    - [XY](https://pulser.readthedocs.io/en/stable/tutorials/xy_spin_chain.html)
- The following properties from a Pulser Sequence are also correctly applied:
    - [hardware modulation](https://pulser.readthedocs.io/en/stable/tutorials/output_mod_eom.html)
    - [SLM mask](https://pulser.readthedocs.io/en/stable/tutorials/slm_mask.html)
    - A complex phase for the omega parameter
- Customizable output, with the folowing inbuilt options:
    - The quantum state in MPS format
    - Bitstrings
    - The fidelity with respect to a given state
    - The expectation of a given operator
    - The qubit density (magnetization)
    - The correlation matrix
    - The mean, second moment and variance of the energy
- Specification of
    - initial state
    - various precision parameters
    - whether to run on cpu or gpu(s)
    - the $U_{ij}$ coefficients from [here](./advanced/hamiltonian.md)
    - A cutoff below which $U_{ij}$ are set to 0 (this makes the computation more memory efficient)

## Planned features

- Parallel TDVP on multiple GPUs
- More noise:
    - the currently unsupported noises in the Pulser `NoiseModel`
- Differentiability

## More Info
Please see the API specification for a list of available config options ([see here](api.md)), and the Observables page for how to compute observables ([see here](observables.md)).
Those configuration options relating to the mathematical functioning of the backend are explained in more detail in the Config page ([see here](advanced/config.md)).
For notebooks with examples for how to do various things, please see the notebooks page ([see here](./notebooks/index.md)).
