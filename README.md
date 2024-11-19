<div align="center">
  <img src="docs/logos/LogoTaglineSoftGreen.svg">

  # Emu-MPS
</div>

**EMU-MPS** is a Pulser backend, designed to **EMU**late the dynamics of programmable arrays of neutral atoms, with matrix product states (**MPS**). It allows users to increase the number of qubits and reduce computation time.

Join us on [Slack](https://pasqalworkspace.slack.com/archives/C07MUV5K7EU) or by [e-mail](mailto:emulation@pasqal.com) to give us feedback about how you plan to use Emu-MPS or if you require specific feature-upgrades.


## Getting started

You can install from source, or download the package from the private pypi registry that pasqal maintains in gitlab.
For developers, we recommend installing from source, for users we recommend installing from the registry.

**Warning:** installing emu-mps will update pulser-core

We always recommend using a virtual environment.

<details>
  <summary>Click me to see how it is done</summary>

  #### Create a virtual environment using python

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

When pip is configured to know about the pasqal registry, Emu-MPS installs as

```bash
pip install emu-mps
```
When pip is not already configured, the easiest way to do so, is to add a file `~/.config/pip/pip.conf` containing:

```
[global]
extra-index-url=https://gitlab.pasqal.com/api/v4/projects/597/packages/pypi/simple
                possible.other.urls
```

As this shows, it is also possible to have multiple extra repositories configured. Note that the order is not important.

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

## Check the tutorial notebooks and example scripts

For more information, you can check the tutorials and examples located in the [examples folder](https://gitlab.pasqal.com/emulation/rydberg-atoms/emu-ct/-/tree/main/examples?ref_type=heads)

## Documentation

Please check the [documentation](./docs/index.md) page for more info about contributing, the API, benchmarks, etc.


![Code Coverage](https://img.shields.io/badge/Coverage-95%25-brightgreen.svg)
