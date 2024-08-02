
<div align="center">
  <img src="docs/images/LogoTaglineSoftGreen.svg">

  # Emu-MPS
</div>



**Emu-mps** is a Pulser backend, designed to **emu**late the dynamics of programmable arrays of neutral atoms, with matrix product states (**MPS**). It allows users to increase the number of qubits and reduce computation time.

Join us on [Slack](https://pasqalworkspace.slack.com/archives/C0389KD4ZKQ) or by [e-mail](mailto:emulation@pasqal.com) to give us feedback about how you plan to use Emu-TN or if you require specific feature-upgrades.

## Features

- Supports more than 40 qubits
- Compatibility with Pulser-core functions
- Adjustable pulse discretization (default: dt ~ 10 ns)

## Getting started

git clone this [repository ](https://gitlab.pasqal.com/emulation/rydberg-atoms/emu-ct)


Inside the `emu_ct` folder

```bash
pip install .
```
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

  Then activate the environment:

  ```
  source .venv/bin/activate
  ```

  Or

  - On Unix or MacOS with bash: source /path/to/new/virtual/environment/bin/activate

  - On Windows: C:\> /path/to/new/virtual/environment/Scripts/activate

  Remember to replace `/path/to/new/virtual/environment` with the actual path to your virtual environment. Once the environment is activated, you can clone emu_ct and install it using

</details>



<details>
  <summary>Guidelines for developers </summary>
  We recommend using an environment, git clone the repository, then inside the `emu_ct` folder

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

##

![Code Coverage](https://img.shields.io/badge/Coverage-95%25-brightgreen.svg)
