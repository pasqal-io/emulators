# Computing observables

All emulators (backends) share a convenient way to define observables to be tracked during the simulation, as defined in [pulser](https://pulser.readthedocs.io/en/stable/apidoc/_autosummary/pulser.backend.html). The default available observables are listed below, together with examples for how to create them.

!!! example
    How to use created observables to obtain results from an emulation is shown in the example notebooks [for emu-sv](./emu_sv/notebooks/getting_started.ipynb) and [for emu-mps](./emu_mps/notebooks/getting_started.ipynb).

!!! warning
    Please, take into account that, for performance reasons, individual emulators may overwrite the default implementation in [pulser](https://pulser.readthedocs.io/en/stable/apidoc/_autosummary/pulser.backend.html).

## StateResult

Stores the quantum state at the evaluation times.

See [Pulser documentation](https://pulser.readthedocs.io/en/stable/apidoc/_autosummary/pulser.backend.StateResult.html).

## BitStrings

Stores bitstrings sampled from the state at the evaluation times.

See [Pulser documentation](https://pulser.readthedocs.io/en/stable/apidoc/_autosummary/pulser.backend.BitStrings.html).

## Fidelity

Stores the fidelity with a pure state at the evaluation times.

See [Pulser documentation](https://pulser.readthedocs.io/en/stable/apidoc/_autosummary/pulser.backend.Fidelity.html).

## Expectation

Stores the expectation of the given operator on the current state.

See [Pulser documentation](https://pulser.readthedocs.io/en/stable/apidoc/_autosummary/pulser.backend.Expectation.html).

## CorrelationMatrix

Stores the correlation matrix for the current state.

See [Pulser documentation](https://pulser.readthedocs.io/en/stable/apidoc/_autosummary/pulser.backend.CorrelationMatrix.html).

## Occupation

Stores the occupation number of an eigenstate on each qudit.

See [Pulser documentation](https://pulser.readthedocs.io/en/stable/apidoc/_autosummary/pulser.backend.Occupation.html).

## Energy

Stores the energy of the system at the evaluation times.

See [Pulser documentation](https://pulser.readthedocs.io/en/stable/apidoc/_autosummary/pulser.backend.Energy.html).

## EnergyVariance

Stores the variance of the Hamiltonian at the evaluation times.

See [Pulser documentation](https://pulser.readthedocs.io/en/stable/apidoc/_autosummary/pulser.backend.EnergyVariance.html).

## SecondMomentOfEnergy

Stores the expectation value of `H(t)²` at the evaluation times.

See [Pulser documentation](https://pulser.readthedocs.io/en/stable/apidoc/_autosummary/pulser.backend.EnergySecondMoment.html).

## Defining your own observable

Most commonly desired information can be obtained using the classes documented above

- Arbitrary observables can be measured using `Expectation(operator, ...)` which requires providing a valid operator for the backend in use. Please check the [tutorial](../examples/emu_mps/creating_observables_tutorial.py) for an example of an implementation.
- Fidelities on arbitrary states can be computed using `Fidelity(state, ...)` which requires providing a valid state for the backend in use.
- Information about the time dependent states and Hamiltonians is available via `StateResult`, `Energy` etc.

If additional behavior is desired (e.g. the kurtosis of the energy, or entanglement entropy), the user can subclass the `Observable` class to implement any behavior only depending on the parameters of its `apply` method ([see here](base_classes.md/#observable)). Computation of the entanglement entropy, for example, cannot be done in a backend-independent manner, so it is unlikely to ever make it into the above default list. However, we do intend to define backend-specific callbacks in the future, which would belong to the API of a specific backend. Callbacks that can be implemented in a backend-independent manner can be added to the above list upon popular request.

## Observables on remote backend

In heavy simulations, the entire quantum state can easily occupy few GBs while operators, might not even fit into the available memory.
For this reason, **only for remote backend execution on [pasaqal-cloud](https://docs.pasqal.cloud/cloud/)**, and when defining a state/operator :

!!! info
    - `state` in `Fidelity(state, ...)`, besides being a proper state for the backend, must also be defined with its [`from_state_amplitudes()`](base_classes.md/#state) class method.
    - `operator` in `Expectation(operator, ...)`, besides being a proper operator for the backend, must also be defined with its [`from_operator_repr()`](base_classes.md/#operator) class method.

The methods above requires the user to locally install a backend to instantiate a state or an operator class.
It might not be desirable in all cases since, for example, `pulser-simulation` will install `qutip`, while `emu-mps/sv` will install `torch`.
To avoid such dependency, the classes [`StateRepr`](base_classes.md/#staterepr), [`OperatorRepr`](base_classes.md/#operatorrepr) just implements the abstract representation of such objects.

!!! tip
    If the full backend is not needed, state/operator in the observable can just be defined with their abstract representation

    - `StateRepr.from_state_amplitudes()`
    - `OperatorRepr.from_operator_repr()`

Moreover, the observable
!!! failure
    `StateResult` is not supported in [pasaqal-cloud](https://docs.pasqal.cloud/cloud/) remote backend emulators.
