# Computing observables

All emulators (backends) share a convenient way to define observables to be tracked during the simulation, as defined in [pulser](https://pulser.readthedocs.io/en/stable/apidoc/_autosummary/pulser.backend.html). The default available observables are listed below, together with examples for how to create them.

!!! info "Examples"
    How to use created observables to obtain results from an emulation is shown in the example notebooks [for emu-sv](./emu_sv/notebooks/getting_started.ipynb) and [for emu-mps](./emu_mps/notebooks/getting_started.ipynb).

!!! warning
    Please, take into account that, for performance reasons, individual emulators may overwrite the default implementation in [pulser](https://pulser.readthedocs.io/en/stable/apidoc/_autosummary/pulser.backend.html).

## StateResult
::: pulser.backend.default_observables.StateResult
    options:
        show_source: false

## BitStrings
::: pulser.backend.default_observables.BitStrings
    options:
        show_source: false

## Fidelity
::: pulser.backend.default_observables.Fidelity
    options:
        show_source: false

## Expectation
::: pulser.backend.default_observables.Expectation
    options:
        show_source: false

## CorrelationMatrix
::: pulser.backend.default_observables.CorrelationMatrix
    options:
        show_source: false

## QubitDensity
::: pulser.backend.default_observables.Occupation
    options:
        show_source: false

## Energy
::: pulser.backend.default_observables.Energy
    options:
        show_source: false

## EnergyVariance
::: pulser.backend.default_observables.EnergyVariance
    options:
        show_source: false

## SecondMomentOfEnergy
::: pulser.backend.default_observables.EnergySecondMoment
    options:
        show_source: false

## Defining your own observable
Most commonly desired information can be obtained using the classes documented above

- arbitrary observables can be measured using `Expectation`
- fidelities on arbitrary states can be computed using `Fidelity`
- Information about the time dependent states and Hamiltonians is available via `StateResult`, `Energy` etc.

If additional behaviour is desired (e.g. the kurtosis of the energy, or entanglement entropy), the user can subclass the `Callback` class to implement any behaviour only depending on the parameters of its `__call__` method ([see here](base_classes.md#callback)). Computation of the entanglement entropy, for example, cannot be done in a backend-independent manner, so it is unlikely to ever make it into the above default list. However, we do intend to define backend-specific callbacks in the future, which would belong to the API of a specific backend. Callbacks that can be implemented in a backend-independent manner can be added to the above list upon popular request.
