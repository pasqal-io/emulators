# API specification

The emu-mps api is based on a series of abstract base classes, which are intended to generalize into a backend independent API.
Currently these classes are defined in emu-mps, and they will be documented here until they are moved into a more general location, probably pulser-core.
While they are in this project, see the specification [here](../base_classes.md).

## SVBackend
::: emu_sv.sv_backend.SVBackend

## SVConfig
::: emu_sv.sv_config.SVConfig

## StateVector
::: emu_sv.state_vector.StateVector

### inner
::: emu_sv.state_vector.inner

## DenseOperator
::: emu_sv.dense_operator.DenseOperator
