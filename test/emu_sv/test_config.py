from emu_sv import SVConfig


def test_serialise_default_config() -> None:
    default_config = SVConfig()
    config_str = default_config.to_abstract_repr()

    deserialized_config = SVConfig.from_abstract_repr(config_str)

    for attr in ["dt", "max_krylov_dim", "krylov_tolerance", "gpu"]:
        assert getattr(deserialized_config, attr) == getattr(default_config, attr), f"{attr} mismatch"


def test_serialise_config() -> None:
    dt = 10
    max_krylov_dim = 100
    krylov_tolerance = 1e-5
    gpu = False

    default_config = SVConfig(
        dt=dt,
        max_krylov_dim = max_krylov_dim,
        krylov_tolerance=krylov_tolerance,
        gpu=gpu,
        )
    config_str = default_config.to_abstract_repr()

    deserialized_config = SVConfig.from_abstract_repr(config_str)

    attributes = ["dt", "max_krylov_dim", "krylov_tolerance", "gpu"]
    values = [dt, max_krylov_dim, krylov_tolerance, gpu]
    for attr, val in zip(attributes, values):
        assert getattr(deserialized_config, attr) == getattr(default_config, attr), f"{attr} mismatch"
        assert getattr(deserialized_config, attr) == val, f"{attr} != {val} mismatch"