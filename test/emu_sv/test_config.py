from emu_sv import SVConfig

# copypaste sv_config.py specific attributes
# attributes from SVConfig._expected_kwargs
sv_attributes = [
    "dt",
    "max_krylov_dim",
    "gpu",
    "krylov_tolerance",
]


def test_serialise_default_config() -> None:
    default_config = SVConfig()
    config_str = default_config.to_abstract_repr()

    deserialized_config = SVConfig.from_abstract_repr(config_str)

    for attr in sv_attributes:
        assert getattr(deserialized_config, attr) == getattr(
            default_config, attr
        ), f"{attr} mismatch"


def test_serialise_config() -> None:
    # Arguments are arbitrary just to be != default, no deep meaning behind
    dt = 1
    max_krylov_dim = 10
    gpu = False
    krylov_tolerance = 1e-5

    default_config = SVConfig(
        dt=dt,
        max_krylov_dim=max_krylov_dim,
        gpu=gpu,
        krylov_tolerance=krylov_tolerance,
    )

    config_str = default_config.to_abstract_repr()
    deserialized_config = SVConfig.from_abstract_repr(config_str)

    values = [dt, max_krylov_dim, gpu, krylov_tolerance]

    for attr, val in zip(sv_attributes, values):
        assert getattr(deserialized_config, attr) == getattr(
            default_config, attr
        ), f"{attr} mismatch"
        assert getattr(deserialized_config, attr) == val, f"{attr} != {val} mismatch"
