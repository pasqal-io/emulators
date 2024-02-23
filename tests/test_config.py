from emu_ct import Config


def test_singleton():
    # Test that Config is actually a singleton

    # first assert on default values
    assert Config().get_bond_precision() == 1e-8
    assert Config().get_max_bond_dim() == 1024
    assert Config().get_krylov_dim() == 80
    assert Config().get_krylov_exp_tolerance() == 1e-10
    assert Config().get_krylov_norm_tolerance() == 1e-7

    # set something else
    Config().set_max_bond_dim(512)
    Config().set_bond_precision(1e-7)
    Config().set_krylov_exp_tolerance(1e-8)

    # see if it sticks
    assert Config().get_bond_precision() == 1e-7
    assert Config().get_max_bond_dim() == 512
    assert Config().get_krylov_exp_tolerance() == 1e-8
