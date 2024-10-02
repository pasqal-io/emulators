import pytest

from emu_mps.base_classes.results import Results


def test_store_get():
    victim = Results()

    victim.store(callback_name="my_observable", time=400, value="Hello world!")
    victim.store(callback_name="my_observable", time=500, value="Hello world 2!")

    with pytest.raises(ValueError) as e:
        victim.store(callback_name="my_observable", time=400, value="Hello world 3!")
        assert (
            str(e.value)
            == "A value is already stored for observable 'my_observable' at time 400"
        )

    assert victim["my_observable"][500] == "Hello world 2!"
    assert victim["my_observable", 500] == "Hello world 2!"
    assert victim["my_observable"][400] == "Hello world!"
    assert victim["my_observable", 400] == "Hello world!"

    with pytest.raises(ValueError) as e:
        victim["my_observabole"][500]
    assert str(e.value) == "No value for observable 'my_observabole' has been stored"

    with pytest.raises(ValueError) as e:
        victim["my_observabole", 500]
    assert str(e.value) == "No value for observable 'my_observabole' has been stored"

    with pytest.raises(ValueError) as e:
        victim["my_observable", 600]
    assert str(e.value) == "No value stored at time 600 for observable 'my_observable'"
