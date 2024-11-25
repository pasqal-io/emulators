from typing import Any

import pytest
import logging
from unittest.mock import MagicMock

from emu_base import Results, AggregationType


def test_store_get():
    victim = Results()

    my_observable = MagicMock()
    my_observable.name = "my_observable"

    victim.store(callback=my_observable, time=400, value="Hello world!")
    victim.store(callback=my_observable, time=500, value="Hello world 2!")

    with pytest.raises(ValueError) as e:
        victim.store(callback=my_observable, time=400, value="Hello world 3!")
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


def test_aggregate_default_aggregator():
    results = [Results() for _ in range(3)]

    my_observable = MagicMock()
    my_observable.name = "my_observable"
    my_observable.default_aggregation_type = AggregationType.MEAN

    def store(*, time: int, values: list[Any]):
        for i, value in enumerate(values):
            results[i].store(callback=my_observable, time=time, value=value)

    store(time=400, values=[0, 6, 15])

    aggregated = Results.aggregate(results)

    assert aggregated.get_result_names() == ["my_observable"]
    assert aggregated.get_result_times("my_observable") == [400]
    assert aggregated.get_result("my_observable", 400) == pytest.approx(7)

    results[1].store(callback=my_observable, time=500, value="whatever")

    with pytest.raises(ValueError) as e:
        Results.aggregate(results)

    assert str(e.value) == (
        "Monte-Carlo results seem to provide from incompatible simulations: "
        "the callbacks are not stored at the same times"
    )


def test_aggregate_custom_aggregator(caplog):
    results = [Results() for _ in range(3)]

    my_observable = MagicMock()
    my_observable.name = "my_observable"
    my_observable.default_aggregation_type = None

    def store(*, time: int, values: list[Any]):
        for i, value in enumerate(values):
            results[i].store(callback=my_observable, time=time, value=value)

    store(time=400, values=["A", "B", "C"])

    Results.aggregate(results)
    assert caplog.record_tuples == [
        ("global_logger", logging.WARNING, "Skipping aggregation of `my_observable`")
    ]

    def string_concat(strings: list[str]):
        return "".join(strings)

    aggregated = Results.aggregate(results, my_observable=string_concat)

    assert aggregated.get_result_names() == ["my_observable"]
    assert aggregated.get_result_times("my_observable") == [400]
    assert aggregated.get_result("my_observable", 400) == "ABC"

    results[1].store(callback=my_observable, time=300, value="whatever")

    with pytest.raises(ValueError) as e:
        Results.aggregate(results, my_observable=string_concat)

    assert str(e.value) == (
        "Monte-Carlo results seem to provide from incompatible simulations: "
        "the callbacks are not stored at the same times"
    )


def test_aggregate_custom_aggregator_override():
    results = [Results() for _ in range(3)]

    my_observable = MagicMock()
    my_observable.name = "my_observable"
    my_observable.default_aggregation_type = AggregationType.MEAN

    def store(*, time: int, values: list[Any]):
        for i, value in enumerate(values):
            results[i].store(callback=my_observable, time=time, value=value)

    store(time=400, values=[5, 15, 10])

    aggregated = Results.aggregate(results, my_observable=max)

    assert aggregated.get_result_names() == ["my_observable"]
    assert aggregated.get_result_times("my_observable") == [400]
    assert aggregated.get_result("my_observable", 400) == 15


def test_aggregate_multiple_observables():
    results = [Results() for _ in range(2)]
    observable1, observable2 = MagicMock(), MagicMock()

    for i, observable in enumerate([observable1, observable2]):
        observable.name = f"observable_{i}"
        observable.default_aggregation_type = AggregationType.MEAN

    results[0].store(callback=observable1, time=100, value=[1, 10])
    results[1].store(callback=observable1, time=100, value=[2, 11])
    results[0].store(callback=observable2, time=100, value=500)

    with pytest.raises(ValueError) as e:
        Results.aggregate(results)

    assert str(e.value) == (
        "Monte-Carlo results seem to provide from incompatible simulations: "
        "they do not all contain the same observables"
    )

    results[1].store(callback=observable2, time=100, value=501)

    aggregated = Results.aggregate(results)

    assert aggregated["observable_0", 100][0] == pytest.approx(1.5)
    assert aggregated["observable_0", 100][1] == pytest.approx(10.5)

    assert aggregated["observable_1", 100] == pytest.approx(500.5)
