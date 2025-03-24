from emu_base.aggregators import (
    mean_aggregator,
    bag_union_aggregator,
    aggregate,
)
from pulser.backend import Results
from typing import Any
from unittest.mock import MagicMock
import logging
import pytest
from collections import Counter


def test_mean_aggregator_list():
    assert mean_aggregator([1, 2, 3]) == pytest.approx(2)


def test_mean_aggregator_list_of_lists():
    listwise_mean = mean_aggregator([[3, 4], [2, 6], [1, 5]])

    assert len(listwise_mean) == 2
    assert listwise_mean[0] == pytest.approx(2)
    assert listwise_mean[1] == pytest.approx(5)


def test_mean_aggregator_list_of_matrices():
    matrixwise_mean = mean_aggregator(
        [
            [[3, 4], [7, 12]],
            [[1, 5], [9, 11]],
            [[2, 6], [8, 10]],
        ]
    )

    assert len(matrixwise_mean) == 2
    assert len(matrixwise_mean[0]) == 2
    assert matrixwise_mean[0][0] == pytest.approx(2)
    assert matrixwise_mean[0][1] == pytest.approx(5)
    assert matrixwise_mean[1][0] == pytest.approx(8)
    assert matrixwise_mean[1][1] == pytest.approx(11)


def test_mean_aggregator_unsupported_data():
    with pytest.raises(Exception):
        mean_aggregator([[], []])

    with pytest.raises(Exception):
        mean_aggregator(
            [
                [
                    [[1, 2], [3, 4]],
                    [[1, 2], [3, 4]],
                    [[1, 2], [3, 4]],
                ]
            ]
        )


def test_bag_union_aggregator():
    assert bag_union_aggregator(
        [
            Counter(["00", "01", "00"]),
            Counter(["00", "01", "00"]),
            Counter(["00", "01", "00", "00", "11"]),
        ]
    ) == Counter(
        {
            "00": 7,
            "01": 3,
            "11": 1,
        }
    )


def test_aggregate_default_aggregator():
    duration = 500
    results = [Results(atom_order=[], total_duration=duration) for _ in range(3)]

    my_observable = MagicMock()
    my_observable.uuid = "my_observable"
    my_observable.tag = "expectation"

    def store(*, time: int, values: list[Any]):
        for i, value in enumerate(values):
            results[i]._store(observable=my_observable, time=time, value=value)

    store(time=duration * 4 / 5, values=[0, 6, 15])

    aggregated = aggregate(results)

    assert aggregated.get_result_tags() == ["expectation"]
    assert aggregated.get_result_times("expectation") == [duration * 4 / 5]
    assert aggregated.get_result("expectation", duration * 4 / 5) == pytest.approx(7)

    results[1]._store(observable=my_observable, time=duration, value="whatever")

    with pytest.raises(ValueError) as e:
        aggregate(results)

    assert str(e.value) == (
        "Monte-Carlo results seem to provide from incompatible simulations: "
        "the callbacks are not stored at the same times"
    )


def test_aggregate_custom_aggregator(caplog):
    duration = 400
    results = [Results(atom_order=[], total_duration=duration) for _ in range(3)]

    my_observable = MagicMock()
    my_observable.uuid = "my_observable"
    my_observable.tag = "my_observable"

    def store(*, time: int, values: list[Any]):
        for i, value in enumerate(values):
            results[i]._store(observable=my_observable, time=time, value=value)

    store(time=duration, values=["A", "B", "C"])

    aggregate(results)
    assert caplog.record_tuples == [
        ("global_logger", logging.WARNING, "Skipping aggregation of `my_observable`")
    ]

    def string_concat(strings: list[str]):
        return "".join(strings)

    aggregated = aggregate(results, my_observable=string_concat)

    assert aggregated.get_result_tags() == ["my_observable"]
    assert aggregated.get_result_times("my_observable") == [duration]
    assert aggregated.get_result("my_observable", duration) == "ABC"


def test_aggregate_unequal_evaluation_times(caplog):
    duration = 400
    results = [Results(atom_order=[], total_duration=duration) for _ in range(3)]

    my_observable = MagicMock()
    my_observable.uuid = "my_observable"
    my_observable.tag = "my_observable"

    def store(*, time: int, values: list[Any]):
        for i, value in enumerate(values):
            results[i]._store(observable=my_observable, time=time, value=value)

    results[1]._store(observable=my_observable, time=duration * 3 / 4, value="whatever")
    store(time=duration, values=["A", "B", "C"])

    aggregate(results)
    assert caplog.record_tuples == [
        ("global_logger", logging.WARNING, "Skipping aggregation of `my_observable`")
    ]

    def string_concat(strings: list[str]):
        return "".join(strings)

    with pytest.raises(ValueError) as e:
        aggregate(results, my_observable=string_concat)

    assert str(e.value) == (
        "Monte-Carlo results seem to provide from incompatible simulations: "
        "the callbacks are not stored at the same times"
    )


def test_aggregate_multiple_observables():
    duration = 100
    results = [Results(atom_order=[], total_duration=duration) for _ in range(2)]
    observable1, observable2 = MagicMock(), MagicMock()

    for i, observable in enumerate([observable1, observable2]):
        observable.uuid = f"my_observable_{i}"
        observable.tag = f"expectation_{i}"

    results[0]._store(observable=observable1, time=duration, value=[1, 10])
    results[1]._store(observable=observable1, time=duration, value=[2, 11])
    results[0]._store(observable=observable2, time=duration, value=500)

    with pytest.raises(ValueError) as e:
        aggregate(results)

    assert str(e.value) == (
        "Monte-Carlo results seem to provide from incompatible simulations: "
        "they do not all contain the same observables"
    )

    results[1]._store(observable=observable2, time=duration, value=501)

    aggregated = aggregate(results)

    assert aggregated.get_result("expectation_0", duration)[0] == pytest.approx(1.5)
    assert aggregated.get_result("expectation_0", duration)[1] == pytest.approx(10.5)
    assert aggregated.get_result("expectation_1", duration) == pytest.approx(500.5)
