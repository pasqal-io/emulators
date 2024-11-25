from emu_base.base_classes.aggregators import mean_aggregator, bag_union_aggregator
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
