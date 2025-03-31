import statistics
import torch
from typing import Any, Callable
import collections
from enum import Enum, auto
from pulser.backend import (
    Results,
)
import logging


_NUMERIC_TYPES = {int, float, complex}


class AggregationType(Enum):
    """
    Defines how to combine multiple values from different simulation results.
    """

    MEAN = auto()  # statistics.fmean or list/matrix-wise equivalent
    BAG_UNION = auto()  # Counter.__add__


def mean_aggregator(
    values: list[Any],
) -> (
    complex
    | float
    | list[complex]
    | list[float]
    | list[list[complex]]
    | list[list[float]]
    | torch.Tensor
):  # FIXME: support tuples?
    if values == []:
        raise ValueError("Cannot average 0 samples")

    element_type = type(values[0])

    if element_type in _NUMERIC_TYPES:
        return statistics.fmean(values)

    if element_type == torch.Tensor:
        acc = torch.zeros_like(values[0])
        for ten in values:
            acc += ten
        return acc / len(values)

    if element_type != list:
        raise NotImplementedError("Cannot average this type of data")

    if values[0] == []:
        raise ValueError("Cannot average list of empty lists")

    sub_element_type = type(values[0][0])

    if sub_element_type in _NUMERIC_TYPES:
        dim = len(values[0])
        return [statistics.fmean(value[i] for value in values) for i in range(dim)]

    if sub_element_type != list:  # FIXME: ABC.Iterable? Collection? subclass?
        raise ValueError(f"Cannot average list of lists of {sub_element_type}")

    if values[0][0] == []:
        raise ValueError("Cannot average list of matrices with no columns")

    if (sub_sub_element_type := type(values[0][0][0])) not in _NUMERIC_TYPES:
        raise ValueError(f"Cannot average list of matrices of {sub_sub_element_type}")

    dim1 = len(values[0])
    dim2 = len(values[0][0])
    return [
        [statistics.fmean(value[i][j] for value in values) for j in range(dim2)]
        for i in range(dim1)
    ]


def bag_union_aggregator(values: list[collections.Counter]) -> collections.Counter:
    return sum(values, start=collections.Counter())


aggregation_types_definitions: dict[AggregationType, Callable] = {
    AggregationType.MEAN: mean_aggregator,
    AggregationType.BAG_UNION: bag_union_aggregator,
}


def _get_aggregation_type(tag: str) -> AggregationType | None:
    if tag.startswith("bitstrings"):
        return AggregationType.BAG_UNION
    if tag.startswith("expectation"):
        return AggregationType.MEAN
    if tag.startswith("fidelity"):
        return AggregationType.MEAN
    if tag.startswith("correlation_matrix"):
        return AggregationType.MEAN
    if tag.startswith("occupation"):
        return AggregationType.MEAN
    if tag.startswith("energy"):
        return AggregationType.MEAN
    if tag.startswith("energy_second_moment"):
        return AggregationType.MEAN
    else:
        return None


def aggregate(
    results_to_aggregate: list[Results],
    **aggregator_functions: Callable[[Any], Any],
) -> Results:
    if len(results_to_aggregate) == 0:
        raise ValueError("no results to aggregate")
    if len(results_to_aggregate) == 1:
        return results_to_aggregate[0]
    stored_callbacks = set(results_to_aggregate[0].get_result_tags())
    if not all(
        set(results.get_result_tags()) == stored_callbacks
        for results in results_to_aggregate
    ):
        raise ValueError(
            "Monte-Carlo results seem to provide from incompatible simulations: "
            "they do not all contain the same observables"
        )
    aggregated = Results(
        atom_order=results_to_aggregate[0].atom_order,
        total_duration=results_to_aggregate[0].total_duration,
    )
    for tag in stored_callbacks:
        aggregation_type = aggregator_functions.get(
            tag,
            _get_aggregation_type(tag),
        )
        if aggregation_type is None:
            logging.getLogger("global_logger").warning(f"Skipping aggregation of `{tag}`")
            continue
        aggregation_function: Any = (
            aggregation_type
            if callable(aggregation_type)
            else aggregation_types_definitions[aggregation_type]
        )
        evaluation_times = results_to_aggregate[0].get_result_times(tag)
        if not all(
            results.get_result_times(tag) == evaluation_times
            for results in results_to_aggregate
        ):
            raise ValueError(
                "Monte-Carlo results seem to provide from incompatible simulations: "
                "the callbacks are not stored at the same times"
            )

        uuid = results_to_aggregate[0]._find_uuid(tag)
        for t in results_to_aggregate[0].get_result_times(tag):
            v = aggregation_function(
                [result.get_result(tag, t) for result in results_to_aggregate]
            )
            aggregated._store_raw(uuid=uuid, tag=tag, time=t, value=v)

    return aggregated
