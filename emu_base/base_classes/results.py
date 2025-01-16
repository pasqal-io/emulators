from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from pathlib import Path
import json
import logging

from emu_base.base_classes.callback import Callback, AggregationType
from emu_base.base_classes.aggregators import aggregation_types_definitions


@dataclass
class Results:
    """
    This class contains emulation results. Since the results written by
    an emulator are defined through callbacks, the contents of this class
    are not known a-priori.
    """

    statistics: Any = None  # Backend-specific data

    _results: dict[str, dict[int, Any]] = field(default_factory=dict)
    _default_aggregation_types: dict[str, Optional[AggregationType]] = field(
        default_factory=dict
    )

    @classmethod
    def aggregate(
        cls,
        results_to_aggregate: list["Results"],
        **aggregator_functions: Callable[[Any], Any],
    ) -> "Results":
        if len(results_to_aggregate) == 0:
            raise ValueError("no results to aggregate")

        if len(results_to_aggregate) == 1:
            return results_to_aggregate[0]

        stored_callbacks = set(results_to_aggregate[0].get_result_names())

        if not all(
            set(results.get_result_names()) == stored_callbacks
            for results in results_to_aggregate
        ):
            raise ValueError(
                "Monte-Carlo results seem to provide from incompatible simulations: "
                "they do not all contain the same observables"
            )

        aggregated: Results = cls()

        for stored_callback in stored_callbacks:
            aggregation_type = aggregator_functions.get(
                stored_callback,
                results_to_aggregate[0].get_aggregation_type(stored_callback),
            )

            if aggregation_type is None:
                logging.getLogger("global_logger").warning(
                    f"Skipping aggregation of `{stored_callback}`"
                )
                continue

            aggregation_function: Any = (
                aggregation_type
                if callable(aggregation_type)
                else aggregation_types_definitions[aggregation_type]
            )

            evaluation_times = results_to_aggregate[0].get_result_times(stored_callback)
            if not all(
                results.get_result_times(stored_callback) == evaluation_times
                for results in results_to_aggregate
            ):
                raise ValueError(
                    "Monte-Carlo results seem to provide from incompatible simulations: "
                    "the callbacks are not stored at the same times"
                )

            aggregated._results[stored_callback] = {
                t: aggregation_function(
                    [result[stored_callback, t] for result in results_to_aggregate]
                )
                for t in evaluation_times
            }

        return aggregated

    def store(self, *, callback: Callback, time: Any, value: Any) -> None:
        self._results.setdefault(callback.name, {})

        if time in self._results[callback.name]:
            raise ValueError(
                f"A value is already stored for observable '{callback.name}' at time {time}"
            )

        self._results[callback.name][time] = value
        self._default_aggregation_types[callback.name] = callback.default_aggregation_type

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, tuple):
            # results["energy", t]
            callback_name, time = key

            if callback_name not in self._results:
                raise ValueError(
                    f"No value for observable '{callback_name}' has been stored"
                )

            if time not in self._results[callback_name]:
                raise ValueError(
                    f"No value stored at time {time} for observable '{callback_name}'"
                )

            return self._results[callback_name][time]

        # results["energy"][t]
        assert isinstance(key, str)
        callback_name = key
        if callback_name not in self._results:
            raise ValueError(f"No value for observable '{callback_name}' has been stored")

        return self._results[key]

    def get_result_names(self) -> list[str]:
        """
        get a list of results present in this object

        Args:

        Returns:
            list of results by name

        """
        return list(self._results.keys())

    def get_result_times(self, name: str) -> list[int]:
        """
        get a list of times for which the given result has been stored

        Args:
            name: name of the result to get times of

        Returns:
            list of times in ns

        """
        return list(self._results[name].keys())

    def get_result(self, name: str, time: int) -> Any:
        """
        get the given result at the given time

        Args:
            name: name of the result to get
            time: time in ns at which to get the result

        Returns:
            the result

        """
        return self._results[name][time]

    def get_aggregation_type(self, name: str) -> Optional[AggregationType]:
        return self._default_aggregation_types[name]

    def dump(self, file_path: Path) -> None:
        with file_path.open("w") as file_handle:
            json.dump(
                {
                    "observables": self._results,
                    "statistics": self.statistics,
                },
                file_handle,
            )
