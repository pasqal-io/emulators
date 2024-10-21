from dataclasses import dataclass, field
from typing import Any


@dataclass
class Results:
    """
    This class contains emulation results. Since the results written by
    an emulator are defined through callbacks, the contents of this class
    are not known a-priori.
    """

    _results: dict[str, dict[int, Any]] = field(default_factory=dict)

    def store(self, *, callback_name: str, time: Any, value: Any) -> None:
        self._results.setdefault(callback_name, {})

        if time in self._results[callback_name]:
            raise ValueError(
                f"A value is already stored for observable '{callback_name}' at time {time}"
            )

        self._results[callback_name][time] = value

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
