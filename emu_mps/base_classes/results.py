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

    def __setitem__(self, name: str, value: Any) -> None:
        self._results[name] = value

    def __getitem__(self, name: str) -> dict[int, Any]:
        if self._results.get(name) is None:
            self._results[name] = {}
        return self._results[name]

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
