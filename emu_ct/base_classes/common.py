import random
from collections import Counter


def readout_with_error(c: str, *, p_false_pos: float, p_false_neg: float) -> str:
    # p_false_pos = false positive, p_false_neg = false negative
    r = random.random()
    if c == "0" and r < p_false_pos:
        return "1"

    if c == "1" and r < p_false_neg:
        return "0"

    return c


def apply_measurement_errors(
    bitstrings: Counter[str], *, p_false_pos: float, p_false_neg: float
) -> Counter[str]:
    """
    Given a bag of sampled bitstrings, returns another bag of bitstrings
    sampled with readout/measurement errors.

        p_false_pos: probability of false positive
        p_false_neg: probability of false negative
    """

    result: Counter[str] = Counter()
    for bitstring, count in bitstrings.items():
        for _ in range(count):
            bitstring_with_error = "".join(
                readout_with_error(c, p_false_pos=p_false_pos, p_false_neg=p_false_neg)
                for c in bitstring
            )

            result[bitstring_with_error] += 1

    return result
