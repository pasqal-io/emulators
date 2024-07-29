from emu_ct.base_classes.common import apply_measurement_errors, readout_with_error
from collections import Counter
import random
from unittest.mock import patch, call


@patch("emu_ct.noise.random.random")
def test_readout_with_error(random_mock):
    random_mock.side_effect = [0.6, 0.08, 0.4, 0.1, 0.04]

    assert readout_with_error("0", p_false_pos=0.1, p_false_neg=0.2) == "0"
    assert readout_with_error("0", p_false_pos=0.1, p_false_neg=0.2) == "1"
    assert readout_with_error("1", p_false_pos=0.1, p_false_neg=0.2) == "1"
    assert readout_with_error("1", p_false_pos=0.1, p_false_neg=0.2) == "0"
    assert readout_with_error("0", p_false_pos=0.1, p_false_neg=0.2) == "1"


def test_add_measurement_errors():
    bitstrings = Counter(
        {
            "1010": 3,
            "0101": 2,
            "1111": 1,
        }
    )

    # Error probability is null
    bitstrings_without_measurement_errors = apply_measurement_errors(
        bitstrings, p_false_pos=0.0, p_false_neg=0.0
    )
    assert bitstrings_without_measurement_errors == bitstrings

    # Error probability is not null
    p_false_pos = 0.1
    p_false_neg = 0.2
    with patch(
        "emu_ct.base_classes.common.readout_with_error"
    ) as readout_with_error_mock:
        readout_with_error_mock.side_effect = [
            "1",
            "0",
            "1",
            "1",
            "0",
            "0",
            "0",
            "1",
            "1",
            "0",
            "1",
            "0",
            "0",
            "1",
            "0",
            "1",
            "0",
            "0",
            "0",
            "1",
            "1",
            "1",
            "1",
            "1",
        ]

        bitstrings_with_measurement_errors = apply_measurement_errors(
            bitstrings, p_false_pos=p_false_pos, p_false_neg=p_false_neg
        )

        ps = {"p_false_pos": p_false_pos, "p_false_neg": p_false_neg}

        # Counter is a subclass of dict.
        # In Python 3.7+, dict iteration order is guaranteed to be the insertion order.
        # Therefore, the loop in add_measurement_errors has a known order.

        readout_with_error_mock.assert_has_calls(
            [call("1", **ps), call("0", **ps), call("1", **ps), call("0", **ps)] * 3
            + [call("0", **ps), call("1", **ps), call("0", **ps), call("1", **ps)] * 2
            + [call("1", **ps), call("1", **ps), call("1", **ps), call("1", **ps)] * 1
        )

        assert bitstrings_with_measurement_errors == Counter(
            {"1011": 1, "0001": 2, "1010": 1, "0101": 1, "1111": 1}
        )


def test_add_measurement_errors_large():
    random.seed(0xDEADBEEF)

    bitstrings = Counter(
        {
            "101010001": 39845,
            "010110001": 2,
            "111100001": 1,
        }
    )

    bitstrings_with_measurement_errors = apply_measurement_errors(
        bitstrings, p_false_pos=0.000001, p_false_neg=0.0000085
    )
    assert bitstrings_with_measurement_errors == Counter(
        {
            "101010001": 39843,
            "101010000": 1,
            "111010001": 1,
            "010110001": 2,
            "111100001": 1,
        }
    )

    assert sum(bitstrings_with_measurement_errors.values()) == sum(bitstrings.values())
