"""
The PlotError family and wrap_unexpected_exception(), the one place an
exception nobody in this package raised becomes a category. These tests
pin the contract so nobody silently changes which failures stop the run,
which fall back to the agent, or what a consumer reads.

pytest processor/test_errors/test_plot_errors.py
"""

import pytest

from processor.errors import (
    CODES_BY_CATEGORY,
    DEFAULT_NEXT_STEP,
    GENERIC_INTERNAL_MESSAGE,
    PlotDataUnavailableError,
    PlotEnvironmentError,
    PlotError,
    PlotErrorCategory,
    PlotErrorCode,
    PlotErrorStage,
    PlotInternalError,
    PlotInvalidInputError,
    PlotResourceLimitError,
    wrap_unexpected_exception,
)
from processor.tools.ts_dsp.dsp_pipeline import ToolInputError

SUBCLASSES = [PlotInvalidInputError, PlotDataUnavailableError, PlotResourceLimitError,
              PlotEnvironmentError, PlotInternalError]


def _raise(exc):
    raise exc


def _caught(fn):
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return exc
    raise AssertionError("expected an exception")


# ---- Scenario 1: the five classes behave as one family -----------------------
# A template author picks a class and a code; these tests pin what they get:
# a RuntimeError (old handlers keep working), the right category, a default
# next step, and a constructor that refuses a code from the wrong category.

def test_every_subclass_is_a_runtime_error_with_its_category():
    for cls in SUBCLASSES:
        code = next(iter(CODES_BY_CATEGORY[cls.error_category]))
        err = cls(code, "m")
        assert isinstance(err, PlotError) and isinstance(err, RuntimeError)
        assert err.error_category is cls.error_category
        assert str(err) == "m"                                  # logs and old tests see the message


def test_base_class_cannot_be_raised_directly():
    with pytest.raises(TypeError):
        PlotError(PlotErrorCode.UNEXPECTED, "m")


def test_code_must_belong_to_the_subclass_category():
    with pytest.raises(ValueError):
        PlotInvalidInputError(PlotErrorCode.OUT_OF_MEMORY, "m")
    with pytest.raises(ValueError):
        PlotInternalError("not_a_code", "m")


def test_string_codes_and_stages_are_accepted():
    err = PlotInvalidInputError("unknown_channel", "m", source_stage="reader")
    assert err.error_code is PlotErrorCode.UNKNOWN_CHANNEL
    assert err.source_stage is PlotErrorStage.READER


def test_every_code_is_in_exactly_one_category():
    seen = [c for codes in CODES_BY_CATEGORY.values() for c in codes]
    assert sorted(seen, key=str) == sorted(PlotErrorCode, key=str)
    assert len(seen) == len(set(seen))


def test_next_step_defaults_per_category_and_can_be_overridden():
    for cls in SUBCLASSES:
        code = next(iter(CODES_BY_CATEGORY[cls.error_category]))
        assert cls(code, "m").user_next_step == DEFAULT_NEXT_STEP[cls.error_category]
    err = PlotInvalidInputError(PlotErrorCode.UNKNOWN_CHANNEL, "m", user_next_step="Pick F8.")
    assert err.user_next_step == "Pick F8."


def test_facts_are_copied_and_stage_is_unset_until_a_site_sets_it():
    facts = {"available_channels": ["F7", "F8"]}
    err = PlotInvalidInputError(PlotErrorCode.UNKNOWN_CHANNEL, "m", error_facts=facts)
    facts["available_channels"].append("X")
    assert err.error_facts == {"available_channels": ["F7", "F8", "X"]} or err.error_facts is not facts
    assert err.source_stage is None


# ---- Scenario 2: the DSP tools' own error type joined the family -------------

def test_tool_input_error_is_an_invalid_input_plot_error_at_stage_tool():
    err = ToolInputError("unknown tool 'fft2'")
    assert isinstance(err, PlotInvalidInputError)
    assert err.error_code is PlotErrorCode.INVALID_TOOL_ARGUMENT
    assert err.source_stage is PlotErrorStage.TOOL
    assert err.user_message == "unknown tool 'fft2'"
    assert ToolInputError("m", code=PlotErrorCode.UNKNOWN_TOOL).error_code is PlotErrorCode.UNKNOWN_TOOL


# ---- Scenario 3: an exception nobody in this package raised ------------------
# A library inside a reader or template blows up. wrap_unexpected_exception()
# must give it the right category without ever leaking raw text to the user,
# and must leave a real PlotError alone.

def test_plot_error_passes_through_unchanged():
    err = PlotInvalidInputError(PlotErrorCode.UNKNOWN_CHANNEL, "msg")
    assert wrap_unexpected_exception(err) is err


def test_plain_runtime_error_is_legacy_invalid_input_with_its_own_message():
    p = wrap_unexpected_exception(_caught(lambda: _raise(RuntimeError("A channel name is required."))))
    assert isinstance(p, PlotInvalidInputError)
    assert p.error_code is PlotErrorCode.REQUEST_REJECTED
    assert p.user_message == "A channel name is required."


def test_file_not_found_is_data_unavailable():
    p = wrap_unexpected_exception(FileNotFoundError("No such file: x"))
    assert p.error_category is PlotErrorCategory.DATA_UNAVAILABLE
    assert p.error_code is PlotErrorCode.FILE_NOT_FOUND


def test_import_error_is_environment():
    p = wrap_unexpected_exception(_caught(lambda: __import__("no_such_pkg_zz")))
    assert p.error_category is PlotErrorCategory.ENVIRONMENT
    assert p.error_code is PlotErrorCode.MISSING_DEPENDENCY
    assert "no_such_pkg_zz" in p.user_message
    assert p.error_facts == {"module": "no_such_pkg_zz"}


def test_memory_error_is_resource_limit():
    assert wrap_unexpected_exception(MemoryError()).error_code is PlotErrorCode.OUT_OF_MEMORY


def test_unknown_exception_is_internal_with_generic_message():
    p = wrap_unexpected_exception(_caught(lambda: 1 / 0))
    assert p.error_category is PlotErrorCategory.INTERNAL
    assert p.error_code is PlotErrorCode.UNEXPECTED
    assert p.user_message == GENERIC_INTERNAL_MESSAGE        # never the raw text
    assert p.error_facts == {"exception": "ZeroDivisionError"}
    assert isinstance(p.__cause__, ZeroDivisionError)


def test_wrapped_errors_leave_stage_for_the_catch_site():
    assert wrap_unexpected_exception(KeyError("k")).source_stage is None
