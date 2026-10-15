"""
processor.errors — the error classes every part of the processor raises, and
one adapter for exceptions nobody in this package raised.

The rule
--------
Raise the PlotError subclass that says what the problem *means for the
person waiting*, at the place that finds the problem and still has the
context (the channel they asked for, the channels the file has). Never
raise a plain RuntimeError for a condition you understand. Translate a
built-in exception only where you know its meaning, with `raise ... from exc`.

    class hierarchy                            error_category
    Exception
    └── RuntimeError
        └── PlotError                          (base; not raised directly)
            ├── PlotInvalidInputError          invalid_input      fix the request
            ├── PlotDataUnavailableError       data_unavailable   choose other data / window
            ├── PlotResourceLimitError         resource_limit     make the request smaller
            ├── PlotEnvironmentError           environment        repair the compute node
            └── PlotInternalError              internal           our bug; report run_id

PlotError is a RuntimeError, so every existing `except RuntimeError` and
`pytest.raises(RuntimeError)` keeps working.

What an error carries (the "pockets")
-------------------------------------
    error_category   set by the subclass; the broad kind, for consumers that
                     see JSON rather than Python classes
    error_code       PlotErrorCode; the exact condition, stable across wording
    user_message     the sentence a person reads; complete on its own
    user_next_step   what the person should do; defaults per category
    error_facts      structured, safe-to-save facts for programs / the chat
                     agent (e.g. available_channels); never a traceback or
                     an internal path
    source_stage     PlotErrorStage; the most specific known place. A raise
                     site may set reader/tool; a catch site fills template/run
                     only when it is still empty.

Two catch sites in main.py call wrap_unexpected_exception(): the template
boundary (which may still fall back to the agent) and main() (which can
only report). Which categories stop the run and which fall back is decided
in main.py, not here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

__all__ = [
    "PlotErrorCategory", "PlotErrorCode", "PlotErrorStage",
    "PlotError", "PlotInvalidInputError", "PlotDataUnavailableError",
    "PlotResourceLimitError", "PlotEnvironmentError", "PlotInternalError",
    "wrap_unexpected_exception", "GENERIC_INTERNAL_MESSAGE",
]


class PlotErrorCategory(str, Enum):
    INVALID_INPUT = "invalid_input"
    DATA_UNAVAILABLE = "data_unavailable"
    RESOURCE_LIMIT = "resource_limit"
    ENVIRONMENT = "environment"
    INTERNAL = "internal"


class PlotErrorStage(str, Enum):
    CONFIG = "config"        # reading the request / env vars
    READER = "reader"        # opening or decoding the input file
    TOOL = "tool"            # a DSP tool in the pipeline
    TEMPLATE = "template"    # the canned template itself
    AGENT = "agent"          # the LLM fallback
    STUB = "stub"            # the placeholder figure
    RUN = "run"              # anywhere else in main()


class PlotErrorCode(str, Enum):
    """Every condition the processor can report. Python uses the member
    (PlotErrorCode.UNKNOWN_CHANNEL); workflow-service stores the value
    ("unknown_channel"). Add a member when the corrective action, the
    program's behaviour, or the facts worth saving differ from every
    existing one. The rejected parameter itself belongs in error_facts."""

    # invalid_input — fix the request
    UNKNOWN_TEMPLATE = "unknown_template"
    TEMPLATE_ARGS_INVALID_JSON = "template_args_invalid_json"
    TEMPLATE_ARGS_NOT_OBJECT = "template_args_not_object"
    CONFIG_NO_REQUEST = "config_no_request"
    TEMPLATE_FAILED_NO_PROMPT = "template_failed_no_prompt"
    MISSING_ARGUMENT = "missing_argument"
    INVALID_ARGUMENT = "invalid_argument"
    UNKNOWN_ARGUMENT = "unknown_argument"
    UNKNOWN_CHANNEL = "unknown_channel"
    UNKNOWN_COLUMN = "unknown_column"
    INVALID_TIME_WINDOW = "invalid_time_window"
    UNSUPPORTED_UNIT = "unsupported_unit"
    INCOMPATIBLE_CHANNELS = "incompatible_channels"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_TOOL_ARGUMENT = "invalid_tool_argument"
    INCOMPATIBLE_PIPELINE = "incompatible_pipeline"
    REQUEST_REJECTED = "request_rejected"      # legacy: a plain RuntimeError not yet converted

    # data_unavailable — choose other data or another window
    FILE_NOT_FOUND = "file_not_found"
    NO_INPUT_FILES = "no_input_files"
    UNSUPPORTED_FILE_FORMAT = "unsupported_file_format"
    UNREADABLE_FILE = "unreadable_file"
    EMPTY_DATA = "empty_data"
    INSUFFICIENT_SAMPLES = "insufficient_samples"
    MISSING_REQUIRED_METADATA = "missing_required_metadata"

    # resource_limit — make the request smaller
    WINDOW_TOO_LARGE = "window_too_large"
    TOO_MANY_CHANNELS = "too_many_channels"
    DATA_TOO_LARGE = "data_too_large"
    OUT_OF_MEMORY = "out_of_memory"

    # environment — repair the compute node
    CONFIG_MISSING_DIRS = "config_missing_dirs"
    MISSING_DEPENDENCY = "missing_dependency"
    STORAGE_UNAVAILABLE = "storage_unavailable"

    # internal — our bug
    TEMPLATE_NO_OUTPUT = "template_no_output"
    AGENT_FAILED = "agent_failed"
    STUB_FAILED = "stub_failed"
    UNEXPECTED = "unexpected"
    ABORTED = "aborted"


CODES_BY_CATEGORY: dict[PlotErrorCategory, frozenset[PlotErrorCode]] = {
    PlotErrorCategory.INVALID_INPUT: frozenset({
        PlotErrorCode.UNKNOWN_TEMPLATE, PlotErrorCode.TEMPLATE_ARGS_INVALID_JSON,
        PlotErrorCode.TEMPLATE_ARGS_NOT_OBJECT, PlotErrorCode.CONFIG_NO_REQUEST,
        PlotErrorCode.TEMPLATE_FAILED_NO_PROMPT, PlotErrorCode.MISSING_ARGUMENT,
        PlotErrorCode.INVALID_ARGUMENT, PlotErrorCode.UNKNOWN_ARGUMENT,
        PlotErrorCode.UNKNOWN_CHANNEL, PlotErrorCode.UNKNOWN_COLUMN,
        PlotErrorCode.INVALID_TIME_WINDOW, PlotErrorCode.UNSUPPORTED_UNIT,
        PlotErrorCode.INCOMPATIBLE_CHANNELS, PlotErrorCode.UNKNOWN_TOOL,
        PlotErrorCode.INVALID_TOOL_ARGUMENT, PlotErrorCode.INCOMPATIBLE_PIPELINE,
        PlotErrorCode.REQUEST_REJECTED,
    }),
    PlotErrorCategory.DATA_UNAVAILABLE: frozenset({
        PlotErrorCode.FILE_NOT_FOUND, PlotErrorCode.NO_INPUT_FILES,
        PlotErrorCode.UNSUPPORTED_FILE_FORMAT, PlotErrorCode.UNREADABLE_FILE,
        PlotErrorCode.EMPTY_DATA, PlotErrorCode.INSUFFICIENT_SAMPLES,
        PlotErrorCode.MISSING_REQUIRED_METADATA,
    }),
    PlotErrorCategory.RESOURCE_LIMIT: frozenset({
        PlotErrorCode.WINDOW_TOO_LARGE, PlotErrorCode.TOO_MANY_CHANNELS,
        PlotErrorCode.DATA_TOO_LARGE, PlotErrorCode.OUT_OF_MEMORY,
    }),
    PlotErrorCategory.ENVIRONMENT: frozenset({
        PlotErrorCode.CONFIG_MISSING_DIRS, PlotErrorCode.MISSING_DEPENDENCY,
        PlotErrorCode.STORAGE_UNAVAILABLE,
    }),
    PlotErrorCategory.INTERNAL: frozenset({
        PlotErrorCode.TEMPLATE_NO_OUTPUT, PlotErrorCode.AGENT_FAILED,
        PlotErrorCode.STUB_FAILED, PlotErrorCode.UNEXPECTED, PlotErrorCode.ABORTED,
    }),
}

DEFAULT_NEXT_STEP: dict[PlotErrorCategory, str] = {
    PlotErrorCategory.INVALID_INPUT: "Check the plot settings (channel, time window, units) and try again.",
    PlotErrorCategory.DATA_UNAVAILABLE: "Choose a different file or a time window inside the recording.",
    PlotErrorCategory.RESOURCE_LIMIT: "Ask for a shorter window or fewer channels.",
    PlotErrorCategory.ENVIRONMENT: "The compute node needs an update; ask your node admin.",
    PlotErrorCategory.INTERNAL: "Unexpected error. Report the run ID if it keeps happening.",
}

GENERIC_INTERNAL_MESSAGE = "The plot failed for an unexpected reason."


class PlotError(RuntimeError):
    """Base class. Raise one of the five subclasses, never this directly."""

    error_category: PlotErrorCategory  # set on each subclass

    def __init__(
        self,
        error_code: PlotErrorCode | str,
        user_message: str,
        *,
        user_next_step: str = "",
        error_facts: dict[str, Any] | None = None,
        source_stage: PlotErrorStage | str = "",
    ) -> None:
        if type(self) is PlotError:
            raise TypeError("raise a PlotError subclass (PlotInvalidInputError, ...), not PlotError itself")
        code = PlotErrorCode(error_code)                 # ValueError on an unknown string
        if code not in CODES_BY_CATEGORY[self.error_category]:
            raise ValueError(f"{code.value!r} is not a {self.error_category.value} code")
        self.error_code = code
        self.user_message = user_message
        self.user_next_step = user_next_step or DEFAULT_NEXT_STEP[self.error_category]
        self.error_facts: dict[str, Any] = dict(error_facts or {})
        self.source_stage: PlotErrorStage | None = PlotErrorStage(source_stage) if source_stage else None
        super().__init__(user_message)   # str(exc) == user_message, for logs and tests

    def __repr__(self) -> str:
        stage = self.source_stage.value if self.source_stage else None
        return f"{type(self).__name__}({self.error_code.value!r}, {self.user_message!r}, stage={stage!r})"


class PlotInvalidInputError(PlotError):
    error_category = PlotErrorCategory.INVALID_INPUT


class PlotDataUnavailableError(PlotError):
    error_category = PlotErrorCategory.DATA_UNAVAILABLE


class PlotResourceLimitError(PlotError):
    error_category = PlotErrorCategory.RESOURCE_LIMIT


class PlotEnvironmentError(PlotError):
    error_category = PlotErrorCategory.ENVIRONMENT


class PlotInternalError(PlotError):
    error_category = PlotErrorCategory.INTERNAL


def wrap_unexpected_exception(exc: BaseException) -> PlotError:
    """The boundary adapter: give an exception nobody in this package raised a
    category so it can still be reported. Returns a PlotError unchanged.

      PlotError           -> unchanged
      MemoryError         -> PlotResourceLimitError(out_of_memory)
      ImportError         -> PlotEnvironmentError(missing_dependency)
      FileNotFoundError   -> PlotDataUnavailableError(file_not_found)
      RuntimeError        -> PlotInvalidInputError(request_rejected)   legacy: a raise
                             not yet converted; keeps its own message
      anything else       -> PlotInternalError(unexpected)  generic message; the
                             class name goes in error_facts, the text stays in the log

    The caller fills source_stage afterwards if it is still empty.
    """
    if isinstance(exc, PlotError):
        return exc
    text = str(exc).strip() or exc.__class__.__name__
    if isinstance(exc, MemoryError):
        err: PlotError = PlotResourceLimitError(
            PlotErrorCode.OUT_OF_MEMORY,
            "The request needed more memory than the plot runner has.")
    elif isinstance(exc, ImportError):
        err = PlotEnvironmentError(
            PlotErrorCode.MISSING_DEPENDENCY,
            f"The compute node is missing a dependency the plot needs ({exc.name or text}).",
            error_facts={"module": exc.name or ""})
    elif isinstance(exc, FileNotFoundError):
        err = PlotDataUnavailableError(PlotErrorCode.FILE_NOT_FOUND, text)
    elif isinstance(exc, RuntimeError):
        err = PlotInvalidInputError(PlotErrorCode.REQUEST_REJECTED, text)
    else:
        err = PlotInternalError(PlotErrorCode.UNEXPECTED, GENERIC_INTERNAL_MESSAGE,
                                error_facts={"exception": exc.__class__.__name__})
    err.__cause__ = exc
    return err
