"""
main() must build one PlotRunReport per invocation and push it on every
exit path — including exits the except handlers cannot see — and the
failure path must carry the category the consumers read.

pytest processor/test_errors/test_main_report.py
"""

import pytest

from processor import main as main_mod
from processor.errors import PlotErrorCode, PlotErrorStage, PlotInvalidInputError


@pytest.fixture
def pushed(monkeypatch):
    """Capture the report main() hands to push_report_to_workflow_service."""
    box = []
    monkeypatch.setattr(main_mod, "push_report_to_workflow_service", lambda r: box.append(r) or True)
    monkeypatch.setenv("EXECUTION_RUN_ID", "run-1")
    return box


# ---- Scenario 1: run() returns normally --------------------------------------

def test_success_report(monkeypatch, pushed):
    def fake_run(report):
        report.figure_generated_by, report.figure_bytes = "template", 1234

    monkeypatch.setattr(main_mod, "run", fake_run)
    assert main_mod.main() == 0
    (r,) = pushed
    assert r.run_status == "succeeded" and r.run_failure_error is None
    assert r.figure_generated_by == "template" and r.figure_bytes == 1234
    assert r.run_id == "run-1"
    assert r.run_duration_seconds >= 0


# ---- Scenario 2: run() raises a PlotError it recognised ----------------------
# The error object itself lands on the report; a stage set at the raise site
# is kept, an empty one is stamped "run" by this catch site.

def test_plot_error_report(monkeypatch, pushed):
    err = PlotInvalidInputError(PlotErrorCode.UNKNOWN_CHANNEL, "Channel 'EEG-7' is not in this file.",
                                error_facts={"available_channels": ["F7", "F8"]},
                                source_stage=PlotErrorStage.READER)

    def fake_run(report):
        raise err

    monkeypatch.setattr(main_mod, "run", fake_run)
    assert main_mod.main() == 1
    (r,) = pushed
    assert r.run_status == "failed"
    assert r.run_failure_error is err                       # the PlotError itself, unchanged
    assert err.source_stage is PlotErrorStage.READER        # a stage set at the raise site is kept


def test_plot_error_without_stage_is_stamped_run(monkeypatch, pushed):
    err = PlotInvalidInputError(PlotErrorCode.UNKNOWN_CHANNEL, "m")
    monkeypatch.setattr(main_mod, "run", lambda report: (_ for _ in ()).throw(err))
    main_mod.main()
    assert pushed[0].run_failure_error.source_stage is PlotErrorStage.RUN


# ---- Scenario 3: run() raises something that is not a PlotError --------------

def test_unexpected_exception_is_wrapped_internal(monkeypatch, pushed):
    def fake_run(report):
        raise KeyError("secret internal detail")

    monkeypatch.setattr(main_mod, "run", fake_run)
    assert main_mod.main() == 1
    (r,) = pushed
    err = r.run_failure_error
    assert err.error_category.value == "internal"
    assert err.source_stage is PlotErrorStage.RUN           # the catch site stamps itself
    assert "secret" not in err.user_message                 # raw text only on __cause__ / in the log
    assert isinstance(err.__cause__, KeyError)


# ---- Scenario 4: the report is pushed no matter how run() ends ---------------
# Including the exits the except blocks cannot see: SystemExit from a library,
# or one of the handlers itself blowing up.

def test_push_is_called_even_when_run_raises_before_anything(monkeypatch, pushed):
    monkeypatch.setattr(main_mod, "run", lambda report: (_ for _ in ()).throw(RuntimeError("x")))
    assert main_mod.main() == 1
    assert len(pushed) == 1


def test_system_exit_from_run_is_still_pushed_and_still_propagates(monkeypatch, pushed):
    # A library calling sys.exit() inside run(): not an Exception, so neither
    # handler catches it. The finally must still push a report, and the
    # SystemExit must keep propagating so the Lambda fails as before.
    def fake_run(report):
        raise SystemExit(3)

    monkeypatch.setattr(main_mod, "run", fake_run)
    with pytest.raises(SystemExit) as ei:
        main_mod.main()
    assert ei.value.code == 3
    (r,) = pushed
    assert r.run_status == "failed"
    err = r.run_failure_error
    assert err.error_category.value == "internal" and err.error_code is PlotErrorCode.ABORTED
    assert err.source_stage is PlotErrorStage.RUN


def test_exception_inside_a_handler_is_still_pushed(monkeypatch, pushed):
    # wrap_unexpected_exception() itself blowing up inside the `except Exception` handler.
    monkeypatch.setattr(main_mod, "run", lambda report: (_ for _ in ()).throw(ValueError("x")))
    monkeypatch.setattr(main_mod, "wrap_unexpected_exception",
                        lambda exc: (_ for _ in ()).throw(TypeError("adapter broke")))
    with pytest.raises(TypeError):
        main_mod.main()
    (r,) = pushed
    assert r.run_failure_error.error_code is PlotErrorCode.ABORTED


# ---- Scenario 5: the log line ------------------------------------------------

def test_log_line_has_the_one_shape(monkeypatch, pushed, caplog):
    err = PlotInvalidInputError(PlotErrorCode.UNKNOWN_CHANNEL, "Channel 'Z' is not in the file.",
                                source_stage=PlotErrorStage.READER)
    monkeypatch.setattr(main_mod, "run", lambda report: (_ for _ in ()).throw(err))
    with caplog.at_level("ERROR", logger="quick-plot"):
        main_mod.main()
    assert "Quick-plot failed [invalid_input/unknown_channel at reader]: Channel 'Z' is not in the file." in caplog.text
