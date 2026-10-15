"""
push_report_to_workflow_service() is the transport for the run report.
These tests pin the key names (they are read by name in pennsieve-mcp and
compute-node-chat), the auth preference, the host fallback order, the
path-stripping of user-facing text, and that nothing here can ever raise
into main().

pytest processor/test_errors/test_push_report.py
"""

import io
import json
import urllib.error

import pytest

from processor import report
from processor.errors import PlotEnvironmentError, PlotErrorCode, PlotErrorStage, PlotInvalidInputError
from processor.report import PlotRunReport, public_message, push_report_to_workflow_service, to_outputs


class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _capture(monkeypatch, status=200, error=None):
    calls = []

    def fake_urlopen(req, timeout):
        calls.append((req, timeout))
        if error is not None:
            raise error
        r = _Resp(b"{}")
        r.status = status
        return r

    monkeypatch.setattr(report.urllib.request, "urlopen", fake_urlopen)
    return calls


ENV = {
    "EXECUTION_RUN_ID": "run-42",
    "SESSION_TOKEN": "sess-token",
    "PENNSIEVE_API_HOST": "https://api.pennsieve.net",
    "PENNSIEVE_API_HOST2": "https://api2.pennsieve.net",
}


def _invalid(msg="Channel 'X' is not in this file.", **kw):
    return PlotInvalidInputError(PlotErrorCode.UNKNOWN_CHANNEL, msg, **kw)


# ---- Scenario 1: what the run record will contain ----------------------------
# The keys are read by name downstream, so they are pinned exactly: field
# names mirrored under "quickplot.", errors expanded into their parts, values
# always strings, long values clipped, paths trimmed from user-facing text.

def test_to_outputs_keys_mirror_the_report_fields():
    r = PlotRunReport(run_status="failed", requested_template="edf_processed_timeseries",
                      run_duration_seconds=3.456,
                      run_failure_error=_invalid(error_facts={"available_channels": ["F7", "F8"]},
                                                 source_stage=PlotErrorStage.READER))
    out = to_outputs(r)
    assert out == {
        "quickplot.run_status": "failed",
        "quickplot.requested_template": "edf_processed_timeseries",
        "quickplot.figure_generated_by": "",
        "quickplot.figure_bytes": "0",
        "quickplot.run_duration_seconds": "3.46",
        "quickplot.run_failure_error.error_category": "invalid_input",
        "quickplot.run_failure_error.error_code": "unknown_channel",
        "quickplot.run_failure_error.source_stage": "reader",
        "quickplot.run_failure_error.user_message": "Channel 'X' is not in this file.",
        "quickplot.run_failure_error.user_next_step":
            "Check the plot settings (channel, time window, units) and try again.",
        "quickplot.run_failure_error.error_facts": '{"available_channels": ["F7", "F8"]}',
    }
    assert all(isinstance(v, str) for v in out.values())   # map[string]string on the wire


def test_fallback_trigger_error_is_reported_alongside_success():
    r = PlotRunReport(run_status="succeeded", figure_generated_by="agent", figure_bytes=99,
                      requested_template="edf_processed_timeseries",
                      fallback_trigger_error=PlotEnvironmentError(
                          PlotErrorCode.MISSING_DEPENDENCY, "missing pyedflib"))
    out = to_outputs(r)
    assert out["quickplot.run_status"] == "succeeded"
    assert out["quickplot.figure_generated_by"] == "agent"
    assert out["quickplot.fallback_trigger_error.error_category"] == "environment"
    assert out["quickplot.fallback_trigger_error.source_stage"] == ""      # unset stage is an empty string
    assert not any(k.startswith("quickplot.run_failure_error.") for k in out)


def test_long_values_are_clipped():
    r = PlotRunReport(run_failure_error=_invalid("x" * 5000))
    assert len(to_outputs(r)["quickplot.run_failure_error.user_message"]) == report.OUTPUTS_VALUE_MAX


@pytest.mark.parametrize("raw,expected", [
    ("Could not read '/mnt/efs/run-1/in/rec.edf' as EDF data: /mnt/efs/run-1/in/rec.edf: a read error occurred",
     "Could not read 'rec.edf' as EDF data: rec.edf: a read error occurred"),
    ("Channel 'F8' is not in this file.", "Channel 'F8' is not in this file."),   # untouched
    ("y-axis range -200/200 has non-numeric bounds", "y-axis range -200/200 has non-numeric bounds"),  # not a path
    ("start at 00:10:00 (300 s)", "start at 00:10:00 (300 s)"),
])
def test_public_message_strips_paths_only(raw, expected):
    assert public_message(raw) == expected


def test_outputs_carry_public_message():
    err = _invalid("Could not read '/mnt/efs/x/rec.edf' as EDF data")
    out = to_outputs(PlotRunReport(run_failure_error=err))
    assert out["quickplot.run_failure_error.user_message"] == "Could not read 'rec.edf' as EDF data"
    assert err.user_message.startswith("Could not read '/mnt")     # the PlotError itself is not mutated


# ---- Scenario 2: the PUT itself ----------------------------------------------
# Which URL, which auth header, which host wins, and that no network or
# configuration problem can ever raise back into main().

def test_put_uses_api2_host_session_token_and_run_id(monkeypatch):
    calls = _capture(monkeypatch)
    assert push_report_to_workflow_service(PlotRunReport(run_status="succeeded"), env=ENV) is True
    req, timeout = calls[0]
    assert req.full_url == "https://api2.pennsieve.net/compute/workflows/runs/run-42/outputs"
    assert req.get_method() == "PUT"
    assert req.get_header("Authorization") == "Bearer sess-token"
    assert timeout == report.OUTPUTS_TIMEOUT_S
    body = json.loads(req.data)
    assert body["quickplot.run_status"] == "succeeded"
    assert body["quickplot.reporter.auth"] == "session_token"
    assert body["quickplot.reporter.host"] == "https://api2.pennsieve.net"


def test_put_prefers_callback_token_when_present(monkeypatch):
    calls = _capture(monkeypatch)
    env = dict(ENV, CALLBACK_TOKEN="cb-token")
    assert push_report_to_workflow_service(PlotRunReport(), env=env)
    req, _ = calls[0]
    assert req.get_header("X-callback-token") == "cb-token"
    assert req.get_header("Authorization") is None
    assert json.loads(req.data)["quickplot.reporter.auth"] == "callback_token"


def test_host_fallback_order(monkeypatch):
    calls = _capture(monkeypatch)
    push_report_to_workflow_service(PlotRunReport(), env=dict(ENV, QUICKPLOT_API_HOST="https://override.example/"))
    assert calls[-1][0].full_url.startswith("https://override.example/compute/")
    env = dict(ENV)
    del env["PENNSIEVE_API_HOST2"]
    push_report_to_workflow_service(PlotRunReport(), env=env)
    assert calls[-1][0].full_url.startswith("https://api.pennsieve.net/compute/")


# ---- Scenario 3: not on the platform, or on it but misconfigured -------------

def test_off_platform_is_a_quiet_no_op(monkeypatch, caplog):
    calls = _capture(monkeypatch)
    env = dict(ENV)
    del env["EXECUTION_RUN_ID"]
    with caplog.at_level("INFO", logger="quick-plot"):
        assert push_report_to_workflow_service(PlotRunReport(), env=env) is False
    assert calls == []
    assert all(r.levelname == "INFO" for r in caplog.records)      # local runs: no warning noise


@pytest.mark.parametrize("missing", ["SESSION_TOKEN", "PENNSIEVE_API_HOST2"])
def test_on_platform_but_misconfigured_warns_and_skips(monkeypatch, caplog, missing):
    calls = _capture(monkeypatch)
    env = {k: v for k, v in ENV.items() if k not in (missing, "PENNSIEVE_API_HOST")}
    with caplog.at_level("WARNING", logger="quick-plot"):
        assert push_report_to_workflow_service(PlotRunReport(), env=env) is False
    assert calls == []
    assert any(r.levelname == "WARNING" for r in caplog.records)


@pytest.mark.parametrize("error", [
    urllib.error.HTTPError("u", 403, "forbidden", {}, io.BytesIO(b"{\"message\":\"no\"}")),
    urllib.error.URLError("name resolution failed"),
    TimeoutError("timed out"),
])
def test_network_failures_return_false_never_raise(monkeypatch, error):
    _capture(monkeypatch, error=error)
    assert push_report_to_workflow_service(PlotRunReport(), env=ENV) is False
