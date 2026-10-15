"""
The fallback gate in run(): which template failures reach the agent loop.

  invalid_input / data_unavailable / resource_limit -> raise (fail fast, agent never called)
  environment / internal                            -> agent runs; fallback_trigger_error recorded
  no PROMPT and template failed                     -> the template's own error is raised

pytest processor/test_errors/test_gate.py
"""

import types

import pytest

from processor import main as main_mod
from processor.errors import (
    PlotDataUnavailableError,
    PlotEnvironmentError,
    PlotError,
    PlotErrorCategory,
    PlotErrorCode,
    PlotErrorStage,
    PlotInternalError,
    PlotInvalidInputError,
    PlotResourceLimitError,
)
from processor.main import FAIL_FAST, CannedOutcome
from processor.report import PlotRunReport

STOPS = [
    PlotInvalidInputError(PlotErrorCode.UNKNOWN_CHANNEL, "the template's own message"),
    PlotDataUnavailableError(PlotErrorCode.INSUFFICIENT_SAMPLES, "the template's own message"),
    PlotResourceLimitError(PlotErrorCode.WINDOW_TOO_LARGE, "the template's own message"),
]
FALLS_BACK = [
    PlotEnvironmentError(PlotErrorCode.MISSING_DEPENDENCY, "layer missing"),
    PlotInternalError(PlotErrorCode.TEMPLATE_NO_OUTPUT, "no figure written"),
]


@pytest.fixture
def env(monkeypatch, tmp_path):
    inp, out = tmp_path / "in", tmp_path / "out"
    inp.mkdir()
    (inp / "rec.edf").write_bytes(b"x")
    monkeypatch.setenv("INPUT_DIR", str(inp))
    monkeypatch.setenv("OUTPUT_DIR", str(out))
    monkeypatch.setenv("TEMPLATE", "edf_processed_timeseries")
    monkeypatch.setenv("PROMPT", "plot it")
    monkeypatch.delenv("STUB_MODE", raising=False)
    monkeypatch.setattr(main_mod, "STUB_MODE", False)
    return out


def _agent_stub(monkeypatch, calls):
    import processor.agent as agent_mod
    import processor.executor as exec_mod

    def run_agent(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(success=True, output_size=99, iterations=2, final_text="", error="")

    monkeypatch.setattr(agent_mod, "run_agent", run_agent)
    monkeypatch.setattr(exec_mod, "resolve_layer_site_packages", lambda: "/nonexistent")


# ---- Scenario 1: the policy itself -------------------------------------------

def test_policy_is_the_three_user_actionable_categories():
    assert FAIL_FAST == {PlotErrorCategory.INVALID_INPUT, PlotErrorCategory.DATA_UNAVAILABLE,
                         PlotErrorCategory.RESOURCE_LIMIT}


# ---- Scenario 2: the template fails for a reason the user can act on ---------
# Stop with the template's own message; the agent is never called.

@pytest.mark.parametrize("err", STOPS, ids=lambda e: e.error_category.value)
def test_user_actionable_template_error_fails_fast(monkeypatch, env, err):
    calls = []
    _agent_stub(monkeypatch, calls)
    monkeypatch.setattr(main_mod, "try_canned_template", lambda *a: CannedOutcome(error=err))

    r = PlotRunReport()
    with pytest.raises(PlotError) as ei:
        main_mod.run(r)
    assert ei.value is err
    assert calls == []                              # agent never ran
    assert r.fallback_trigger_error is None         # it IS the error, not a side note
    assert r.requested_template == "edf_processed_timeseries"


# ---- Scenario 3: the template fails for a platform reason --------------------
# The agent gets a turn; the reason is kept on the report as the fallback trigger.

@pytest.mark.parametrize("err", FALLS_BACK, ids=lambda e: e.error_category.value)
def test_platform_template_error_falls_back_and_is_recorded(monkeypatch, env, err):
    calls = []
    _agent_stub(monkeypatch, calls)
    monkeypatch.setattr(main_mod, "try_canned_template", lambda *a: CannedOutcome(error=err))

    r = PlotRunReport()
    main_mod.run(r)                                 # no raise: the agent rescued it
    assert len(calls) == 1
    assert r.figure_generated_by == "agent" and r.figure_bytes == 99
    assert r.fallback_trigger_error is err          # the user still learns why


# ---- Scenario 4: there is no PROMPT, so nothing can be tried instead ---------

def test_no_prompt_surfaces_the_template_error_not_a_generic_one(monkeypatch, env):
    monkeypatch.setenv("PROMPT", "")
    err = FALLS_BACK[0]
    monkeypatch.setattr(main_mod, "try_canned_template", lambda *a: CannedOutcome(error=err))
    r = PlotRunReport()
    with pytest.raises(PlotError) as ei:
        main_mod.run(r)
    assert ei.value is err
    assert r.fallback_trigger_error is None         # no fallback was attempted, so not a trigger


# ---- Scenario 5: the template simply works -----------------------------------

def test_template_success_skips_agent(monkeypatch, env):
    calls = []
    _agent_stub(monkeypatch, calls)
    fig = env / "figure.png"

    def produce(config, target, out_path):
        env.mkdir(exist_ok=True)
        fig.write_bytes(b"PNG")
        return CannedOutcome(produced=True)

    monkeypatch.setattr(main_mod, "try_canned_template", produce)
    r = PlotRunReport()
    main_mod.run(r)
    assert r.figure_generated_by == "template" and r.figure_bytes == 3
    assert calls == []


# ---- Scenario 6: what try_canned_template() hands back -----------------------
# Its own checks raise typed errors; an error from render() keeps the stage
# the raise site set, and a plain exception gets wrapped and stamped "template".

def test_unknown_template_is_invalid_input(monkeypatch, tmp_path):
    monkeypatch.setattr(main_mod, "setup_layer_python_path", lambda: None)
    out = main_mod.try_canned_template({"template": "nope", "template_args": ""},
                                       str(tmp_path / "f.edf"), str(tmp_path / "o.png"))
    assert out.produced is False
    assert isinstance(out.error, PlotInvalidInputError)
    assert out.error.error_code is PlotErrorCode.UNKNOWN_TEMPLATE
    assert out.error.source_stage is PlotErrorStage.TEMPLATE
    assert "edf_processed_timeseries" in out.error.user_message
    assert "edf_processed_timeseries" in out.error.error_facts["known_templates"]


def _canned_with(monkeypatch, tmp_path, exc):
    class Boom:
        NAME = "boom"
        SUPPORTED_EXTENSIONS = (".edf",)

        def render(self, *a, **k):
            raise exc

    monkeypatch.setattr("processor.templates.get", lambda name: Boom())
    monkeypatch.setattr(main_mod, "setup_layer_python_path", lambda: None)
    return main_mod.try_canned_template({"template": "boom", "template_args": ""},
                                        str(tmp_path / "f.edf"), str(tmp_path / "o.png"))


def test_typed_error_from_render_keeps_its_own_stage(monkeypatch, tmp_path):
    exc = PlotInvalidInputError(PlotErrorCode.UNKNOWN_CHANNEL, "Channel 'Z' is not in this file.",
                                source_stage=PlotErrorStage.READER)
    out = _canned_with(monkeypatch, tmp_path, exc)
    assert out.error is exc                                   # not re-wrapped
    assert out.error.source_stage is PlotErrorStage.READER    # the raise site knew better


def test_plain_exception_from_render_is_wrapped_and_stamped_template(monkeypatch, tmp_path):
    out = _canned_with(monkeypatch, tmp_path, RuntimeError("Channel 'Z' is not in this file."))
    assert out.error.error_code is PlotErrorCode.REQUEST_REJECTED      # legacy adapter path
    assert out.error.source_stage is PlotErrorStage.TEMPLATE           # the site stamps itself
    assert out.error.error_category in FAIL_FAST
