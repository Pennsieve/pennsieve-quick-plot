"""
Tests for try_canned_template's handling of TEMPLATE_ARGS — the JSON blob
MCP's plot_file forwards as processorParams.template_args. Verifies the
processor decodes it and splats it into the template's render(**kwargs),
preserves non-string types, and reports a malformed / non-object blob as
an invalid_input PlotError (run() falls back to the agent and records it).

pytest processor/test_templates/test_dispatch_template_args.py
"""

import json

from processor import main


class _StubTemplate:
    NAME = "stub_template"
    SUPPORTED_EXTENSIONS = (".edf",)

    def __init__(self):
        self.received = None

    def render(self, target_file_path, output_path, **kwargs):
        self.received = kwargs
        with open(output_path, "wb") as fh:
            fh.write(b"PNGDATA")


def _run(monkeypatch, tmp_path, template_args):
    """Drive try_canned_template against a stub template; return (result, stub, out_path)."""
    stub = _StubTemplate()
    monkeypatch.setattr("processor.templates.get", lambda name: stub)
    # The stub carries its own (no-op) deps; skip the EFS sys.path shuffle.
    monkeypatch.setattr(main, "setup_layer_python_path", lambda: None)

    target = tmp_path / "rec.edf"
    target.write_bytes(b"x")
    out = tmp_path / "figure.png"
    config = {"template": "stub_template", "template_args": template_args}
    result = main.try_canned_template(config, str(target), str(out))
    return result, stub, out


def test_args_are_parsed_and_splatted(monkeypatch, tmp_path):
    args = {"channel": "F8", "start_time": 10, "duration": 5,
            "time_unit": "s", "y_range": [-200, 200], "y_unit": "uV"}
    result, stub, out = _run(monkeypatch, tmp_path, json.dumps(args))
    assert result.produced is True and result.error is None
    assert out.exists() and out.stat().st_size > 0
    # Types survive the JSON round-trip: numbers stay numbers, lists stay lists.
    assert stub.received == args


def test_empty_args_render_with_no_kwargs(monkeypatch, tmp_path):
    result, stub, out = _run(monkeypatch, tmp_path, "")
    assert result.produced is True
    assert stub.received == {}


def test_malformed_json_is_invalid_input(monkeypatch, tmp_path):
    result, stub, out = _run(monkeypatch, tmp_path, "{not valid json")
    assert result.produced is False
    assert result.error.error_category.value == "invalid_input"
    assert result.error.error_code.value == "template_args_invalid_json"
    assert stub.received is None          # render never reached
    assert not out.exists()


def test_non_object_json_is_invalid_input(monkeypatch, tmp_path):
    result, stub, out = _run(monkeypatch, tmp_path, "[1, 2, 3]")
    assert result.produced is False
    assert result.error.error_code.value == "template_args_not_object"
    assert stub.received is None
