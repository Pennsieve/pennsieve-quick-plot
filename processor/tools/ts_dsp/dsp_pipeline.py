"""
ts_dsp.dsp_pipeline — the tool contract and the pipeline runner.

Two things live here:

  * `@dsp_tool(...)` — the decorator every DSP tool wears. It attaches a
    `ToolSpec` (name, required parameters, input/output domains) to the
    function as `fn.spec`. It does NOT register anything: the list of
    tools is written out explicitly in `ts_dsp/__init__.py` (`ALL_TOOLS`),
    the same way `templates/__init__.py` lists the templates.

  * `apply_dsp_pipeline(signal, steps)` — the runner. It walks an ordered
    list of {tool, params} steps, looks each tool up by name, validates the
    step against the *current* Signal (known tool, required params present,
    domains compatible) and only then runs it.

Because each tool declares `requires` / `produces`, an illegal ordering
(e.g. a time-domain filter after an FFT that produced a frequency-domain
signal) is rejected automatically — there is no need to enumerate forbidden
combinations.

`produces` is a delta: list an axis/domain only if the applied tool changes
it; otherwise the original axis/domain passes through untouched (a tool that
changes nothing declares `produces={}`, mirroring how the tools themselves
use dataclasses.replace to set only the fields they change).

This module imports only the standard library.
"""

from __future__ import annotations

from dataclasses import dataclass


class ToolInputError(RuntimeError):
    """A DSP request that is invalid because of the *user's* input — an
    unknown tool, a missing/nonsensical parameter, or an illegal step order.

    Distinct from a generic failure so the caller can choose to surface it to
    the user rather than silently falling back to the agent loop. Subclasses
    RuntimeError so existing `except RuntimeError` handlers still catch it.
    """


@dataclass(frozen=True)
class ParamSpec:
    """One parameter a tool accepts."""

    name: str                 # e.g. cutoff
    required: bool = True
    description: str = ""
    unit: str = ""            # e.g. Hz


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool: its callable plus the metadata the runner and the
    (generated) MCP schema read."""

    name: str                 # e.g. highpass_filter
    func: "object"            # Callable[[Signal, ...], Signal]
    requires: dict            # axes that MUST match, e.g. {"x_domain": "time"}
    produces: dict            # axes this tool sets on its output, e.g. {"x_domain": "frequency"}
    params: tuple             # tuple[ParamSpec, ...]
    description: str = ""


def dsp_tool(name: str, *, requires: dict, produces: dict,
             params: tuple = (), description: str = ""):
    """Attach a ToolSpec to a Signal -> Signal function as `fn.spec`.

    Registration is separate and explicit: add the function to `ALL_TOOLS`
    in `ts_dsp/__init__.py`.
    """

    def deco(fn):
        fn.spec = ToolSpec(
            name=name, func=fn, requires=dict(requires), produces=dict(produces),
            params=tuple(params), description=description,
        )
        return fn

    return deco


def _check_params(spec: ToolSpec, params: dict) -> None:
    given = set(params)
    allowed = {p.name for p in spec.params}
    for p in spec.params:
        # did the user miss a required parameter?
        if p.required and p.name not in given:
            hint = f" ({p.description})" if p.description else ""
            unit = f" [{p.unit}]" if p.unit else ""
            raise ToolInputError(
                f"{spec.name} requires parameter {p.name!r}{unit}{hint}."
            )
    unknown = given - allowed
    # did the user pass a parameter this tool does not have?
    if unknown:
        raise ToolInputError(
            f"{spec.name} got unknown parameter(s): {', '.join(sorted(unknown))}. "
            f"Allowed: {', '.join(sorted(allowed)) or '(none)'}."
        )


def _check_domains(spec: ToolSpec, signal) -> None:
    for axis, need in spec.requires.items():
        have = getattr(signal, axis, None)
        if have != need:
            raise ToolInputError(
                f"{spec.name} expects {axis}={need!r}, but the signal is "
                f"{axis}={have!r} at this point in the pipeline."
            )


def apply_dsp_pipeline(signal, steps, registry: "dict[str, ToolSpec] | None" = None):
    """Run an ordered list of {tool, params} steps on `signal`.

    e.g.
    [ {tool: "highpass_filter", params: {cutoff: 1.0}},
      {tool: "energy",          params: {window_size: 1, window_stride: 0.5}} ]

    Each step is validated (known tool, required params present, domains
    compatible with the running signal) *before* it executes, so a bad
    request fails fast with a specific ToolInputError. Returns the processed
    Signal. An empty / falsy `steps` returns the signal unchanged.

    `registry` defaults to the package's tool table (`ts_dsp._REGISTRY`);
    it is a parameter so tests can run the driver against a tiny fake table.
    """
    if registry is None:
        from . import _REGISTRY  # lazy: __init__ imports this module
        registry = _REGISTRY

    if not steps:
        return signal
    if not isinstance(steps, (list, tuple)):
        raise ToolInputError(
            f"pipeline must be a list of steps; got {type(steps).__name__}."
        )
    for i, step in enumerate(steps):
        # 1. shape of the step
        if not isinstance(step, dict) or "tool" not in step:
            raise ToolInputError(
                f"pipeline step {i} must be an object with a 'tool' name; got {step!r}."
            )
        # 2. known tool?
        spec = registry.get(step["tool"])
        if spec is None:
            raise ToolInputError(
                f"Unknown processing tool {step['tool']!r}. "
                f"Available tools: {', '.join(sorted(registry)) or '(none)'}."
            )
        # 3. parameters
        params = step.get("params") or {}
        if not isinstance(params, dict):
            raise ToolInputError(
                f"params for step {i} ({spec.name}) must be an object; got "
                f"{type(params).__name__}."
            )
        _check_params(spec, params)
        # 4. is the current signal the right kind of data for this tool?
        _check_domains(spec, signal)
        # 5. run
        signal = spec.func(signal, **params)
    return signal
