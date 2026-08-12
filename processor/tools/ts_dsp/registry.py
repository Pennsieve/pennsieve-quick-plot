"""
ts_dsp.registry 

A notebook (_REGISTRY) where every tool writes down its name and its rules.
A foreman (apply_dsp_pipeline) who takes the user's command ("first filter, then compute energy"), looks each step up in the notebook, checks it makes sense, and only then lets it run.

Every DSP tool registers itself here via the `@dsp_tool(...)` decorator,
declaring its required parameters and its input/output domains. The registry
is flat (name -> ToolSpec): callers dispatch a tool by *name* and never need
to know which module it lives in. `apply_dsp_pipeline` walks an ordered list
of {tool, params} steps, validating each step against the running Signal's
domains before executing it, and returns the processed Signal.

Because each tool declares `requires` / `produces`, an illegal ordering
(e.g. a time-domain filter after an FFT that produced a frequency-domain
signal) is rejected automatically — there is no need to enumerate forbidden
combinations.

This module imports only the standard library. Tools read a Signal's fields
by attribute and return a modified copy via dataclasses.replace, so nothing
here needs to import the Signal class — which keeps the dependency one-way
(tools/templates import the registry; the registry imports nothing back).
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

    name: str                 # e.g. highpass-filter cutoff
    required: bool = True
    description: str = ""
    unit: str = ""            # e.g. Hz 


@dataclass(frozen=True)
class ToolSpec:
    """A registered tool: its callable plus the metadata the driver and the
    (generated) MCP schema read."""

    name: str                 # e.g. highpass_filter
    func: "object"            # Callable[[Signal, ...], Signal]
    requires: dict            # axes that MUST match, e.g. {"x_domain": "time"}
    produces: dict            # axes this tool sets on its output Signal, e.g. {"x_domain": "frequency"}
    params: tuple             # tuple[ParamSpec, ...]; the list of ParaSpec from above
    description: str = ""


_REGISTRY: dict[str, ToolSpec] = {}


def dsp_tool(name: str, *, requires: dict, produces: dict,
             params: tuple = (), description: str = ""):
    """Register a Signal -> Signal function as a named, domain-tagged tool."""

    def deco(fn):
        # rejects if a second tool tries to register under an already-taken name
        if name in _REGISTRY:
            raise RuntimeError(f"Duplicate DSP tool registration: {name!r}.")
        _REGISTRY[name] = ToolSpec(
            name=name, func=fn, requires=dict(requires), produces=dict(produces),
            params=tuple(params), description=description,
        )
        return fn

    return deco


def get(name: str) -> "ToolSpec | None":
    return _REGISTRY.get(name)


def known_names() -> list[str]:
    return sorted(_REGISTRY)


def specs() -> list[ToolSpec]:
    """All registered specs, name-sorted — used by the schema dumper."""
    return [_REGISTRY[n] for n in known_names()]


def _check_params(spec: ToolSpec, params: dict) -> None:
    given = set(params)
    allowed = {p.name for p in spec.params}
    for p in spec.params:
        # did user miss/skip any required parameter to run the tool?
        if p.required and p.name not in given:
            hint = f" ({p.description})" if p.description else ""
            unit = f" [{p.unit}]" if p.unit else ""
            raise ToolInputError(
                f"{spec.name} requires parameter {p.name!r}{unit}{hint}."
            )
    unknown = given - allowed
    # did user input any parameter that does not exist?
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


def apply_dsp_pipeline(signal, steps):
    """Run an ordered list of {tool, params} steps on `signal`.
    e.g. 
    [ {tool: "highpass_filter", params: {cutoff: 1.0}},
      {tool: "energy",          params: {}} ]

    Each step is validated (known tool, required params present, domains
    compatible with the running signal) *before* it executes, so a bad
    request fails fast with a specific ToolInputError. Returns the processed
    Signal. An empty / falsy `steps` returns the signal unchanged.
    """
    # if the list is empty, return the signal untouched (i.e. no processing)
    if not steps:
        return signal
    # if list malformed, raise the error
    if not isinstance(steps, (list, tuple)):
        raise ToolInputError(
            f"pipeline must be a list of steps; got {type(steps).__name__}."
        )
    # Then, for each step in the list
    for i, step in enumerate(steps):
        # is the step/tool name shaped correctly?
        if not isinstance(step, dict) or "tool" not in step:
            raise ToolInputError(
                f"pipeline step {i} must be an object with a 'tool' name; got {step!r}."
            )
        spec = get(step["tool"])
        # is the tool registered?
        if spec is None:
            raise ToolInputError(
                f"Unknown processing tool {step['tool']!r}. "
                f"Available tools: {', '.join(known_names()) or '(none)'}."
            )
        params = step.get("params") or {}
        # is the param shaped correctly?
        if not isinstance(params, dict):
            raise ToolInputError(
                f"params for step {i} ({spec.name}) must be an object; got "
                f"{type(params).__name__}."
            )
        _check_params(spec, params)
        _check_domains(spec, signal)
        signal = spec.func(signal, **params)
    return signal