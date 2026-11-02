"""Robot backends behind the agent MCP. The frontend never imports a backend directly.

A backend is constructed once per session by the frontend process (the only robot
owner) and implements:
    tools() -> list[dict]            MCP tool definitions (name, description, inputSchema)
    call(name, arguments) -> (value, images)
                                     value: JSON-safe dict; images: MCP image content
                                     blocks {'type': 'image', 'mimeType', 'data'(base64)}
    status() -> dict                 live robot summary merged into the `status` tool
    close()                          release hardware; must be safe to call twice
Optional: ``mutating`` (set of tool names that can move hardware; a failure in one
marks control uncertain) and ``validate(name, arguments)`` (raise before any motion).

A robot may also register read-only telemetry for the operator dashboard: a class whose
``snapshot()`` returns a JSON-safe dict with ``kind`` and ``ok``. It must never command
the robot, since it runs next to a session's backend.
"""
from importlib import import_module
# registry key -> "module:Class"; the module is imported only when selected.
BACKENDS = {
    'rby1': 'services.rby1.backend:Rby1Backend',
    'mock': 'services.mock_backend:MockBackend',
}
# registry key -> "module:Class" of the dashboard telemetry; robots without one show no live state.
TELEMETRY = {
    'rby1': 'services.rby1.telemetry:Telemetry',
}


def _load(spec):
    module, _, attribute = spec.partition(':')
    return getattr(import_module(module), attribute)


def make_telemetry(name):
    """The robot's read-only dashboard telemetry, or None when it registers none."""
    return _load(TELEMETRY[name])() if name in TELEMETRY else None

def backend_class(name):
    try:
        module, _, attribute = BACKENDS[name].partition(':')
    except KeyError:
        raise ValueError(f"Unknown robot backend {name!r}; choose one of {', '.join(sorted(BACKENDS))}") from None
    return getattr(import_module(module), attribute)


def make_backend(config):
    """The configured robot's backend, constructed with the operator configuration."""
    return backend_class(config.robot)(config)
