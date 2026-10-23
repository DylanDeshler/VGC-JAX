"""Imports every content module so their handlers are registered (effects.hooks registry) before any tracing."""
from vgcjax.effects import conditions, field, fieldevent, protect, sideconds, statuses, volatiles  # noqa: F401
