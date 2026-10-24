"""Imports every content module so their handlers are registered (effects.hooks registry) before any tracing."""
from vgcjax.effects import abilities, conditions, field, fieldevent, protect, sideconds, statuses, terrain, volatiles, weather  # noqa: F401
