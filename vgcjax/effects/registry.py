"""Imports every content module so their handlers are registered (effects.hooks registry) before any tracing."""
from vgcjax.effects import abilities, ability_mods, ability_reactions, berries, conditions, disables, field, items, locked, mega, move_callbacks, moves_b1, moves_b2, fieldevent, protect, sideconds, statuses, terrain, volatiles, weather  # noqa: F401
