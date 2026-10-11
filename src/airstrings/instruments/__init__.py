"""Importing this package registers every built-in instrument."""

from airstrings.instruments import guitar, piano  # noqa: F401
from airstrings.instruments.base import (
    Instrument,
    PlayContext,
    create_instrument,
    instrument_names,
    register_instrument,
)

__all__ = ["Instrument", "PlayContext", "create_instrument", "instrument_names", "register_instrument"]
