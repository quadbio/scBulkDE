"""Logging setup for scbulkde."""

from __future__ import annotations

import logging
from typing import Literal


def _setup_logger() -> logging.Logger:
    """Set up the scbulkde logger with rich formatting."""
    from rich.console import Console
    from rich.logging import RichHandler

    logger = logging.getLogger("scbulkde")
    logger.setLevel(logging.INFO)

    console = Console(force_terminal=True)
    if console.is_jupyter is True:
        console.is_jupyter = False

    ch = RichHandler(
        show_path=False,
        console=console,
        show_time=False,
    )
    logger.addHandler(ch)

    # Prevent double outputs
    logger.propagate = False
    return logger


def set_log_level(
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | Literal[10, 20, 30, 40, 50],
) -> None:
    """Set the logging level for scbulkde.

    Parameters
    ----------
    level
        Logging level. Can be a string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        or logging constants (logging.DEBUG=10, logging.INFO=20, etc.).

    Examples
    --------
    >>> import scbulkde
    >>> scbulkde.set_log_level("DEBUG")
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


logger = _setup_logger()
