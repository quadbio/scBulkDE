from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

from . import _constants as config


def _in_notebook():
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
        return shell == "ZMQInteractiveShell"
    except Exception:  # noqa: BLE001
        return False


def _setup_logger() -> logging.Logger:
    """Set up the scbulkde logger with rich formatting."""
    logger = logging.getLogger("scbulkde")
    logger.setLevel(getattr(logging, config.LOG_LEVEL.upper()))

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


def set_log_level(level):
    """Set the logging level for scbulkde."""
    if isinstance(level, str):
        level_str = level.upper()
        level_value = getattr(logging, level_str)
    else:
        level_value = level
        level_str = logging.getLevelName(level_value)
        if not isinstance(level_str, str):
            level_str = "INFO"
    logger.setLevel(level_value)
    for handler in logger.handlers:
        handler.setLevel(level_value)
    config.LOG_LEVEL = level_str


logger = _setup_logger()
