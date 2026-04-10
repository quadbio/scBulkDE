"""Logging utilities for scbulkde."""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

# Create package-level logger
logger = logging.getLogger("scbulkde")
logger.setLevel(logging.INFO)

# Add console handler if not already present
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)


def set_log_level(level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | int) -> None:
    """
    Set the logging level for scbulkde.

    Parameters
    ----------
    level
        Logging level. Can be a string ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        or an integer from the logging module.

    Examples
    --------
    >>> import scbulkde as scb
    >>> scb.ut.set_log_level("DEBUG")  # Enable debug logging
    >>> scb.ut.set_log_level("WARNING")  # Only show warnings and errors
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    logger.setLevel(level)


def _in_notebook() -> bool:
    """
    Check if code is running in a Jupyter notebook.

    Returns
    -------
    bool
        True if running in a notebook environment, False otherwise.
    """
    try:
        from IPython import get_ipython

        if get_ipython() is None:
            return False
        if "IPKernelApp" in get_ipython().config:
            return True
        return False
    except (ImportError, AttributeError):
        return False
