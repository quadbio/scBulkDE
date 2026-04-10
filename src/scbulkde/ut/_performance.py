"""Performance monitoring utilities."""

from __future__ import annotations

import functools
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable

# Global flag to enable/disable performance logging
_PERFORMANCE_ENABLED = False


def set_performance_enabled(enabled: bool) -> None:
    """
    Enable or disable performance logging.

    When enabled, functions decorated with @performance will log their execution time.

    Parameters
    ----------
    enabled
        Whether to enable performance logging.

    Examples
    --------
    >>> import scbulkde as sb
    >>> sb.ut.set_performance_enabled(True)  # Enable performance logging
    >>> # ... run your analysis ...
    >>> sb.ut.set_performance_enabled(False)  # Disable performance logging
    """
    global _PERFORMANCE_ENABLED
    _PERFORMANCE_ENABLED = enabled


def performance(*, logger: logging.Logger):
    """
    Decorator to log function execution time.

    Parameters
    ----------
    logger
        Logger instance to use for logging performance metrics.

    Returns
    -------
    Callable
        Decorated function that logs its execution time when performance monitoring is enabled.

    Examples
    --------
    >>> from scbulkde.ut import logger, performance
    >>> @performance(logger=logger)
    ... def my_function():
    ...     # ... do something ...
    ...     pass
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _PERFORMANCE_ENABLED:
                return func(*args, **kwargs)

            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(f"{func.__name__} took {elapsed:.3f}s")
            return result

        return wrapper

    return decorator
