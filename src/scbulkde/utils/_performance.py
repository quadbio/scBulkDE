"""Utility to print or log function performance."""

from __future__ import annotations

import time
from functools import wraps


def performance(logger=None, prefix="Performance: "):
    """Decorator factory to print or log function performance at DEBUG level."""
    from . import _constants as config

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not config.PERFORMANCE_ENABLED:
                return func(*args, **kwargs)
            t_start = time.perf_counter()
            result = func(*args, **kwargs)
            t_end = time.perf_counter()
            msg = f"{prefix}{func.__name__} took {t_end - t_start:.4f} seconds."
            if logger:
                logger.info(msg)
            else:
                print(msg)
            return result

        return wrapper

    return decorator


def set_performance_enabled(flag: bool):
    """Set global performance reporting."""
    from . import _constants as config

    config.PERFORMANCE_ENABLED = flag
