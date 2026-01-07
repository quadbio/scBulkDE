"""Utility to print or log function performance."""

from __future__ import annotations

import time
from functools import wraps


def performance(enabled=True, logger=None, prefix="Performance: "):
    """Decorator factory to print or log function performance at DEBUG level."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)
            t_start = time.perf_counter()
            result = func(*args, **kwargs)
            t_end = time.perf_counter()
            msg = f"{prefix}{func.__name__} took {t_end - t_start:.4f} seconds."
            if logger:
                logger.debug(msg)
            else:
                print(msg)
            return result

        return wrapper

    return decorator
