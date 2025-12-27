"""DE engine backends."""

from __future__ import annotations

from ._base import DEEngine
from ._pydeseq2 import PyDESeq2Engine

_ENGINES: dict[str, type[DEEngine]] = {
    "pydeseq2": PyDESeq2Engine,
}


def get_engine(name: str) -> DEEngine:
    """Get a DE engine by name.

    Parameters
    ----------
    name
        Engine name. Currently supported: "pydeseq2".

    Returns
    -------
    DEEngine
        Instantiated engine.

    Raises
    ------
    ValueError
        If engine not found.
    """
    if name not in _ENGINES:
        available = list(_ENGINES.keys())
        raise ValueError(f"Unknown engine '{name}'. Available: {available}")
    return _ENGINES[name]()


def list_engines() -> list[str]:
    """List available DE engines.

    Returns
    -------
    list[str]
        Names of available engines.
    """
    return list(_ENGINES.keys())


__all__ = ["DEEngine", "get_engine", "list_engines"]
