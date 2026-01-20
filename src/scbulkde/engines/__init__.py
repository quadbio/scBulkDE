"""Public interface for DE engine components."""

from ._base import DEEngineBase

# from ._edger import EdgeRDEEngine  # Uncomment if you add more engines
# from ._limma import LimmaDEEngine
from ._factory import get_engine_instance
from ._pydeseq2 import PyDESeq2DEEngine

__all__ = [
    "DEEngineBase",
    "PyDESeq2Engine",
    "get_engine_instance",
]
