"""scbulkde:  Single-cell to bulk differential expression analysis."""

from importlib.metadata import PackageNotFoundError, version

from scbulkde import pp, tl
from scbulkde.pp import PseudobulkResult, pseudobulk
from scbulkde.tl import DEResult, de

try:
    __version__ = version("scbulkde")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    # Submodules
    "pp",
    "tl",
    # Classes
    "PseudobulkResult",
    "DEResult",
    # Functions (convenience imports)
    "pseudobulk",
    "de",
]
