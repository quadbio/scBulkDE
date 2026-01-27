"""scbulkde:  Single-cell to bulk differential expression analysis."""

from importlib.metadata import PackageNotFoundError, version

from scbulkde import pp, tl
from scbulkde.pp import pseudobulk
from scbulkde.tl import de, rank_genes_groups
from scbulkde.ut._containers import DEResult, PseudobulkResult

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
    # Functions
    "pseudobulk",
    "de",
    "rank_genes_groups",
]
