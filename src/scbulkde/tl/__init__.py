"""Tools for differential expression analysis."""

from ._rank_genes_groups import rank_genes_groups
from .tl_basic import de

__all__ = [
    "rank_genes_groups",
    "de",
]
