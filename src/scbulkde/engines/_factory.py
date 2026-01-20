"""Factory for creating differential expression engine instances."""

from scbulkde.engines._pydeseq2 import PyDESeq2Engine
from scbulkde.ut._logging import logger


def get_engine_instance(adata, engine_name: str, engine_params: dict):
    """
    Create an instance of the specified differential expression engine.

    Parameters
    ----------
    adata
        Annotated data object (e.g., AnnData, DataFrame).
    engine_name
        Name of the DE engine.
    engine_params
        Parameters for the engine.

    Returns
    -------
    DEEngineBase
        Instance of the DE engine.

    Examples
    --------
    >>> from scbulkde.engines._factory import get_engine_instance
    >>> engine = get_engine_instance(adata, "pydeseq2", {"min_counts": 20})
    """
    # Available DE engines
    engine_map = {
        "pydeseq2": PyDESeq2Engine,
    }

    engine_name_lower = engine_name.lower()
    engine_class = engine_map.get(engine_name_lower)

    if engine_class is None:
        available_engines = ", ".join(sorted(engine_map.keys()))
        raise ValueError(f"Unknown DE engine: {engine_name}. Available engines: {available_engines}")

    logger.info("Creating %s engine instance", engine_name)
    return engine_class(adata, **engine_params)
