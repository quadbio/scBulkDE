"""Factory for creating differential expression engine instances."""

from scbulkde.engines._parametric import AnovaEngine  # , EbayesEngine
from scbulkde.engines._pydeseq2 import PyDESeq2Engine
from scbulkde.ut._logging import logger


def get_engine_instance(engine_name: str):
    """Create an instance of the specified differential expression engine."""
    # Available DE engines
    engine_map = {
        "pydeseq2": PyDESeq2Engine,
        "anova": AnovaEngine,
        # "ebayes": EbayesEngine,
    }

    engine_name_lower = engine_name.lower()
    engine_class = engine_map.get(engine_name_lower)

    if engine_class is None:
        available_engines = ", ".join(sorted(engine_map.keys()))
        raise ValueError(f"Unknown DE engine: {engine_name}. Available engines: {available_engines}")

    logger.info("Creating %s engine instance", engine_name)
    return engine_class()
