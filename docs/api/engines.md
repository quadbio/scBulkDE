# Engines

Statistical backends for differential expression testing. Custom engines can
be created by subclassing {class}`~scbulkde.engines.DEEngineBase`.

## Engine Factory

```{eval-rst}
.. module:: scbulkde.engines
.. currentmodule:: scbulkde.engines

.. autosummary::
    :toctree: generated

    get_engine_instance
```

## Base Class

```{eval-rst}
.. autosummary::
    :toctree: generated

    DEEngineBase
```

## Built-in Engines

```{eval-rst}
.. autosummary::
    :toctree: generated

    AnovaEngine
    PyDESeq2Engine
```
