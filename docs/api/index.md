# API

Import scBulkDE as:

```python
import scbulkde as scb
```

## Preprocessing: `pp`

Preprocessing functions for pseudobulk aggregation.

```{eval-rst}
.. module:: scbulkde.pp
.. currentmodule:: scbulkde

.. autosummary::
    :toctree: generated

    pp.pseudobulk
```

## Tools: `tl`

Tools for differential expression analysis.

```{eval-rst}
.. module:: scbulkde.tl
.. currentmodule:: scbulkde

.. autosummary::
    :toctree: generated

    tl.de
    tl.rank_genes_groups
```

## Classes

Container objects returned by the core functions.

### PseudobulkResult

```{eval-rst}
.. currentmodule:: scbulkde

.. autosummary::
    :toctree: generated

    PseudobulkResult
```

#### Attributes

```{eval-rst}
.. autosummary::
    :toctree: generated

    PseudobulkResult.adata_sub
    PseudobulkResult.pb_counts
    PseudobulkResult.grouped
    PseudobulkResult.sample_table
    PseudobulkResult.design_matrix
    PseudobulkResult.design_formula
    PseudobulkResult.group_key
    PseudobulkResult.group_key_internal
    PseudobulkResult.query
    PseudobulkResult.reference
    PseudobulkResult.strata
    PseudobulkResult.layer
    PseudobulkResult.layer_aggregation
    PseudobulkResult.categorical_covariates
    PseudobulkResult.continuous_covariates
    PseudobulkResult.continuous_aggregation
    PseudobulkResult.min_cells
    PseudobulkResult.min_fraction
    PseudobulkResult.min_coverage
    PseudobulkResult.qualify_strategy
    PseudobulkResult.n_cells
```

#### Properties

```{eval-rst}
.. autosummary::
    :toctree: generated

    PseudobulkResult.n_samples
    PseudobulkResult.collapsed
```

### DEResult

```{eval-rst}
.. autosummary::
    :toctree: generated

    DEResult
```

#### Attributes

```{eval-rst}
.. autosummary::
    :toctree: generated

    DEResult.results
    DEResult.query
    DEResult.reference
    DEResult.design
    DEResult.engine
    DEResult.used_pseudoreplicates
    DEResult.used_single_cell
    DEResult.n_repetitions
    DEResult.repetition_results
    DEResult.repetition_stats
```

#### Properties

```{eval-rst}
.. autosummary::
    :toctree: generated

    DEResult.n_significant
    DEResult.n_genes
    DEResult.fallback_used
```

#### Methods

```{eval-rst}
.. autosummary::
    :toctree: generated

    DEResult.summary
    DEResult.get_repetition_stats
    DEResult.get_repetition_results
```

## Engines

Statistical backends for differential expression testing. Custom engines can
be created by subclassing {class}`~scbulkde.engines.DEEngineBase`.

### Engine Factory

```{eval-rst}
.. module:: scbulkde.engines
.. currentmodule:: scbulkde.engines

.. autosummary::
    :toctree: generated

    get_engine_instance
```

### Base Class

```{eval-rst}
.. autosummary::
    :toctree: generated

    DEEngineBase
```

### Built-in Engines

```{eval-rst}
.. autosummary::
    :toctree: generated

    AnovaEngine
    PyDESeq2Engine
```

## Settings

Configure logging and performance tracing globally.

```{eval-rst}
.. currentmodule:: scbulkde.ut

.. autosummary::
    :toctree: generated

    set_log_level
    set_performance_enabled
```
