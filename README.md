# scBulkDE

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]
[![PyPI][badge-pypi]][pypi]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/quadbio/scBulkDE/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/scBulkDE
[badge-pypi]: https://img.shields.io/pypi/v/scbulkde.svg

scBulkDE performs differential expression testing on pseudobulked single-cell data. It aggregates cells into pseudobulk samples, infers a full-rank design matrix and performs differential gene expression analysis while accounting for categorical and continuous covariates. Currently PyDeseq2 and ANOVA backends are supported for DE testing.

## Key features

- **Pseudobulk aggregation** with quality control
- **Covariate-aware design** — categorical and continuous covariates with automated resolution of confounding factors
- **Multiple DE engines** — ANOVA F-test and PyDESeq2
- **Fallback strategies** — pseudoreplicate generation or single-cell testing when biological replicates are insufficient
- **Scanpy drop-in** — `tl.rank_genes_groups` stores results in `adata.uns` for seamless integration with scanpy.

## Installation

You need to have Python 3.11 or newer installed on your system.

Install from PyPI:

```bash
pip install scbulkde
```

Or install the latest development version:

```bash
pip install git+https://github.com/quadbio/scBulkDE.git@main
```

## Quick start

```python
import scbulkde as scb

# One-step: pseudobulk + DE
de_result = scb.tl.de(
    adata,
    group_key="cell_type",
    query="B cells",
    reference="rest",
    replicate_key="donor",
    engine="anova",
)

de_result.results.head()
```

Or separate pseudobulking from testing for more control:

```python
# Step 1 — Pseudobulk
pb_result = scb.pp.pseudobulk(
    adata,
    group_key="cell_type",
    query="B cells",
    reference="rest",
    replicate_key="donor",
)

# Step 2 — DE
de_result = scb.tl.de(pb_result, engine="anova")
```

### Scanpy drop-in replacement

```python
import scanpy as sc
import scbulkde as scb

scb.tl.rank_genes_groups(
    adata,
    groupby="cell_type",
    de_kwargs=dict(replicate_key="donor", engine="anova"),
)

sc.pl.rank_genes_groups(adata, n_genes=20)
```

## Documentation

Please refer to the [documentation][], in particular the [API documentation][].

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/quadbio/scBulkDE/issues
[tests]: https://github.com/quadbio/scBulkDE/actions/workflows/test.yaml
[documentation]: https://scBulkDE.readthedocs.io
[changelog]: https://scBulkDE.readthedocs.io/en/latest/changelog.html
[api documentation]: https://scBulkDE.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/scBulkDE
