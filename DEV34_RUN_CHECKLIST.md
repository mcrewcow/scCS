# scCS dev33 final run checklist

## 1. Install the built wheel

```bash
pip install --force-reinstall --no-deps \
  sccs_py-0.8.0.dev34-py3-none-any.whl
```

Restart the notebook kernel and verify:

```python
import scCS
print(scCS.__version__)
print(scCS.__file__)
```

Expected version: `0.8.0.dev34`.

## 2. Run the focused real-data gates

Run these notebooks from the beginning:

1. `04_schwann_pair_scorer_dev33.ipynb`
2. `06_schwann_multi_scorer_dev33.ipynb`
3. `09_package_decision_reproduction_dev33.ipynb`

Expected outcomes:

- PairScorer: positive ChC CFA effect with replicate-level inference.
- MultiScorer: significant ChC omnibus result and positive planned ordered
  contrast.
- Notebook 09: `FINAL: READY_TO_FREEZE`.

Negative Gut Signed Ordering Flux is biological information and is not a
failure condition.

## 3. Confirm separate downstream exports

Run:

- `10_pancreas_downstream_analysis_dev33.ipynb`
- `11_schwann_downstream_analysis_dev33.ipynb`

Expected directories:

- `tutorial_outputs/pancreas_downstream`
- `tutorial_outputs/schwann_downstream`

## 4. Standard scalability run

Run `12_scalability_no_chunking_dev33.ipynb` normally. It measures complete
problems that fit on the current host and records larger problems as skipped.
It never substitutes a chunked solve.

## 5. Exact 200-million-cell full graph run

On a high-memory host:

```bash
export SCCS_SCALABILITY_PROFILE=exact_200m
jupyter nbconvert --to notebook --execute \
  12_scalability_no_chunking_dev33.ipynb \
  --output 12_scalability_no_chunking_dev33_executed.ipynb \
  --ExecutePreprocessor.timeout=-1
```

Alternatively use the command-line runner:

```bash
python run_dffp_200m_no_chunk_dev33.py \
  --cells 200000000 \
  --degree 4 \
  --outcomes 3 \
  --effective-horizon 64 \
  --output dffp_200m_degree4_result.json
```

The runner raises `MemoryError` before allocation if the entire problem cannot
fit with the requested headroom. It never chunks or partitions the scientific
calculation.

## Release decision

Proceed to the release candidate when:

- notebook 09 prints `READY_TO_FREEZE`;
- the controlled Schwann condition effects are recovered clearly;
- all tutorial exports complete;
- the exact 200M result JSON reports `cells=200000000`,
  `scientific_scoring_chunked=false`, and `converged=true`.
