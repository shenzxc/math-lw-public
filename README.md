# Uncertainty-Calibrated KV Cache Compression

This is the public research snapshot for:

**Uncertainty-Calibrated KV Cache Compression for Efficient LLM Inference**

The repository contains the LaTeX manuscript draft, reproducible local
experiment code, synthetic prompt generators, public pilot-result summaries, and
figures. It intentionally omits private infrastructure notes, remote-machine
operational scripts, model weights, and raw private execution logs.

## Current Status

The current manuscript is conservative. It reports pilot evidence and negative
confirmation results honestly: uncertainty-gated heavy-hitter scoring produced
a small-model signal but did not transfer reliably to a 7B setting. The next
research step is to redesign uncertainty as a calibrated budget/risk controller
over strong KV-cache backbones rather than as the main token-ranking signal.

## Contents

- `main.tex`: top-level LaTeX document.
- `sections/`: manuscript sections.
- `references.bib`: bibliography.
- `experiments/run_kv_cache_pilot.py`: core KV-cache compression pilot runner.
- `experiments/generate_synthetic_benchmark.py`: controlled retrieval prompt generator.
- `experiments/analyze_*.py`: aggregation and plotting scripts.
- `experiments/prompts/`: deterministic synthetic prompt sets.
- `figures/`: manuscript figures.
- `outputs/analysis_*`: public aggregate summaries and plots.

## Build

With `tectonic`:

```bash
make pdf
```

With a traditional LaTeX setup:

```bash
make pdf-latexmk
```

## Reproducibility Notes

The public repository is designed for local or self-managed experiments. Any
large-model or remote-hardware runs should be adapted to the user's own compute
environment. Private infrastructure automation has been deliberately excluded.

## Citation

This is an in-progress research draft. Please cite the repository only as a
work in progress unless a formal preprint is released.
