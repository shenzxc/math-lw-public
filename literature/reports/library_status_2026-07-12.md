# Literature Library Status Report

Generated: 2026-07-12 14:45:55

## Summary

- Records in catalog: 100
- Local PDFs retained: 100
- Structural PDF validation: 100/100 passed with `pdfinfo`
- PDF directory is ignored by git via `.gitignore`; metadata and scripts can be committed.
- Sources used: Scopus candidate search, arXiv API, OpenAlex optional hooks, and manual public PDF links for key papers.

## Category Quotas

- kv_cache_compression: 45
- serving_systems: 15
- uncertainty_calibration: 20
- long_context_reliability: 12
- theory_interpretability: 8

## Core Papers To Read First

- [7] H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models (kv_cache_compression)
- [1] Efficient Memory Management for Large Language Model Serving with PagedAttention (kv_cache_compression)
- [11] Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time (kv_cache_compression)
- [3] CacheGen: KV Cache Compression and Streaming for Fast Large Language Model Serving (kv_cache_compression)
- [4] KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization (kv_cache_compression)
- [15] KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache (kv_cache_compression)
- [6] NACL: A General and Effective KV Cache Eviction Framework for LLMs at Inference Time (kv_cache_compression)
- [26] SnapKV: LLM Knows What You are Looking for Before Generation (kv_cache_compression)
- [61] Conformal Risk Control (uncertainty_calibration)
- [62] On Calibration of Modern Neural Networks (uncertainty_calibration)
- [64] Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation (uncertainty_calibration)
- [65] SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models (uncertainty_calibration)
- [81] Lost in the Middle: How Language Models Use Long Contexts (long_context_reliability)
- [82] LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding (long_context_reliability)
- [86] RULER: What's the Real Context Size of Your Long-Context Language Models? (long_context_reliability)
- [83] NeedleBench: Can LLMs Do Retrieval and Reasoning in 1 Million Context Window? (long_context_reliability)
- [94] Attention Is All You Need (theory_interpretability)

## Web-Only References Kept Outside The 100-PDF Library

- A Mathematical Framework for Transformer Circuits: https://transformer-circuits.pub/2021/framework/index.html
- Causal Scrubbing: https://www.alignmentforum.org/posts/JvZhhzycHu2Yd57RN/causal-scrubbing-a-method-for-rigorously-testing
- Original Needle-in-a-Haystack implementation: https://github.com/gkamradt/needle-in-a-haystack

## Files

- `literature/catalog.csv`: sortable catalog with category, source URL, local PDF path, status, and relevance score.
- `literature/catalog.json`: machine-readable full metadata.
- `literature/library.bib`: BibTeX draft for manuscript integration.
- `literature/pdfs/`: local PDFs; intentionally not tracked by git.
- `literature/scripts/build_literature_library.py`: reproducible builder/downloader.

## Notes For The Paper

- Related work should be split into KV cache compression, LLM serving, uncertainty/calibration, conformal risk control, and long-context reliability.
- The most thesis-critical bridge is the small set around compression preserving uncertainty, conformal risk control, and failure/risk of KV cache compression.
- For public GitHub, commit metadata/scripts/report but keep PDFs local unless permissions are explicitly checked.
