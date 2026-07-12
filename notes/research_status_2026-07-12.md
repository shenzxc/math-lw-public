# Research Status: 2026-07-12

## Current evidence

The first uncertainty-gated heavy-hitter design produced a local small-model
signal but failed to transfer to the Qwen2.5-7B controlled retrieval setting.
The project therefore does not currently support a claim that uncertainty-
weighted token ranking improves KV-cache compression.

The main mechanism concern is a target mismatch: next-token uncertainty
measures the difficulty of the current prediction, while cache retention needs
to estimate the future marginal value of each historical token. These quantities
need not align.

## Evaluation correction

The original H2O and UCKV2 comparisons used different protected recent windows
and different eviction cadences. Periodic UCKV2 eviction could also exceed its
nominal budget between eviction events. Historical results are therefore useful
diagnostics but are not final strictly matched-memory evidence.

The public runner now provides `h2o_matched_B`, which uses the same recent-token
rule, probe layers, and eviction cadence as UCKV2. It also reports:

- average and maximum retained tokens;
- budget-overrun step count;
- maximum token overrun;
- average and peak cache bytes;
- selector and decode timing fields.

`experiments/test_kv_cache_selection.py` verifies shared selection and exact
post-eviction budgets.

## Prior-art correction

Broad claims that uncertainty-aware or input-adaptive budget control is new are
not defensible. Nearby work includes ZigZagKV, UNCOMP, DBudgetKV, GVote,
CompilerKV, and CONF-KV. Reliability and risk framing must also account for
DefensiveKV, *The Pitfalls of KV Cache Compression*, and *The risk of KV cache
compression*.

The remaining constructive hypothesis is narrower:

> Can a fixed strong KV-compression backbone be wrapped with a finite-sample
> audited budget-selection and abstention policy that provides a useful
> memory-risk trade-off under a deployment-valid feature contract?

This is not yet an established contribution. It must pass a primary-source
novelty audit and a 7B standard-benchmark gate.

## Next decision gates

1. Reproduce full KV, H2O, and at least one modern strong method in a
   standardized harness on small RULER and LongBench slices.
2. Compare token-utility variants only under the repaired evaluation contract.
3. Freeze the backbone and test a prompt-level budget controller with held-out
   calibration, explicit coverage, abstention, and risk upper bounds.
4. Report in-distribution and shifted results with realized memory and selector
   overhead.
5. Use a constructive paper route only if the controller passes on a 7B model;
   otherwise center the paper on fixed-contract cross-scale transfer failure
   and reliability auditing.

## Primary sources for the next literature pass

- ZigZagKV: https://aclanthology.org/2025.coling-main.596/
- UNCOMP: https://aclanthology.org/2025.emnlp-main.209/
- DBudgetKV: https://arxiv.org/abs/2502.16886
- GVote: https://openreview.net/forum?id=0yLdDZMutq
- CompilerKV: https://arxiv.org/abs/2602.08686
- CONF-KV: https://arxiv.org/abs/2605.24786
- DefensiveKV: https://arxiv.org/abs/2510.13334
- The Pitfalls of KV Cache Compression:
  https://aclanthology.org/2026.acl-long.1926/
- The risk of KV cache compression: https://arxiv.org/abs/2607.01520
- Expected Attention and KVPress: https://arxiv.org/abs/2510.00636
- Conformal Risk Control: https://openreview.net/forum?id=33XGfHLtZg
