# UCKV2 Ablation Results

Evaluation split only. Rows are grouped by ablation tag and matched budget.

## Best Ablation By Budget

| ablation   |   budget |   h2o_answer_contains |   uckv2_answer_contains |   answer_delta |   h2o_avg_kept_tokens |   uckv2_avg_kept_tokens |   kept_delta |   uckv2_lambda |   uckv2_beta |   uckv2_evict_every |   uckv2_selector_decode_fraction |
|:-----------|---------:|----------------------:|------------------------:|---------------:|----------------------:|------------------------:|-------------:|---------------:|-------------:|--------------------:|---------------------------------:|
| base       |      256 |                0.1667 |                  0.1667 |         0.0000 |              256.0000 |                259.5000 |       3.5000 |         1.0000 |       0.5000 |                  16 |                           0.0941 |

## Matched-Budget Deltas

| ablation   |   budget |   h2o_answer_contains |   uckv2_answer_contains |   answer_delta |   h2o_avg_kept_tokens |   uckv2_avg_kept_tokens |   kept_delta |   uckv2_lambda |   uckv2_beta |   uckv2_evict_every |   uckv2_selector_decode_fraction |
|:-----------|---------:|----------------------:|------------------------:|---------------:|----------------------:|------------------------:|-------------:|---------------:|-------------:|--------------------:|---------------------------------:|
| base       |      256 |                0.1667 |                  0.1667 |         0.0000 |              256.0000 |                259.5000 |       3.5000 |         1.0000 |       0.5000 |                  16 |                           0.0941 |
| beta0      |      256 |                0.1667 |                  0.1667 |         0.0000 |              256.0000 |                259.5000 |       3.5000 |         1.0000 |       0.0000 |                  16 |                           0.1389 |

## Policy Summary

| ablation   | policy          |   budget |   uckv2_lambda |   uckv2_beta |   uckv2_evict_every |   uckv2_min_recent |   uckv2_recent_fraction |   runs |   answer_contains |   answer_contains_se |   avg_kept_tokens |   decode_tokens_per_s |   selector_decode_fraction |
|:-----------|:----------------|---------:|---------------:|-------------:|--------------------:|-------------------:|------------------------:|-------:|------------------:|---------------------:|------------------:|----------------------:|---------------------------:|
| base       | h2o_hh_256      | 256.0000 |         1.0000 |       0.5000 |                  16 |                 64 |                  0.2500 |      1 |            0.1667 |                  nan |          256.0000 |                4.8349 |                   nan      |
| base       | uckv2_fixed_256 | 256.0000 |         1.0000 |       0.5000 |                  16 |                 64 |                  0.2500 |      1 |            0.1667 |                  nan |          259.5000 |                3.6957 |                     0.0941 |
| base       | full            | nan      |         1.0000 |       0.5000 |                  16 |                 64 |                  0.2500 |      1 |            0.1667 |                  nan |         1333.3333 |                5.9852 |                   nan      |
| beta0      | h2o_hh_256      | 256.0000 |         1.0000 |       0.0000 |                  16 |                 64 |                  0.2500 |      1 |            0.1667 |                  nan |          256.0000 |                4.7334 |                   nan      |
| beta0      | uckv2_fixed_256 | 256.0000 |         1.0000 |       0.0000 |                  16 |                 64 |                  0.2500 |      1 |            0.1667 |                  nan |          259.5000 |                3.8440 |                     0.1389 |
| beta0      | full            | nan      |         1.0000 |       0.0000 |                  16 |                 64 |                  0.2500 |      1 |            0.1667 |                  nan |         1333.3333 |                5.5476 |                   nan      |
