# UCKV2 L1 Matched-Budget Results

Evaluation split only. Scores are averaged over completed seeds.

## Policy Summary

| policy          |   seeds |   answer_contains |   answer_contains_se |   min_answer_contains |   max_answer_contains |   avg_kept_tokens |   decode_tokens_per_s |   selector_decode_fraction |
|:----------------|--------:|------------------:|---------------------:|----------------------:|----------------------:|------------------:|----------------------:|---------------------------:|
| full            |       5 |            0.9458 |               0.0141 |              0.895833 |              0.979167 |           285.95  |                7.8686 |                            |
| h2o_hh_96       |       5 |            0.4917 |               0.0106 |              0.458333 |              0.520833 |            96     |                6.9791 |                            |
| uckv2_fixed_96  |       5 |            0.4667 |               0.0051 |              0.458333 |              0.479167 |           101.875 |                7.4136 |                     0.0446 |
| h2o_hh_128      |       5 |            0.4917 |               0.0106 |              0.458333 |              0.520833 |           128     |                7.1365 |                            |
| uckv2_fixed_128 |       5 |            0.5167 |               0.0179 |              0.479167 |              0.583333 |           133.875 |                7.49   |                     0.0405 |
| h2o_hh_192      |       5 |            0.5958 |               0.0193 |              0.541667 |              0.645833 |           187.804 |                7.2261 |                            |
| uckv2_fixed_192 |       5 |            0.6042 |               0.0174 |              0.5625   |              0.645833 |           192.21  |                7.4429 |                     0.0468 |
| h2o_hh_256      |       5 |            0.6708 |               0.0153 |              0.625    |              0.708333 |           233.95  |                7.3253 |                            |
| uckv2_fixed_256 |       5 |            0.8708 |               0.0167 |              0.8125   |              0.916667 |           237.065 |                7.3968 |                     0.0523 |

## Matched-Budget Deltas

|   budget |   h2o_answer_contains |   uckv2_answer_contains |   answer_delta |   h2o_avg_kept_tokens |   uckv2_avg_kept_tokens |   kept_delta |   h2o_decode_tokens_per_s |   uckv2_decode_tokens_per_s |   decode_tokens_per_s_delta |   uckv2_selector_decode_fraction |
|---------:|----------------------:|------------------------:|---------------:|----------------------:|------------------------:|-------------:|--------------------------:|----------------------------:|----------------------------:|---------------------------------:|
|       96 |                0.4917 |                  0.4667 |        -0.025  |                96     |                 101.875 |       5.875  |                   6.97907 |                     7.41361 |                      0.4345 |                           0.0446 |
|      128 |                0.4917 |                  0.5167 |         0.025  |               128     |                 133.875 |       5.875  |                   7.13649 |                     7.49005 |                      0.3536 |                           0.0405 |
|      192 |                0.5958 |                  0.6042 |         0.0083 |               187.804 |                 192.21  |       4.4062 |                   7.22614 |                     7.44285 |                      0.2167 |                           0.0468 |
|      256 |                0.6708 |                  0.8708 |         0.2    |               233.95  |                 237.065 |       3.1142 |                   7.3253  |                     7.39676 |                      0.0715 |                           0.0523 |

## Paired Prompt Counts

|   budget |   paired_prompts |   both_correct |   h2o_only |   uckv2_only |   both_wrong |   paired_answer_delta |   paired_answer_delta_ci95_low |   paired_answer_delta_ci95_high |   mcnemar_exact_p |
|---------:|-----------------:|---------------:|-----------:|-------------:|-------------:|----------------------:|-------------------------------:|--------------------------------:|------------------:|
|       96 |              240 |            112 |          6 |            0 |          122 |           -0.025      |                    -0.0458333  |                     -0.00833333 |       0.03125     |
|      128 |              240 |            117 |          1 |            7 |          115 |            0.025      |                     0.00416667 |                      0.05       |       0.0703125   |
|      192 |              240 |            143 |          0 |            2 |           95 |            0.00833333 |                     0          |                      0.0208333  |       0.5         |
|      256 |              240 |            161 |          0 |           48 |           31 |            0.2        |                     0.15       |                      0.254167   |       7.10543e-15 |

## Main Takeaway

UCKV2 has mixed matched-budget behavior in this run. The best point is budget 256 with answer-containment delta +0.2000; the weakest point is budget 96 with delta -0.0250. Positive local gates should therefore be confirmed on larger models before being used as a paper claim. The paired comparison at budget 256 has 240 prompts, H2O-only correct=0, UCKV2-only correct=48, and McNemar p=7.11e-15.
