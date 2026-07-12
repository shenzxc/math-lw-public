# Long-Context Matched Tolerance Selection

Selection rule: `match_full_accuracy_then_min_tokens`
Full-cache evaluation accuracy: `0.972222`
Selected tolerance: `0.35`

## Actual UCKV Tolerance Sweep

|   uckv_risk_tolerance |   answer_contains |   avg_kept_tokens |      avg_kl |   top1_mismatch_rate |   fallback_steps |
|----------------------:|------------------:|------------------:|------------:|---------------------:|-----------------:|
|                  0.05 |          0.972222 |           1141.94 | 0           |            0         |                0 |
|                  0.1  |          0.972222 |           1141.94 | 0           |            0         |                0 |
|                  0.15 |          0.972222 |           1141.94 | 0           |            0         |                0 |
|                  0.2  |          0.972222 |           1141.94 | 0           |            0         |                0 |
|                  0.25 |          0.972222 |           1138.49 | 7.75565e-05 |            0         |                0 |
|                  0.3  |          0.972222 |           1138.44 | 7.91174e-05 |            0         |                0 |
|                  0.35 |          0.972222 |           1138.4  | 9.86136e-05 |            0         |                0 |
|                  0.4  |          0.805556 |           1028.58 | 0.0540961   |            0.0150463 |                0 |
|                  0.45 |          0.805556 |           1021.86 | 0.0807619   |            0.0219907 |                0 |
|                  0.5  |          0.805556 |           1020.65 | 0.0945442   |            0.025463  |                0 |
