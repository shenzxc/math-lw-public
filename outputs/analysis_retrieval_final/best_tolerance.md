# Best Tolerance Selection

Selection rule: `target:match_best_fixed_accuracy_then_min_tokens`
Best fixed-window accuracy: `0.930556`
Selected tolerance: `0.36`

## Selected UCKV Row

| run_label                                                      | policy      |   uckv_risk_tolerance |   answer_contains |   avg_kept_tokens |     avg_kl |   top1_mismatch_rate |   fallback_steps | replay_metric_source       |
|:---------------------------------------------------------------|:------------|----------------------:|------------------:|------------------:|-----------:|---------------------:|-----------------:|:---------------------------|
| final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384_tol036 | uckv_budget |                  0.36 |          0.930556 |           259.212 | 0.00108506 |           0.00405093 |              159 | fixed_candidate_simulation |

## Main Sweep

| run_label                                                       | policy           |   uckv_risk_tolerance |   answer_contains |   avg_kept_tokens |      avg_kl |   top1_mismatch_rate |   fallback_steps | replay_metric_source       |
|:----------------------------------------------------------------|:-----------------|----------------------:|------------------:|------------------:|------------:|---------------------:|-----------------:|:---------------------------|
| final_smollm2_chat_retrieval144_seed29_thr01_tol05_w384         | full             |                 0.5   |          0.930556 |           287.597 | 0           |           0          |                0 | actual_replay              |
| final_smollm2_chat_retrieval144_seed29_thr01_tol05_w384         | sink4_window_128 |                 0.5   |          0.513889 |           132     | 0.374817    |           0.100694   |                0 | actual_replay              |
| final_smollm2_chat_retrieval144_seed29_thr01_tol05_w384         | sink4_window_192 |                 0.5   |          0.736111 |           190.361 | 0.205563    |           0.0584491  |                0 | actual_replay              |
| final_smollm2_chat_retrieval144_seed29_thr01_tol05_w384         | sink4_window_256 |                 0.5   |          0.763889 |           235.964 | 0.129961    |           0.0364583  |                0 | actual_replay              |
| final_smollm2_chat_retrieval144_seed29_thr01_tol05_w384         | sink4_window_320 |                 0.5   |          0.847222 |           267.137 | 0.0646875   |           0.015625   |                0 | actual_replay              |
| final_smollm2_chat_retrieval144_seed29_thr01_tol05_w384         | sink4_window_384 |                 0.5   |          0.930556 |           283.925 | 0.000634874 |           0          |                0 | actual_replay              |
| final_smollm2_chat_retrieval144_seed29_thr01_tol05_w384         | sink4_window_64  |                 0.5   |          0.305556 |            68     | 0.567499    |           0.13831    |                0 | actual_replay              |
| final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384_tol03   | uckv_budget      |                 0.3   |          0.930556 |           283.382 | 0.000635603 |           0          |              411 | fixed_candidate_simulation |
| final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384_tol032  | uckv_budget      |                 0.32  |          0.930556 |           282.446 | 0.000649631 |           0          |              385 | fixed_candidate_simulation |
| final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384_tol034  | uckv_budget      |                 0.34  |          0.930556 |           274.126 | 0.000794023 |           0          |              240 | fixed_candidate_simulation |
| final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384_tol035  | uckv_budget      |                 0.35  |          0.930556 |           266.417 | 0.000925798 |           0          |              181 | fixed_candidate_simulation |
| final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384_tol036  | uckv_budget      |                 0.36  |          0.930556 |           259.212 | 0.00108506  |           0.00405093 |              159 | fixed_candidate_simulation |
| final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384_tol0365 | uckv_budget      |                 0.365 |          0.888889 |           255.511 | 0.00132867  |           0.00520833 |              153 | fixed_candidate_simulation |
| final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384_tol037  | uckv_budget      |                 0.37  |          0.875    |           253.441 | 0.00770662  |           0.00868056 |              151 | fixed_candidate_simulation |
| final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384_tol0375 | uckv_budget      |                 0.375 |          0.847222 |           250.748 | 0.0251676   |           0.0127315  |              150 | fixed_candidate_simulation |
| final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384_tol038  | uckv_budget      |                 0.38  |          0.847222 |           248.59  | 0.0397257   |           0.0173611  |              150 | fixed_candidate_simulation |
| final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384_tol039  | uckv_budget      |                 0.39  |          0.833333 |           243.94  | 0.0499715   |           0.0196759  |              106 | fixed_candidate_simulation |
| final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384_tol04   | uckv_budget      |                 0.4   |          0.791667 |           237.646 | 0.064675    |           0.0231481  |               53 | fixed_candidate_simulation |
| final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384_tol042  | uckv_budget      |                 0.42  |          0.763889 |           233.353 | 0.0984057   |           0.0300926  |               26 | fixed_candidate_simulation |
| final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384_tol044  | uckv_budget      |                 0.44  |          0.763889 |           232.935 | 0.109599    |           0.0324074  |               24 | fixed_candidate_simulation |
| final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384_tol046  | uckv_budget      |                 0.46  |          0.75     |           231.432 | 0.111858    |           0.0335648  |               22 | fixed_candidate_simulation |
| final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384_tol048  | uckv_budget      |                 0.48  |          0.736111 |           206.822 | 0.123793    |           0.0358796  |               22 | fixed_candidate_simulation |
| final_smollm2_chat_retrieval144_seed29_thr01_tol05_w384         | uckv_budget      |                 0.5   |          0.680556 |           190.614 | 0.204961    |           0.0607639  |               20 | actual_replay              |
| final_smollm2_chat_retrieval144_seed29_reuse_thr01_w384_tol05   | uckv_budget      |                 0.5   |          0.680556 |           190.613 | 0.138875    |           0.0399306  |               20 | fixed_candidate_simulation |
