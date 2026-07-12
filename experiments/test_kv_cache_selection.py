#!/usr/bin/env python3

import unittest

import torch

from experiments.run_kv_cache_pilot import (
    select_score_heavy_hitter_indices,
    select_uckv2_heavy_hitter_indices,
)


class HeavyHitterSelectionTest(unittest.TestCase):
    def test_shared_contract_produces_identical_selection(self) -> None:
        positions = list(range(20))
        scores = torch.tensor(
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0] * 2
        )
        kwargs = {
            "sink": 2,
            "budget": 8,
            "current_abs_pos": 19,
            "min_recent": 3,
            "recent_fraction": 0.25,
        }

        baseline = select_score_heavy_hitter_indices(
            scores,
            positions,
            **kwargs,
        )
        uckv2 = select_uckv2_heavy_hitter_indices(
            scores,
            positions,
            **kwargs,
        )

        self.assertEqual(baseline, uckv2)
        self.assertEqual(len(baseline), kwargs["budget"])

    def test_selection_respects_hard_post_eviction_budget(self) -> None:
        positions = list(range(30))
        scores = torch.arange(30, dtype=torch.float32)

        selected = select_score_heavy_hitter_indices(
            scores,
            positions,
            sink=4,
            budget=10,
            current_abs_pos=29,
            min_recent=4,
            recent_fraction=0.4,
        )

        self.assertEqual(len(selected), 10)
        self.assertTrue(set(range(4)).issubset(selected))
        self.assertTrue(set(range(26, 30)).issubset(selected))

    def test_mandatory_overflow_still_returns_exact_budget(self) -> None:
        positions = list(range(12))
        scores = torch.zeros(12)

        selected = select_score_heavy_hitter_indices(
            scores,
            positions,
            sink=4,
            budget=5,
            current_abs_pos=11,
            min_recent=5,
            recent_fraction=1.0,
        )

        self.assertEqual(selected, [7, 8, 9, 10, 11])


if __name__ == "__main__":
    unittest.main()
