import unittest

import pandas as pd

import nfl_week_analysis as analysis


class WeekAnalysisTests(unittest.TestCase):
    def test_week_groups_cover_requested_ranges_without_overlap(self):
        for week in range(1, 19):
            matches = [
                label
                for label, start, end in analysis.WEEK_GROUPS
                if start <= week <= end
            ]
            self.assertEqual(len(matches), 1)

    def test_group_metrics_keep_pushes_out_of_ats_win_percentage(self):
        frame = pd.DataFrame(
            [
                {
                    "season": 2024,
                    "week": 17,
                    "straight_up_decision": True,
                    "straight_up_correct": True,
                    "absolute_margin_error": 7.0,
                    "model_ats_result": "WIN",
                    "absolute_spread_disagreement": 3.0,
                },
                {
                    "season": 2024,
                    "week": 18,
                    "straight_up_decision": True,
                    "straight_up_correct": False,
                    "absolute_margin_error": 14.0,
                    "model_ats_result": "PUSH",
                    "absolute_spread_disagreement": 5.0,
                },
                {
                    "season": 2024,
                    "week": 16,
                    "straight_up_decision": True,
                    "straight_up_correct": True,
                    "absolute_margin_error": 3.0,
                    "model_ats_result": "LOSS",
                    "absolute_spread_disagreement": 1.0,
                },
            ]
        )

        result = analysis.build_week_analysis(frame)
        late = result.loc[
            (result["record_type"] == "overall_week_group")
            & (result["week_group"] == "Weeks 17-18")
        ].iloc[0]
        comparison = result.loc[
            result["record_type"] == "late_vs_earlier_comparison"
        ].iloc[0]

        self.assertEqual(late["game_count"], 2)
        self.assertEqual(late["ats_wins"], 1)
        self.assertEqual(late["ats_losses"], 0)
        self.assertEqual(late["ats_pushes"], 1)
        self.assertEqual(late["ats_win_percentage"], 1.0)
        self.assertAlmostEqual(late["average_absolute_margin_error"], 10.5)
        self.assertEqual(late["average_absolute_spread_disagreement"], 4.0)
        self.assertEqual(late["median_absolute_spread_disagreement"], 4.0)
        self.assertIn("late 2", comparison["comparison_sample_sizes"])

    def test_wilson_interval_handles_empty_and_finite_samples(self):
        self.assertTrue(pd.isna(analysis._wilson_interval(0, 0)[0]))
        lower, upper = analysis._wilson_interval(5, 10)
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)


if __name__ == "__main__":
    unittest.main()
