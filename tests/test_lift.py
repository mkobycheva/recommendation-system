import unittest

from src.evaluation.lift import compute_lift_table


class ComputeLiftTableTest(unittest.TestCase):
    def test_columns_and_metric_order(self):
        model_results = {"map@5": 0.4, "ndcg@10": 0.5, "recall@10": 0.6, "n_users": 10}
        popularity_results = {"map@5": 0.1, "ndcg@10": 0.2, "recall@10": 0.3, "n_users": 10}
        random_results = {"map@5": 0.05, "ndcg@10": 0.05, "recall@10": 0.05, "n_users": 10}

        table = compute_lift_table(model_results, popularity_results, random_results)

        self.assertEqual(
            list(table.columns),
            ["Metric", "Model", "Top-Popular", "Random", "Lift vs Popular", "Lift vs Random"],
        )
        self.assertEqual(list(table["Metric"]), ["MAP@5", "NDCG@10", "Recall@10"])

    def test_lift_values(self):
        model_results = {"map@5": 0.4, "ndcg@10": 0.5, "recall@10": 0.6}
        popularity_results = {"map@5": 0.2, "ndcg@10": 0.25, "recall@10": 0.3}
        random_results = {"map@5": 0.1, "ndcg@10": 0.1, "recall@10": 0.1}

        table = compute_lift_table(model_results, popularity_results, random_results)
        row = table[table["Metric"] == "MAP@5"].iloc[0]

        self.assertAlmostEqual(row["Lift vs Popular"], 2.0)
        self.assertAlmostEqual(row["Lift vs Random"], 4.0)

    def test_zero_baseline_gives_infinite_lift(self):
        model_results = {"map@5": 0.4}
        popularity_results = {"map@5": 0.0}
        random_results = {"map@5": 0.0}

        table = compute_lift_table(model_results, popularity_results, random_results)
        row = table.iloc[0]

        self.assertEqual(row["Lift vs Popular"], float("inf"))
        self.assertEqual(row["Lift vs Random"], float("inf"))


if __name__ == "__main__":
    unittest.main()
