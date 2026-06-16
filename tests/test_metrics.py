import unittest

from src.evaluation.metrics import average_precision_at_k, map_at_k


class MapAtKTest(unittest.TestCase):
    def test_average_precision_perfect_ranking_returns_one(self):
        self.assertEqual(average_precision_at_k(["a", "b"], {"a", "b"}, k=10), 1.0)

    def test_average_precision_partial_hits(self):
        result = average_precision_at_k(["x", "a", "b"], {"a", "b"}, k=10)

        self.assertAlmostEqual(result, (1 / 2 + 2 / 3) / 2)

    def test_average_precision_no_hits_returns_zero(self):
        self.assertEqual(average_precision_at_k(["x", "y"], {"a"}, k=10), 0.0)

    def test_average_precision_duplicate_recommendations_count_once(self):
        result = average_precision_at_k(["a", "a", "b"], {"a", "b"}, k=10)

        self.assertEqual(result, 1.0)

    def test_average_precision_no_relevant_items_returns_zero(self):
        self.assertEqual(average_precision_at_k(["a", "b"], set(), k=10), 0.0)

    def test_map_at_k_averages_multiple_users(self):
        recommended = {
            "u1": ["a", "b"],
            "u2": ["x", "c"],
            "u3": ["z"],
        }
        relevant = {
            "u1": {"a", "b"},
            "u2": {"c"},
            "u3": set(),
        }

        self.assertAlmostEqual(map_at_k(recommended, relevant, k=10), (1.0 + 0.5 + 0.0) / 3)


if __name__ == "__main__":
    unittest.main()
