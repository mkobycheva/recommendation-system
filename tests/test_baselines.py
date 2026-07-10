import unittest

import numpy as np
import pandas as pd

from src.evaluation.baselines import make_popularity_recommender, make_random_recommender


def _domain_item_indices():
    return {
        "books": np.array([0, 1, 2], dtype=np.int64),
        "movies": np.array([3, 4], dtype=np.int64),
    }


def _idx2item():
    return {
        0: "books::B1",
        1: "books::B2",
        2: "books::B3",
        3: "movies::M1",
        4: "movies::M2",
    }


def _train_df():
    # books::B1 x3, books::B2 x2, books::B3 x1, movies::M1 x2, movies::M2 x1
    return pd.DataFrame({
        "item_id": (
            ["books::B1"] * 3
            + ["books::B2"] * 2
            + ["books::B3"] * 1
            + ["movies::M1"] * 2
            + ["movies::M2"] * 1
        )
    })


class PopularityRecommenderTest(unittest.TestCase):
    def test_returns_most_frequent_items_in_order(self):
        recommend = make_popularity_recommender(_train_df(), _domain_item_indices(), _idx2item())

        result = recommend(["u1", "u2"], "books", k=2)

        self.assertEqual(result["u1"], ["books::B1", "books::B2"])
        self.assertEqual(result["u2"], ["books::B1", "books::B2"])

    def test_returns_fewer_than_k_if_domain_has_fewer_items(self):
        recommend = make_popularity_recommender(_train_df(), _domain_item_indices(), _idx2item())

        result = recommend(["u1"], "movies", k=10)

        self.assertEqual(result["u1"], ["movies::M1", "movies::M2"])

    def test_same_recommendations_regardless_of_user(self):
        recommend = make_popularity_recommender(_train_df(), _domain_item_indices(), _idx2item())

        result = recommend(["u1", "u2", "u3"], "books", k=3)

        self.assertEqual(result["u1"], result["u2"])
        self.assertEqual(result["u2"], result["u3"])


class RandomRecommenderTest(unittest.TestCase):
    def test_returns_k_valid_domain_items_without_duplicates(self):
        recommend = make_random_recommender(_domain_item_indices(), _idx2item(), seed=42)

        result = recommend(["u1", "u2"], "books", k=2)

        books_items = {"books::B1", "books::B2", "books::B3"}
        for items in result.values():
            self.assertEqual(len(items), 2)
            self.assertEqual(len(set(items)), 2)
            self.assertTrue(set(items).issubset(books_items))

    def test_returns_fewer_than_k_if_domain_has_fewer_items(self):
        recommend = make_random_recommender(_domain_item_indices(), _idx2item(), seed=42)

        result = recommend(["u1"], "movies", k=10)

        self.assertEqual(len(result["u1"]), 2)
        self.assertEqual(set(result["u1"]), {"movies::M1", "movies::M2"})

    def test_deterministic_for_same_seed(self):
        recommend_a = make_random_recommender(_domain_item_indices(), _idx2item(), seed=42)
        recommend_b = make_random_recommender(_domain_item_indices(), _idx2item(), seed=42)

        result_a = recommend_a(["u1", "u2"], "books", k=2)
        result_b = recommend_b(["u1", "u2"], "books", k=2)

        self.assertEqual(result_a, result_b)


if __name__ == "__main__":
    unittest.main()
