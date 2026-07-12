import unittest

import torch

from app.recommenders.sequential import (
    BERT4RecModel,
    BERT4RecRecommender,
    SASRecModel,
    SASRecRecommender,
)

BOOK_ITEMS = [f"books::b{i}" for i in range(7)]
MOVIE_ITEMS = [f"movies::m{i}" for i in range(5)]
ALL_ITEMS = BOOK_ITEMS + MOVIE_ITEMS
NUM_ITEMS = len(ALL_ITEMS) + 1  # +1 for padding, matching NUM_ITEMS = len(encoder.classes_) + 1
MAX_LEN = 8


def make_item2idx():
    return {item_id: idx + 1 for idx, item_id in enumerate(ALL_ITEMS)}  # 0 reserved for padding


def make_domain_item_indices(item2idx):
    return {
        "books": [item2idx[item_id] for item_id in BOOK_ITEMS],
        "movies": [item2idx[item_id] for item_id in MOVIE_ITEMS],
    }


class SASRecRecommenderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.item2idx = make_item2idx()
        self.domain_item_indices = make_domain_item_indices(self.item2idx)
        model = SASRecModel(num_items=NUM_ITEMS, max_len=MAX_LEN, d_model=8, n_heads=2, n_layers=1, dropout=0.0)
        self.recommender = SASRecRecommender(model, self.item2idx, self.domain_item_indices, MAX_LEN)

    def test_recommendations_are_from_target_domain_only(self):
        result = self.recommender.recommend(["books::b0", "books::b1"], target_domain="movies", k=10)

        self.assertTrue(result)
        self.assertTrue(all(item.startswith("movies::") for item in result))

    def test_selected_items_are_excluded(self):
        selected = ["movies::m0", "movies::m1"]

        result = self.recommender.recommend(selected, target_domain="movies", k=10)

        self.assertFalse(set(selected) & set(result))

    def test_unknown_selection_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.recommender.recommend(["books::unknown"], target_domain="books", k=5)

    def test_empty_selection_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.recommender.recommend([], target_domain="books", k=5)

    def test_result_length_at_most_k(self):
        result = self.recommender.recommend(["books::b0"], target_domain="books", k=3)

        self.assertLessEqual(len(result), 3)

    def test_cart_longer_than_max_len_does_not_crash(self):
        selected = BOOK_ITEMS * 3  # 21 items, > MAX_LEN=8

        result = self.recommender.recommend(selected, target_domain="movies", k=5)

        self.assertTrue(all(item.startswith("movies::") for item in result))

    def test_unknown_target_domain_returns_empty(self):
        result = self.recommender.recommend(["books::b0"], target_domain="music", k=5)

        self.assertEqual(result, [])


class BERT4RecRecommenderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.item2idx = make_item2idx()
        self.domain_item_indices = make_domain_item_indices(self.item2idx)
        mask_token = NUM_ITEMS  # matches encoder convention: MASK_TOKEN = NUM_ITEMS
        model = BERT4RecModel(
            num_items=NUM_ITEMS, mask_token=mask_token, max_len=MAX_LEN,
            d_model=8, n_heads=2, n_layers=1, dropout=0.0,
        )
        self.recommender = BERT4RecRecommender(model, self.item2idx, self.domain_item_indices, MAX_LEN, mask_token)

    def test_recommendations_are_from_target_domain_only(self):
        result = self.recommender.recommend(["books::b0", "books::b1"], target_domain="movies", k=10)

        self.assertTrue(result)
        self.assertTrue(all(item.startswith("movies::") for item in result))

    def test_selected_items_are_excluded(self):
        selected = ["movies::m0", "movies::m1"]

        result = self.recommender.recommend(selected, target_domain="movies", k=10)

        self.assertFalse(set(selected) & set(result))

    def test_unknown_selection_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.recommender.recommend(["books::unknown"], target_domain="books", k=5)

    def test_empty_selection_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.recommender.recommend([], target_domain="books", k=5)

    def test_result_length_at_most_k(self):
        result = self.recommender.recommend(["books::b0"], target_domain="books", k=3)

        self.assertLessEqual(len(result), 3)

    def test_cart_longer_than_max_len_does_not_crash(self):
        selected = BOOK_ITEMS * 3  # 21 items, > MAX_LEN=8

        result = self.recommender.recommend(selected, target_domain="movies", k=5)

        self.assertTrue(all(item.startswith("movies::") for item in result))

    def test_unknown_target_domain_returns_empty(self):
        result = self.recommender.recommend(["books::b0"], target_domain="music", k=5)

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
