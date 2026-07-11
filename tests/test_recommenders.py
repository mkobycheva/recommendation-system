import unittest

import numpy as np
from gensim.models import Word2Vec

from app.recommenders.item2vec import Item2VecRecommender
from app.recommenders.svd_als import FoldInRecommender

BOOK_ITEMS = [f"books::b{i}" for i in range(7)]
MOVIE_ITEMS = [f"movies::m{i}" for i in range(5)]
ALL_ITEMS = BOOK_ITEMS + MOVIE_ITEMS


def make_item2idx():
    return {item_id: idx for idx, item_id in enumerate(ALL_ITEMS)}


def make_domain_item_indices(item2idx):
    return {
        "books": [item2idx[item_id] for item_id in BOOK_ITEMS],
        "movies": [item2idx[item_id] for item_id in MOVIE_ITEMS],
    }


def make_item_factors(seed=0, n_factors=8):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(len(ALL_ITEMS), n_factors)).astype(np.float32)


class FoldInRecommenderTest(unittest.TestCase):
    def setUp(self):
        self.item2idx = make_item2idx()
        self.domain_item_indices = make_domain_item_indices(self.item2idx)
        self.item_factors = make_item_factors()
        self.recommender = FoldInRecommender(self.item_factors, self.item2idx, self.domain_item_indices)

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

    def test_partial_unknown_selection_still_works(self):
        result = self.recommender.recommend(["books::b0", "books::unknown"], target_domain="movies", k=5)

        self.assertTrue(all(item.startswith("movies::") for item in result))

    def test_unknown_target_domain_returns_empty(self):
        result = self.recommender.recommend(["books::b0"], target_domain="music", k=5)

        self.assertEqual(result, [])


def make_word2vec_model():
    rng = np.random.default_rng(1)
    # Guarantee every item appears at least once regardless of min_count.
    sequences = [ALL_ITEMS]
    for _ in range(50):
        seq_len = rng.integers(3, 6)
        sequences.append(rng.choice(ALL_ITEMS, size=seq_len, replace=False).tolist())

    return Word2Vec(
        sentences=sequences, vector_size=8, window=3, sg=1, min_count=1, workers=1, seed=1,
    )


def make_item2vec_domain_indices(model):
    return {
        "books": [model.wv.key_to_index[i] for i in model.wv.index_to_key if i.startswith("books::")],
        "movies": [model.wv.key_to_index[i] for i in model.wv.index_to_key if i.startswith("movies::")],
    }


class Item2VecRecommenderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = make_word2vec_model()
        cls.domain_item_indices = make_item2vec_domain_indices(cls.model)

    def setUp(self):
        self.recommender = Item2VecRecommender(self.model, self.domain_item_indices)

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
            self.recommender.recommend(["books::totally_unknown_item"], target_domain="books", k=5)

    def test_empty_selection_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.recommender.recommend([], target_domain="books", k=5)

    def test_result_length_at_most_k(self):
        result = self.recommender.recommend(["books::b0"], target_domain="books", k=2)

        self.assertLessEqual(len(result), 2)

    def test_unknown_target_domain_returns_empty(self):
        result = self.recommender.recommend(["books::b0"], target_domain="music", k=5)

        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
