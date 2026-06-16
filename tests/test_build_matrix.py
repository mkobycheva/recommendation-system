import unittest

import pandas as pd

from src.data.build_matrix import add_domain_item_ids, build_user_item_matrix


class BuildMatrixTest(unittest.TestCase):
    def test_add_domain_item_ids_creates_prefixed_item_ids(self):
        df = pd.DataFrame({"parent_asin": ["B1", "B2"]})

        result = add_domain_item_ids(df, "books")

        self.assertEqual(result["domain"].tolist(), ["books", "books"])
        self.assertEqual(result["item_id"].tolist(), ["books::B1", "books::B2"])
        self.assertNotIn("domain", df.columns)

    def test_build_user_item_matrix_indexes_users_items_and_domains(self):
        train_df = pd.DataFrame(
            {
                "user_id": ["u1", "u1", "u2"],
                "item_id": ["books::B1", "movies::M1", "books::B2"],
                "domain": ["books", "movies", "books"],
            }
        )

        matrix = build_user_item_matrix(train_df, domains=["books", "movies"])

        self.assertEqual(matrix.user_item_train.shape, (2, 3))
        self.assertEqual(matrix.user_item_train.nnz, 3)
        self.assertEqual(set(matrix.user2idx), {"u1", "u2"})
        self.assertEqual(set(matrix.item2idx), {"books::B1", "movies::M1", "books::B2"})

        book_items = {matrix.idx2item[idx] for idx in matrix.domain_item_indices["books"]}
        movie_items = {matrix.idx2item[idx] for idx in matrix.domain_item_indices["movies"]}
        self.assertEqual(book_items, {"books::B1", "books::B2"})
        self.assertEqual(movie_items, {"movies::M1"})

        u1_seen = {matrix.idx2item[idx] for idx in matrix.train_seen_idx_by_user[matrix.user2idx["u1"]]}
        self.assertEqual(u1_seen, {"books::B1", "movies::M1"})

    def test_build_user_item_matrix_requires_domain_column(self):
        train_df = pd.DataFrame({"user_id": ["u1"], "item_id": ["books::B1"]})

        with self.assertRaisesRegex(ValueError, "domain"):
            build_user_item_matrix(train_df)


if __name__ == "__main__":
    unittest.main()
