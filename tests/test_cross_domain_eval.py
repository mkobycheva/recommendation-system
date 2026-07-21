import unittest

import pandas as pd

from src.evaluation.cross_domain_eval import evaluate_multi_domain


class EvaluateMultiDomainTest(unittest.TestCase):
    def _split_dfs(self):
        valid_df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "item_id": ["books::B1", "movies::M1"],
            "domain": ["books", "movies"],
        })
        test_df = pd.DataFrame({
            "user_id": ["u1", "u2"],
            "item_id": ["books::B2", "movies::M2"],
            "domain": ["books", "movies"],
        })
        return {"valid": valid_df, "test": test_df}

    def _recommend_func(self, user_ids, target_domain, k=10, **kwargs):
        if target_domain == "books":
            items = ["books::B1", "books::B2", "books::B3"]
        else:
            items = ["movies::M1", "movies::M2", "movies::M3"]
        return {user_id: items[:k] for user_id in user_ids}

    def test_default_keys_are_ndcg10_recall10_map5(self):
        valid_results, test_results = evaluate_multi_domain(
            self._split_dfs(), self._recommend_func
        )

        for results in (valid_results, test_results):
            for domain in ("books", "movies"):
                self.assertEqual(
                    set(results[domain].keys()),
                    {"ndcg@10", "recall@10", "map@5", "n_users"},
                )

    def test_custom_k_and_map_k_change_keys(self):
        valid_results, _ = evaluate_multi_domain(
            self._split_dfs(), self._recommend_func, k=20, map_k=3
        )

        self.assertEqual(
            set(valid_results["books"].keys()),
            {"ndcg@20", "recall@20", "map@3", "n_users"},
        )

    def test_recommend_func_is_called_with_max_of_k_and_map_k(self):
        seen_k = {}

        def recording_recommend_func(user_ids, target_domain, k=10, **kwargs):
            seen_k[target_domain] = k
            return {user_id: [] for user_id in user_ids}

        evaluate_multi_domain(self._split_dfs(), recording_recommend_func, k=10, map_k=5)

        self.assertEqual(seen_k["books"], 10)
        self.assertEqual(seen_k["movies"], 10)

        seen_k.clear()
        evaluate_multi_domain(self._split_dfs(), recording_recommend_func, k=5, map_k=20)

        self.assertEqual(seen_k["books"], 20)
        self.assertEqual(seen_k["movies"], 20)


if __name__ == "__main__":
    unittest.main()
