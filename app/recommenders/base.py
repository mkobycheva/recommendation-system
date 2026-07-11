"""Abstract interface for cold-start recommenders used by the app backend."""

from abc import ABC, abstractmethod


class Recommender(ABC):
    @abstractmethod
    def recommend(self, selected_items: list[str], target_domain: str, k: int = 10) -> list[str]:
        """selected_items -- item_id with a domain prefix (as in add_domain_item_ids).

        Returns a ranked list of item_id of length <= k, only from target_domain,
        excluding selected_items from the output.
        """
        raise NotImplementedError
