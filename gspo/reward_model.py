from abc import ABC, abstractmethod
from typing import List
import torch


class BaseRewardModel(ABC):
    """Abstract base class for reward models."""

    @abstractmethod
    def __call__(self, queries: List[str], responses: List[str]) -> torch.Tensor:
        """
        Compute rewards for query-response pairs.

        Args:
            queries: List of query strings
            responses: List of response strings

        Returns:
            Tensor of rewards with shape (batch_size,)
        """
        pass
