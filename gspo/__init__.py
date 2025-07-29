"""
GSPO: Group Sequence Policy Optimization for PyTorch

A PyTorch implementation of Group Sequence Policy Optimization (GSPO) for training
large language models with reinforcement learning.
"""

from .trainer import GSPOTrainer
from .config import GSPOConfig
from .reward_model import BaseRewardModel

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "GSPOTrainer",
    "GSPOConfig",
    "BaseRewardModel",
]
