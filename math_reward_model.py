import torch
import re
from fractions import Fraction
from gspo.reward_model import BaseRewardModel
from typing import List

class MathRewardModel(BaseRewardModel):
    def __init__(self, tolerance=1e-5):
        super().__init__()
        self.tolerance = tolerance

    def __call__(self, queries: List[str], responses: List[str]) -> torch.Tensor:
        rewards = []
        for query, response in zip(queries, responses):
            # Extract ground truth from query (assuming query contains [answer])
            gt_match = re.search(r'\[([^\]]+)\]', query)
            if not gt_match:
                rewards.append(-1.0)
                continue

            gt_str = gt_match.group(1)
            try:
                # Handle different answer formats (int, float, fraction)
                if '/' in gt_str:
                    ground_truth = float(Fraction(gt_str))
                else:
                    ground_truth = float(gt_str)
            except:
                rewards.append(-1.0)
                continue

            # Check formatting
            has_thinking = '<thinking>' in response and '</thinking>' in response
            bracket_match = re.search(r'\[([^\]]+)\]', response)
            has_brackets = bool(bracket_match)

            # Formatting reward components
            formatting_reward = 0.5 * has_thinking + 0.5 * has_brackets

            # Correctness reward
            correctness_reward = -1.0  # Default to incorrect
            if bracket_match:
                try:
                    ans_str = bracket_match.group(1)
                    if '/' in ans_str:
                        answer = float(Fraction(ans_str))
                    else:
                        answer = float(ans_str)

                    # Cap the error to avoid extreme values
                    error = min(abs(answer - ground_truth), 10.0)
                    correctness_reward = 1.0 - min(error / (abs(ground_truth) + 1e-5), 1.0)
                except:
                    pass

            total_reward = formatting_reward + correctness_reward
            rewards.append(total_reward)

        return torch.tensor(rewards, dtype=torch.float32)
