"""
Basic example of using GSPO for language model training.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from gspo import GSPOTrainer, GSPOConfig, BaseRewardModel
from torch.utils.data import DataLoader, Dataset
import pandas as pd

class MathRewardModel(BaseRewardModel):
    """Reward model for math tasks."""

    def __call__(self, queries, responses):
        # In practice, this would be your actual reward computation
        rewards = []
        for query, response in zip(queries, responses):
            # Extract the answer from the response
            answer = response.split('[')[1].split(']')[0]
            # Compute the reward based on the correctness of the answer
            reward = 1.0 if self.is_correct_answer(query, answer) else 0.0
            rewards.append(reward)
        return torch.tensor(rewards)

    def is_correct_answer(self, query, answer):
        # Implement your logic to check if the answer is correct
        # This is a placeholder implementation
        return True

class MathDataset(Dataset):
    """Dataset for math tasks."""

    def __init__(self, tokenizer, csv_file, num_samples=1000):
        self.tokenizer = tokenizer
        self.num_samples = num_samples

        # Load data from CSV
        data = pd.read_csv(csv_file)
        self.queries = data['task'].tolist()[:num_samples]
        self.responses = data['answer'].tolist()[:num_samples]

    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx):
        query = self.queries[idx]
        response = self.responses[idx]

        # Tokenize query + response
        full_text = f"{query} {response}"

        # Tokenize
        encoded = self.tokenizer(
            full_text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        # Create response masks (1 for response tokens, 0 for query tokens)
        query_encoded = self.tokenizer(query, return_tensors="pt")
        query_length = query_encoded['input_ids'].shape[1]

        response_mask = torch.zeros_like(encoded['input_ids'])
        response_mask[:, query_length:] = 1

        # Compute reward
        reward = self.reward_model([query], [response])

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'response_mask': response_mask,
            'rewards': reward,
        }


def main():
    # Initialize model and tokenizer
    model_name = "Qwen/Qwen3-0.6B"  # Replace with your model
    model = AutoModel.from_pretrained(model_name)
    ref_model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configuration
    config = GSPOConfig(
        group_size=4,
        clip_range_left=3e-4,
        clip_range_right=4e-4,
        learning_rate=1e-5,
        batch_size=1,
        num_epochs=1,
        logging_steps=5,
        output_dir="./GSPO_output"
    )

    # Initialize trainer
    reward_model = MathRewardModel()
    trainer = GSPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=config
    )

    # Create dataset and dataloader
    dataset = MathDataset(tokenizer, "train.csv", num_samples=20)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Train
    print("Starting GSPO training...")
    trainer.train(dataloader)
    print("Training completed!")


if __name__ == "__main__":
    main()
