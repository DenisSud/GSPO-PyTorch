"""
Basic example of using GSPO for language model training.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from gspo import GSPOTrainer, GSPOConfig, BaseRewardModel
from torch.utils.data import DataLoader, Dataset


class DummyRewardModel(BaseRewardModel):
    """Example reward model that returns random rewards."""

    def __call__(self, queries, responses):
        # In practice, this would be your actual reward computation
        return torch.rand(len(queries))


class DummyDataset(Dataset):
    """Example dataset for demonstration."""

    def __init__(self, tokenizer, num_samples=100):
        self.tokenizer = tokenizer
        self.num_samples = num_samples

        # Dummy data - replace with your actual data
        self.queries = ["What is the capital of France?"] * num_samples
        self.responses = [
            ["Paris is the capital.", "The capital is Paris.", "It's Paris.", "Paris."]
        ] * (num_samples // 4)

    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx):
        query = self.queries[idx * 4]  # Each query has 4 responses
        responses = self.responses[idx]

        # Tokenize query + responses
        full_texts = [f"{query} {response}" for response in responses]

        # Tokenize
        encoded = self.tokenizer(
            full_texts,
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

        # Dummy rewards
        rewards = torch.rand(4)

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'],
            'response_mask': response_mask,
            'rewards': rewards,
        }


def main():
    # Initialize model and tokenizer
    model_name = "gpt2"  # Replace with your model
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
        output_dir="./example_outputs"
    )

    # Initialize trainer
    reward_model = DummyRewardModel()
    trainer = GSPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=config
    )

    # Create dataset and dataloader
    dataset = DummyDataset(tokenizer, num_samples=20)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Train
    print("Starting GSPO training...")
    trainer.train(dataloader)
    print("Training completed!")


if __name__ == "__main__":
    main()
