from gspo.config import GSPOConfig
from gspo.trainer import GSPOTrainer
from math_reward_model import MathRewardModel
from math_dataset import MathDataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, random_split
import logging



# Setup logging
logging.basicConfig(level=logging.INFO)

def main():
    # Configuration
    config = GSPOConfig(
        group_size=8,
        learning_rate=1e-5,
        num_epochs=3,
        batch_size=8,
        max_length=512,
        report_benchmarks=True,
        benchmark_warmup_steps=10,
        logging_steps=10,
        output_dir="./math_gspo_output"
    )

    # Load model and tokenizer
    model_name = "models/base/Qwen3-0.6B"  # Using 0.5B as proxy for 0.6B
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Initialize reward model
    reward_model = MathRewardModel()

    # Prepare dataset
    full_dataset = MathDataset("data/train.csv", tokenizer, test_size=100)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    def collate_fn(batch):
        # Pad sequences
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item['input_ids']) for item in batch],
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        )
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item['attention_mask']) for item in batch],
            batch_first=True,
            padding_value=0
        )
        padded_response_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(item['response_mask']) for item in batch],
            batch_first=True,
            padding_value=0
        )

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask,
            'response_mask': padded_response_mask,
            'queries': [item['query'] for item in batch],
            'answers': [item['answer'] for item in batch]
        }

    # Initialize trainer
    trainer = GSPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        tokenizer=tokenizer,
        config=config
    )

    # Update DataLoader to use collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Training
    trainer.train(train_loader)

    # Save final model
    trainer.save_checkpoint("math_gspo_final")

if __name__ == "__main__":
    main()
