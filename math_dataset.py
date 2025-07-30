import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np

class MathDataset(Dataset):
    def __init__(self, csv_path, tokenizer, split='train', test_size=100):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(csv_path)

        # Split data
        if split == 'train':
            self.data = self.data[:-test_size]
        else:  # test
            self.data = self.data[-test_size:]

        # Pre-calculate queries
        self.queries = [
            f"Solve: {row['task']} \nReason step-by-step and put your final answer in brackets."
            for _, row in self.data.iterrows()
        ]

        # Tokenize all queries upfront
        self.tokenized_inputs = tokenizer(
            self.queries,
            padding=False,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors=None
        )

        # Create response masks (1 for response tokens, 0 for query tokens)
        self.response_masks = []
        for input_ids in self.tokenized_inputs['input_ids']:
            # Find the position where the response starts
            # This is a heuristic: response starts after "Solve: ... \n"
            response_start = self.queries[0].find('\n') + 1
            if response_start == 0:
                response_start = len(self.queries[0])  # fallback: no response tokens

            # Tokenize the query prefix
            prefix = self.queries[0][:response_start]
            prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)

            # Create mask (1 for tokens after prefix)
            mask = [0] * len(prefix_tokens) + [1] * (len(input_ids) - len(prefix_tokens))
            self.response_masks.append(mask)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.tokenized_inputs['input_ids'][idx],
            'attention_mask': self.tokenized_inputs['attention_mask'][idx],
            'response_mask': self.response_masks[idx],
            'query': self.queries[idx],
            'answer': self.data.iloc[idx]['answer']
        }
