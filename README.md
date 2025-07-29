
# GSPO (Group Sequence Policy Optimization) Implementation
**⚠️ Warning: This is an experimental implementation. It is not optimized for production use and is not fully featured. Use at your own risk.**

A PyTorch implementation of Group Sequence Policy Optimization (GSPO) for training large language models with reinforcement learning, based on the paper ["Group Sequence Policy Optimization"](https://arxiv.org/abs/2507.18071) by the Qwen Team at Alibaba Inc.

## 🚀 Features

- **Stable Training**: Sequence-level importance sampling eliminates token-level variance issues
- **MoE Support**: Native support for Mixture-of-Experts models without routing replay
- **Flexible Configuration**: Easy-to-use configuration system
- **GSPO-token Variant**: Support for token-level advantage customization
- **Memory Efficient**: Simplified infrastructure compared to traditional PPO

## 📦 Installation

```bash
git clone https://github.com/denissud/gspo-pytorch.git
cd gspo-pytorch
pip install -r requirements.txt
```

Or install directly:
```bash
pip install torch transformers numpy
```

## 🔧 Quick Start

### Basic Usage

```python
from gspo import GSPOTrainer, GSPOConfig, BaseRewardModel
from transformers import AutoModel, AutoTokenizer
import torch

# Load your models
model = AutoModel.from_pretrained("your-model")
ref_model = AutoModel.from_pretrained("your-model")  # Reference model
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Define your reward model
class MyRewardModel(BaseRewardModel):
    def __call__(self, queries, responses):
        # Your reward computation logic
        return torch.tensor([0.8, 0.6, 0.9, 0.7])  # Example rewards

# Configuration
config = GSPOConfig(
    group_size=4,
    clip_range_left=3e-4,
    clip_range_right=4e-4,
    learning_rate=1e-5,
    batch_size=1,
    num_epochs=3
)

# Initialize trainer
reward_model = MyRewardModel()
trainer = GSPOTrainer(
    model=model,
    ref_model=ref_model,
    reward_model=reward_model,
    tokenizer=tokenizer,
    config=config
)

# Train
trainer.train(train_dataloader)
```

### Configuration Options

```python
config = GSPOConfig(
    # Core GSPO parameters
    clip_range_left=3e-4,        # Left clipping range
    clip_range_right=4e-4,       # Right clipping range
    group_size=8,                # Number of responses per query

    # Training parameters
    learning_rate=1e-5,
    num_epochs=1,
    batch_size=1,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,

    # Advanced options
    use_token_level=False,       # Use GSPO-token variant
    normalize_advantages=True,
    advantage_eps=1e-8,

    # I/O
    output_dir="./gspo_outputs",
    logging_steps=10,
    save_steps=1000,
)
```

## 📊 Key Differences from GRPO

| Aspect | GRPO | GSPO |
|--------|------|------|
| Importance Ratio | Token-level | Sequence-level |
| Clipping | Per token | Per sequence |
| Stability | Prone to collapse | Stable training |
| MoE Support | Requires routing replay | Native support |
| Variance | High (accumulates over length) | Low (length-normalized) |

## 🧠 Algorithm Overview

GSPO addresses the fundamental issues in GRPO by:

1. **Sequence-level Importance Sampling**: Using `π_θ(y|x)/π_θ_old(y|x)` instead of token-level ratios
2. **Length Normalization**: Importance ratio is `(π_θ(y|x)/π_θ_old(y|x))^(1/|y|)`
3. **Unified Clipping**: Clips entire sequences rather than individual tokens
4. **Group-based Advantages**: Normalizes rewards within response groups

### Mathematical Formulation

The GSPO objective is:

```
J_GSPO(θ) = E[1/G ∑_{i=1}^G min(s_i(θ) * A_i, clip(s_i(θ), 1-ε, 1+ε) * A_i)]
```

Where:
- `s_i(θ) = (π_θ(y_i|x) / π_θ_old(y_i|x))^(1/|y_i|)` is the sequence importance ratio
- `A_i` is the group-normalized advantage
- `ε` is the clipping range

## 📈 Performance Benefits

Based on the original paper, GSPO shows:
- **Superior training efficiency** compared to GRPO
- **Stable convergence** for large models and long sequences
- **Native MoE support** without complex workarounds
- **Simplified infrastructure** requirements

## 🔬 Advanced Usage

### GSPO-token Variant

For scenarios requiring token-level advantage customization:

```python
config = GSPOConfig(
    use_token_level=True,  # Enable GSPO-token
    # ... other parameters
)

# Your batch should include 'token_advantages'
batch = {
    'input_ids': input_ids,
    'attention_mask': attention_mask,
    'response_mask': response_mask,
    'rewards': rewards,
    'token_advantages': token_level_advantages  # Shape: (batch_size, seq_len)
}
```

### Custom Reward Models

```python
class CustomRewardModel(BaseRewardModel):
    def __init__(self, reward_model_path):
        self.model = AutoModel.from_pretrained(reward_model_path)

    def __call__(self, queries, responses):
        # Your custom reward computation
        inputs = self.tokenizer(
            [f"{q} {r}" for q, r in zip(queries, responses)],
            return_tensors="pt", padding=True, truncation=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            rewards = torch.sigmoid(outputs.logits.squeeze())

        return rewards
```

## 📋 Data Format

Your training data should be formatted as:

```python
batch = {
    'input_ids': torch.tensor,      # Shape: (batch_size, seq_len)
    'attention_mask': torch.tensor, # Shape: (batch_size, seq_len)
    'response_mask': torch.tensor,  # Shape: (batch_size, seq_len) - 1s for response tokens
    'rewards': torch.tensor,        # Shape: (batch_size,) - rewards in [0, 1]
    'token_advantages': torch.tensor  # Optional, for GSPO-token variant
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
git clone https://github.com/denissud/gspo-pytorch.git
cd gspo-pytorch
pip install -e .
pip install -r requirements-dev.txt
```

### Running Tests

```bash
python -m pytest tests/
```

## 📄 Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{zheng2025gspo,
  title={Group Sequence Policy Optimization},
  author={Zheng, Chujie and Liu, Shixuan and Li, Mingze and Chen, Xiong-Hui and Yu, Bowen and Gao, Chang and Dang, Kai and Liu, Yuqiong and Men, Rui and Yang, An and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2507.18071},
  year={2025}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original GSPO paper and implementation by the Qwen Team at Alibaba Inc.
- PyTorch and Transformers libraries for the underlying infrastructure
- The open-source community for continuous feedback and improvements

## 📞 Support

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/denissud/gspo-pytorch/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/denissud/gspo-pytorch/discussions)
- 📧 **Email**: your.email@example.com

---

⭐ **Star this repository** if you find it useful!
