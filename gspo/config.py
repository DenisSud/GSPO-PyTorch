from dataclasses import dataclass
from typing import Dict, Any, Union
from pathlib import Path
import json


@dataclass
class GSPOConfig:
    """Configuration class for GSPO training."""

    # Clipping parameters
    clip_range_left: float = 3e-4
    clip_range_right: float = 4e-4

    # Group parameters
    group_size: int = 8

    # Training parameters
    learning_rate: float = 1e-5
    num_epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 0

    # Advantage computation
    normalize_advantages: bool = True
    advantage_eps: float = 1e-8

    # Logging and checkpointing
    logging_steps: int = 10
    save_steps: int = 1000
    eval_steps: int = 500
    output_dir: str = "./gspo_outputs"

    # Model parameters
    max_length: int = 512

    # Optimization
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Scheduler
    lr_scheduler_type: str = "linear"

    # Token-level variant
    use_token_level: bool = False  # Whether to use GSPO-token variant

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}

    def save_pretrained(self, save_directory: Union[str, Path]):
        """Save config to directory."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        config_file = save_directory / "gspo_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_pretrained(cls, load_directory: Union[str, Path]):
        """Load config from directory."""
        load_directory = Path(load_directory)
        config_file = load_directory / "gspo_config.json"

        with open(config_file, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)
