import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict
import time
from contextlib import contextmanager
from .config import GSPOConfig
from .reward_model import BaseRewardModel

logger = logging.getLogger(__name__)


class GSPOTrainer:
    """
    Group Sequence Policy Optimization (GSPO) Trainer.

    Implements the GSPO algorithm from the paper for training language models
    with reinforcement learning.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        reward_model: BaseRewardModel,
        tokenizer,
        config: GSPOConfig,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler = None,
    ):
        """
        Initialize GSPO trainer.

        Args:
            model: The policy model to train
            ref_model: The reference model (frozen copy of initial model)
            reward_model: Model to compute rewards
            tokenizer: Tokenizer for the models
            config: GSPO configuration
            optimizer: Optimizer (created automatically if None)
            lr_scheduler: Learning rate scheduler
        """
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        self.config = config

        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Benchmarking state
        self.benchmark_data = defaultdict(list)
        self.benchmark_count = 0

    @contextmanager
    def benchmark_section(self, name: str):
        """
        Context manager for timing code sections with CUDA synchronization.

        Args:
            name: Name of the section to benchmark
        """
        if not self.config.report_benchmarks:
            yield
            return

        # Skip during warmup
        if self.global_step < self.config.benchmark_warmup_steps:
            yield
            return

        # Initialize events
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Synchronize and record start time
        torch.cuda.synchronize()
        start_event.record()
        cpu_start = time.perf_counter()

        try:
            yield
        finally:
            # Record end time and synchronize
            end_event.record()
            torch.cuda.synchronize()

            # Calculate times
            cuda_time = start_event.elapsed_time(end_event) / 1000.0  # ms to seconds
            cpu_time = time.perf_counter() - cpu_start

            # Store benchmark data
            self.benchmark_data[name].append({
                "cuda": cuda_time,
                "cpu": cpu_time
            })

    def log_benchmarks(self):
        """Log averaged benchmark results for the current accumulation"""
        if not self.benchmark_data:
            return

        logger.info("⏱️ Benchmark Results (avg over last {} steps):".format(
            self.config.logging_steps))

        # Calculate averages
        for section, times in self.benchmark_data.items():
            if times:
                avg_cuda = sum(t['cuda'] for t in times) / len(times)
                avg_cpu = sum(t['cpu'] for t in times) / len(times)
                logger.info(f"  {section}:")
                logger.info(f"    CUDA: {avg_cuda:.6f}s")
                logger.info(f"    CPU:  {avg_cpu:.6f}s")
                logger.info(f"    Samples: {len(times)}")

    def compute_sequence_likelihood_ratio(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sequence-level importance ratio s_i(θ) as defined in Equation (7).

        Args:
            input_ids: Token IDs with shape (batch_size, seq_len)
            attention_mask: Attention mask with shape (batch_size, seq_len)
            response_mask: Mask indicating response tokens with shape (batch_size, seq_len)

        Returns:
            Sequence importance ratios with shape (batch_size,)
        """
        batch_size = input_ids.shape[0]

        with torch.no_grad():
            # Get reference model logits
            ref_outputs = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            ref_logits = ref_outputs.logits

        # Get current model logits
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits

        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_response_mask = response_mask[..., 1:].contiguous()

        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        ref_log_probs = F.log_softmax(shift_ref_logits, dim=-1)

        # Gather log probabilities for the actual tokens
        log_probs_tokens = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        ref_log_probs_tokens = torch.gather(
            ref_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Compute log ratio for response tokens only
        log_ratio = (log_probs_tokens - ref_log_probs_tokens) * shift_response_mask

        # Sum over sequence length and normalize by response length
        response_lengths = shift_response_mask.sum(dim=-1)
        sequence_log_ratio = log_ratio.sum(dim=-1) / (response_lengths + 1e-8)

        # Convert to importance ratio with length normalization
        importance_ratio = torch.exp(sequence_log_ratio)

        return importance_ratio

    def compute_token_level_importance_ratio(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
        sequence_importance_ratio: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute token-level importance ratio for GSPO-token variant.

        Args:
            input_ids: Token IDs with shape (batch_size, seq_len)
            attention_mask: Attention mask
            response_mask: Response token mask
            sequence_importance_ratio: Pre-computed sequence importance ratios

        Returns:
            Token importance ratios with shape (batch_size, seq_len)
        """
        # Get current model logits (no gradients for the denominator)
        with torch.no_grad():
            outputs_detached = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits_detached = outputs_detached.logits

        # Get current model logits (with gradients for numerator)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits

        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_logits_detached = logits_detached[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Compute probabilities
        probs = F.softmax(shift_logits, dim=-1)
        probs_detached = F.softmax(shift_logits_detached, dim=-1)

        # Gather probabilities for actual tokens
        probs_tokens = torch.gather(
            probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        probs_tokens_detached = torch.gather(
            probs_detached, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Token importance ratio (numerically equals sequence ratio)
        token_ratio = sequence_importance_ratio.unsqueeze(-1) * (
            probs_tokens / (probs_tokens_detached + 1e-8)
        )

        return token_ratio

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        group_size: int
    ) -> torch.Tensor:
        """
        Compute group-based advantages as in Equation (6).

        Args:
            rewards: Rewards with shape (batch_size,)
            group_size: Number of responses per query

        Returns:
            Advantages with shape (batch_size,)
        """
        batch_size = rewards.shape[0]
        num_groups = batch_size / group_size

        # Reshape to group format
        rewards_grouped = rewards.view(num_groups, group_size)

        # Compute group statistics
        group_means = rewards_grouped.mean(dim=1, keepdim=True)
        group_stds = rewards_grouped.std(dim=1, keepdim=True, unbiased=False)

        # Compute advantages
        advantages_grouped = (rewards_grouped - group_means) / (
            group_stds + self.config.advantage_eps
        )

        # Reshape back
        advantages = advantages_grouped.view(batch_size)

        if self.config.normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + self.config.advantage_eps)

        return advantages

    def compute_gspo_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: torch.Tensor,
        advantages: torch.Tensor,
        token_level_advantages: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GSPO loss.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            response_mask: Response token mask
            advantages: Sequence-level advantages
            token_level_advantages: Token-level advantages for GSPO-token variant

        Returns:
            Loss tensor and metrics dictionary
        """
        # Compute sequence importance ratios
        importance_ratios = self.compute_sequence_likelihood_ratio(
            input_ids, attention_mask, response_mask
        )

        if self.config.use_token_level and token_level_advantages is not None:
            # GSPO-token variant
            token_importance_ratios = self.compute_token_level_importance_ratio(
                input_ids, attention_mask, response_mask, importance_ratios
            )

            # Apply clipping to token ratios
            clipped_ratios = torch.clamp(
                token_importance_ratios,
                min=1.0 - self.config.clip_range_left,
                max=1.0 + self.config.clip_range_right
            )

            # Compute token-level objectives
            obj1 = token_importance_ratios * token_level_advantages.unsqueeze(-1)
            obj2 = clipped_ratios * token_level_advantages.unsqueeze(-1)

            # Take minimum and average over response tokens
            response_lengths = response_mask[:, 1:].sum(dim=-1)
            policy_loss = -torch.min(obj1, obj2)
            policy_loss = (policy_loss * response_mask[:, 1:]).sum(dim=-1) / (response_lengths + 1e-8)
            policy_loss = policy_loss.mean()
        else:
            # Standard GSPO
            # Apply clipping to sequence ratios
            clipped_ratios = torch.clamp(
                importance_ratios,
                min=1.0 - self.config.clip_range_left,
                max=1.0 + self.config.clip_range_right
            )

            # Compute sequence-level objectives
            obj1 = importance_ratios * advantages
            obj2 = clipped_ratios * advantages

            policy_loss = -torch.min(obj1, obj2).mean()

        # Compute metrics
        metrics = {
            'policy_loss': policy_loss.item(),
            'importance_ratio_mean': importance_ratios.mean().item(),
            'importance_ratio_std': importance_ratios.std().item(),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
            'clipped_fraction': (
                (importance_ratios < (1.0 - self.config.clip_range_left)) |
                (importance_ratios > (1.0 + self.config.clip_range_right))
            ).float().mean().item()
        }

        return policy_loss, metrics

    def training_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform a single training step.

        Args:
            batch: Batch containing 'input_ids', 'attention_mask', 'queries'

        Returns:
            Loss and metrics
        """
        # Generate responses
        with self.benchmark_section("response_generation"):
            # Generate responses for each query
            generated = self.model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=self.config.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                num_return_sequences=self.config.group_size
            )

            # Create response mask
            query_lengths = batch['attention_mask'].sum(dim=1)
            response_mask = torch.zeros_like(generated, dtype=torch.float32)
            for i, q_len in enumerate(query_lengths):
                # For each group member
                for j in range(self.config.group_size):
                    idx = i * self.config.group_size + j
                    response_mask[idx, q_len:] = 1

            # Decode responses
            responses = self.tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )

        # Compute rewards
        with self.benchmark_section("reward_computation"):
            # Repeat queries for group_size
            repeated_queries = [
                q for q in batch['queries']
                for _ in range(self.config.group_size)
            ]
            rewards = self.reward_model(repeated_queries, responses)

        # Update batch with generated data
        batch = {
            'input_ids': generated,
            'attention_mask': (generated != self.tokenizer.pad_token_id).long(),
            'response_mask': response_mask,
            'rewards': rewards
        }

        # Track benchmark count
        if self.config.report_benchmarks:
            self.benchmark_count += 1

        # Compute advantages
        with self.benchmark_section("advantage_calculation"):
            advantages = self.compute_advantages(rewards, self.config.group_size)

        # Get token-level advantages if using GSPO-token
        token_level_advantages = batch.get('token_advantages', None)

        # Compute loss
        with self.benchmark_section("loss_computation"):
            loss, metrics = self.compute_gspo_loss(
                batch['input_ids'],
                batch['attention_mask'],
                batch['response_mask'],
                advantages,
                token_level_advantages
            )

        # Backward pass
        with self.benchmark_section("backward_pass"):
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()

        # Log benchmarks at specified intervals
        if (self.config.report_benchmarks and
            self.benchmark_count >= self.config.logging_steps and
            self.global_step > self.config.benchmark_warmup_steps):

            self.log_benchmarks()
            self.benchmark_data.clear()
            self.benchmark_count = 0

        return loss, metrics

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        print("Model set to training mode.")
        epoch_metrics = {}
        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            print(f"Processing step {step}, batch {batch}.")

            # Move batch to device
            device = next(self.model.parameters()).device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            print(f"Batch moved to device {device}.")

            # Forward pass
            loss, metrics = self.training_step(batch)
            print(f"Loss: {loss.item()}, Metrics: {metrics}.")

            # Accumulate loss
            total_loss += loss.item()
            print(f"Total loss accumulated: {total_loss}.")

            # Update epoch metrics
            for k, v in metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)
            print(f"Epoch metrics updated: {epoch_metrics}.")

            # Optimizer step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                with self.benchmark_section("optimizer_step"):
                    print("Performing optimizer step.")

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    print("Gradient clipping performed.")

                    # Optimizer step
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    print("Optimizer step completed.")

                self.global_step += 1
                print(f"Global step incremented: {self.global_step}.")

            # Logging
            if self.global_step % self.config.logging_steps == 0:
                avg_loss = total_loss / (step + 1)
                print(f"Average loss at step {self.global_step}: {avg_loss:.6f}.")
                logger.info(f"Step {self.global_step}: Loss = {avg_loss:.6f}")
                for k, v in metrics.items():
                    print(f"Metric {k}: {v:.6f}.")
                    logger.info(f"  {k}: {v:.6f}")

        # Average metrics over epoch
        epoch_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}
        epoch_metrics['epoch_loss'] = total_loss / len(dataloader)
        print(f"Final epoch metrics: {epoch_metrics}.")

        return epoch_metrics

    def train(self, train_dataloader: DataLoader, num_epochs: Optional[int] = None):
        """
        Main training loop.

        Args:
            train_dataloader: Training data loader
            num_epochs: Number of epochs (uses config if None)
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        logger.info(f"Starting GSPO training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")

            epoch_metrics = self.train_epoch(train_dataloader)

            # Log epoch results
            logger.info(f"Epoch {epoch + 1} completed:")
            for k, v in epoch_metrics.items():
                logger.info(f"  {k}: {v:.6f}")

            self.epoch += 1

            # Save checkpoint
            if (epoch + 1) % self.config.save_steps == 0:
                self.save_checkpoint(f"checkpoint-epoch-{epoch + 1}")

        logger.info("GSPO training completed!")

    def save_checkpoint(self, checkpoint_name: str):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_pretrained(checkpoint_dir / "model")
        self.tokenizer.save_pretrained(checkpoint_dir / "tokenizer")

        # Save training state
        torch.save({
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
        }, checkpoint_dir / "training_state.pt")

        # Save config
        self.config.save_pretrained(checkpoint_dir)

        logger.info(f"Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self, checkpoint_path: Union[str, Path]):
        """Load training checkpoint."""
        checkpoint_path = Path(checkpoint_path)

        # Load training state
        training_state = torch.load(checkpoint_path / "training_state.pt")
        self.global_step = training_state['global_step']
        self.epoch = training_state['epoch']
        self.optimizer.load_state_dict(training_state['optimizer_state_dict'])

        if self.lr_scheduler and training_state['lr_scheduler_state_dict']:
            self.lr_scheduler.load_state_dict(training_state['lr_scheduler_state_dict'])

        logger.info(f"Checkpoint loaded from {checkpoint_path}")
