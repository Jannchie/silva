"""Typed training configuration loaded from YAML (configs/*.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    embedding_dim: int = 1152  # SigLIP2-SO400M pooled feature width
    dropout: float = 0.1
    hidden_dims: list[int] = Field(default_factory=list)  # MLP trunk before head; [] = linear probe


class DataConfig(BaseModel):
    manifest_path: str = "data/manifest.parquet"
    num_workers: int = 4


class TrainConfig(BaseModel):
    batch_size: int = 256
    grad_accum: int = 1
    epochs: int = 8
    lr_head: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    use_pos_weight: bool = True  # auto per-threshold class balancing (compute_pos_weight)
    ranking_weight: float = 0.0  # weight on pairwise ranking loss (directly optimises Spearman)
    soft_spearman_weight: float = 0.0  # weight on global soft-Spearman loss (ranking + calibration)
    mixed_precision: str = "bf16"  # accelerate mixed precision; "no" for CPU
    report_to: str = "none"        # "wandb" to log to Weights & Biases, "none" to disable
    project_name: str = "silva"
    run_name: str | None = None
    eval_every: int = 1
    early_stop_metric: str = "spearman"
    early_stop_patience: int = 3
    seed: int = 42
    output_dir: str = "outputs/v1_stage1_head"


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with Path(path).open(encoding="utf-8") as f:
            return cls(**(yaml.safe_load(f) or {}))
