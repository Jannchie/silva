"""Typed training configuration loaded from YAML (configs/*.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str = "google/siglip2-so400m-patch14-384"
    dropout: float = 0.1


class DataConfig(BaseModel):
    manifest_path: str = "data/manifest.parquet"
    num_workers: int = 4


class TrainConfig(BaseModel):
    freeze_backbone: bool = True
    batch_size: int = 16
    grad_accum: int = 2
    epochs: int = 8
    lr_head: float = 3e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    smooth_l1_weight: float = 0.2
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
