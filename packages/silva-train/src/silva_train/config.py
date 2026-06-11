"""Typed training configuration loaded from YAML (configs/*.yaml)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    embedding_dim: int = 1152  # SigLIP2-SO400M pooled feature width
    dropout: float = 0.1
    hidden_dims: list[int] = Field(default_factory=list)  # MLP trunk before head; [] = linear probe
    n_residual_blocks: int = 0  # pre-norm residual blocks appended after the MLP trunk


class DataConfig(BaseModel):
    manifest_path: str | list[str] = "data/manifest.parquet"  # one parquet, or a list to merge several sources for training
    # 0: the dataset is fully resident in memory tensors, so worker processes only add
    # cost — on Windows (spawn) each would pickle the whole dataset.
    num_workers: int = 0
    pair_manifest_path: str | None = None  # parquet of explicit preference pairs (embedding_a/embedding_b/target); enables the pairwise margin term


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
    qwk_weight: float = 0.0  # weight on quadratic-weighted-kappa loss (crushes large-gap blunders, improves MAE)
    pairwise_weight: float = 0.0  # weight on the explicit-preference margin loss (data.pair_manifest_path); 0 = disabled
    pair_margin: float = 0.5  # decisive pairs: winner's ordinal_score must lead by >= this margin
    tie_margin: float = 0.0  # tie pairs: |score_a - score_b| above this is penalised
    pair_batch: int = 256  # pairs sampled per micro-batch for the margin term
    score_anchors: list[float] | Literal["auto"] | None = None  # non-uniform QWK category positions; "auto" = kNN pseudo-retest estimate
    label_smoothing: float = 0.0  # soften ordinal targets {0,1}->{eps,1-eps}; keeps latent finite, kills the 0~1 tail saturation
    mixup_alpha: float = 0.0  # Beta(alpha,alpha) embedding mixup; 0 = disabled
    ema_decay: float = 0.0  # exponential moving average of weights; 0 = disabled, typical 0.999
    loss_truncation: float = 0.0  # drop the top-k% highest-loss samples per batch; 0 = disabled, typical 0.05
    cosine_restarts: int = 0  # number of warm restarts; 0 = single cosine decay (default)
    mixed_precision: str = "no"  # accelerate mixed precision; head 极小用 fp32 数值更稳，bf16 无明显收益；"no" 同样适用 CPU
    report_to: str = "pandm"       # default: local-first tracking (writes ./.pandm); "none" to disable, or any accelerate tracker name
    project_name: str = "silva"
    run_name: str | None = None
    eval_every: int = 1
    train_eval_samples: int = 0  # fixed train-split subset evaluated each eval for the train-val gap; 0 = match val size
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
