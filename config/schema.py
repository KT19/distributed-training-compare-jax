from dataclasses import dataclass
from typing import Any

PyTree = Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    max_seq_len: int
    dropout: float
    parallel: str


@dataclass(frozen=True)
class OptimConfig:
    lr: float
    weight_decay: float
    grad_clip: float


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    parallel: str
    # batching
    batch: int
    # steps
    steps: int
    log_every: int
    # output
    output_dir: str
    # pipeline parallelism
    pp_microbatches: int = 1
