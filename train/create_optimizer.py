from typing import Any

import optax

from config.schema import OptimConfig


def create_optimizer(cfg: OptimConfig) -> Any:
    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.adamw(learning_rate=cfg.lr, weight_decay=cfg.weight_decay),
    )
