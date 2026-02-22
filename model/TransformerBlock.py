import jax
from flax import linen as nn

from config.schema import ModelConfig
from model.CausalSelfAttention import CausalSelfAttention
from model.MLP import MLP


class TransformerBlock(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array | None) -> jax.Array:
        # Attention Path
        residual = x
        x = nn.LayerNorm()(x)
        x = CausalSelfAttention(self.config)(x, mask)
        x = x + residual

        # MLP Path
        residual = x
        x = nn.LayerNorm()(x)
        x = MLP(self.config)(x)
        x = x + residual

        return x
