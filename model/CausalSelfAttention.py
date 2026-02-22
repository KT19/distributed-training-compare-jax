import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P

from config.schema import ModelConfig


class CausalSelfAttention(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array, mask: jax.Array | None) -> jax.Array:
        B, T, D = x.shape
        head_dim = self.config.d_model // self.config.n_heads

        # QKV
        q = nn.Dense(self.config.d_model, name="q_proj")(x)
        k = nn.Dense(self.config.d_model, name="k_proj")(x)
        v = nn.Dense(self.config.d_model, name="v_proj")(x)

        # Reshape
        q = q.reshape(B, T, self.config.n_heads, head_dim)
        k = k.reshape(B, T, self.config.n_heads, head_dim)
        v = v.reshape(B, T, self.config.n_heads, head_dim)

        # TP: shard the head dimension across devices
        if self.config.parallel == "tp":
            q = jax.lax.with_sharding_constraint(q, P(None, None, "data", None))
            k = jax.lax.with_sharding_constraint(k, P(None, None, "data", None))
            v = jax.lax.with_sharding_constraint(v, P(None, None, "data", None))

        # Attention
        attn_weights = jnp.einsum("bthd,bshd->bhts", q, k) * (head_dim**-0.5)

        if mask is not None:
            attn_weights = attn_weights + mask

        attn_weights = nn.softmax(attn_weights, axis=-1)

        # output
        out = jnp.einsum("bhts,bshd->bthd", attn_weights, v)

        out = out.reshape(B, T, D)  # flatten heads

        out = nn.Dense(self.config.d_model, name="out_proj")(out)

        # TP: all-reduce after row-parallel out_proj
        if self.config.parallel == "tp":
            out = jax.lax.with_sharding_constraint(out, P(None, None, None))

        return out
