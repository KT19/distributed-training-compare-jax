import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.sharding import PartitionSpec as P

from config.schema import ModelConfig
from model.TransformerBlock import TransformerBlock


class ScanBlock(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, carry, _):
        h, mask = carry
        h = TransformerBlock(self.config)(h, mask)

        return (h, mask), None


class GPTModel(nn.Module):
    config: ModelConfig

    @nn.compact
    def embed_forward(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        # x: (batch, T)
        _, T = x.shape

        # Token Embedding
        token_embed = nn.Embed(self.config.vocab_size, self.config.d_model, name="wte")(x)

        # Pos Embedding
        pos_indices = jnp.arange(T)[None, :]  # (1, T)
        pos_embed = nn.Embed(self.config.max_seq_len, self.config.d_model, name="wpe")(pos_indices)

        # Embedding
        h = token_embed + pos_embed
        h = nn.Dropout(self.config.dropout, deterministic=False)(h)

        # Constrain activation sharding based on parallel strategy
        if self.config.parallel == "dp":
            # DP: batch dimension sharded across devices
            h = jax.lax.with_sharding_constraint(h, P("data", None, None))

        elif self.config.parallel in ("tp", "pp"):
            # TP/PP: activations replicated (params are sharded)
            h = jax.lax.with_sharding_constraint(h, P(None, None, None))

        # Causal mask
        mask = jnp.tril(jnp.ones((T, T)))
        mask = jnp.where(mask == 1, 0, -1e9)[None, None, :, :]  # (1, 1, T, T)

        return h, mask

    @nn.compact
    def stage_forward(self, h: jax.Array, mask: jax.Array, n_layers: int) -> jax.Array:
        # Transformer Block
        ScannedBlock = nn.scan(
            ScanBlock,
            variable_axes={"params": 0},
            split_rngs={"params": True, "dropout": True},
            length=n_layers,
            metadata_params={nn.PARTITION_NAME: "layers"},
        )
        (h, _), _ = ScannedBlock(self.config)((h, mask), None)

        return h

    @nn.compact
    def head_forward(self, h: jax.Array) -> jax.Array:
        h = nn.LayerNorm()(h)
        logits = nn.Dense(self.config.vocab_size, name="lm_head")(h)

        return logits

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        h, mask = self.embed_forward(x)
        h = self.stage_forward(h, mask, self.config.n_layers)
        logits = self.head_forward(h)

        return logits
