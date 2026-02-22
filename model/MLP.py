import jax
from flax import linen as nn
from jax.sharding import PartitionSpec as P

from config.schema import ModelConfig


class MLP(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.Dense(self.config.d_ff, name="fc1")(x)
        x = nn.gelu(x)

        # TP: column
        if self.config.parallel == "tp":
            x = jax.lax.with_sharding_constraint(x, P(None, None, "data"))

        x = nn.Dense(self.config.d_model, name="fc2")(x)

        # TP: all-reduce
        if self.config.parallel == "tp":
            x = jax.lax.with_sharding_constraint(x, P(None, None, None))

        return x
