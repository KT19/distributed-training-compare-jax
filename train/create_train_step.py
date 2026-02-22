from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as tree_util
import optax
from flax.training.train_state import TrainState
from jax.sharding import PartitionSpec as P

from config.schema import PyTree


@dataclass(frozen=True)
class Batch:
    x: jax.Array
    y: jax.Array


tree_util.register_dataclass(Batch)


def create_train_step(parallel: str = "dp") -> Any:
    if parallel == "pp":
        raise ValueError("`parallel=pp` must use `create_pp_train_step`")

    @jax.jit
    def train_step(state: TrainState, batch: Batch, rng: jax.Array) -> tuple[TrainState, jax.Array]:
        def loss_fn(params: PyTree, x: jax.Array, y: jax.Array, rng: jax.Array):
            logits = state.apply_fn({"params": params}, x, rngs={"dropout": rng})
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)

            return loss.mean()

        # Shard or replicate input data based on parallel strategy
        if parallel == "dp":
            # DP: split batch across devices
            x = jax.lax.with_sharding_constraint(batch.x, P("data", None))
            y = jax.lax.with_sharding_constraint(batch.y, P("data", None))
        else:
            # TP: replicate data on all devices
            x = jax.lax.with_sharding_constraint(batch.x, P(None, None))
            y = jax.lax.with_sharding_constraint(batch.y, P(None, None))

        loss, grads = jax.value_and_grad(loss_fn)(state.params, x, y, rng)

        state = state.apply_gradients(grads=grads)

        return state, loss

    return train_step


def create_pp_train_step(
    apply_fn: Any,
    embed_method: Any,
    stage_method: Any,
    head_method: Any,
    tx: Any,
    num_microbatches: int,
    num_stages: int,
    layers_per_stage: int,
    d_model: int,
) -> Any:
    # Stage-to-stage transfer map: stage i sends data to stage i+1.
    perm = [(i, i + 1) for i in range(num_stages - 1)]

    # reuse params & opt state
    @partial(jax.pmap, axis_name="pipe", in_axes=(0, 0, None, 0), donate_argnums=(0, 1))
    def train_step(params: PyTree, opt_state: PyTree, batch: Batch, rng: jax.Array):
        stage_id = jax.lax.axis_index("pipe")
        is_first_stage = stage_id == 0  # injects embed
        is_last_stage = stage_id == (num_stages - 1)  # compute loss

        microbatch_size = batch.x.shape[0] // num_microbatches
        seq_len = batch.x.shape[1]
        x_micro = batch.x.reshape((num_microbatches, microbatch_size, seq_len))
        y_micro = batch.y.reshape((num_microbatches, microbatch_size, seq_len))

        # placeholder
        h_zeros = jnp.zeros((microbatch_size, seq_len, d_model), dtype=jnp.float32)
        y_zeros = jnp.zeros((microbatch_size, seq_len), dtype=y_micro.dtype)

        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.float32))
        mask = jnp.where(mask == 1, 0.0, -1e9)[None, None, :, :]  # (1, 1, T, T)

        def pp_loss_fn(stage_params: PyTree) -> jax.Array:
            # carry = (activation_buffer, target_buffer, valid_flag, accumulated_loss)
            def body_fn(carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array], t: jax.Array):
                h_buf, y_buf, valid_buf, loss_sum = carry

                # Only the first `num_microbatches` clocks inject new data.
                inject_valid = t < num_microbatches
                t_clamped = jnp.minimum(t, num_microbatches - 1)

                x_in = jax.lax.dynamic_index_in_dim(x_micro, t_clamped, keepdims=False)
                y_in = jax.lax.dynamic_index_in_dim(y_micro, t_clamped, keepdims=False)

                # Make RNG different across both stage and clock.
                stage_rng = jax.random.fold_in(rng, stage_id)
                embed_rng = jax.random.fold_in(stage_rng, t)

                # Apply embedding only to the first stage
                should_embed = jnp.logical_and(is_first_stage, inject_valid)

                def embed_inputs(_):
                    h_embed, _ = apply_fn(
                        {"params": stage_params["embed"]},
                        x_in,
                        method=embed_method,
                        rngs={"dropout": embed_rng},
                    )
                    return h_embed

                h_embed = jax.lax.cond(should_embed, embed_inputs, lambda _: h_zeros, operand=None)

                def first_stage_inputs(_):
                    return h_embed, y_in, mask, inject_valid

                def non_first_stage_inputs(_):
                    return h_buf, y_buf, mask, valid_buf

                h_cur, y_cur, mask_cur, valid_cur = jax.lax.cond(
                    is_first_stage,
                    first_stage_inputs,
                    non_first_stage_inputs,
                    operand=None,
                )

                def run_stage(_):
                    return apply_fn(
                        {"params": stage_params["stage"]},
                        h_cur,
                        mask_cur,
                        layers_per_stage,
                        method=stage_method,
                    )

                h_out = jax.lax.cond(valid_cur, run_stage, lambda _: h_zeros, operand=None)

                def compute_last_stage_loss(_):
                    logits = apply_fn(
                        {"params": stage_params["head"]},
                        h_out,
                        method=head_method,
                    )
                    return optax.softmax_cross_entropy_with_integer_labels(logits, y_cur).mean()

                loss_add = jax.lax.cond(
                    jnp.logical_and(is_last_stage, valid_cur),
                    compute_last_stage_loss,
                    lambda _: jnp.array(0.0, dtype=jnp.float32),
                    operand=None,
                )

                # Shift this stage's output to the next stage for the next clock.
                if num_stages == 1:
                    h_next = h_zeros
                    y_next = y_zeros
                    valid_next = jnp.array(False)
                else:
                    # pass to the next stage
                    h_next = jax.lax.ppermute(h_out, axis_name="pipe", perm=perm)
                    y_next = jax.lax.ppermute(y_cur, axis_name="pipe", perm=perm)
                    valid_next = jax.lax.ppermute(valid_cur, axis_name="pipe", perm=perm)

                return (h_next, y_next, valid_next, loss_sum + loss_add), None

            init_carry = (
                h_zeros,
                y_zeros,
                jnp.array(False),
                jnp.array(0.0, dtype=jnp.float32),
            )

            num_clock_steps = num_microbatches + num_stages - 1
            (_, _, _, loss_sum), _ = jax.lax.scan(
                body_fn,
                init_carry,
                jnp.arange(num_clock_steps),
            )

            local_loss = loss_sum / num_microbatches

            # Replicate scalar loss on all stages (easier logging on host).
            return jax.lax.psum(local_loss, axis_name="pipe")

        loss, grads = jax.value_and_grad(pp_loss_fn)(params)
        updates, new_opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_opt_state, loss

    return train_step
