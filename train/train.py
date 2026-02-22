import os
import time
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax.training import train_state
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from config.schema import ModelConfig, OptimConfig, TrainConfig
from data.fineweb_edu import get_batch_iterator
from model.GPTModel import GPTModel
from parallel.sharding import get_sharded_params
from train.create_optimizer import create_optimizer
from train.create_train_step import Batch, create_pp_train_step, create_train_step


def train_dp_tp(
    train_config: TrainConfig,
    model_config: ModelConfig,
    opt_config: OptimConfig,
    parallel_strategy: str,
    num_devices: int,
) -> None:
    mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), axis_names=("data",))
    init_config = replace(model_config, parallel="none")
    init_model = GPTModel(init_config)
    dummy_input = jnp.ones((1, model_config.max_seq_len), dtype=jnp.int32)
    params = init_model.init(jax.random.PRNGKey(train_config.seed), dummy_input)["params"]

    model = GPTModel(model_config)
    start_time = 0

    with mesh:
        params, _ = get_sharded_params(params, mesh, parallel=parallel_strategy)

        tx = create_optimizer(opt_config)
        state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        def _infer_sharding(x):
            if not hasattr(x, "ndim") or x.ndim == 0:
                return NamedSharding(mesh, P())
            if hasattr(x, "sharding") and hasattr(x.sharding, "spec"):
                return NamedSharding(mesh, x.sharding.spec)
            return NamedSharding(mesh, P(*([None] * x.ndim)))

        state_sharding = jax.tree.map(_infer_sharding, state)
        state = jax.device_put(state, state_sharding)

        train_step = create_train_step(parallel=parallel_strategy)
        data_iterator = get_batch_iterator(batch_size=train_config.batch, seq_len=model_config.max_seq_len + 1)
        key = jax.random.PRNGKey(train_config.seed)

        # log
        running_losses = []
        loss_history = []
        elapsed_times = []

        print("Warmup")
        for _ in range(5):
            x_np = next(data_iterator)
            x_jnp = jnp.array(x_np[:, :-1])
            y_jnp = jnp.array(x_np[:, 1:])
            batch = Batch(x=x_jnp, y=y_jnp)
            key, subkey = jax.random.split(key)
            state, loss = train_step(state, batch, subkey)

        print("Start measuring")
        start_time = time.perf_counter()
        for step in range(1, train_config.steps + 1, 1):
            x_np = next(data_iterator)
            x_jnp = jnp.array(x_np[:, :-1])
            y_jnp = jnp.array(x_np[:, 1:])
            batch = Batch(x=x_jnp, y=y_jnp)

            key, subkey = jax.random.split(key)
            state, loss = train_step(state, batch, subkey)
            running_losses.append(float(np.asarray(loss)))
            loss_history.append(float(np.asarray(loss)))
            cur_time = time.perf_counter()
            elapsed_times.append(cur_time - start_time)

            if step % train_config.log_every == 0:
                print(
                    f"Step: {step} | Avg loss: {np.mean(running_losses):.4f} | Average step time: {(cur_time - start_time) / step:.4f}"
                )

                running_losses = []

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time}")
    print("End")
    # Save
    os.makedirs(train_config.output_dir, exist_ok=True)
    # csv
    log_info = {"step": np.arange(train_config.steps), "elapsed_time": elapsed_times, "loss": loss_history}
    df = pd.DataFrame(log_info)
    df.to_csv(os.path.join(train_config.output_dir, "log.csv"), index=False)

    return


def train_pp(
    train_config: TrainConfig,
    model_config: ModelConfig,
    opt_config: OptimConfig,
    parallel_strategy: str,
    num_devices: int,
) -> None:
    assert parallel_strategy == "pp", "Should be called if `PP`"

    pp_config = replace(model_config, parallel="none")
    pp_model = GPTModel(pp_config)
    layers_per_stage = model_config.n_layers // num_devices

    init_key = jax.random.PRNGKey(train_config.seed)
    init_key, embed_param_key, embed_dropout_key, stage_key, head_key = jax.random.split(init_key, 5)

    dummy_input = jnp.ones((1, model_config.max_seq_len), dtype=jnp.int32)
    embed_params = pp_model.init(
        {"params": embed_param_key, "dropout": embed_dropout_key},
        dummy_input,
        method=pp_model.embed_forward,
    )["params"]
    dummy_h, dummy_mask = pp_model.apply(
        {"params": embed_params},
        dummy_input,
        method=pp_model.embed_forward,
        rngs={"dropout": embed_dropout_key},
    )
    head_params = pp_model.init(
        head_key,
        dummy_h,
        method=pp_model.head_forward,
    )["params"]

    # `stage` is stage-specific (contiguous layer chunk) while `embed` and `head` are replicated on all stages
    local_params = []
    for stage_idx in range(num_devices):
        stage_init_key = jax.random.fold_in(stage_key, stage_idx)

        # TransformerBlock partitioned
        stage_params = pp_model.init(
            stage_init_key,
            dummy_h,
            dummy_mask,
            layers_per_stage,
            method=pp_model.stage_forward,
        )["params"]

        local_params.append(
            {
                "embed": embed_params,
                "stage": stage_params,
                "head": head_params,
            }
        )

    tx = create_optimizer(opt_config)
    local_opt_state = [tx.init(p) for p in local_params]

    params = jax.device_put_sharded(local_params, jax.devices())
    opt_state = jax.device_put_sharded(local_opt_state, jax.devices())

    train_step = create_pp_train_step(
        apply_fn=pp_model.apply,
        embed_method=pp_model.embed_forward,
        stage_method=pp_model.stage_forward,
        head_method=pp_model.head_forward,
        tx=tx,
        num_microbatches=train_config.pp_microbatches,
        num_stages=num_devices,
        layers_per_stage=layers_per_stage,
        d_model=model_config.d_model,
    )

    data_iterator = get_batch_iterator(batch_size=train_config.batch, seq_len=model_config.max_seq_len + 1)
    key = jax.random.PRNGKey(train_config.seed)

    # log
    running_losses = []
    loss_history = []
    elapsed_times = []

    print("Warmup")
    for _ in range(5):
        x_np = next(data_iterator)
        x_jnp = jnp.array(x_np[:, :-1])
        y_jnp = jnp.array(x_np[:, 1:])
        batch = Batch(x=x_jnp, y=y_jnp)
        key, subkey = jax.random.split(key)
        device_rngs = jax.random.split(subkey, num_devices)
        params, opt_state, loss = train_step(params, opt_state, batch, device_rngs)
        _ = loss

    print("Start measuring")
    start_time = time.perf_counter()
    for step in range(1, train_config.steps + 1, 1):
        x_np = next(data_iterator)
        x_jnp = jnp.array(x_np[:, :-1])
        y_jnp = jnp.array(x_np[:, 1:])
        batch = Batch(x=x_jnp, y=y_jnp)

        key, subkey = jax.random.split(key)
        device_rngs = jax.random.split(subkey, num_devices)
        params, opt_state, loss = train_step(params, opt_state, batch, device_rngs)

        running_losses.append(float(np.asarray(loss)[0]))
        loss_history.append(float(np.asarray(loss)[0]))
        cur_time = time.perf_counter()
        elapsed_times.append(cur_time - start_time)

        if step % train_config.log_every == 0:
            print(
                f"Step: {step} | Avg loss: {np.mean(running_losses):.4f} | Average step time: {(cur_time - start_time) / step:.4f}"
            )
            running_losses = []

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time}")
    print("End")
    # Save
    os.makedirs(train_config.output_dir, exist_ok=True)
    # csv
    log_info = {"step": np.arange(train_config.steps), "elapsed_time": elapsed_times, "loss": loss_history}
    df = pd.DataFrame(log_info)
    df.to_csv(os.path.join(train_config.output_dir, "log.csv"), index=False)

    return
