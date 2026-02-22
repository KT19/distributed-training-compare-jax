import click
import jax
import yaml

from config.schema import ModelConfig, OptimConfig, TrainConfig
from data.fineweb_edu import get_tokenizer
from train.train import train_dp_tp, train_pp


@click.command()
@click.option("--train_config_path", default="configs/train_config_dp.yaml")
def main(train_config_path: str):
    # Config
    with open("configs/model_config.yaml") as f:
        config_dict = yaml.safe_load(f)
    # get vocab size
    tokenizer = get_tokenizer()
    config_dict["vocab_size"] = len(tokenizer)

    with open(train_config_path) as f:
        train_dict = yaml.safe_load(f)
    train_config = TrainConfig(**train_dict)

    # Model config with parallel strategy for sharding constraints during training
    config_dict["parallel"] = train_config.parallel
    model_config = ModelConfig(**config_dict)

    with open("configs/optim_config.yaml") as f:
        config_dict = yaml.safe_load(f)
    opt_config = OptimConfig(**config_dict)

    parallel_strategy = train_config.parallel

    num_devices = len(jax.devices())

    print(f"Running `{parallel_strategy}` on {num_devices} devices.")

    if parallel_strategy == "pp":
        train_pp(
            train_config=train_config,
            model_config=model_config,
            opt_config=opt_config,
            parallel_strategy=parallel_strategy,
            num_devices=num_devices,
        )

    elif parallel_strategy == "dp" or parallel_strategy == "tp":
        train_dp_tp(
            train_config=train_config,
            model_config=model_config,
            opt_config=opt_config,
            parallel_strategy=parallel_strategy,
            num_devices=num_devices,
        )

    else:
        raise ValueError(f"Unsupported strategy `{parallel_strategy}`")


if __name__ == "__main__":
    main()
