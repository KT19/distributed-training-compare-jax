from typing import Any

import jax
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import tree_map_with_path

from config.schema import PyTree


def get_sharded_params(params: PyTree, mesh: Any, parallel: str) -> PyTree:
    """
    Define mesh-based parameter sharding for DP/TP modes.
    """

    # Define the rules
    def get_spec(path, leaf):
        """
        path: tuple of keys
        leaf: The actual parameter tensor
        """
        path_str = "/".join([str(k) for k in path])

        # Parallel strategy
        if parallel == "dp":
            # replicate
            return P(None)

        if parallel == "tp":
            # Only shard kernel weights (rank 3: Layers, In, Out)
            # rank<=2 params (e.g., bias) are replicated
            if leaf.ndim < 3:
                if "lm_head" in path_str:
                    if "kernel" in path_str:
                        return P(None, "data")
                    if "bias" in path_str:
                        return P(
                            "data",
                        )
                # Other (e.g., embed) are replicated
                return P(None)

            # attention layers (Layers, d_model, heads*head_dim)
            if "q_proj" in path_str or "k_proj" in path_str or "v_proj" in path_str:
                return P(None, None, "data")

            if "out_proj" in path_str:
                # Row parallel (Layers, heads*head_dim, d_model)
                return P(None, "data", None)

            # MLP
            if "fc1" in path_str:
                # Column parallel
                return P(None, None, "data")
            if "fc2" in path_str:
                # Row parallel
                return P(None, "data", None)

            # Other rank-3 params
            return P(None)

        return P(None)

    spec = tree_map_with_path(get_spec, params)

    # Apply sharding
    sharding_tree = jax.tree.map(lambda s: NamedSharding(mesh, s), spec)
    sharded_params = jax.device_put(params, sharding_tree)

    return sharded_params, spec
