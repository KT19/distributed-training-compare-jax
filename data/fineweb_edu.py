from typing import Any

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def get_tokenizer():
    """Create a GPT-2 tokenizer with a dedicated <pad> token added."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
    return tokenizer


def get_batch_iterator(batch_size: int, seq_len: int) -> Any:
    """
    Streams batches of tokens from huggingface
    """
    tokenizer = get_tokenizer()

    ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)

    buffer = []

    for item in ds:
        text = item["text"]
        tokens = tokenizer.encode(text)
        buffer.extend(tokens)

        while len(buffer) >= batch_size * seq_len:
            chunk_size = batch_size * seq_len
            chunk = buffer[:chunk_size]
            buffer = buffer[chunk_size:]

            batch_np = np.array(chunk, dtype=np.int32).reshape(batch_size, seq_len)

            x = batch_np

            yield x
