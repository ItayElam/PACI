from __future__ import annotations
import sys
sys.path.append("../")

import os
from pathlib import Path
from functools import partial
from itertools import chain
import torch
from datasets import load_dataset, load_from_disk
    
from .base import BaseDataset


def _tokenize_batch(batch, tokenizer):
    return tokenizer(
        batch["text"],
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )


def _group_texts(tokenized_ds, block_size: int, num_proc: int, map_batch_size: int, eos_id: int | None):
    def group_texts(examples):
        if eos_id is not None:
            seq = []
            for row in examples["input_ids"]:
                seq.extend(row)
                seq.append(eos_id)
        else:
            seq = list(chain.from_iterable(examples["input_ids"]))

        total_length = (len(seq) // block_size) * block_size
        if total_length == 0:
            return {"input_ids": [], "labels": []}

        input_ids = [seq[i:i+block_size] for i in range(0, total_length, block_size)]
        return {"input_ids": input_ids, "labels": input_ids.copy()}

    return tokenized_ds.map(
        group_texts,
        batched=True,
        batch_size=map_batch_size,
        num_proc=num_proc,
        load_from_cache_file=False,
    )



def build_openwebtext_splits(
    *,
    tokenizer,
    tokenizer_id: str,
    block_size: int = 512,
    cache_dir: str | Path = "cache/openwebtext",
    val_frac: float = 0.02,
    split_seed: int = 1234,
    num_proc: int = 48,
    map_batch_size: int = 1000,
    shuffle_before_split: bool = True,
):
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer.eos_token_id is None; cannot insert EOS separators.")

    cache_dir = Path(cache_dir)
    root = cache_dir / tokenizer_id / f"block_{block_size}"
    train_path = root / "train_grouped"
    val_path = root / "val_grouped"

    if train_path.exists() and val_path.exists():
        return

    print(f"Saving data to {os.path.join(os.getcwd(), root)}")
    raw = load_dataset("Skylion007/openwebtext", split="train", trust_remote_code=True)

    if shuffle_before_split:
        raw = raw.shuffle(seed=split_seed)

    raw = raw.filter(
        lambda x: [len(text) > 20 for text in x["text"]], 
        batched=True,
        num_proc=num_proc
    )

    # Split raw BEFORE tokenize/group
    n_raw = len(raw)
    n_val_raw = int(n_raw * val_frac)
    val_raw = raw.select(range(n_val_raw))
    train_raw = raw.select(range(n_val_raw, n_raw))

    # Tokenize each split
    fn = partial(_tokenize_batch, tokenizer=tokenizer)
    train_tok = train_raw.map(
        fn, batched=True, remove_columns=["text"],
        num_proc=num_proc, batch_size=map_batch_size,
        load_from_cache_file=True, desc="Tokenizing OWT train",
    )
    val_tok = val_raw.map(
        fn, batched=True, remove_columns=["text"],
        num_proc=num_proc, batch_size=map_batch_size,
        load_from_cache_file=True, desc="Tokenizing OWT val",
    )

    train_ds = _group_texts(train_tok, block_size, num_proc, map_batch_size, eos_id=eos_id)
    val_ds   = _group_texts(val_tok,   block_size, num_proc, map_batch_size, eos_id=eos_id)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(train_path)
    val_ds.save_to_disk(val_path)



class OpenWebText(BaseDataset):
    def __init__(
        self,
        batch_size,
        num_workers=4,
        tokenizer=None,
        block_size=512,
        cache_dir="cache/openwebtext",
        tokenizer_id=None,
    ):
        self.block_size = block_size
        self.cache_dir = Path(cache_dir)

        if tokenizer_id is not None:
            self.tokenizer_id = tokenizer_id
        else:
            self.tokenizer_id = getattr(tokenizer, "name_or_path", "custom_tokenizer")

        super().__init__(batch_size=batch_size, num_workers=num_workers, tokenizer=tokenizer)

    @property
    def name(self):
        return "OpenWebText"

    def _root(self) -> Path:
        return self.cache_dir / self.tokenizer_id / f"block_{self.block_size}"

    def _train_path(self) -> Path:
        return self._root() / "train_grouped"

    def _val_path(self) -> Path:
        return self._root() / "val_grouped"

    @property
    def train_dataset(self):
        p = self._train_path()
        if not p.exists():
            raise FileNotFoundError(
                f"Missing prebuilt dataset at {p}. Run build_openwebtext_splits(...) first."
            )
        return load_from_disk(p)

    @property
    def test_dataset(self):
        p = self._val_path()
        if not p.exists():
            raise FileNotFoundError(
                f"Missing prebuilt dataset at {p}. Run build_openwebtext_splits(...) first."
            )
        return load_from_disk(p)

    def get_collate_fn(self):
        return self.collate_fn

    def collate_fn(self, batch):
        input_ids = torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long)

        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100

        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        return input_ids, labels, attention_mask
