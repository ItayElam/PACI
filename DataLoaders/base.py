from abc import ABC, abstractmethod
import numpy as np
import random
import torch
from torch.utils.data import Sampler
import math


def seed_init_fn(x):
   seed = 0 + x
   np.random.seed(seed)
   random.seed(seed)
   torch.manual_seed(seed)
   return


class ResumableRandomBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, drop_last, base_seed=0):
        self.n = int(dataset_len)
        self.bs = int(batch_size)
        self.drop_last = bool(drop_last)
        self.base_seed = int(base_seed)
        self.epoch = 0
        self.batch_pos = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self.batch_pos = 0

    def epoch_done(self):
        self.epoch += 1
        self.batch_pos = 0

    def state_dict(self):
        return {"epoch": self.epoch, "batch_pos": self.batch_pos}

    def load_state_dict(self, state):
        self.epoch = int(state["epoch"])
        self.batch_pos = int(state["batch_pos"])

    def __len__(self):
        if self.drop_last:
            return self.n // self.bs
        return math.ceil(self.n / self.bs)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.base_seed + self.epoch)

        perm = torch.randperm(self.n, generator=g).tolist()
        total_batches = len(self)

        for b in range(self.batch_pos, total_batches):
            s = b * self.bs
            e = s + self.bs
            batch = perm[s:e]
            if len(batch) < self.bs and self.drop_last:
                break
            self.batch_pos += 1
            yield batch

class BaseDataset(ABC):
    def __init__(self, batch_size, num_workers=32, tokenizer=None, base_seed=1234, drop_last=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_batch_size = batch_size
        print("Loading Data")

        self.train_batch_sampler = ResumableRandomBatchSampler(
            dataset_len=len(self.train_dataset),
            batch_size=self.data_batch_size,
            drop_last=drop_last,
            base_seed=base_seed,
        )

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_sampler=self.train_batch_sampler, 
                                                        num_workers=num_workers, prefetch_factor=4,
                                                        pin_memory=True, persistent_workers=True,
                                                        collate_fn=self.get_collate_fn(), worker_init_fn = seed_init_fn)
        print("Training Data Loaded")
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.data_batch_size,
                                                       shuffle=False, num_workers=num_workers, prefetch_factor=4,
                                                       pin_memory=True, persistent_workers=True,
                                                       collate_fn=self.get_collate_fn(), worker_init_fn = seed_init_fn)
        print("Validation Data Loaded")

    @property
    def batch_size(self):
        return self.data_batch_size

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def train_dataset(self):
        pass

    @property
    @abstractmethod
    def test_dataset(self):
        pass

    def get_collate_fn(self):
        return None
