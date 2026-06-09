import threading

import torch
import torch.distributed.rpc as rpc

from .Constants import DistTypes

 
work_storage = {}
rank_storage = {}
rank_locks = {}
rank_conds = {}

class P2PQueue:
    def __init__(self, src_rank: int, dst_rank: int, curr_rank: int,
                 curr_device: torch.device, dst_device: torch.device, tag="", put_max_size=0):
        self.dst_rank = dst_rank
        self.curr_rank = curr_rank
        self.curr_device = curr_device
        self.dst_device = dst_device
        self.tag = tag

        self.my_name = f"{src_rank}|{curr_rank}|{self.tag}"
        self.dst_name = f"{self.curr_rank}|{self.dst_rank}|{self.tag}"

        rank_storage.setdefault(self.my_name, {})
        rank_storage.setdefault(self.dst_name, {})

        rank_locks.setdefault(self.my_name, threading.Lock())
        rank_locks.setdefault(self.dst_name, threading.Lock())

        rank_conds.setdefault(self.my_name, threading.Condition(rank_locks[self.my_name]))
        rank_conds.setdefault(self.dst_name, threading.Condition(rank_locks[self.dst_name]))

        self.current_put = 0
        self.current_get = 0

        self._pending = []

    @staticmethod
    def _internal_put(dst_name: str, idx: int, tensor, extra_data):
        # Runs on destination worker.
        with rank_conds[dst_name]:
            rank_storage[dst_name][idx] = (tensor, extra_data)
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                torch.cuda.synchronize(tensor.device)
            rank_conds[dst_name].notify_all()

    def put(self, tensor, extra_data=None):
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().clone().contiguous()
            if tensor.is_cuda:
                torch.cuda.synchronize(tensor.device)
        if isinstance(extra_data, torch.Tensor):
            extra_data = extra_data.detach().clone().contiguous()
            if extra_data.is_cuda:
                torch.cuda.synchronize(extra_data.device)

        while len(self._pending) and self._pending[0][0].done(): 
            done, *_ = self._pending.pop(0)
            done.wait()
        w = rpc.rpc_async(
            f"worker{self.dst_rank}",
            P2PQueue._internal_put,
            args=(self.dst_name, self.current_put, tensor, extra_data),
        )
        self._pending.append((w, tensor, extra_data))
        self.current_put += 1

    def get(self):
        with rank_conds[self.my_name]:
            while self.current_get not in rank_storage[self.my_name]:
                    rank_conds[self.my_name].wait()
            
            tensor, extra_data = rank_storage[self.my_name].pop(self.current_get)
            self.current_get += 1
            data_type = DistTypes.tensor if isinstance(tensor, torch.Tensor) else DistTypes.other
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.clone().contiguous().to(self.curr_device)
            if isinstance(extra_data, torch.Tensor):
                extra_data = extra_data.clone().contiguous().to(self.curr_device)
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                torch.cuda.synchronize(self.curr_device)

            return data_type, tensor, extra_data

    def empty(self):
        with rank_locks[self.my_name]:
            return self.current_get not in rank_storage[self.my_name]

    def close(self):
        pass