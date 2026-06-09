import time
import numpy as np
import torch
from contextlib import contextmanager
from torch.optim import AdamW


def format_time(time_s):
    time_ms = time_s * 1000
    if time_ms < 1:
        return f"{time_ms * 1000:.2f} µs"
    elif time_ms < 1000:
        return f"{time_ms:.2f} ms"
    else:
        return f"{time_ms / 1000:.2f} s"


def time_model(model, device, input_sample, timing_repeat=100):
    model = model.to(device)
    if isinstance(input_sample, tuple):
        input_sample = tuple([t.to(device) for t in input_sample])
    else:
        input_sample = input_sample.to(device)
    timings = []

    for layer in model.layers:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for i in range(3):
            with torch.no_grad():
                y = layer(input_sample)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        for i in range(timing_repeat):
            with torch.no_grad():
                y = layer(input_sample)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.perf_counter()
        input_sample = y

        timings.append(end_time - start_time)
    return timings

def get_shape(x):
    if isinstance(x, torch.Tensor):
        return x.shape
    elif isinstance(x, (tuple, list)):
        return tuple([get_shape(x[i]) for i in range(len(x))])
    elif x is None:
        return 1
    else:
        raise Exception(f"cant get shape, got type {type(x)}")



def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def fmt_bytes(n: int) -> str:
    x = float(n)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if x < 1024 or unit == "TiB":
            return f"{int(x)} B" if unit == "B" else f"{x:.2f} {unit}"
        x /= 1024
    return f"{x:.2f} PiB"


def param_and_buffer_bytes(model) -> int:
    return sum(tensor_bytes(p) for p in model.parameters()) + sum(
        tensor_bytes(b) for b in model.buffers()
    )


def grad_bytes(model) -> int:
    return sum(tensor_bytes(p.grad) for p in model.parameters() if p.grad is not None)


def optimizer_state_bytes(optim: torch.optim.Optimizer) -> int:
    total = 0
    for _p, state in optim.state.items():
        for v in state.values():
            if torch.is_tensor(v):
                total += tensor_bytes(v)
            elif isinstance(v, dict):
                for vv in v.values():
                    if torch.is_tensor(vv):
                        total += tensor_bytes(vv)
    return total


@contextmanager
def count_saved_tensors(group_mult):
    saved = {
        "bytes": 0,
        "count": 0,
        "by_dtype": {},
        "by_device": {},
    }

    def pack(t: torch.Tensor):
        b = tensor_bytes(t) * group_mult
        saved["bytes"] += b
        saved["count"] += group_mult
        dt = str(t.dtype)
        dv = str(t.device)
        saved["by_dtype"][dt] = saved["by_dtype"].get(dt, 0) + b
        saved["by_device"][dv] = saved["by_device"].get(dv, 0) + b
        return t

    def unpack(t: torch.Tensor):
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        yield saved


def make_optim(params):
    return AdamW(params, lr=1e-3)


def summarize_run(model, group_mult, example_input: torch.Tensor, device: torch.device):
    model = model.to(device)
    x = example_input.to(device)

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    optim = make_optim(model.parameters())

    with count_saved_tensors(group_mult) as saved:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=True):
            optim.zero_grad(set_to_none=True)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optim.step()

    summary = {}

    summary["model_params+buffers_bytes"] = param_and_buffer_bytes(model)
    summary["grads_bytes"] = grad_bytes(model)
    summary["optimizer_state_bytes"] = optimizer_state_bytes(optim)

    summary["autograd_saved_bytes"] = saved["bytes"]
    summary["autograd_saved_count"] = saved["count"]
    summary["autograd_saved_by_dtype"] = saved["by_dtype"]
    summary["autograd_saved_by_device"] = saved["by_device"]
    summary['total_memory_used'] = summary['model_params+buffers_bytes'] + summary['grads_bytes'] + summary['optimizer_state_bytes'] + summary['autograd_saved_bytes']

    if use_cuda:
        summary["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        summary["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return summary


def get_mem_per_layer(model, device, current_x):
    layer_static = []
    layer_activation = []
    batch_size = current_x.shape[0]
    current_x = current_x[:1].to(device)
    model = model.to(device)
    for idx in range(len(model.layers)):
        s = summarize_run(model.layers[idx:idx+1], batch_size, current_x, torch.device(device))
        layer_static.append(
            s['model_params+buffers_bytes']
            + s['grads_bytes']
            + s['optimizer_state_bytes']
        )
        layer_activation.append(s['autograd_saved_bytes'])
        torch.cuda.empty_cache()
        with torch.no_grad():
            current_x = model.layers[idx:idx+1](current_x)

    del model
    del current_x
    torch.cuda.empty_cache()

    return layer_static, layer_activation


class GetPartitions:
    def __init__(self, config
                    ):
        self.config = config
        self.device = config.device
        self.split_to = config.split_to

    def split_by_indices(self, model, indices, shapes):
        if not (0 < self.split_to <= len(model.layers)):
            raise ValueError(f"Number of groups must be between 1 and the number of layers ({len(model.layers)}), requested {self.split_to}.")

        new_layers = []
        block_gpu_mapping = []
        block_indices = []
        block_io_shapes = []
        for gpu_id, group_indices in enumerate(indices):
            block = torch.nn.Sequential()
            block.append(model.layers[group_indices[0]])
            block_gpu_mapping.append(gpu_id)
            block_indices.append(group_indices[0])
            for i in range(1, len(group_indices)):
                if group_indices[i] == group_indices[i - 1] + 1:
                    block.append(model.layers[group_indices[i]])
                else:
                    raise ValueError(f"Layer {group_indices[i]} is not consecutive to layer {group_indices[i - 1]}")

            block_io_shapes.append((shapes[f"{group_indices[0]}_in"], shapes[f"{group_indices[-1]}_out"]))
            new_layers.append(block)

        combined = list(zip(block_indices, new_layers, block_gpu_mapping))
        combined_sorted = sorted(combined, key=lambda x: x[0])
        list1_sorted, list2_sorted, list3_sorted = zip(*combined_sorted)

        new_layers = torch.nn.Sequential(*list2_sorted)
        block_gpu_mapping = list(list3_sorted)

        model.layers = new_layers
        model.layer_gpu_id = block_gpu_mapping
        model.block_io_shapes = block_io_shapes

    def partition_model(self, model, input_sample, indices=None, timing_repeat=100, max_mem_usage = float("inf")):
        if isinstance(input_sample, tuple):
            input_sample_micro = tuple([i[:i.shape[0] // self.config.grad_accumulation] for i in input_sample])
        else:
            input_sample_micro = input_sample[:input_sample.shape[0] // self.config.grad_accumulation]
        if indices is None:
            timings = time_model(model, self.device, input_sample_micro, timing_repeat)
            statics, activations = get_mem_per_layer(model, self.device, input_sample_micro)
            indices = self.constrained_partition(timings, statics, activations, max_mem_usage)
            mem_usages_calculated = [
                sum(statics[min(g): max(g)+1])
                + (self.split_to - i) * sum(activations[min(g): max(g)+1])
                for i, g in enumerate(indices)
            ]
            print(f"Partitioner predicited max memory usage: {max(mem_usages_calculated)/(1024**3):.2f} GB at layer {np.argmax(mem_usages_calculated)}")
        print(f"Initial model has {len(model.layers)} layers, splitting to {self.split_to}")
        if isinstance(input_sample_micro, tuple):
            x = tuple([t.to(self.device) for t in input_sample_micro])
        else:
            x = input_sample_micro.to(self.device)
        shapes = {"0_in": get_shape(x)}
        model.to(self.device)
        with torch.no_grad():
            for i, layer in enumerate(model.layers):                                        
                x = layer(x)
                x_shape = get_shape(x)
                shapes[f"{i}_out"] = x_shape
                shapes[f"{i+1}_in"] = x_shape
        shapes["res_out"] = x_shape

        self.split_by_indices(model, indices, shapes)

        shapes.clear()
        
        if isinstance(input_sample_micro, tuple):
            x = tuple([t.to(self.device) for t in input_sample_micro])
        else:
            x = input_sample_micro.to(self.device)
        shapes = {"0_in": get_shape(x)}

        print("input shape", get_shape(x))
        with torch.no_grad():
            for i, layer in enumerate(model.layers):                                        
                x = layer(x)
                x_shape = get_shape(x)
                print(f"layer {i} output shape", x_shape)            
                shapes[f"{i}_out"] = x_shape
                shapes[f"{i+1}_in"] = x_shape
        shapes["res_out"] = x_shape
                                                                        
        model.to('cpu')
        torch.cuda.empty_cache()
        return indices, shapes
    
    def constrained_partition(self, times, statics, activations, memory_budget):
        n = len(times)
        if len(statics) != n or len(activations) != n:
            raise ValueError("times, statics, and activations must have the same length")
        if not (1 <= self.split_to <= n):
            raise ValueError(f"split_to must be between 1 and {n}")

        N = self.split_to

        pt = [0.0] * (n + 1)
        ps = [0] * (n + 1)
        pa = [0] * (n + 1)
        for i in range(n):
            pt[i + 1] = pt[i] + float(times[i])
            ps[i + 1] = ps[i] + int(statics[i])
            pa[i + 1] = pa[i] + int(activations[i])

        def tsum(a, b):
            return pt[b] - pt[a]

        def ssum(a, b):
            return ps[b] - ps[a]

        def asum(a, b):
            return pa[b] - pa[a]

        # ---------- PASS 1: HARD CONSTRAINT (no violations allowed) ----------
        INF = float("inf")
        bestT = [[INF] * (N + 1) for _ in range(n + 1)]
        S1 = [[-1] * (N + 1) for _ in range(n + 1)]

        # j=1 (weight = N)
        weight = N
        for i in range(1, n + 1):
            eff_m = ssum(0, i) + weight * asum(0, i)
            if eff_m <= memory_budget:
                bestT[i][1] = tsum(0, i)
                S1[i][1] = 0

        for j in range(2, N + 1):
            weight = N - j + 1
            for i in range(j, n + 1):
                best = INF
                best_x = -1
                for x in range(j - 1, i):
                    if bestT[x][j - 1] == INF:
                        continue
                    eff_m = ssum(x, i) + weight * asum(x, i)
                    if eff_m > memory_budget:
                        continue  # hard constraint
                    t = tsum(x, i)
                    cand = max(bestT[x][j - 1], t)
                    if cand < best:
                        best = cand
                        best_x = x
                bestT[i][j] = best
                S1[i][j] = best_x

        if bestT[n][N] != INF:
            groups = self._reconstruct_groups(S1, n, N)
            return groups
        
        # ---------- PASS 2: FALLBACK (minimize max overshoot, then max time) ----------4
        print("Pass 2: FALLBACK (minimize max overshoot, then max time)")
        best = [[(INF, INF)] * (N + 1) for _ in range(n + 1)]
        S2 = [[-1] * (N + 1) for _ in range(n + 1)]

        # j=1
        weight = N
        for i in range(1, n + 1):
            t = tsum(0, i)
            eff_m = ssum(0, i) + weight * asum(0, i)
            o = max(0, eff_m - memory_budget)
            best[i][1] = (o, t)
            S2[i][1] = 0

        for j in range(2, N + 1):
            weight = N - j + 1
            for i in range(j, n + 1):
                best_ij = (INF, INF)
                best_x = -1
                for x in range(j - 1, i):
                    prev = best[x][j - 1]
                    if prev[0] == INF:
                        continue
                    t = tsum(x, i)
                    eff_m = ssum(x, i) + weight * asum(x, i)
                    o = max(0, eff_m - memory_budget)
                    cand = (max(prev[0], o), max(prev[1], t))
                    if cand < best_ij:
                        best_ij = cand
                        best_x = x
                best[i][j] = best_ij
                S2[i][j] = best_x

        groups = self._reconstruct_groups(S2, n, N)
        return groups

    def _reconstruct_groups(self, S, n, N):
        cuts = []
        i, j = n, N
        while j > 0:
            x = S[i][j]
            if x < 0:
                raise RuntimeError("Reconstruction failed")
            cuts.append(x)
            i, j = x, j - 1
        cuts.reverse()
        cuts.append(n)
        return [list(range(cuts[k], cuts[k + 1])) for k in range(len(cuts) - 1)]

