import torch
from tqdm import tqdm
import time
import sys
import math
import subprocess
import os
from collections import deque
sys.path.append("../")

from DistPACI import TrainingConfig, PACIModel

BATCHES_PER_TIMING_SAMPLE = 15
TIMING_WINDOW_SIZE = 6
REL_TPS_ERR_THRESHOLD = 0.005 

def set_random(seed):
    import numpy as np
    import random
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_adamw(model, lr=1e-4, weight_decay=1e-2, betas=(0.9, 0.95), eps=1e-8):
    # Credit: https://github.com/minhnguyent546/pre-training-gpt2/blob/master/gpt2/utils.py
    param_list = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in param_list if p.dim() >= 2]
    no_decay_params = [p for p in param_list if p.dim() < 2]

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)

def make_static_minibatch(
    batch_size=256,
    seq_len=1024,
    vocab_size=50257,
    device="cuda",
    seed=1234,
):
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    input_ids = torch.randint(
        low=0, high=vocab_size, size=(batch_size, seq_len),
        generator=g, device=device, dtype=torch.long
    )

    targets = input_ids.clone()
    targets[:, :-1] = input_ids[:, 1:]
    targets[:, -1] = -100

    return input_ids, targets


def split_to_microbatchs(tensor, n, dim=0):
    total = tensor.size(dim)
    base = total // n
    extra = total % n

    sizes = [base + 1 if i < extra else base for i in range(n)]
    return list(torch.split(tensor, sizes, dim=dim))


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = {f"cuda:{i}": int(x.split()[0]) for i, x in enumerate(memory_free_info)}
    return memory_free_values


def benchmark_model(
    model,
    config: TrainingConfig,
    input_sample,
    target_sample,
    return_dict,
    live_progress=None,
):
    progress = None
    paci_model = None
    try:
        bu_stdout = sys.stdout
        os.environ["paci_suppress_output"] = "true"
        
        paci_model = PACIModel(model, config, measure_iterations=50)
        experiment_dir = f"speed_experiments/{torch.cuda.get_device_name()}/split_to_{config.split_to}"
        os.makedirs(experiment_dir, exist_ok=True)
        split_file_path = os.path.join(experiment_dir, f"split_{input_sample.shape[0]//config.grad_accumulation}.pth")
        if os.path.exists(split_file_path):
            print("Found existing checkpoint, loading..")
            state_dict = torch.load(split_file_path, map_location='cpu')
            paci_model.init(input_sample, state_dict["split_indices"])
        else:
            paci_model.init(input_sample)
        
        paci_model.half()
        
        paci_model.train()
        
        for i in range(4): # Warmup
            paci_model(input_sample, target_sample, False)

        paci_model.epoch_done()

        tokens_per_minibatch = input_sample.numel()
        chunk_times_sec = deque(maxlen=TIMING_WINDOW_SIZE)
        measured_batches = 0
        batches_in_chunk = 0

        start_time = time.perf_counter()
        rel_tps_err = float("inf")
        stable_tps_estimate = 0.0
        chunk_start_time = start_time
        max_mem = get_gpu_memory()

        if live_progress is None:
            progress = tqdm(desc="Measuring speed", total=None)
        else:
            live_batches, live_stable, live_rel = live_progress
        while True:
            paci_model(input_sample, target_sample, False)
            measured_batches += 1

            [max_mem.update({k: v}) for k, v in get_gpu_memory().items() if max_mem[k] < v]
            if progress is not None:
                progress.update(1)
                progress.set_description(
                    f"[Stable tps {stable_tps_estimate:.2f} | TPS rel SE {rel_tps_err:.2%}]"
                )
            else:
                live_batches.value = measured_batches
                live_stable.value = stable_tps_estimate
                live_rel.value = rel_tps_err

            batches_in_chunk += 1
            if batches_in_chunk >= BATCHES_PER_TIMING_SAMPLE:
                now = time.perf_counter()
                chunk_dt = max(now - chunk_start_time, 1e-9)
                chunk_times_sec.append(chunk_dt)
                chunk_start_time = now
                batches_in_chunk = 0

                if len(chunk_times_sec) >= 2:
                    mean_dt = sum(chunk_times_sec) / len(chunk_times_sec)
                    if mean_dt > 0:
                        var = sum((x - mean_dt) ** 2 for x in chunk_times_sec) / (len(chunk_times_sec) - 1)
                        std = math.sqrt(max(var, 0.0))
                        stderr_mean_dt = std / math.sqrt(len(chunk_times_sec))
                        tokens_per_chunk = BATCHES_PER_TIMING_SAMPLE * tokens_per_minibatch
                        stable_tps_estimate = tokens_per_chunk / mean_dt
                        se_tps = (tokens_per_chunk / (mean_dt**2)) * stderr_mean_dt
                        rel_tps_err = se_tps / stable_tps_estimate
                        if (len(chunk_times_sec) == TIMING_WINDOW_SIZE and rel_tps_err <= REL_TPS_ERR_THRESHOLD):
                            break

        torch.cuda.synchronize()
        end_time = time.perf_counter()

        paci_model.epoch_done()
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        if not os.path.exists(split_file_path):
            state_dict = {"split_indices": paci_model.split_indices}
            torch.save(state_dict, split_file_path)
        
        runtime = end_time - start_time
        mean_dt = sum(chunk_times_sec) / len(chunk_times_sec)
        var = sum((x - mean_dt) ** 2 for x in chunk_times_sec) / (len(chunk_times_sec) - 1)
        std = math.sqrt(max(var, 0.0))
        stderr_mean_dt = std / math.sqrt(len(chunk_times_sec))
        tokens_per_chunk = BATCHES_PER_TIMING_SAMPLE * tokens_per_minibatch
        stable_tps = tokens_per_chunk / mean_dt
        se_tps = (tokens_per_chunk / (mean_dt**2)) * stderr_mean_dt
        final_rel_tps_err = se_tps / stable_tps
        stable_batches_per_second = BATCHES_PER_TIMING_SAMPLE / mean_dt
        stable_tokens_per_second = stable_batches_per_second * tokens_per_minibatch

        return_dict["runtime"] = runtime
        return_dict["measured_batches"] = measured_batches
        return_dict["overall_batches_per_second"] = measured_batches / runtime
        return_dict["overall_tokens_per_second"] = (measured_batches * tokens_per_minibatch) / runtime
        return_dict["max_mem_usage"] = max_mem
        return_dict["status"] = "ok"
        return_dict["batches_per_timing_sample"] = BATCHES_PER_TIMING_SAMPLE
        return_dict["timing_window_size"] = TIMING_WINDOW_SIZE
        return_dict["tps_err_threshold"] = REL_TPS_ERR_THRESHOLD
        return_dict["timing_sample_count"] = len(chunk_times_sec)
        return_dict["final_rel_tps_err"] = final_rel_tps_err
        return_dict["stable_batches_per_second"] = stable_batches_per_second
        return_dict["stable_tokens_per_second"] = stable_tokens_per_second
        return_dict["batches_per_second"] = stable_batches_per_second
        return_dict["tokens_per_second"] = stable_tokens_per_second

        print(return_dict)
    except Exception as e:
        sys.stdout = bu_stdout
        print(e)
        return_dict["exception"] = e
        return_dict["status"] = "exception"
    finally:
        if progress is not None:
            progress.close()
        if paci_model is not None:
            paci_model.stop_training()
