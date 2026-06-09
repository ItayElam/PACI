from __future__ import annotations
import csv
import math
import time
import sys

import os
import torch
from tqdm import tqdm
from prettytable import PrettyTable
from utils import benchmark_model, build_adamw, make_static_minibatch, set_random
import warnings

sys.path.append("../")
from Utils.metrics import LanguageModelingLoss   # noqa: E402
from DistPACI import TrainingConfig, PipelineMethods, PACIMode   # noqa: E402
from Models import GPT2_medium  # noqa: E402
warnings.filterwarnings("ignore", category=UserWarning) 


_BATCH_PARAM_UNSET = object()

def save_experiment_records(all_records, folder_path, gpu_name, experiment_name):
    gpu_dir = os.path.join(folder_path, gpu_name)
    os.makedirs(gpu_dir, exist_ok=True)

    csv_path = os.path.join(gpu_dir, f"{experiment_name}.csv")
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_records[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(all_records)

def _iter_batch_configs(
    micro_batch_sizes,
    num_micro_batches,
    mini_batch_sizes,
):
    if micro_batch_sizes is not None and num_micro_batches is not None:
        for micro_batch_size in micro_batch_sizes:
            for num_micros in num_micro_batches:
                yield micro_batch_size, num_micros, micro_batch_size * num_micros
        return
    if micro_batch_sizes is not None and mini_batch_sizes is not None:
        for micro_batch_size in micro_batch_sizes:
            for mini_batch_size in mini_batch_sizes:
                num_micros = math.ceil(mini_batch_size / micro_batch_size)
                yield micro_batch_size, num_micros, mini_batch_size
        return
    if num_micro_batches is not None and mini_batch_sizes is not None:
        for num_micros in num_micro_batches:
            for mini_batch_size in mini_batch_sizes:
                yield mini_batch_size / num_micros, num_micros, mini_batch_size
        return
    raise RuntimeError("unreachable batch config branch")


def generate_sweep(pp_mode_list = [PipelineMethods.flush, PipelineMethods.paci],
                    seq_len_list=[1024], num_gpus=[8], flush_virtual_mult=[1],
                    micro_batch_sizes=_BATCH_PARAM_UNSET,
                    num_micro_batches=_BATCH_PARAM_UNSET,
                    mini_batch_sizes=_BATCH_PARAM_UNSET,
                    ):
    unset = (
        micro_batch_sizes is _BATCH_PARAM_UNSET,
        num_micro_batches is _BATCH_PARAM_UNSET,
        mini_batch_sizes is _BATCH_PARAM_UNSET,
    )
    n_unset = sum(unset)
    if n_unset == 1:
        micro_batch_sizes = None if unset[0] else micro_batch_sizes
        num_micro_batches = None if unset[1] else num_micro_batches
        mini_batch_sizes = None if unset[2] else mini_batch_sizes
    else:
        raise ValueError("Provide exactly two of micro_batch_sizes, num_micro_batches, mini_batch_sizes")

    for pp in num_gpus:
        for micro_batch_size, num_micros, mini_batch_size in _iter_batch_configs(
            micro_batch_sizes, num_micro_batches, mini_batch_sizes
        ):
            for seq_len in seq_len_list:
                for pp_mode in pp_mode_list:
                    for v in flush_virtual_mult:
                        if pp_mode != PipelineMethods.flush and v != 1:
                            continue

                        yield {
                            "pp_mode": pp_mode,

                            # Model / data
                            "seq_len": seq_len,

                            # Batch configuration
                            "mini_batch_size": mini_batch_size,
                            "num_micro_batches": num_micros,
                            "micro_batch_size": micro_batch_size,

                            # Pipeline configuration
                            "split_to": pp * v,
                            "virt": v,

                            # Derived info
                            "tokens_per_step": mini_batch_size * seq_len,
                        }


def run_benchmark(benchmark, experiment_type):

    set_random(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    total_steps = 300_000                
    paci_mode = PACIMode.perf
    device="cuda"
    model, tokenizer = GPT2_medium(context_length=1024)
    
    
    loss_function = LanguageModelingLoss(tokenizer.vocab_size, -100)

    optimizer_func = build_adamw
    optim_params = {
        "lr": 1e-4,             
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 1e-2,
    }


    all_records = []
    ctx = torch.multiprocessing.get_context('spawn')
    live_batches = ctx.Value('i', 0)
    live_stable = ctx.Value('d', 0.0)
    live_rel = ctx.Value('d', float('inf'))

    table = PrettyTable()
    table.field_names = [
        "mode",
        'GPUs',
        "stages",
        "num_micro",
        "micro_size",
        "seq_len",
        "tokens/sec",
        "tps_norm",
        "max_mem_usage",
    ]

    loop = tqdm(list(benchmark))
    for cfg in loop:
        pipeline_mode = cfg["pp_mode"]
        benchmark_target = benchmark_model

        split_to = cfg["split_to"]
        grad_accumulation = cfg["num_micro_batches"]
        virtual_pipelines = cfg["virt"]
        micro_batch_size =cfg['micro_batch_size']
        seq_len = cfg["seq_len"]
        desc = f"{pipeline_mode} {split_to//virtual_pipelines} GPUs, {split_to} stages | num micro {cfg['num_micro_batches']}, micro size {micro_batch_size}, seq len {seq_len}"
        loop.set_description(desc)
        # print(desc)
        

        config = TrainingConfig(model, build_optimizer_func=optimizer_func, device=device, grad_accumulation=grad_accumulation,
                                optimizer_params=optim_params, total_steps=total_steps, pipeline_mode=pipeline_mode, paci_mode=paci_mode,
                                split_to=split_to, virtual_stages=virtual_pipelines, loss_function=loss_function, 
                                output_verbose=False, ignore_token=-100)

        manager = ctx.Manager()
        return_dict = manager.dict()
        input_sample, target_sample = make_static_minibatch(cfg["mini_batch_size"], cfg["seq_len"], tokenizer.vocab_size, device="cpu")

        live_batches.value = 0
        live_stable.value = 0.0
        live_rel.value = float("inf")
        bench_args = (
            model,
            config,
            input_sample,
            target_sample,
            return_dict,
            (live_batches, live_stable, live_rel),
        )

        process = ctx.Process(target=benchmark_target, args=bench_args)
        process.start()

        while process.is_alive():
            nb = live_batches.value
            stable = live_stable.value
            rel = live_rel.value
            rel_s = f"{rel:.2%}" if rel != float("inf") else "inf"
            loop.set_postfix_str(
                f"batches={nb} | stable_tps={stable:.2f} | rel_SE={rel_s}",
                refresh=True,
            )
            loop.refresh()
            time.sleep(0.05)
        process.join()
        loop.set_postfix_str("", refresh=True)
        loop.refresh()
        if return_dict["status"] == "ok":
            tokens_per_second = return_dict["tokens_per_second"]
            max_mem = max(list(return_dict["max_mem_usage"].values()))
            tps_norm = tokens_per_second / (split_to // virtual_pipelines)
            tokens_per_second = f"{tokens_per_second:.2f}"
            tps_norm = f"{tps_norm:.2f}"
        else:
            tokens_per_second = "0 (OOM)"
            tps_norm = "0 (OOM)"
            max_mem = "OOM"
        table.add_row([
                f"{pipeline_mode} V={virtual_pipelines}",
                split_to // virtual_pipelines,
                split_to,
                grad_accumulation,
                micro_batch_size,
                seq_len,
                tokens_per_second,
                tps_norm,
                max_mem,
            ])
        all_records.append(
            {
            "mode": f"{pipeline_mode}" + (f" V={virtual_pipelines}" if pipeline_mode == PipelineMethods.flush else ""),
            'GPUs': split_to // virtual_pipelines,
            "stages": split_to,
            "num_micro": grad_accumulation,
            "micro_size": micro_batch_size,
            "seq_len": seq_len,
            "tokens/sec": tokens_per_second,
            "tps_norm": tps_norm,
            "max_mem_usage": max_mem,
            }
        )
        save_experiment_records([all_records[-1]], "plots", torch.cuda.get_device_name(), experiment_type)    
        
    
    return all_records, table

if __name__ == '__main__':

    for _ in range(3):
        all_records, table1 = run_benchmark(generate_sweep(num_micro_batches=[4, 8, 16, 24, 32, 40], micro_batch_sizes=[4, 8, 16, 32],
                                                        num_gpus=[8],
                                                        pp_mode_list=[PipelineMethods.paci, PipelineMethods.flush], flush_virtual_mult=[1]),
                                            experiment_type="microbatch_sweep"
                                            )

        all_records, table1 = run_benchmark(generate_sweep(num_micro_batches=[4, 8, 16, 24, 32, 40], mini_batch_sizes=[128, 256],
                                                        num_gpus=[8],
                                                        pp_mode_list=[PipelineMethods.paci, PipelineMethods.flush], flush_virtual_mult=[1]),
                                            experiment_type="minibatch_sweep"
                                            )
