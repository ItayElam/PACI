import os
import math
import random
import numpy as np

import torch
from torch.optim.lr_scheduler import LambdaLR

from Utils.metrics import LanguageModelingLoss
from DataLoaders import OpenWebText
from DataLoaders.OpenWebText import build_openwebtext_splits
from Utils.ChangeFDlimit import set_ulimit
from DistPACI import TrainingConfig, PACIModel, PACIMode, LRstepMode, PipelineMethods
from Models import GPT2_medium
from LLModelTraininer import model_trainer

def set_random(seed):

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



class gpt2_like_lr_lambda:
    def __init__(self, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1) -> float:
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

    def lr_lambda(self, step: int):
        if step < self.warmup_steps:
            return step / float(max(1, self.warmup_steps))

        if step >= self.total_steps:
            return float(self.min_lr_ratio)

        decay_steps = max(1, self.total_steps - self.warmup_steps)
        t = (step - self.warmup_steps) / float(decay_steps)  # in [0, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))    # 1 -> 0
        return float(self.min_lr_ratio) + (1.0 - float(self.min_lr_ratio)) * cosine


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    random_seed = 123
    grad_accumulation = 8
    pipeline_mode = PipelineMethods.paci
    base_early_stopping_step=None
    global_batch_size = 256
    
    base_peak_lr = 3e-4
    create_model = GPT2_medium

    base_total_steps = 190_000                
    warmup_steps_frac = 0.01
    base_resumable_checkpoints_at = [80_000, 160_000]
    sequence_length = 1024
    base_log_training_metrics_every = 500
    base_checkpoint_every = 1000
    base_evaluate_every = 5000

    base_global_batch_size = 256
    peak_lr = math.sqrt(global_batch_size / base_global_batch_size) * base_peak_lr

    steps_multiplier = (base_global_batch_size / global_batch_size)
    total_steps = int(base_total_steps * steps_multiplier)
    warmup_steps = int(warmup_steps_frac * total_steps)
    early_stopping_step = int(base_early_stopping_step * steps_multiplier) if base_early_stopping_step is not None else None
    log_training_metrics_every = int(base_log_training_metrics_every * steps_multiplier)
    checkpoint_every = int(base_checkpoint_every * steps_multiplier)
    evaluate_every = int(base_evaluate_every * steps_multiplier)
    resumable_checkpoints_at = [int(i * steps_multiplier) for i in base_resumable_checkpoints_at]

    paci_mode = PACIMode.perf
    device = 'cuda'
    split_to = torch.cuda.device_count()

    set_random(random_seed)
    model, tokenizer = create_model(context_length=sequence_length)
    unique_id = os.getenv("SLURM_JOB_ID", f"R{os.getpid()}")
    
    basedir = "."

    build_openwebtext_splits(
        tokenizer=tokenizer,
        tokenizer_id="gpt2_tok",
        block_size=sequence_length,
        cache_dir=f"{basedir}/cache/openwebtext/"
    )

    dataset = OpenWebText(global_batch_size, 4, tokenizer, block_size=sequence_length, cache_dir=f"{basedir}/cache/openwebtext/", tokenizer_id="gpt2_tok")
    loss_function = LanguageModelingLoss(tokenizer.vocab_size, -100)

    optimizer_func = build_adamw
    optim_params = {
        "lr": peak_lr,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "weight_decay": 0.1,
    }    

    lr_scheduler_class = LambdaLR
    lr_scheduler_params = {
        "lr_lambda": gpt2_like_lr_lambda(warmup_steps=warmup_steps, total_steps=total_steps).lr_lambda
    }
    lr_step_on = LRstepMode.step

    output_path = f'{basedir}/outputs/{model.model_name}_{split_to}-Layers_{dataset.name}_{unique_id}_{random_seed}'

    set_ulimit(len(dataset.train_loader) * 5)

    config = TrainingConfig(model, build_optimizer_func=optimizer_func, device=device, grad_accumulation=grad_accumulation,
                            log_training_metrics_every=log_training_metrics_every, checkpoint_every=checkpoint_every, 
                            evaluate_every=evaluate_every, optimizer_params=optim_params, total_steps=total_steps,
                            pipeline_mode=pipeline_mode, paci_mode=paci_mode, split_to=split_to, loss_function=loss_function, 
                            lr_scheduler_class=lr_scheduler_class, lr_scheduler_params=lr_scheduler_params, lr_step_on=lr_step_on,
                            output_verbose=False, ignore_token=-100)
    paci_model = PACIModel(model, config, measure_iterations=10)
    
    model_trainer(paci_model, config, dataset, output_path, early_stopping_step, resumable_checkpoints_at)

if __name__ == '__main__':
    main()
