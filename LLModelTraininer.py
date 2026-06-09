import os
import time
import platform
import torch
from tqdm import tqdm
from Utils.metrics import evaluate_perplexity
from Utils.Logging import TensorBoardLogger, redirect_output_to_file
from DistPACI import TrainingConfig, PACIModel
import gc



def log_training_metrics(tb_logger, results, current_step, tokens_per_second, split_to, total_steps):
    metrics_win = evaluate_perplexity(results)
    tb_logger.log_metrics(metrics_win, step=current_step, prefix="train_average")
    perf_dict = {
        "tokens/second": tokens_per_second,
        }
    for i in range(split_to):
        perf_dict[f"max_cuda{i}_mem"] = torch.cuda.max_memory_allocated(f"cuda:{i}")
    torch.cuda.reset_peak_memory_stats()
    tb_logger.log_metrics(perf_dict, step=current_step, prefix="perf")

    print(f"\nstep {current_step}/{total_steps}")
    print("Training:")
    [print(f"\t{k}: {v}") for k, v in metrics_win.items()]


def model_trainer(paci_model: PACIModel, config: TrainingConfig, dataset, output_path, early_stopping_step=None, resumable_checkpoints_at=None):
    total_steps = config.total_steps
    split_to = config.split_to
    log_training_metrics_every = config.log_training_metrics_every
    checkpoint_every = config.checkpoint_every
    evaluate_every = config.evaluate_every


    latest_path = os.path.join(output_path, 'checkpoint_latest.pth')
    os.makedirs(output_path, exist_ok=True)

    redirect_output_to_file(os.path.join(output_path, "run.log"))
    
    info = (f'\n\npython: {platform.python_version()}, torch: {torch.__version__}, cudnn: {torch.backends.cudnn.version()}, ' 
          f'cuda: {torch.version.cuda}, gpus: {torch.cuda.device_count()} X {torch.cuda.get_device_name(0)}\n')
    print(info)

    with open(f"{output_path}/config.txt", 'w') as f:
        f.write(info)
        f.write(str(config))

    start_step = 1
    input_sample, _, _ = next(iter(dataset.train_loader))

    if os.path.exists(latest_path):
        print("Found existing checkpoint, loading..")
        state_dict = torch.load(latest_path, map_location='cpu')
        paci_model.load_state_dict(state_dict, input_sample)

        last_finished_step = int(state_dict["last_step"])  # 1-based, last FINISHED
        steps_per_epoch = len(dataset.train_loader)
        epoch = last_finished_step // steps_per_epoch
        batch_pos = last_finished_step % steps_per_epoch
        dataset.train_batch_sampler.load_state_dict({"epoch": epoch, "batch_pos": batch_pos})

        start_step = last_finished_step + 1


    tb_logger = TensorBoardLogger(os.path.join(output_path, 'tensorboard'), purge_step=start_step)

    paci_model.init(input_sample)
    paci_model.half()

    paci_model.train()
    results = []
    
    print(f"Starting training | {len(dataset.train_loader)} steps in epoch totaling {total_steps / (len(dataset.train_loader)):.2f} epochs")
    print(f"Saving results to {output_path}")

    loader = iter(dataset.train_loader)
    loop = tqdm(range(start_step, total_steps + 1), desc="Training", smoothing=0)
    
    training_start_time = time.time()
    total_tokens = 0
    for current_step in loop:
        try:
            input_ids, target_ids, _ = next(loader)
        except StopIteration:
            print("\nFinished epoch")
            dataset.train_batch_sampler.epoch_done()
            loader = iter(dataset.train_loader)
            input_ids, target_ids, _ = next(loader)

        total_tokens += input_ids.size(0) * input_ids.size(1)
        res_list = paci_model(input_ids, target_ids)
        for res in res_list:
            results.append(res)
            
        tokens_per_second = total_tokens / (time.time() - training_start_time)
        loop.set_description(f'Training step {current_step}/{total_steps} | {tokens_per_second:.3f} tokens/second (estimated)')
        
        if current_step % log_training_metrics_every == 0:
            log_training_metrics(tb_logger, results, current_step, tokens_per_second, split_to, total_steps)
            results = []

        if current_step % evaluate_every == 0:
            paci_model.eval()
            eval_results = []
            for input_ids, target_ids, _ in tqdm(dataset.test_loader):
                result_list = paci_model(input_ids, target_ids)
                for res in result_list:
                    eval_results.append(res)
            metrics_win = evaluate_perplexity(eval_results)
            tb_logger.log_metrics(metrics_win, step=current_step, prefix="validation")
            print("\nEvaluation:")
            [print(f"\t{k}: {v}") for k, v in metrics_win.items()]
            
            eval_results = []
            paci_model.train()

        if current_step % checkpoint_every == 0:
            print(f"Saving checkpoint {current_step}")
            st = time.time()
            state_dict = paci_model.state_dict()
            print(f"Time Fetching state: {time.time()-st:.2f} seconds")
            state_dict['last_step'] = current_step
            st = time.time()
            torch.save(state_dict, latest_path)
            if resumable_checkpoints_at is not None and current_step in resumable_checkpoints_at:
                resumable_path = os.path.join(output_path, f'resumeable_checkpoint_{current_step}.pth')
                torch.save(state_dict, resumable_path)

            if current_step == total_steps:
                model_dict = {
                        "model_state_dict": state_dict["model_state_dict"],
                        "split_indices": state_dict["split_indices"],
                    }
                checkpoint_path = os.path.join(output_path, f'checkpoint_{current_step}.pth')
                torch.save(model_dict, checkpoint_path)
                del model_dict

            print(f"Time saving to disk: {time.time()-st:.2f} seconds")
            
            del state_dict
            gc.collect()
        
        if early_stopping_step is not None and current_step == early_stopping_step:
            break
    paci_model.stop_training()
    os._exit(0)
