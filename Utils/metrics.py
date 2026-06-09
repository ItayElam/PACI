import math
import torch
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)



def evaluate_perplexity(model_results):
    total_loss = 0.0
    total_tokens = 0
    last_lr = []

    for res_obj in model_results:
        i = res_obj.get_data(wait=True)
        loss = float(i["loss"])
        tokens = int(i["tokens"])
        total_loss += loss * tokens
        total_tokens += tokens
        last_lr = [float(j) for j in i["lr"]]

    if total_tokens == 0:
        print("\nERROR: Got total tokens = 0")
        return {"avg_loss": float("nan"), "perplexity": float("nan")}

    avg_loss = total_loss / total_tokens
    
    metrics = {"avg_loss": avg_loss, "perplexity": math.exp(avg_loss), }
    for j, lr in enumerate(last_lr):
        metrics[f"lr group {j}"] = lr
    return metrics
  
class LanguageModelingLoss:
    def __init__(self, vocab_size, ignore_index=0):
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.vocab_size = vocab_size
    def __call__(self, output, target):
        logits = output.view(-1, output.size(-1))

        target_ids = target.view(-1)  # (Batch * Seq Length)
        loss = self.loss_fn(logits, target_ids)
        return loss