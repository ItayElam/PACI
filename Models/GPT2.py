from __future__ import annotations

import sys
sys.path.append("../")

from Models.base import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
from transformers import AutoTokenizer, GPT2Tokenizer


@dataclass
class GPTConfig:
    context_length: int = 512
    vocab_size: int = 50257
    num_layers: int = 12
    embd_size: int = 768
    num_heads: int = 12
    embd_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))
    

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # 'embd_size' sized vector divided into 'num_heads' heads
        assert config.embd_size % config.num_heads == 0, "embedding dim should be divisible by number of heads"
        self.num_heads = config.num_heads
        self.embd_size = config.embd_size
        # batched key, query, and value projections for all heads
        self.c_attn = nn.Linear(config.embd_size, 3 * config.embd_size)
        self.c_proj = nn.Linear(config.embd_size, config.embd_size)
        self.c_proj.SCALE_INIT = 1.0

        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.config = config

        # not really a bias, more of a mask, but following OpenAI/HF naming convention
        # self.register_buffer("bias", torch.tril(torch.ones(config.context_length, config.context_length)).view(1, 1, config.context_length, config.context_length))

    def forward(self, x):
        B, T, C = x.shape
        # calculate query, key, values for all heads in a batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels
        qkv = self.c_attn(x)    # (B, T, 3C)
        q, k, v = qkv.split(self.embd_size, dim=-1)    # (B,T,C), (B,T,C), (B,T,C)
        q = q.view(B, T, self.num_heads, self.embd_size // self.num_heads).transpose(1, 2)    # (B,nh,T,hs)
        k = k.view(B, T, self.num_heads, self.embd_size // self.num_heads).transpose(1, 2)    # (B,nh,T,hs)
        v = v.view(B, T, self.num_heads, self.embd_size // self.num_heads).transpose(1, 2)    # (B,nh,T,hs)
        # attn = q @ k.transpose(-2, -1) / np.sqrt(k.shape[-1])    # (B,nh,T,hs) @ (B,nh,hs,T) --> (B,nh,T,T)
        # attn = attn.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        # attn = F.softmax(attn, dim=-1)
        # out = attn @ v    # (B,nh,T,T) @ (B,nh,T,hs) --> (B,nh,T,hs)
        # flash-attention paper (significantly faster, but logically the same as above 4 lines)
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.config.attn_pdrop if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.c_proj(out)
        out = self.resid_drop(out)
        return out

    def spill(self):
        return [self]

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.embd_size, 4 * config.embd_size)
        # self.gelu = nn.GELU(approximate='tanh')    # approximate='tanh' used to try to reproduce gpt2 paper
        self.gelu = NewGELU()
        self.c_proj = nn.Linear(4 * config.embd_size, config.embd_size)
        self.c_proj.SCALE_INIT = 1.0
        self.dropout = nn.Dropout(config.resid_pdrop)


    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

    def spill(self):
        return [self.c_fc, self.gelu, self.c_proj, self.dropout]

class Block(nn.Module):
    """ Transformer Encoder block """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.embd_size, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.embd_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    class Block1(nn.Module):
        def __init__(self, block: Block):
            super().__init__()
            self.ln_1 = block.ln_1
            self.attn = block.attn

        def forward(self, x):
            return x + self.attn(self.ln_1(x))

    class Block2(nn.Module):
        def __init__(self, block: Block):
            super().__init__()
            self.ln_2 = block.ln_2
            self.mlp = block.mlp

        def forward(self, x):
            return x + self.mlp(self.ln_2(x))

    def spill(self):
        return [Block.Block1(self), Block.Block2(self)]

class EmbeddingBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(self.config.vocab_size, self.config.embd_size)
        self.wpe = nn.Embedding(self.config.context_length, self.config.embd_size)
        self.drop = nn.Dropout(self.config.embd_pdrop)


    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.config.context_length, f'sequence length {T} should be <= {self.config.context_length}'
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)    # (T,)
        pos_embd = self.wpe(pos)    # (T, embd_size)
        tok_embd = self.wte(idx)    # (B, T, embd_size)
        x = pos_embd + tok_embd    # (B, T, embd_size)
        x = self.drop(x)
        return x

    def spill(self):
        return [self]
    
class GPT(nn.Module):
    """ adapted from https://github.com/saqib1707/gpt2-from-scratch/blob/master/src/model.py"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.embedding_block = EmbeddingBlock(config)
        self.transformer_blocks = nn.ModuleList([Block(self.config) for _ in range(self.config.num_layers)]) 
        self.ln_f = nn.LayerNorm(self.config.embd_size, eps=config.layer_norm_epsilon)

        # language modeling head
        self.lm_head = nn.Linear(self.config.embd_size, self.config.vocab_size, bias=False)
        # init params (iterates over all submodules and applies _init_weights)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'SCALE_INIT'):
                std /= (2 * self.config.num_layers)**0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)    # as per openai gpt-2 source code
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, idx):
        
        x = self.embedding_block(idx)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.ln_f(x)    # (B, T, embd_size)
        logits = self.lm_head(x)    # (B, T, vocab_size)
        return logits

    

# Sequential Model Construction
class GPT2SequentialModel(BaseModel):
    def __init__(self, model: GPT = None, config: GPTConfig = None):
        super().__init__()
        assert model is not None or config is not None, "At least one of model or config should be provided"
        if model is None:
            model = GPT(config)
        self.config = model.config
        self.layers = nn.Sequential()
        [self.layers.append(i) for i in model.embedding_block.spill()]
        for block in model.transformer_blocks:
            for b in block.spill():
                self.layers.append(b)
        self.layers.append(model.ln_f)
        self.layers.append(model.lm_head)
        
        self.total_params = 0
        for param in model.parameters():
            self.total_params += torch.numel(param)
        self.total_params /= 1e6
        
    def create_layers(self, *args, **kwargs):
        pass
    
    @property
    def model_name(self):
        return f"GPT-{self.total_params:.1f}M"

    def forward(self, input_ids):
        return self.layers(input_ids)

    @classmethod
    def from_pretrained(cls, model_type: str) -> "GPT":
        """
        Load weights from Hugging Face GPT2LMHeadModel into this GPT implementation,
        accounting for:
          - different module naming (embedding_block.*, transformer_blocks.*)
          - Conv1D vs Linear weight layout (transpose needed)
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        from transformers import GPT2LMHeadModel

        print(f"loading weights from pretrained gpt: {model_type}")

        cfg_map = {
            "gpt2":        dict(num_layers=12, num_heads=12, embd_size=768),
            "gpt2-medium": dict(num_layers=24, num_heads=16, embd_size=1024),
            "gpt2-large":  dict(num_layers=36, num_heads=20, embd_size=1280),
            "gpt2-xl":     dict(num_layers=48, num_heads=25, embd_size=1600),
        }
        cfg_args = cfg_map[model_type]
        cfg_args["vocab_size"] = 50257
        # must be 1024 to match HF GPT-2's positional embeddings
        cfg_args["context_length"] = 1024

        config = GPTConfig(**cfg_args)
        model = GPT(config)
        sd = model.state_dict()

        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = hf_model.state_dict()

        transposed = (
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        )

        new_sd = {}

        for k, v in sd_hf.items():
            # skip masks/biases that don't exist in our implementation
            if k.endswith(".attn.masked_bias") or k.endswith(".attn.bias"):
                print(f"Skipping {k}")
                continue

            new_k = k
            # map HF names -> our names
            new_k = new_k.replace("transformer.wte.", "embedding_block.wte.")
            new_k = new_k.replace("transformer.wpe.", "embedding_block.wpe.")
            new_k = new_k.replace("transformer.h.", "transformer_blocks.")
            new_k = new_k.replace("transformer.ln_f.", "ln_f.")

            # (lm_head.* remains lm_head.*)

            if new_k not in sd:
                print(sd.keys())
                raise KeyError(f"Key {new_k} (from {k}) not found in target model state_dict")

            if any(new_k.endswith(w) for w in transposed):
                assert v.shape[::-1] == sd[new_k].shape, (k, v.shape, sd[new_k].shape)
                new_sd[new_k] = v.t()
            else:
                assert v.shape == sd[new_k].shape, (k, v.shape, sd[new_k].shape)
                new_sd[new_k] = v

        model.load_state_dict(new_sd, strict=True)
        model.config = config
        return cls(model=model)


def check_size():
    config = GPTConfig(num_layers=24, embd_size=1024, num_heads=16)
    model = GPT2SequentialModel(config=config)
    total = 0
    for param in model.parameters():
        total += torch.numel(param)
    print(f"Total params: {total / 1e6} M, {len(model.layers)}")

def check_correct():
    import torch
    from transformers import GPT2LMHeadModel
    import os
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    hf = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="sdpa",).to(device=device, dtype=dtype).eval()
    model = GPT2SequentialModel.from_pretrained("gpt2").to(device=device, dtype=dtype).eval()
    print("HF eps:", hf.config.layer_norm_epsilon)
    print("Seq eps:", model.config.layer_norm_epsilon)
    print(type(hf.transformer.h[0].mlp.act))
    print(hf.transformer.h[0].mlp.act)

    # Same random input
    torch.manual_seed(42)
    x = torch.randint(0, 50257, (2, 32), device=device)

    with torch.no_grad():
        logits_hf = hf(x).logits
        logits_ours = model(x)
    
    assert logits_hf.shape == logits_ours.shape

    print("max abs diff:", (logits_hf - logits_ours).abs().max())



if __name__ == "__main__":
    check_correct()
    
    
def GPT2(context_length=512, num_layers=24, embd_size=1024, num_heads=16) -> tuple[BaseModel, GPT2Tokenizer]:
    config = GPTConfig(context_length=context_length, num_layers=num_layers, embd_size=embd_size, num_heads=num_heads)
    model = GPT2SequentialModel(config=config)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    return model, tokenizer

def GPT2_medium(context_length=512) -> tuple[BaseModel, GPT2Tokenizer]:
    return GPT2(context_length, num_layers=24, embd_size=1024, num_heads=16)

def GPT2_large(context_length=512):
    return GPT2(context_length, num_layers=36, embd_size=1280, num_heads=20)

def GPT2_xlarge(context_length=512):
    return GPT2(context_length, num_layers=48, embd_size=1600, num_heads=25)
