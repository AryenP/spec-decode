"""GPT-2 implemented from scratch with an explicit KV cache.

Only the weights and tokenizer come from HuggingFace; the forward pass, the
cache, and all decoding logic live here so every line can be defended.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_positions: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    layer_norm_eps: float = 1e-5


class KVCache:
    """Pre-allocated key/value cache for every layer.

    Shapes: k, v are (n_layer, B, n_head, max_len, head_dim). `length` is the
    number of valid positions. `rollback` is what speculative decoding needs:
    after the target rejects a draft token, both caches drop the tail.
    """

    def __init__(self, cfg: GPTConfig, batch: int, max_len: int, device, dtype):
        hd = cfg.n_embd // cfg.n_head
        shape = (cfg.n_layer, batch, cfg.n_head, max_len, hd)
        self.k = torch.empty(shape, device=device, dtype=dtype)
        self.v = torch.empty(shape, device=device, dtype=dtype)
        self.max_len = max_len
        self.length = 0

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new k/v for `layer` at the current position; return full views."""
        T = k.shape[2]
        end = self.length + T
        if end > self.max_len:
            raise RuntimeError(f"KV cache overflow: {end} > {self.max_len}")
        self.k[layer, :, :, self.length:end] = k
        self.v[layer, :, :, self.length:end] = v
        return self.k[layer, :, :, :end], self.v[layer, :, :, :end]

    def commit(self, n: int) -> None:
        self.length += n

    def rollback(self, n: int) -> None:
        self.length = max(0, self.length - n)

    def bytes(self) -> int:
        return self.k.numel() * self.k.element_size() * 2


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.layer_idx = layer_idx
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

    def forward(self, x: torch.Tensor, cache: KVCache | None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)

        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            past = cache.length
            k_all, v_all = cache.append(self.layer_idx, k, v)  # (B, nh, past+T, hd)
            if T == 1:
                # Single new token attends to everything cached: no mask needed.
                y = F.scaled_dot_product_attention(q, k_all, v_all)
            else:
                # New tokens see all past positions plus a causal pattern among
                # themselves. Mask shape (T, past+T), True = may attend.
                S = past + T
                mask = torch.ones(T, S, dtype=torch.bool, device=x.device)
                mask[:, past:] = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
                y = F.scaled_dot_product_attention(q, k_all, v_all, attn_mask=mask)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.c_proj(F.gelu(self.c_fc(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, layer_idx: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cache):
        x = x + self.attn(self.ln_1(x), cache)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.n_positions, cfg.n_embd)
        self.h = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, eps=cfg.layer_norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tied, as in GPT-2

    @property
    def device(self):
        return self.wte.weight.device

    @property
    def dtype(self):
        return self.wte.weight.dtype

    def new_cache(self, batch: int, max_len: int) -> KVCache:
        return KVCache(self.cfg, batch, max_len, self.device, self.dtype)

    @torch.no_grad()
    def forward(self, idx: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        """Return logits (B, T, V) for `idx`.

        With a cache, `idx` holds only the NEW tokens; positions start at
        cache.length. The cache is advanced by T on return.
        """
        B, T = idx.shape
        start = cache.length if cache is not None else 0
        if start + T > self.cfg.n_positions:
            raise RuntimeError(f"sequence length {start + T} exceeds n_positions {self.cfg.n_positions}")
        pos = torch.arange(start, start + T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.h:
            x = block(x, cache)
        if cache is not None:
            cache.commit(T)
        return self.lm_head(self.ln_f(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_pretrained(cls, name: str, device="cpu", dtype=torch.float32) -> "GPT":
        """Load HF GPT-2 weights into this implementation.

        HF stores c_attn/c_proj/c_fc as Conv1D with weight (in, out); nn.Linear
        wants (out, in), hence the transposes.
        """
        from transformers import GPT2LMHeadModel

        hf = GPT2LMHeadModel.from_pretrained(name, torch_dtype=torch.float32)
        hc = hf.config
        cfg = GPTConfig(
            vocab_size=hc.vocab_size,
            n_positions=hc.n_positions,
            n_layer=hc.n_layer,
            n_head=hc.n_head,
            n_embd=hc.n_embd,
            layer_norm_eps=hc.layer_norm_epsilon,
        )
        model = cls(cfg)
        sd = model.state_dict()
        hsd = hf.state_dict()
        transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
        for k in sd:
            if k == "lm_head.weight":
                continue  # tied to wte
            hk = "transformer." + k
            w = hsd[hk]
            if k.endswith(transposed):
                w = w.t()
            assert w.shape == sd[k].shape, (k, w.shape, sd[k].shape)
            sd[k].copy_(w)
        del hf
        return model.to(device=device, dtype=dtype).eval()
