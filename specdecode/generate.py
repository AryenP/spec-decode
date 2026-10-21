"""Three decoders over the same model: no cache, KV cache, speculative.

Speculative decoding follows Leviathan, Kalman & Matias (2023), "Fast
Inference from Transformers via Speculative Decoding": a small draft model
proposes gamma tokens, the target verifies them in ONE forward pass, and the
accept/reject rule guarantees the output distribution equals the target's.
Greedy (temperature 0) falls out as the special case where acceptance means
"draft token == target argmax", so the output is bit-identical to target greedy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .model import GPT, KVCache


def _sample(probs: torch.Tensor, gen: torch.Generator | None) -> torch.Tensor:
    """probs: (..., V) -> token ids (...)."""
    flat = probs.reshape(-1, probs.shape[-1])
    # torch.multinomial on MPS lacks generator support; sample on CPU.
    out = torch.multinomial(flat.cpu(), 1, generator=gen).squeeze(-1)
    return out.view(probs.shape[:-1]).to(probs.device)


def _probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return F.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
    return F.softmax(logits.float() / temperature, dim=-1).to(logits.dtype)


@dataclass
class GenStats:
    new_tokens: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    drafted: int = 0
    accepted: int = 0
    rounds: int = 0
    acceptance_per_round: list[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    @property
    def mean_accepted_per_round(self) -> float:
        return self.accepted / self.rounds if self.rounds else 0.0


@torch.no_grad()
def generate_no_cache(model: GPT, prompt: torch.Tensor, max_new: int, temperature: float = 0.0,
                      gen: torch.Generator | None = None) -> tuple[torch.Tensor, GenStats]:
    """Baseline: re-run the full sequence every step. O(n^2) in sequence length."""
    stats = GenStats()
    ids = prompt
    for _ in range(max_new):
        logits = model(ids)[:, -1]
        stats.target_forwards += 1
        nxt = _sample(_probs(logits, temperature), gen)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        stats.new_tokens += 1
    return ids, stats


@torch.no_grad()
def generate_kv(model: GPT, prompt: torch.Tensor, max_new: int, temperature: float = 0.0,
                gen: torch.Generator | None = None) -> tuple[torch.Tensor, GenStats]:
    """Prefill the prompt once, then one forward per new token over a single position."""
    stats = GenStats()
    B, T = prompt.shape
    cache = model.new_cache(B, T + max_new)
    logits = model(prompt, cache)[:, -1]
    stats.target_forwards += 1
    ids = prompt
    for _ in range(max_new):
        nxt = _sample(_probs(logits, temperature), gen)
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        stats.new_tokens += 1
        if stats.new_tokens == max_new:
            break
        logits = model(nxt[:, None], cache)[:, -1]
        stats.target_forwards += 1
    return ids, stats


@torch.no_grad()
def generate_speculative(target: GPT, draft: GPT, prompt: torch.Tensor, max_new: int, gamma: int = 4,
                         temperature: float = 0.0, gen: torch.Generator | None = None,
                         ) -> tuple[torch.Tensor, GenStats]:
    """Speculative decoding with KV caches on both models. Batch size 1 only.

    Invariant: each cache covers exactly a prefix of `ids` (the committed
    sequence). Tokens of `ids` the cache lacks are fed together with the next
    round's drafts, so the target does ONE forward per round and nothing is
    ever recomputed.

    Each round:
      1. draft: feed the committed tokens its cache lacks, then propose gamma
         tokens autoregressively (gamma forwards in total)
      2. target: one forward over [missing committed tokens] + [gamma drafts];
         its gamma+1 output positions are p(.|prefix), p(.|prefix,d0), ...
      3. accept d_i with prob min(1, p_i(d_i)/q_i(d_i)); on the first
         rejection resample from norm(max(0, p_i - q_i)) and stop
      4. if all gamma accepted, the last target position gives a bonus token
      5. roll both caches back over the rejected drafts only
    """
    assert prompt.shape[0] == 1, "speculative decoder is written for batch size 1"
    stats = GenStats()
    T = prompt.shape[1]
    cap = T + max_new + gamma + 2
    tc = target.new_cache(1, cap)
    dc = draft.new_cache(1, cap)
    ids = prompt

    # Prefill all but the last prompt token; the last one rides along with
    # the first round's forward so no position is processed twice.
    if T > 1:
        target(prompt[:, :-1], tc)
        draft(prompt[:, :-1], dc)
        stats.target_forwards += 1
        stats.draft_forwards += 1

    n_done = 0
    while n_done < max_new:
        g = min(gamma, max_new - n_done)

        # ---- 1. draft proposes
        logits = draft(ids[:, dc.length:], dc)[:, -1]
        stats.draft_forwards += 1
        q_list, toks = [], []
        for i in range(g):
            q = _probs(logits, temperature)
            tok = _sample(q, gen)
            q_list.append(q)
            toks.append(tok)
            if i < g - 1:
                logits = draft(tok[:, None], dc)[:, -1]
                stats.draft_forwards += 1
        d_tok = torch.stack(toks, dim=1)  # (1, g); draft cache covers ids + d_0..d_{g-2}
        stats.drafted += g

        # ---- 2. target verifies in one pass
        pending = ids[:, tc.length:]  # >= 1 token
        t_logits = target(torch.cat([pending, d_tok], dim=1), tc)
        stats.target_forwards += 1
        off = pending.shape[1] - 1  # index whose output is p(.|ids)
        p_list = [_probs(t_logits[:, off + i], temperature) for i in range(g + 1)]
        # target cache covers ids + d_0..d_{g-1}

        # ---- 3. accept / reject
        n_acc = 0
        resampled = None
        for i in range(g):
            tok = d_tok[0, i]
            p_i, q_i = p_list[i][0, tok], q_list[i][0, tok]
            if temperature == 0.0:
                accept = bool(p_i > 0)  # argmax match
            else:
                r = torch.rand((), generator=gen).item()
                accept = r < min(1.0, (p_i / q_i).item())
            if accept:
                n_acc += 1
            else:
                resid = torch.clamp(p_list[i] - q_list[i], min=0)
                s = resid.sum()
                resid = resid / s if s > 0 else p_list[i]
                resampled = _sample(resid, gen)
                break
        stats.accepted += n_acc
        stats.rounds += 1
        stats.acceptance_per_round.append(n_acc)

        # ---- 4. commit
        last = _sample(p_list[g], gen) if resampled is None else resampled
        new = torch.cat([d_tok[0, :n_acc], last])[: max_new - n_done]
        ids = torch.cat([ids, new[None]], dim=1)
        n_done += new.numel()
        stats.new_tokens = n_done

        # ---- 5. roll back over rejected drafts only
        tc.rollback(g - n_acc)                 # keep ids_old + accepted drafts
        dc.rollback(max(0, (g - 1) - n_acc))   # draft never saw d_{g-1}

    return ids, stats
