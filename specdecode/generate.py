"""Decoders over the from-scratch GPT: no cache, KV cache, speculative
(single and batched), and prompt-lookup speculative with no draft model.

Speculative decoding follows Leviathan, Kalman & Matias (2023). A draft
proposes gamma tokens, the target scores all of them in ONE forward pass, and
the accept/reject rule makes the output distribution exactly the target's.
Greedy is the special case "accept while draft token == target argmax", so
greedy output is bit-identical to the target decoding alone.

Host-side discipline matters as much as the algorithm: v2 of this file did a
GPU sync per draft token (`.item()` in the accept loop) and sampled on the
CPU. On an RTX 4090 that overhead was larger than a 1.5B forward pass and
speculative came out slower than plain KV decoding. v3 does one sync per
round, samples on the device, and writes into a preallocated buffer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .model import GPT


# ----------------------------------------------------------------------------- sampling helpers

def _softmax(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.softmax(logits.float() / temperature, dim=-1)


def _multinomial(probs: torch.Tensor, gen: torch.Generator | None) -> torch.Tensor:
    """probs (N, V) -> (N,) token ids.

    MPS has no generator support in multinomial, so sample on the CPU there.
    A CPU generator with device tensors (the tests do this) also goes via CPU.
    Everywhere else the draw stays on the device: no sync, no transfer.
    """
    if probs.device.type == "mps" or (gen is not None and gen.device.type != probs.device.type):
        return torch.multinomial(probs.cpu(), 1, generator=gen).squeeze(-1).to(probs.device)
    return torch.multinomial(probs, 1, generator=gen).squeeze(-1)


def _rand(shape, gen: torch.Generator | None, device) -> torch.Tensor:
    if gen is not None:
        return torch.rand(shape, generator=gen, device=gen.device)
    return torch.rand(shape, device=device)


def _next_tokens(logits: torch.Tensor, temperature: float, gen) -> torch.Tensor:
    """(..., V) logits -> (...) tokens."""
    if temperature == 0.0:
        return logits.argmax(-1)
    p = _softmax(logits, temperature)
    return _multinomial(p.reshape(-1, p.shape[-1]), gen).view(p.shape[:-1])


def _verify(d: torch.Tensor, tl: torch.Tensor, q: torch.Tensor | None, temperature: float, gen,
            ) -> tuple[int, torch.Tensor]:
    """One row's accept/reject. d: (g,) drafts; tl: (g+1, V) target logits,
    tl[i] is the distribution for position i (tl[g] is the bonus slot);
    q: (g, V) draft probabilities, or None in greedy mode.
    Returns (n_accepted, final token). Exactly one host sync.
    """
    g = d.numel()
    if temperature == 0.0:
        tgt = tl.argmax(-1)                                   # (g+1,)
        match = (d == tgt[:g]).float()
        n_acc = int(torch.cumprod(match, 0).sum())            # leading run of matches; the sync
        # In greedy mode the residual max(0, p - q) is one-hot at the target
        # argmax, and the bonus is the argmax too: both are tgt[n_acc].
        return n_acc, tgt[n_acc]
    p = _softmax(tl, temperature)                             # (g+1, V)
    if g == 0:
        return 0, _multinomial(p[g:g + 1], gen)[0]
    ar = torch.arange(g, device=d.device)
    ratio = (p[ar, d] / q[ar, d]).clamp(max=1.0)              # min(1, p/q) per draft
    r = _rand(g, gen, d.device)
    accept = (r < ratio.to(r.device)).float()
    n_acc = int(torch.cumprod(accept, 0).sum())               # the sync
    if n_acc == g:
        return g, _multinomial(p[g:g + 1], gen)[0]
    resid = (p[n_acc] - q[n_acc]).clamp(min=0)
    s = resid.sum()
    resid = resid / s if s > 0 else p[n_acc]
    return n_acc, _multinomial(resid[None], gen)[0]


# ----------------------------------------------------------------------------- stats

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


# ----------------------------------------------------------------------------- baselines

@torch.no_grad()
def generate_no_cache(model: GPT, prompt: torch.Tensor, max_new: int, temperature: float = 0.0,
                      gen: torch.Generator | None = None) -> tuple[torch.Tensor, GenStats]:
    """Baseline: re-run the full sequence every step. O(n^2) in sequence length."""
    stats = GenStats()
    ids = prompt
    for _ in range(max_new):
        logits = model(ids)[:, -1]
        stats.target_forwards += 1
        nxt = _next_tokens(logits, temperature, gen)
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
    out = torch.empty((B, T + max_new), dtype=prompt.dtype, device=prompt.device)
    out[:, :T] = prompt
    logits = model(prompt, cache)[:, -1]
    stats.target_forwards += 1
    for i in range(max_new):
        nxt = _next_tokens(logits, temperature, gen)
        out[:, T + i] = nxt
        stats.new_tokens += 1
        if i + 1 < max_new:
            logits = model(nxt[:, None], cache)[:, -1]
            stats.target_forwards += 1
    return out, stats


# ----------------------------------------------------------------------------- speculative, batch 1

@torch.no_grad()
def generate_speculative(target: GPT, draft: GPT, prompt: torch.Tensor, max_new: int, gamma: int = 4,
                         temperature: float = 0.0, gen: torch.Generator | None = None,
                         ) -> tuple[torch.Tensor, GenStats]:
    """Speculative decoding with KV caches on both models. Batch size 1.

    Invariant: each cache covers a prefix of the committed sequence. Tokens
    the cache lacks are fed together with the next round's drafts, so the
    target does ONE forward per round and nothing is recomputed.

    Each round:
      1. draft: feed the committed tokens its cache lacks, propose gamma tokens
      2. target: one forward over [missing committed tokens] + [gamma drafts]
      3. accept a prefix of the drafts (`_verify`), append the final token
      4. roll both caches back over the rejected drafts only
    """
    assert prompt.shape[0] == 1, "use generate_speculative_batched for B > 1"
    stats = GenStats()
    T = prompt.shape[1]
    cap = T + max_new + gamma + 1
    tc = target.new_cache(1, cap)
    dc = draft.new_cache(1, cap)
    buf = torch.empty((1, cap), dtype=prompt.dtype, device=prompt.device)
    buf[:, :T] = prompt
    L = T
    if T > 1:  # the last prompt token rides along with round 1
        target(prompt[:, :-1], tc)
        draft(prompt[:, :-1], dc)
        stats.target_forwards += 1
        stats.draft_forwards += 1

    n_done = 0
    while n_done < max_new:
        g = min(gamma, max_new - n_done)
        ids = buf[:, :L]

        # 1. draft
        logits = draft(ids[:, dc.length:], dc)[:, -1]
        stats.draft_forwards += 1
        toks, qs = [], []
        for i in range(g):
            if temperature == 0.0:
                tok = logits.argmax(-1)
            else:
                q = _softmax(logits, temperature)
                tok = _multinomial(q, gen)
                qs.append(q)
            toks.append(tok)
            if i < g - 1:
                logits = draft(tok[:, None], dc)[:, -1]
                stats.draft_forwards += 1
        d_tok = torch.stack(toks, dim=1)  # (1, g); draft cache covers ids + d_0..d_{g-2}
        stats.drafted += g

        # 2. verify in one target pass
        pending = ids[:, tc.length:]
        t_logits = target(torch.cat([pending, d_tok], dim=1), tc)
        stats.target_forwards += 1
        off = pending.shape[1] - 1
        n_acc, final = _verify(d_tok[0], t_logits[0, off:off + g + 1],
                               torch.cat(qs, 0) if qs else None, temperature, gen)

        # 3. commit
        new = torch.cat([d_tok[0, :n_acc], final.view(1)])[: max_new - n_done]
        m = new.numel()
        buf[0, L:L + m] = new
        L += m
        n_done += m
        stats.accepted += n_acc
        stats.rounds += 1
        stats.acceptance_per_round.append(n_acc)

        # 4. roll back over rejected drafts only
        tc.rollback(g - n_acc)
        dc.rollback(max(0, (g - 1) - n_acc))

    stats.new_tokens = n_done
    return buf[:, :L], stats


# ----------------------------------------------------------------------------- speculative, batched

@torch.no_grad()
def generate_speculative_batched(target: GPT, draft: GPT, prompts: torch.Tensor, max_new: int, gamma: int = 4,
                                 temperature: float = 0.0, gen: torch.Generator | None = None,
                                 ) -> tuple[list[torch.Tensor], GenStats]:
    """Speculative decoding over a batch of equal-length prompts.

    Rows accept different numbers of drafts per round, so sequences become
    ragged and each cache row is rolled back by its own amount. To keep every
    row's pending input to exactly one token, the draft also processes its own
    last proposal (one extra cheap forward per round); the batch-1 decoder
    skips that.

    Rows that reach max_new keep being computed (their tokens are discarded)
    until every row is done. That waste is real and is part of why batched
    speculative gains are smaller; it is reported, not hidden.
    """
    B, T = prompts.shape
    stats = GenStats()
    cap = T + max_new + gamma + 2
    tc = target.new_cache(B, cap)
    dc = draft.new_cache(B, cap)
    buf = torch.empty((B, cap), dtype=prompts.dtype, device=prompts.device)
    buf[:, :T] = prompts
    L = [T] * B
    if T > 1:
        target(prompts[:, :-1], tc)
        draft(prompts[:, :-1], dc)
        stats.target_forwards += 1
        stats.draft_forwards += 1
    # invariant: cache row b covers buf[b, :L[b]-1]
    bidx = torch.arange(B, device=prompts.device)
    n_done = [0] * B
    while min(n_done) < max_new:
        last = buf[bidx, torch.tensor(L, device=prompts.device) - 1][:, None]  # (B, 1)

        # 1. draft gamma tokens, and process the last one so every row is 1 token behind
        logits = draft(last, dc)[:, -1]
        stats.draft_forwards += 1
        toks, qs = [], []
        for _ in range(gamma):
            if temperature == 0.0:
                tok = logits.argmax(-1)
            else:
                q = _softmax(logits, temperature)
                tok = _multinomial(q, gen)
                qs.append(q)
            toks.append(tok)
            logits = draft(tok[:, None], dc)[:, -1]
            stats.draft_forwards += 1
        d_tok = torch.stack(toks, dim=1)  # (B, gamma)
        stats.drafted += gamma * B

        # 2. one target pass over [last] + drafts
        tl = target(torch.cat([last, d_tok], dim=1), tc)  # (B, gamma+1, V)
        stats.target_forwards += 1

        # 3. vectorised accept/reject, one sync for the whole batch
        if temperature == 0.0:
            tgt = tl.argmax(-1)                                          # (B, gamma+1)
            match = (d_tok == tgt[:, :gamma]).float()
            n_acc = torch.cumprod(match, 1).sum(1).long()                # (B,)
            final = tgt[bidx, n_acc]
        else:
            p = _softmax(tl, temperature)                                # (B, gamma+1, V)
            q = torch.stack(qs, dim=1)                                   # (B, gamma, V)
            p_d = p[:, :gamma].gather(2, d_tok[..., None]).squeeze(2)
            q_d = q.gather(2, d_tok[..., None]).squeeze(2)
            ratio = (p_d / q_d).clamp(max=1.0)
            r = _rand((B, gamma), gen, d_tok.device)
            accept = (r < ratio.to(r.device)).float().to(d_tok.device)
            n_acc = torch.cumprod(accept, 1).sum(1).long()
            p_sel = p[bidx, n_acc]                                       # bonus slot when n_acc == gamma
            q_sel = torch.where((n_acc < gamma)[:, None], q[bidx, n_acc.clamp(max=gamma - 1)], torch.zeros_like(p_sel))
            resid = (p_sel - q_sel).clamp(min=0)
            s = resid.sum(1, keepdim=True)
            resid = torch.where(s > 0, resid / s.clamp(min=1e-30), p_sel)
            final = _multinomial(resid, gen)
        n_acc_l = n_acc.tolist()                                         # the sync

        # 4. commit per row, roll back per row
        rollback = torch.zeros(B, dtype=torch.long)
        for b in range(B):
            remaining = max_new - n_done[b]
            if remaining <= 0:
                rollback[b] = gamma + 1
                continue
            n = n_acc_l[b]
            m = min(n + 1, remaining)
            if n:
                buf[b, L[b]:L[b] + min(n, m)] = d_tok[b, :min(n, m)]
            if m > n:
                buf[b, L[b] + n] = final[b]
            L[b] += m
            n_done[b] += m
            stats.accepted += n
            rollback[b] = gamma + 1 - m  # keep buf[b, :L[b]-1]
        stats.rounds += 1
        tc.rollback(rollback)
        dc.rollback(rollback)
    stats.new_tokens = sum(n_done)
    return [buf[b, :L[b]] for b in range(B)], stats


# ----------------------------------------------------------------------------- prompt lookup

def _lookup_draft(seq: torch.Tensor, ngram: int, max_draft: int) -> torch.Tensor:
    """Find the most recent earlier occurrence of the last `ngram` tokens and
    propose what followed it. (k,) with k <= max_draft, possibly empty."""
    Lq = seq.numel()
    if Lq <= ngram or max_draft == 0:
        return seq.new_empty(0)
    key = seq[-ngram:]
    windows = seq.unfold(0, ngram, 1)[:-1]
    hits = (windows == key).all(dim=1).nonzero().flatten()
    if hits.numel() == 0:
        return seq.new_empty(0)
    start = int(hits[-1]) + ngram
    return seq[start:start + max_draft]


@torch.no_grad()
def generate_lookup(target: GPT, prompt: torch.Tensor, max_new: int, gamma: int = 4, ngram: int = 3,
                    temperature: float = 0.0, gen: torch.Generator | None = None) -> tuple[torch.Tensor, GenStats]:
    """Speculative decoding with no draft model: drafts come from n-gram
    matches in the sequence so far (Saxena 2023, "prompt lookup decoding").

    Same verify rule with q = one-hot on the proposed token, so acceptance
    probability is p(token) and the output distribution is still the
    target's. Rounds with no match degenerate to a plain cached step.
    """
    assert prompt.shape[0] == 1
    stats = GenStats()
    T = prompt.shape[1]
    cap = T + max_new + gamma + 1
    tc = target.new_cache(1, cap)
    buf = torch.empty((1, cap), dtype=prompt.dtype, device=prompt.device)
    buf[:, :T] = prompt
    L = T
    if T > 1:
        target(prompt[:, :-1], tc)
        stats.target_forwards += 1
    n_done = 0
    while n_done < max_new:
        ids = buf[:, :L]
        d = _lookup_draft(ids[0], ngram, min(gamma, max_new - n_done))
        g = d.numel()
        stats.drafted += g
        pending = ids[:, tc.length:]
        t_logits = target(torch.cat([pending, d[None]], dim=1), tc)
        stats.target_forwards += 1
        off = pending.shape[1] - 1
        tl = t_logits[0, off:off + g + 1]
        q = None if temperature == 0.0 or g == 0 else F.one_hot(d, tl.shape[-1]).float()
        n_acc, final = _verify(d, tl, q, temperature, gen)
        new = torch.cat([d[:n_acc], final.view(1)])[: max_new - n_done]
        m = new.numel()
        buf[0, L:L + m] = new
        L += m
        n_done += m
        stats.accepted += n_acc
        stats.rounds += 1
        stats.acceptance_per_round.append(n_acc)
        tc.rollback(g - n_acc)
    stats.new_tokens = n_done
    return buf[:, :L], stats
