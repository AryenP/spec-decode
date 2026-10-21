import torch
import pytest

from specdecode.model import GPT
from specdecode.generate import generate_no_cache, generate_kv, generate_speculative

DEV = "cpu"


@pytest.fixture(scope="module")
def small():
    return GPT.from_pretrained("gpt2", device=DEV)


@pytest.fixture(scope="module")
def tok():
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained("gpt2")


def enc(tok, s):
    return torch.tensor([tok.encode(s)], device=DEV)


def test_matches_huggingface_logits(small, tok):
    from transformers import GPT2LMHeadModel
    hf = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    ids = enc(tok, "The quick brown fox jumps over the lazy dog because")
    with torch.no_grad():
        ref = hf(ids).logits
    ours = small(ids)
    assert torch.allclose(ours, ref, atol=1e-3, rtol=1e-3)


def test_kv_cache_matches_full_recompute(small, tok):
    ids = enc(tok, "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole")
    full = small(ids)
    cache = small.new_cache(1, ids.shape[1])
    split = 7
    a = small(ids[:, :split], cache)
    b = small(ids[:, split:], cache)
    cached = torch.cat([a, b], dim=1)
    assert torch.allclose(full, cached, atol=1e-4, rtol=1e-4)
    # one-token-at-a-time path
    cache = small.new_cache(1, ids.shape[1])
    parts = [small(ids[:, i:i + 1], cache) for i in range(ids.shape[1])]
    assert torch.allclose(full, torch.cat(parts, dim=1), atol=1e-4, rtol=1e-4)


def test_greedy_kv_equals_greedy_no_cache(small, tok):
    p = enc(tok, "The history of the Roman Empire")
    a, _ = generate_no_cache(small, p, 24)
    b, sb = generate_kv(small, p, 24)
    assert torch.equal(a, b)
    assert sb.target_forwards == 24  # prefill + 23 single-token steps


def test_speculative_greedy_is_exactly_target_greedy(small, tok):
    # draft == target is the degenerate case: every draft token must be accepted
    p = enc(tok, "Speculative decoding works because")
    ref, _ = generate_kv(small, p, 30)
    for gamma in (1, 3, 5):
        out, st = generate_speculative(small, small, p, 30, gamma=gamma)
        assert torch.equal(out, ref), f"gamma={gamma}"
        assert st.acceptance_rate == 1.0
        assert st.target_forwards == st.rounds + 1
        assert out.shape[1] == p.shape[1] + 30


def test_speculative_with_real_draft_matches_target_greedy(small, tok):
    target = GPT.from_pretrained("gpt2-medium", device=DEV)
    p = enc(tok, "The capital of France is")
    ref, _ = generate_kv(target, p, 40)
    out, st = generate_speculative(target, small, p, 40, gamma=4)
    assert torch.equal(out, ref)
    assert 0.0 < st.acceptance_rate < 1.0
    assert st.target_forwards == st.rounds + 1  # one verify pass per round, plus prefill
    assert st.target_forwards < 40


def test_speculative_sampling_distribution_matches_target(small, tok):
    """With temperature > 0 the output distribution must equal the target's.

    Check the first generated token: run many single-token speculative draws
    (draft = gpt2, target = gpt2 but with a perturbed draft via different
    temperature is not allowed by the algorithm, so use a real pair) and
    compare the empirical frequency of the target's top token to its prob.
    """
    target = GPT.from_pretrained("gpt2-medium", device=DEV)
    p = enc(tok, "My favourite colour is")
    temp = 1.0
    with torch.no_grad():
        probs = torch.softmax(target(p)[0, -1] / temp, -1)
    top = probs.argmax().item()
    gen = torch.Generator().manual_seed(0)
    n, hits = 400, 0
    for _ in range(n):
        out, _ = generate_speculative(target, small, p, 1, gamma=1, temperature=temp, gen=gen)
        hits += int(out[0, -1].item() == top)
    emp = hits / n
    se = (probs[top].item() * (1 - probs[top].item()) / n) ** 0.5
    assert abs(emp - probs[top].item()) < 4 * se + 0.01, (emp, probs[top].item())
