"""Correctness tests for the mid-layer loop ablation in ``lotus.py``.

Run from the repo root:

    uv run python scripts/test_mid_loop.py

Checks, for tiny randomly initialised Llama and GPT-2 backbones:

1. ``Lotus(loop_layer_start=0, loop_layer_end=L, mid_loop_injection_mode="add_final_norm")``
   reproduces the original full-model loop (loss / logits / intermediate loss / generate).
2. Every mid-layer configuration matches an independent hook-based reference that
   re-runs the *whole* HF model on the *whole* sequence each iteration (no KV cache,
   no hand-built masks), injecting at the input of ``loop_layer_start`` and reading
   the recurrent state at the output of ``loop_layer_end - 1``.
3. Training-mode forward/backward produces finite gradients (incl. the new RMSNorm gain),
   generation runs, and the KV cache handed to decoding covers the full input.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lotus import Lotus  # noqa: E402

from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB = 256
LATENT_ID, START_ID, END_ID = 253, 254, 255
PAD_ID = EOS_ID = 1
C_THOUGHT = 3
N_LOOPS = 3          # R
N_STEPS = 2          # replaced CoT steps (K used for supervision)
N_LAYERS = 4


def make_backbone(kind, attn_impl, seed=0):
    torch.manual_seed(seed)
    if kind == "llama":
        cfg = LlamaConfig(
            vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
            num_hidden_layers=N_LAYERS, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=256, tie_word_embeddings=False,
        )
        cfg._attn_implementation = attn_impl
        model = LlamaForCausalLM(cfg)
    else:
        cfg = GPT2Config(
            vocab_size=VOCAB, n_embd=64, n_layer=N_LAYERS, n_head=4, n_positions=256,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        )
        cfg._attn_implementation = attn_impl
        model = GPT2LMHeadModel(cfg)
    return model.to(DEVICE)


def make_batch(seed=1):
    """Collator-style batch: left pad to align latents, right pad to equal length."""
    g = torch.Generator().manual_seed(seed)

    def rand_tokens(n):
        return torch.randint(2, 250, (n,), generator=g).tolist()

    n_latent = N_LOOPS * C_THOUGHT
    q_lens, a_lens = [5, 7], [5, 3]
    seqs, labels, pos, masks = [], [], [], []
    for ql, al in zip(q_lens, a_lens):
        q = rand_tokens(ql)
        ans = rand_tokens(al) + [EOS_ID]
        ids = q + [START_ID] + [LATENT_ID] * n_latent + [END_ID] + ans
        lab = [-100] * (len(q) + 1 + n_latent + 1) + ans
        seqs.append(ids)
        labels.append(lab)
    latest = max(len(q) for q in [s[: s.index(LATENT_ID)] for s in seqs])
    out_ids, out_lab, out_pos, out_mask = [], [], [], []
    for ids, lab in zip(seqs, labels):
        n_pad = latest - ids.index(LATENT_ID)
        out_ids.append([PAD_ID] * n_pad + ids)
        out_lab.append([-100] * n_pad + lab)
        out_pos.append([0] * n_pad + list(range(len(ids))))
        out_mask.append([0] * n_pad + [1] * len(ids))
    max_len = max(len(x) for x in out_ids)
    for i in range(len(out_ids)):
        r = max_len - len(out_ids[i])
        out_ids[i] += [PAD_ID] * r
        out_lab[i] += [-100] * r
        out_pos[i] += [0] * r
        out_mask[i] += [0] * r
    steps = torch.randint(2, 250, (2, N_STEPS, C_THOUGHT), generator=g)
    steps[0, 1, 2] = PAD_ID  # some padding inside a step
    t = lambda x: torch.tensor(x, device=DEVICE)
    return dict(
        input_ids=t(out_ids), attention_mask=t(out_mask), labels=t(out_lab),
        position_ids=t(out_pos), replaced_cot_steps=steps.to(DEVICE),
    )


def make_lotus(base, after_loop=True, **kw):
    return Lotus(
        base, LATENT_ID, START_ID, END_ID, EOS_ID, pad_token_id=PAD_ID, c_thought=C_THOUGHT,
        intermediate_loss_weight=0.5, intermediate_loss_after_loop=after_loop, **kw,
    ).to(DEVICE)


# ----------------------------------------------------------------------------
# Independent reference: hook-based mid-layer recurrence on the full sequence
# ----------------------------------------------------------------------------
def naive_reference(base, kind, batch, ls, le, mode, term_fn, n_loops):
    """Returns (loop_region_logits_after_last_iter, suffix_logits) using full-sequence
    HF forwards with hooks; no KV cache and no custom masks."""
    layers = base.transformer.h if kind == "gpt2" else base.model.layers
    ids = batch["input_ids"]
    latent_mask = (ids == LATENT_ID).unsqueeze(-1)
    state = {"rec": None, "captured": None}

    def pre_hook(module, args, kwargs):
        h = args[0]
        if state["rec"] is None:
            return None
        term = term_fn(state["rec"])
        if term.shape[1] < h.shape[1]:  # final pass includes the suffix
            term = torch.cat([term, torch.zeros_like(h[:, term.shape[1]:])], dim=1)
        new = term if mode == "replace" else h + term
        h = torch.where(latent_mask[:, : h.shape[1]], new, h)
        return (h,) + tuple(args[1:]), kwargs

    def post_hook(module, args, out):
        state["captured"] = out[0] if isinstance(out, tuple) else out

    h1 = layers[ls].register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = layers[le - 1].register_forward_hook(post_hook)
    try:
        latent_pos = (ids == LATENT_ID).nonzero()[:, 1]
        loop_start = latent_pos.min().item() - 1
        loop_end = latent_pos.max().item() + 2  # one past END? no: last latent + 1 (exclusive)
        loop_end = latent_pos.max().item() + 1
        # iterations 0..R-1 on [prefix, loop region]
        for _ in range(n_loops):
            base(
                inputs_embeds=base.get_input_embeddings()(ids[:, :loop_end]),
                attention_mask=batch["attention_mask"][:, :loop_end],
                position_ids=batch["position_ids"][:, :loop_end],
                use_cache=False,
            )
            state["rec"] = state["captured"].detach().clone()
        # final full-sequence pass injecting h_rec^(R-1)
        out = base(
            inputs_embeds=base.get_input_embeddings()(ids),
            attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"],
            use_cache=False,
        )
    finally:
        h1.remove()
        h2.remove()
    return out.logits[:, loop_start:loop_end], out.logits[:, loop_end:]


def assert_close(a, b, name, atol=2e-4, rtol=1e-4):
    if a.shape != b.shape:
        raise AssertionError(f"{name}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    diff = (a - b).abs().max().item()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    print(f"    {name:<28s} max|diff|={diff:.2e} {'OK' if ok else 'FAIL'}")
    if not ok:
        raise AssertionError(f"{name}: max diff {diff}")


def test_equivalence_with_legacy(kind, attn_impl):
    print(f"[{kind}/{attn_impl}] (0, L, add_final_norm) == legacy full-model loop")
    base = make_backbone(kind, attn_impl)
    batch = make_batch()
    for after_loop in (True, False):
        legacy = make_lotus(base, after_loop=after_loop).eval()
        mid = make_lotus(
            base, after_loop=after_loop,
            loop_layer_start=0, loop_layer_end=N_LAYERS, mid_loop_injection_mode="add_final_norm",
        ).eval()
        with torch.no_grad():
            o1 = legacy(**batch, n_looped_iters=N_LOOPS)
            o2 = mid(**batch, n_looped_iters=N_LOOPS)
        tag = "after" if after_loop else "per-iter"
        assert_close(o1.logits, o2.logits, f"suffix logits ({tag})")
        assert_close(o1.loss, o2.loss, f"loss ({tag})")
        assert_close(o1.intermediate_loss, o2.intermediate_loss, f"intermediate loss ({tag})")
        assert_close(o1.intermediate_logits[-1], o2.intermediate_logits[-1], f"last loop logits ({tag})")
        # generate on a single example (batch size 1 only)
        one = {k: v[:1] for k, v in batch.items() if k in ("input_ids", "attention_mask")}
        with torch.no_grad():
            g1 = legacy.generate(one["input_ids"], one["attention_mask"], max_new_tokens=6, n_looped_iters=N_LOOPS)
            g2 = mid.generate(one["input_ids"], one["attention_mask"], max_new_tokens=6, n_looped_iters=N_LOOPS)
        assert torch.equal(g1, g2), f"generate mismatch: {g1.tolist()} vs {g2.tolist()}"
        print(f"    generate tokens equal ({tag})   OK")


def test_against_reference(kind, attn_impl):
    base = make_backbone(kind, attn_impl)
    batch = make_batch()
    ranges = [(0, N_LAYERS), (1, 3), (0, 2), (2, N_LAYERS)]
    for (ls, le) in ranges:
        for mode in ("add", "add_norm", "add_final_norm", "replace"):
            print(f"[{kind}/{attn_impl}] layers [{ls},{le}) mode={mode} vs hook reference")
            mid = make_lotus(base, loop_layer_start=ls, loop_layer_end=le, mid_loop_injection_mode=mode).eval()
            if mode == "add_norm":
                with torch.no_grad():  # non-trivial gain so the norm actually matters
                    mid.mid_loop_norm.weight.copy_(torch.linspace(0.5, 1.5, mid.mid_loop_norm.weight.numel()))
            term_fn = {
                "add": lambda r: r,
                "replace": lambda r: r,
                "add_norm": lambda r: mid.mid_loop_norm(r),
                "add_final_norm": lambda r: mid._final_norm()(r),
            }[mode]
            with torch.no_grad():
                out = mid(**batch, n_looped_iters=N_LOOPS)
                ref_loop, ref_suffix = naive_reference(base, kind, batch, ls, le, mode, term_fn, N_LOOPS)
            assert_close(out.intermediate_logits[-1], ref_loop, "loop-region logits")
            assert_close(out.logits, ref_suffix, "suffix logits")
            kv = mid._last_kv_cache
            kv_len = kv.get_seq_length() if hasattr(kv, "get_seq_length") else kv[0][0].shape[-2]
            assert kv_len == batch["input_ids"].shape[1], f"kv len {kv_len} != {batch['input_ids'].shape[1]}"


def test_train_and_generate(kind, attn_impl):
    print(f"[{kind}/{attn_impl}] train-mode forward/backward + generate on mid-layer loop")
    base = make_backbone(kind, attn_impl)
    batch = make_batch()
    for mode in ("add", "add_norm", "replace"):
        for after_loop in (True, False):
            mid = make_lotus(base, after_loop=after_loop, loop_layer_start=1, loop_layer_end=3,
                             mid_loop_injection_mode=mode).train()
            mid.zero_grad(set_to_none=True)
            out = mid(**batch, n_looped_iters=N_LOOPS)
            out.loss.backward()
            assert torch.isfinite(out.loss).all(), "non-finite loss"
            grads = [p.grad for p in mid.parameters() if p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads), "non-finite grads"
            if mode == "add_norm":
                assert mid.mid_loop_norm.weight.grad is not None and mid.mid_loop_norm.weight.grad.abs().sum() > 0, \
                    "mid_loop_norm received no gradient"
            # gradient must reach the prelude, the recurrent block and the coda
            layers = base.transformer.h if kind == "gpt2" else base.model.layers
            for i in range(N_LAYERS):
                g = next(p.grad for p in layers[i].parameters() if p.grad is not None)
                assert g.abs().sum() > 0, f"layer {i} got zero gradient"
            mid.eval()
            one = {k: v[:1] for k, v in batch.items() if k in ("input_ids", "attention_mask")}
            with torch.no_grad():
                g = mid.generate(one["input_ids"], one["attention_mask"], max_new_tokens=5, n_looped_iters=N_LOOPS)
            assert g.shape[1] > one["input_ids"].shape[1]
    print("    OK")


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    print(f"device={DEVICE}")
    for kind in ("llama", "gpt2"):
        for attn_impl in ("eager", "sdpa"):
            test_equivalence_with_legacy(kind, attn_impl)
            test_against_reference(kind, attn_impl)
            test_train_and_generate(kind, attn_impl)
    print("\nALL TESTS PASSED")
