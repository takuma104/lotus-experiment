"""Diagnostic: does free-running generate() agree with teacher-forced forward()?
#
# Usage (from the repo root):
#   uv run python scripts/check_gen_vs_forward.py --config args/midloop_gpt2/mid3-9.yaml \
#       --checkpoint outputs/gsm-lotus-gpt2-mid3-9/checkpoint_4 --stage 4 --n 100

For N validation samples at a given curriculum stage:
  * teacher-forced accuracy: argmax of forward() logits over the visible suffix
    (remaining CoT steps + answer) equals the gold tokens (answer tokens only, and all suffix tokens)
  * generation accuracy: generate() answer == gold answer (run.py's extraction)
  * first-token check: logits at the EoT position from the train-format input vs the
    gen-format input (must match: same prefix, causal model)
"""
import argparse, os, sys, json
import torch
sys.path.insert(0, "scripts")
from transformers import AutoModelForCausalLM, AutoTokenizer
from run import load_hierarchical_yaml
from utils import Config
from dataset import get_dataset, get_cot_latent_dataset, get_question_latent_dataset
from lotus import Lotus

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--stage", type=int, required=True)
ap.add_argument("--n", type=int, default=100)
args = ap.parse_args()

cfg = Config(load_hierarchical_yaml(args.config))
tok = AutoTokenizer.from_pretrained(cfg.model_id); tok.pad_token = tok.eos_token
for t in ("<|start-latent|>", "<|end-latent|>", "<|latent|>"): tok.add_tokens(t)
latent_id, start_id, end_id = (tok.convert_tokens_to_ids(t) for t in ("<|latent|>", "<|start-latent|>", "<|end-latent|>"))
base = AutoModelForCausalLM.from_pretrained(cfg.model_id); base.resize_token_embeddings(len(tok))
model = Lotus(base, latent_id, start_id, end_id, tok.eos_token_id, pad_token_id=tok.pad_token_id, c_thought=cfg.c_thought,
              intermediate_loss_weight=getattr(cfg, "intermediate_loss_weight", 0.0),
              intermediate_loss_after_loop=getattr(cfg, "intermediate_loss_after_loop", False),
              loop_layer_start=getattr(cfg, "loop_layer_start", None), loop_layer_end=getattr(cfg, "loop_layer_end", None),
              mid_loop_injection_mode=getattr(cfg, "mid_loop_injection_mode", "add_norm"))
sd = torch.load(os.path.join(args.checkpoint, "model.pt") if os.path.isdir(args.checkpoint) else args.checkpoint, map_location="cpu", weights_only=False)
info = model.load_state_dict(sd, strict=False); print("missing:", [k for k in info.missing_keys if not k.startswith("teacher")], "unexpected:", info.unexpected_keys)
model = model.cuda().eval(); print("mid_loop:", model.mid_loop, getattr(model, "loop_layer_start", None), getattr(model, "loop_layer_end", None))

raw = json.load(open(cfg.val_path))
answers = [d["answer"].replace(",", "").strip() for d in raw]
base_val = get_dataset(cfg.val_path, tok, max_size=args.n)
ds_tf = get_cot_latent_dataset(args.stage, base_val, cfg, start_id, latent_id, end_id)
ds_gen = get_question_latent_dataset(args.stage, base_val, cfg, start_id, latent_id, end_id)

tf_ans_ok = tf_all_ok = gen_ok = 0; first_tok_maxdiff = 0.0; first_tok_agree = 0
with torch.no_grad():
    for i in range(len(ds_tf)):
        s = ds_tf[i]; g = ds_gen[i]; idx = s["idx"]
        ids = torch.tensor([s["input_ids"]]).cuda(); labels = torch.tensor([s["labels"]]).cuda()
        pos = torch.arange(ids.shape[1]).view(1, -1).cuda()
        out = model(ids, torch.ones_like(ids), labels, pos, n_looped_iters=args.stage)
        loop_end = s["input_ids"].index(end_id)
        logits = out.logits[0]                      # covers positions [loop_end:], predicts loop_end+1...
        pred = logits[:-1].argmax(-1)               # predictions for positions loop_end+1 .. end
        gold = ids[0, loop_end + 1:]
        lab = labels[0, loop_end + 1:]
        ans_len = len(s["answer_labels"]) - s["answer_labels"].count(-100)
        all_ok = bool((pred == gold).all()); ans_ok = bool((pred[-ans_len:] == gold[-ans_len:]).all())
        tf_all_ok += all_ok; tf_ans_ok += ans_ok
        # gen-format forward: first-token logits must equal the train-format ones at the EoT position
        gids = torch.tensor([g["input_ids"]]).cuda(); gpos = torch.arange(gids.shape[1]).view(1, -1).cuda()
        gout = model(gids, torch.ones_like(gids), gids.clone(), gpos, n_looped_iters=args.stage)
        d = (gout.logits[0, -1] - out.logits[0, 0]).abs().max().item(); first_tok_maxdiff = max(first_tok_maxdiff, d)
        first_tok_agree += int(gout.logits[0, -1].argmax().item() == out.logits[0, 0].argmax().item())
        # free-running generation, run.py-style extraction
        gen = model.generate(gids, torch.ones_like(gids), max_new_tokens=64, n_looped_iters=args.stage)
        text = tok.decode(gen[0], skip_special_tokens=True)
        a = text.split("#")[-1].replace(",", "").strip(); ok = (a == answers[idx]); gen_ok += ok
        if i < 4:
            print(f"[{i}] gold={answers[idx]!r} gen={a!r} ok={ok} | tf_ans_ok={ans_ok} tf_all_ok={all_ok} | gold_suffix={tok.decode(gold)!r}")
            print(f"     tf_pred={tok.decode(pred)!r}")
            print(f"     gen_text={text.split(chr(10),1)[-1][:160]!r}")
n = len(ds_tf)
print(f"\nN={n} stage={args.stage}: teacher-forced answer-exact={tf_ans_ok/n:.3f}  teacher-forced suffix-exact={tf_all_ok/n:.3f}  "
      f"generate-exact={gen_ok/n:.3f}  first-token logits maxdiff={first_tok_maxdiff:.2e} agree={first_tok_agree}/{n}")
