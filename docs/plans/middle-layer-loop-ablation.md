# 中間層のみループ (Middle-Layer Recurrence) アブレーション計画

作成日: 2026-09-04

## 目的

LOTUS ([2606.31779](../paper/2606.31779.md)) は、latent プレフィックスに対して LM 全体 `f_θ` を R 回反復する
「全層ループ」を採用している。一方 T2MLR ([2607.15178](../paper/2607.15178.md)) は、
再帰を中間層の一部 (全体の 20% 程度) に限定した方が全層再帰より良い、と報告している
(Table 11 の recurrence-location ablation)。Huginn / Ouro など他の looped transformer でも
prelude / recurrent block / coda の 3 段構成が一般的である。

LOTUS 論文にはループ対象レイヤー範囲のアブレーションが無いため、本リポジトリに
**「層 `[ℓ_start, ℓ_end)` のみを R 回反復する」** モードを追加し、
全層ループ (既存実装) と比較する。

## 現状の実装 (全層ループ)

`scripts/lotus.py` の `Lotus.forward` は次の流れになっている。

| Step | 内容 | コード |
| --- | --- | --- |
| 1 | prefix `[Q, BoT]` を 1 回 forward して KV キャッシュ `C_pre` を作る | `lotus.py` Step 1 |
| 2 | ループ領域 `[BoT, lat…lat]` を全層 forward し `h^(0)` を得る | `efficient_forward` |
| 3 | R 回: `E + h^(t-1)` を latent 位置の **入力埋め込み** に注入し、全層 forward | `inject_latent_embeddings` |
| 4 | 最終イテレーションの KV キャッシュで suffix `[EoT, A]` を forward | Step 4 |

注入元は最終 norm 後の `last_hidden_state`、注入先は入力埋め込みなので、
再帰信号は必ず「最終層 → 埋め込み」を経由する。これが T2MLR 論文が指摘する
「再帰情報が中間層の外側に押し出される」構造そのものである。

## 提案する構成 (中間層ループ)

層数 L のモデルに対し、`loop_layer_start = ℓ_s`, `loop_layer_end = ℓ_e` (0 ≤ ℓ_s < ℓ_e ≤ L) を指定する。

```
prelude : 層 [0, ℓ_s)     ループ領域に 1 回だけ適用 → h_pre (残差ストリーム)
loop    : 層 [ℓ_s, ℓ_e)   R+1 回 (初回 + R 回の注入付き反復)
            h_rec^(0) = mid(h_pre)
            h_rec^(t) = mid( inject(h_pre, h_rec^(t-1)) )   t = 1..R
coda    : 層 [ℓ_e, L) + 最終 norm + lm_head   最終イテレーション後に 1 回
            (per-iter 監督 / IA loss を使う場合は毎イテレーション)
```

- 注入は既存実装と同じく latent 位置のみ (BoT 位置は固定)。
- prefix キャッシュ `C_pre` は全層分あるので、prelude / loop / coda の各段でそのまま使える。
  `DynamicCache` は layer_idx ごとに独立に append されるため、
  「prelude 後のキャッシュを毎イテレーション clone → mid 層が append → 最後に coda 層が append」で
  suffix 用の全層キャッシュが自然に完成する。
- `ℓ_s = 0, ℓ_e = L, mode = add_final_norm` は既存の全層ループと数値的に一致する
  (等価性テストに使う)。

### 注入モード (`mid_loop_injection_mode`)

残差ストリームへ直接足すためスケールの扱いが本質的な設計判断になる。以下を切り替え可能にする。

| モード | 式 | 備考 |
| --- | --- | --- |
| `add` | `h_pre + h_rec` | 生の残差同士の加算。イテレーションごとに残差ノルムが伸びる |
| `add_norm` (デフォルト) | `h_pre + RMSNorm_new(h_rec)` | 新規の学習可能 RMSNorm (gain 初期値 1)。スケールを有界化 |
| `add_final_norm` | `h_pre + norm_final(h_rec)` | モデル自身の最終 norm を流用。`(0, L)` で既存実装と等価 |
| `replace` | `h_rec` | 入力注入なしの純粋な再帰 (既存の `latent_injection_mode=replace` に対応) |

T2MLR 流のゲート付き融合 `Φ(h, r)` は拡張候補として残す (未実装)。

## 実装項目

1. `scripts/lotus.py`
   - `_run_layers(hidden, layer_start, layer_end, attention_mask, position_ids, past_len, cache)`:
     Llama (`model.layers`, rotary, 4D causal mask) と GPT-2 (`transformer.h`, tuple cache) の両方で
     レイヤー範囲を手動 forward する。マスクは `past_len` を明示して自前で作る
     (`DynamicCache.get_seq_length()` は prelude 後に層 0 がループ領域分伸びているため使えない)。
   - `_embed_to_residual`: Llama は恒等、GPT-2 は `drop(wte + wpe)`。
   - `_readout(h)`: coda 層 + 最終 norm + lm_head → `OutputWrapper(logits, last_hidden_state, cache)`。
   - `Lotus.__init__` に `loop_layer_start`, `loop_layer_end`, `mid_loop_injection_mode`,
     `mid_loop_readout_every_iter` を追加。未指定なら既存の全層ループ (後方互換)。
   - `forward` の Step 2/3 に中間層ループ分岐を追加。Step 4 以降 (suffix, loss) は共通。
   - 非対応の組み合わせは assert で弾く: `sample_stages`, `freeze_supervised_blocks`,
     `intermediate_loss_type=cosine` 以外は動くが、cosine / IA loss は per-iter readout を強制する。
2. `scripts/run.py`, `scripts/eval.py`: 設定値のプラミング。
3. `args/`: 例として GPT-2 / Llama-1B の中間層ループ設定を追加。
4. `scripts/test_mid_loop.py`: ランダム初期化の小型 Llama / GPT-2 で
   - `(0, L, add_final_norm)` と既存パスの logits / loss が一致すること
   - 中間層設定で forward / backward / generate が通ること
   - KV キャッシュ長が suffix と整合すること
   を検証する。

## アブレーション設計案

T2MLR Table 11 に倣い、幅固定で位置を振る。Llama-3.2-1B (16 層) の例:

| 設定 | ℓ_s | ℓ_e | ループ対象 |
| --- | --- | --- | --- |
| full (再現ベースライン) | 0 | 16 | 100% |
| middle 50% | 4 | 12 | 50% |
| middle 25% | 6 | 10 | 25% |
| early 25% | 0 | 4 | 25% |
| late 25% | 12 | 16 | 25% |

GPT-2 (12 層) なら `(0,12)`, `(3,9)`, `(4,8)`, `(0,4)`, `(8,12)`。
Llama-3.2-3B (28 層) なら `(0,28)`, `(7,21)`, `(10,17)`, `(0,7)`, `(21,28)`。

各設定について `add_norm` を基本とし、余裕があれば `add` / `replace` も比較する。

記録する指標:
- GSM8K 精度 (既存の eval)
- thought フェーズのレイテンシ (`model._last_timing["thought"]`)。
  全層ループでは `(R+1)` 回の全層 pass、中間層ループでは `1 回の全層 pass + R 回の部分 pass` になるので
  精度と同時に速度への影響も見る。

## 実装状況 (2026-09-04)

上記の実装項目 1〜4 は完了している。

| ファイル | 内容 |
| --- | --- |
| `scripts/lotus.py` | `_run_layers` / `_mid_loop_readout` などの層範囲 forward、`forward` の中間層ループ分岐、`_RMSNorm` |
| `scripts/run.py` | `loop_layer_start` / `loop_layer_end` / `mid_loop_injection_mode` / `mid_loop_readout_every_iter` を config から渡す |
| `scripts/eval.py` | 同名の CLI 引数 (`--loop_layer_start` など) |
| `args/gsm8k_lotus_gpt2_midloop.yaml` | GPT-2 の例 (層 [3, 9)) |
| `args/gsm8k_lotus_llama1b_midloop.yaml` | Llama-3.2-1B の例 (層 [4, 12)) |
| `scripts/test_mid_loop.py` | 等価性 / 参照実装との一致 / 学習・生成の動作テスト |

検証結果:

- `uv run python scripts/test_mid_loop.py`: Llama / GPT-2 × eager / sdpa の全ケースで合格。
  `(0, L, add_final_norm)` は既存パスと logits・loss・生成トークンが完全一致 (max diff 0)。
  中間層設定 4 種 × 注入モード 4 種はフック実装の参照と 2e-4 以内で一致。
- GPT-2 実データのスモーク学習 (debug モード、stage 2、1 epoch、層 [3, 9)、`add_norm`) が
  end-to-end で完走 (loss 4.9 → 1.1、eval loss / 生成も動作)。

実行例:

```bash
# 学習 (GPT-2, 中間層ループ)
CONFIG=args/gsm8k_lotus_gpt2_midloop.yaml NPROC_PER_NODE=2 bash launch_train.sh

# 評価 (学習時と同じ層範囲 / 注入モードを指定すること)
uv run python scripts/eval.py --model_id openai-community/gpt2 --checkpoint <ckpt> --fp32 \
  --c_thought 13 --n_looped_iters 6 --loop_layer_start 3 --loop_layer_end 9 --mid_loop_injection_mode add_norm
```

次のステップ: 上のアブレーション格子に沿って設定ファイルを増やし、精度と thought レイテンシを記録する。

## 注意点 / 既知の制約

- `use_kv_cache=True` 前提 (既存実装も同じ assert がある)。
- GPT-2 では埋め込み dropout の適用位置が既存実装と微妙に異なる (train モードのみ)。eval では等価。
- `add_norm` は新規パラメータ `mid_loop_norm.weight` を持つ。既存チェックポイントの読み込みは
  `strict=False` なので初期値 (1) のまま学習される。
- `generate(output_embedding=True)` が返す埋め込みは、中間層ループでは注入後ではなく元の埋め込みになる。
- FSDP は `LlamaDecoderLayer` 単位でラップされるため、層を個別に呼んでも問題ない。
  ただし全ランクで同じ層列を同じ回数呼ぶ必要がある (現状の分岐は入力に依存しないので満たす)。

## 実験ログ

### 2026-09-04: GPT-2 アブレーション格子を開始

環境: RTX 5090 (32 GB) × 1、fp32 (論文と同じ)。

- 論文の全体バッチ 128 (64 × 2 GPU) は 1 GPU では stage 6 でメモリ不足 (micro-batch 64/32 とも OOM) のため、
  `scripts/run.py` に勾配累積 `grad_accum_steps` を追加し、`batch_size_training: 128`, `grad_accum_steps: 8`
  (micro-batch 16) で全体バッチ 128 を維持した。最長 5000 サンプルでの最悪ケースはピーク 19.4 GB。
  micro-batch ごとの損失は `1/grad_accum_steps` 倍して累積する (micro-batch ごとの token 平均の平均。
  2 GPU 版の「全体 token 平均」とは正規化がごく僅かに異なる)。
- 設定ファイル: `args/midloop_gpt2/` (共通の 1 GPU 設定は `_single_gpu.yaml`)。
- 実行: `scripts/run_midloop_ablation_gpt2.sh` が以下を順に学習 → `checkpoint_final` を
  `eval.py` で GSM8K test / GSM-Hard / MultiArith / SVAMP 評価 (`outputs/<run>/results_{gsm8k,ood}.json`)。
  進捗は `outputs/midloop_gpt2_queue.log` と `outputs/<run>/train.log`、W&B project `lotus`。

| 順 | run | ℓ_s | ℓ_e | 注入 | 備考 |
| --- | --- | --- | --- | --- | --- |
| 1 | `gsm-lotus-gpt2-full-legacy` | – | – | (既存の埋め込み注入) | 再現ベースライン (論文 GPT-2 LOTUS: 44.1 ± 0.7) |
| 2 | `gsm-lotus-gpt2-mid3-9` | 3 | 9 | add_norm | middle 50% |
| 3 | `gsm-lotus-gpt2-mid4-8` | 4 | 8 | add_norm | middle 25% |
| 4 | `gsm-lotus-gpt2-early0-4` | 0 | 4 | add_norm | early 25% |
| 5 | `gsm-lotus-gpt2-late8-12` | 8 | 12 | add_norm | late 25% |
| 6 | `gsm-lotus-gpt2-full0-12` | 0 | 12 | add_norm | 注入モードの対照 (層範囲は全層) |

debug モードでの stage-6 速度 (3013 step/epoch): 全層ループ 1.63 s/step (≈ 82 分/epoch)、
[3, 9) ループ 1.07 s/step (≈ 54 分/epoch)。30 epoch のうち stage 6 が 24 epoch なので、
全層ループ 1 本あたり ≈ 1.5 日、6 本の合計は ≈ 7 日程度の見込み。

### 結果 1/6: `gsm-lotus-gpt2-full-legacy` (全層ループ、再現ベースライン) — 2026-09-06 完了

学習 2026-09-04 19:14 → 09-06 10:05 (約 39 時間、うち 1 回の手動再起動)。stage 6 は約 1 時間/epoch。

| 指標 | 値 | 論文 (Table 1, GPT-2 LOTUS) |
| --- | --- | --- |
| val best | 46.0% (epoch 27) | – |
| val last (epoch 30) | 45.8% | – |
| GSM8K test | **43.7%** (576/1319) | 44.1 ± 0.7 |
| GSM-Hard | 9.9% (130/1319) | 9.5 ± 0.2 |
| MultiArith | 91.1% (164/180) | 92.4 ± 1.4 |
| SVAMP | 42.1% (421/1000) | 41.8 ± 0.9 |
| OOD 平均 | 47.7 | 47.9 |
| thought レイテンシ (GSM8K test) | 27.5 ms/例 (推論全体 38.8 ms/例) | – |
| 学習ピークメモリ | 20.1 GB | – |

論文の GPT-2 LOTUS と誤差範囲で一致しており、1 GPU + 勾配累積 (16 × 8) の設定で再現できている。
validation 精度の推移: stage 0 で 41.4% → stage 1〜5 で 33〜36% に低下 → stage 6 (epoch 7 以降) で
回復し epoch 13 以降は 41〜46% で推移。

### 結果 2/6: `gsm-lotus-gpt2-mid3-9` (層 [3, 9) ループ, add_norm) — 2026-09-07 完了

学習 2026-09-06 10:08 → 09-07 12:25 (約 26 時間、stage 6 は約 40 分/epoch)。

validation 精度の推移: stage 0 で 41.4% (ベースラインと同一) → stage 1〜5 で 19.8〜30.6% →
stage 6 (epoch 7〜30) では 22.4〜28.4% の帯で横ばい (最終 epoch 25.6%)。
ベースラインの同時期 (41〜46%) に遠く及ばない。
**stage 0 の 41.4% を一度も上回らなかったため、best-val 選択の `checkpoint_final` は latent 学習前 (epoch 1) の重み**になっており、
それを 78 latent + R=6 で評価した値は無意味 (GSM8K 3.6%)。
そこで各 run について最終 epoch のチェックポイントも評価するようにした (`results_*_last.json`、`scripts/eval_midloop_ckpt_gpt2.sh`)。

| チェックポイント | GSM8K test | GSM-Hard | MultiArith | SVAMP | OOD 平均 |
| --- | --- | --- | --- | --- | --- |
| mid3-9 checkpoint_final (= epoch 1, stage 0 重み) | 3.6% (47/1319) | 0.9 | 10.6 | 3.9 | 5.1 |
| mid3-9 checkpoint_30 (最終 epoch) | 28.8% (380/1319) | 6.4 | 52.8 | 29.0 | 29.4 |
| 参考: full-legacy checkpoint_30 (最終 epoch) | 43.6% (575/1319) | 9.5 | 89.4 | 41.8 | 46.9 |

学習ピークメモリ 14.8 GB (全層ループ 20.1 GB)。thought レイテンシは同時に走っている次の学習ジョブと GPU を共有した状態で
計測したため参考にならない (checkpoint_final 評価 17.1 ms/例、checkpoint_30 評価 33.3 ms/例)。
全 run 終了後に GPU 空き状態でレイテンシを再計測する。

生成と teacher-forced forward の整合性は `scripts/check_gen_vs_forward.py` で確認済み (stage 4 の checkpoint で
first-token logits 一致 100/100、teacher-forced 26% vs 生成 20%) なので、推論経路のバグではなく
「層 [3, 9) のみを add_norm 注入でループする構成が GPT-2 では学習しにくい」という結果と解釈する。

### 結果 3/6: `gsm-lotus-gpt2-mid4-8` (層 [4, 8) ループ, add_norm) — 2026-09-08 完了

学習 2026-09-07 12:30 → 09-08 11:11 (約 23 時間、stage 6 は約 35 分/epoch)。

validation 精度の推移: stage 0 で 41.4% → stage 1〜5 で 20.0〜32.0% →
stage 6 では 23.2〜28.4% (最終 epoch 26.2%)。mid3-9 とほぼ同じ曲線で、
ループ幅を 50% → 25% に狭めても変化なし。stage 0 を上回らなかったため `checkpoint_final` は epoch 1 の重み。

| チェックポイント | GSM8K test | GSM-Hard | MultiArith | SVAMP | OOD 平均 |
| --- | --- | --- | --- | --- | --- |
| mid4-8 checkpoint_final (= epoch 1, stage 0 重み) | 3.8% (50/1319) | 1.1 | 11.7 | 4.0 | 5.6 |
| mid4-8 checkpoint_30 (最終 epoch) | 26.9% (355/1319) | 6.6 | 43.3 | 27.0 | 25.6 |
| 参考: mid3-9 checkpoint_30 | 28.8% (380/1319) | 6.4 | 52.8 | 29.0 | 29.4 |
| 参考: full-legacy checkpoint_30 | 43.6% (575/1319) | 9.5 | 89.4 | 41.8 | 46.9 |

学習ピークメモリ 13.3 GB。学習済み `mid_loop_norm.weight` は 0.76〜1.16 (初期値 1) で、mid3-9 同様ほとんど動いていない。

### 結果 4/6: `gsm-lotus-gpt2-early0-4` (層 [0, 4) ループ, add_norm を入力埋め込みに注入) — 2026-09-09 完了

学習 2026-09-08 11:16 → 09-09 09:50 (約 22.5 時間、stage 6 は約 30 分/epoch)。

注入先は既存実装と同じ入力埋め込み (ℓ_s = 0) で、ループの出口だけが層 4。
validation 精度の推移: stage 0 で 41.4% → stage 1〜5 で 17.2〜30.0% →
stage 6 では 21.6〜25.8% (最終 epoch 24.2%)。mid 2 本と同じ形で、
入口を入力側に戻しても回復しない。つまり「入口が中間層であること」単独が原因ではなく、
**再帰信号が最終層 (ln_f 後) 以外から取られること、あるいは add_norm 注入そのもの**が疑わしい。
この切り分けは最後の `full0-12` (全層 + add_norm) で確定する。

| チェックポイント | GSM8K test | GSM-Hard | MultiArith | SVAMP | OOD 平均 |
| --- | --- | --- | --- | --- | --- |
| early0-4 checkpoint_final (= epoch 1, stage 0 重み) | 8.1% (107/1319) | 1.8 | 21.1 | 12.5 | 11.8 |
| early0-4 checkpoint_30 (最終 epoch) | 25.2% (333/1319) | 5.8 | 39.4 | 25.5 | 23.6 |
| 参考: mid4-8 checkpoint_30 | 26.9% (355/1319) | 6.6 | 43.3 | 27.0 | 25.6 |
| 参考: mid3-9 checkpoint_30 | 28.8% (380/1319) | 6.4 | 52.8 | 29.0 | 29.4 |
| 参考: full-legacy checkpoint_30 | 43.6% (575/1319) | 9.5 | 89.4 | 41.8 | 46.9 |

学習ピークメモリ 13.3 GB。

### 結果 5/6: `gsm-lotus-gpt2-late8-12` (層 [8, 12) ループ, add_norm) — 2026-09-10 完了

学習 2026-09-09 09:57 → 09-10 08:36 (約 22.5 時間、stage 6 は約 30 分/epoch)。

再帰信号は最終ブロック出力 (ln_f 前の残差) で、注入先は層 8。
validation 精度の推移: stage 0 で 41.4% → stage 1〜5 で 19.6〜31.0% →
stage 6 では 22.2〜26.0% (最終 epoch 24.0%)。他の 3 変種と同じ形。
出口を最終層にしても回復しないので、「再帰信号の取り出し位置」も単独では原因ではない。
残る候補は add_norm 注入そのもの (ln_f を通さず RMSNorm(ゲイン 1 初期化) で残差に足すこと) で、
`full0-12` で確定する。

| チェックポイント | GSM8K test | GSM-Hard | MultiArith | SVAMP | OOD 平均 |
| --- | --- | --- | --- | --- | --- |
| late8-12 checkpoint_final (= epoch 1, stage 0 重み) | 3.0% (39/1319) | 0.8 | 12.8 | 4.7 | 6.1 |
| late8-12 checkpoint_30 (最終 epoch) | 25.9% (342/1319) | 5.8 | 38.3 | 26.5 | 23.6 |
| 参考: early0-4 checkpoint_30 | 25.2% (333/1319) | 5.8 | 39.4 | 25.5 | 23.6 |
| 参考: mid4-8 checkpoint_30 | 26.9% (355/1319) | 6.6 | 43.3 | 27.0 | 25.6 |
| 参考: mid3-9 checkpoint_30 | 28.8% (380/1319) | 6.4 | 52.8 | 29.0 | 29.4 |
| 参考: full-legacy checkpoint_30 | 43.6% (575/1319) | 9.5 | 89.4 | 41.8 | 46.9 |

学習ピークメモリ 13.3 GB。
