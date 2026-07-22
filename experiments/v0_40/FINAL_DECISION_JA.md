# 最終判断メモ 日本語版

## 2026-07-22 追補: 6D・3軸・内部介入の分離

NeuroState 6Dを全モデル共通の内部数値制御として扱う仮説は支持されなかった。数値だけの単調制御は不安定で、6Dは神経伝達物質メタファーを使ったプロンプト生成DSLとして扱う。

外部状態は `approach / vigilance / arousal` の3軸で整理できるが、内部介入に使うのはモデルごとのrandom-control検査を通過した方向だけとする。

```text
Qwen2.5 Instruct: approach成立
Gemma 3 Instruct: approach成立、arousal候補
Qwen3 Instruct: approach成立、arousal成立
OLMo base: 3軸とも意味選択性を確認できず
```

`vigilance` は現時点で独立内部軸として未確認。数学的な直交化は意味的な独立性を保証しなかった。安全制御は3軸や残差介入へ委ねず、モデル外部の権限・承認・検査層で保証する。

詳細は `NEUROSTATE_3AXIS_RESULTS_JA.md` を参照。

## 結論

Qwen での NeuroState residual は、応答開始直前の境界状態に再現性のある因果効果を持つ。
ただし「進む/ためらう」を一発で切り替えるスイッチではない。

一番安全な理解は次の通り。

```text
NeuroState residual は、Qwen の応答境界で生成を少しだけ実用的・能動的・続行志向の表現へ寄せる方向である。
ただし一意な proceed/hesitate スイッチではなく、残差空間には似た効果を持つ他の方向もある。
```

## 強く言えること

- 主効果は Qwen の layer 13 / token position -1 付近に局在する。
- 元 split では 10/10 prompt で正方向に出た。
- split reversal でも効果は残った。
- 実生成では NeuroState / NeuroState residual が practical / active な表現を増やす。
- manual review と stricter second-pass review の一致率は 0.955 で、解釈は変わらなかった。

## 言いすぎになること

- NeuroState は agency vector である。
- NeuroState は proceed switch である。
- NeuroState だけが応答境界を動かせる。
- 強い random direction はただのノイズである。

これらは避ける。

## ランダム比較から分かったこと

`random_residual_253` は生成レベルではかなり強い action-starting comparator だった。
つまり NeuroState residual は意味のある方向だが、唯一の方向ではない。

`random_residual_293` は token-bank では強かったが、実生成では弱かった。
このため、logit/token probe だけで概念名を付けるのは危険。

## SmolLM2 の扱い

SmolLM2 でもモデル内で作った境界方向は広く正方向に効いた。

```text
初回 smoke: 4/4 prompt positive, specificity_rank_p = 0.333333
拡張 smoke: 19/20 prompt positive, specificity_rank_p = 0.267327
```

これは「他モデルでも似た境界介入現象がありそう」という補助証拠にはなる。
ただし random-control separation が弱いので、Qwen 結果の再現とは言わない。

## 最終的な言い方

日本語:

```text
Qwen では NeuroState residual は応答境界で再現性のある因果効果を持ち、生成を実用的・能動的・続行志向に弱く寄せる。ただし一意な proceed/hesitate スイッチではない。SmolLM2 でも類似した境界方向の正方向効果は広く見えたが、random control との分離は弱いため、現時点では再現ではなく cross-model plausibility の補強にとどまる。
```

英語:

```text
NeuroState residual is a robust response-boundary direction in Qwen. In generation, it softly biases outputs toward more practical, active, continuation-oriented phrasing. It is not unique: other residual-space directions can produce related effects. A SmolLM2 smoke test shows broad positive orientation of a model-local boundary direction, but weak random-control separation, so it supports cross-model plausibility rather than replication.
```

## 次にやるなら

1. 独立ブラインドレビューを追加する。
2. 採点済みファイルを `parse_blind_pairwise_review_scores.py` で標準スコアへ戻す。
3. `compare_manual_pairwise_scores.py` で manual score と比較する。
4. 解釈が変わらなければ `FINAL_REPORT.md` をそのまま完成版にする。

独立レビューがない場合は、現状の `FINAL_REPORT.md` と `FINAL_DECISION.md` で完成扱いでよい。

## Gemma Layer Localization

Gemma 3 1B Instruct was tested with the same model-local direction procedure. The direction was extracted at a source layer and injected two layers later at the final token position. Each setting used 20 prompts and 20 random controls.

| source -> target | mean slope | random controls >= model |
|---|---:|---:|
| 2 -> 4 | +0.001953 | 16/20 |
| 6 -> 8 | +0.002839 | 11/20 |
| 10 -> 12 | +0.004041 | 5/20 |
| 12 -> 14 | +0.007298 | 0/20 |
| 14 -> 16 | +0.006103 | 0/20 |
| 16 -> 18 | +0.006380 | 0/20 |
| 18 -> 20 | +0.006704 | 0/20 |

The effect is weak in early layers and becomes strong from the middle layers onward. The strongest tested setting was source layer 12 to target layer 14, with token position -1. The effect persists through target layers 16, 18, and 20, so this should be described as a middle-to-deep-layer band rather than a single uniquely causal layer.

The 12 -> 14 setting produced positive slopes for all 20 prompts. With 20 random controls, none exceeded the model-local direction. The earlier 100-control run at the same setting gave mean slope +0.007298 and 1/100 random controls at least as large in absolute value (rank p=0.0198).

This supports a cross-model internal-steering result: Qwen and Gemma both show a reproducible directionally consistent effect from the proceed-versus-hesitate contrast. It does not prove that the six numeric dimensions have identical meanings in both models.
