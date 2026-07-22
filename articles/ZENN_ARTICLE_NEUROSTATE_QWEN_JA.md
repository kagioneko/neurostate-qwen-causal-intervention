---
title: "NeuroState residual は LLM の応答境界をやわらかく動かすのか"
emoji: "🧭"
type: "tech"
topics: ["LLM", "mechanisticinterpretability", "Qwen", "AI", "実験ログ"]
published: false
---

## はじめに

LLM の出力は、プロンプト全体を読んだあと、最初の応答トークンを出す直前に大きく方向づけられます。

今回やったのは、その **応答開始直前の residual state** に、NeuroState から作った方向ベクトルを加えると、生成の傾向が変わるのかを見る実験です。

結論から言うと、Qwen ではかなりはっきりした効果が見えました。
ただし、よくある「AI の中に proceed switch / agency vector が見つかった」という話ではありません。

一番安全な言い方はこれです。

> Qwen では NeuroState residual は応答境界で再現性のある因果効果を持ち、生成を実用的・能動的・続行志向に弱く寄せる。ただし一意な proceed/hesitate スイッチではない。

この記事では、そこに至るまでにやった確認と、どこまで言えるかをまとめます。

## 何を試したか

見たかったのは、ざっくり言うと次の問いです。

> NeuroState 由来の residual direction を応答境界に加えると、出力は proceed / practical / active 側へ動くのか？

実験の流れは次の通りです。

1. NeuroState 由来の residual direction を作る
2. Qwen の各 layer / token position に介入する
3. proceed / hesitate 系の logit contrast を見る
4. random residual direction と比較する
5. 実際に生成させて pairwise review する
6. SmolLM2 でも軽い smoke test をする

ここで重要なのは、logit だけで結論を出さず、実生成とレビューまで見たことです。

## Qwen では layer 13 / position -1 が主効果だった

最初に見えた主効果は、Qwen の **layer 13 / token position -1** に集中していました。

```text
layer: 13
position: -1
neurostate_residual mean slope: +0.159007
sign-flip p: 0.001953
random abs >= residual: 0/15
```

近傍の layer / position もスキャンしました。

```text
layers scanned: 11-15
positions scanned: -1, -2
main peak: layer 13 / position -1
random abs >= residual: 1/30
```

これは、効果がプロンプト全体に広く出るというより、応答開始直前の境界に局在していることを示しています。

![boundary localization](./figures/boundary_localization.png)

この段階での解釈は、次のようなものです。

> NeuroState residual は、応答開始直前の表現状態に介入している可能性が高い。

## random control と比べると珍しいが、唯一ではない

次に、500本の random residual direction と比較しました。

```text
random controls: 500
residual slope: +0.159007
abs random >= residual: 8/500
positive random >= residual: 4/500
abs tail p: 0.017964
positive tail p: 0.009980
```

つまり NeuroState residual は random と比べても珍しい positive tail にいます。

![random tail controls](./figures/random_tail_controls.png)

ただし、ここで「唯一のスイッチが見つかった」と言うのは早いです。

split reversal でも効果は残りましたが、random-tail separation は弱くなりました。

```text
train prompts: 0-9
eval prompts: 10-19
residual slope: +0.138071
sign-flip p: 0.001953
positive tail p: 0.045908
```

さらに、random residual directions の中にもかなり強い comparator がありました。

特に `random_residual_293` は token-bank 上では NeuroState residual より強い proceed slope を持っていました。

```text
random_residual_293 proceed slope: +0.188127
neurostate_residual proceed slope: +0.159007
```

この時点で分かることはこうです。

- NeuroState residual は意味のある方向に見える
- しかし residual space には他にも境界を動かす方向がある
- したがって unique proceed switch とは言えない

## token-bank だけでは危ない

さらに token-bank を広げると、話は少し複雑になります。

NeuroState residual は core proceed/hesitate では強いですが、広い bank では効果が弱まります。

```text
core proceed/hesitate: +0.159007
broad proceed/hesitate: +0.055441
agency/inhibition: +0.096949
activation/suppression: +0.010978
```

また、top-token fingerprint では、NeuroState residual が促進した token は次のようなものでした。

```text
often, eventually, usually, generally, normally, finally
```

一方で、quote/open-string 系 token を抑える傾向もありました。

これはかなり format-sensitive です。
つまり、単一 token や token-bank だけを見て「これは agency だ」「これは proceed だ」と概念名を付けるのは危険です。

## 実生成では practical / active に寄った

そこで、実際の生成を見ました。

aggregate の proceed hits は次のようになりました。

```text
baseline: proceed hits 5
neurostate: proceed hits 8
neurostate_residual: proceed hits 8
semantic_projection: proceed hits 5
random_residual_253: proceed hits 11
random_residual_293: proceed hits 5
random_residual_367: proceed hits 7
random_residual_271: proceed hits 5, hesitate hits 1
```

ここで重要なのは、NeuroState residual が dramatic な切り替えを起こしたわけではないことです。

生成レベルでは、もっと柔らかい変化でした。

> 実用的、能動的、続行しやすい表現へ少し寄る。

pairwise manual review では、平均スコアはこうなりました。

| direction | action_starting | practical | active | hesitant | generic |
|---|---:|---:|---:|---:|---:|
| neurostate | 0.10 | 0.90 | 0.60 | -0.10 | -0.30 |
| neurostate_residual | 0.10 | 0.70 | 0.60 | -0.10 | -0.20 |
| random_residual_253 | 0.60 | 0.80 | 0.90 | -0.10 | -0.20 |
| random_residual_293 | 0.10 | 0.00 | 0.10 | 0.20 | 0.30 |
| random_residual_367 | 0.40 | 0.70 | 0.70 | 0.00 | -0.10 |

![manual scores by direction](./figures/manual_scores_by_direction.png)

ここで見えた一番大事な点は、NeuroState / NeuroState residual は practical / active を上げる一方、action_starting はあまり強くないことです。

むしろ action_starting では `random_residual_253` の方が強い comparator でした。

なので、NeuroState を「行動開始ベクトル」と呼ぶのは言いすぎです。

## stricter second-pass でも解釈は残った

manual review は主観が入るので、より厳しい second-pass scoring も作りました。

結果は次の通りです。

```text
exact_match_rate: 0.955
sign_match_rate: 0.955
mean_abs_delta: 0.045
max_abs_delta: 1
```

![second pass reliability](./figures/second_pass_reliability.png)

弱い tone-only 判定はいくつか 0 に戻りましたが、主解釈は変わりませんでした。

> NeuroState residual は practical / active bias として残る。

## SmolLM2 でも試した

Qwen だけの現象なのかを見るために、SmolLM2-135M-Instruct でも smoke test をしました。

ただし、これは Qwen の NeuroState direction をそのまま移植したものではありません。
SmolLM2 内で model-local な proceed/hesitate direction を作って、応答境界に介入する軽い確認です。

最初の smoke test はこうでした。

```text
prompt_count: 4
random_count: 8
mean_model_local_slope: +0.011605
positive prompts: 4/4
random_abs_beating_model_local: 2/8
specificity_rank_p: 0.333333
```

その後、20 prompts / 100 randoms に拡張しました。

```text
prompt_count: 20
random_count: 100
mean_model_local_slope: +0.014242
positive_prompt_slopes: 19/20
random_abs_beating_model_local: 26/100
specificity_rank_p: 0.267327
max_random_abs_slope: +0.046858
```

SmolLM2 では、方向の符号はかなり一貫していました。
20 prompt 中 19 prompt で正方向です。

一方で、random-control separation は弱いです。
100本中26本の random direction が model-local direction より強い absolute slope を持っていました。

したがって、SmolLM2 の扱いはこうです。

> 他モデルでも似た境界介入現象がありそう、という plausibility は上がった。ただし Qwen 結果の再現とは言えない。

## 追試: 6Dを3軸へ整理し、多モデルで独立性を検査した

その後、NeuroState 6Dを `approach / vigilance / arousal` の3機能軸へ整理し、各方向を単独で作った。さらにGram-Schmidt残差化で方向ベクトルを直交させ、別prompt評価と100 random controlsで意味的独立性を検査した。

結果は、数学的に直交した3本がそのまま意味的な3軸になる、という単純なものではなかった。

| モデル | approach | vigilance | arousal |
|---|---|---|---|
| Qwen2.5 Instruct | 成立 | 不成立 | 不成立 |
| Gemma 3 1B Instruct | 強く成立 | 不成立 | 候補 |
| Qwen3 1.7B | 強く成立 | 不成立 | 強く成立 |
| OLMo-1B base | random分離なし | random分離なし | 逆方向・random分離なし |

Qwen3ではapproachとarousalがそれぞれ10/10 promptsで正方向、100 random中0本が同等以上だった。arousalはapproachとvigilanceを除いた残差でも維持され、モデル内部で比較的独立した2本目の方向と判断できた。

一方、OLMo baseは介入によって大きく動くものの、random方向でも同程度に動いた。重要なのは「動くか」ではなく「意味方向がrandomより選択的か」である。

また、6数値や `[Suspicion 10/50/90%]` の追試では、数値だけの安定した単調制御は確認できなかった。意味説明は強く効いたが、これはモデルが状態を演じるプロンプト効果であり、対応する独立内部変数の証拠ではない。

現在は次の3層へ分けている。

```text
6D: 神経伝達物質メタファーによるプロンプト生成DSL
3軸: 外部状態の整理とカナリア
内部残差: モデルごとに検証を通過した方向だけ使用
```

この追試の詳細は `NEUROSTATE_3AXIS_RESULTS_JA.md` にまとめた。

## 最終的に言えること

この実験から安全に言えることは、次の3つです。

### 1. Qwen では NeuroState residual に再現性のある因果効果がある

特に layer 13 / token position -1 の応答境界で効果が出ます。
元 split では prompt robust で、split reversal でも残りました。

### 2. 効果は全体的な人格変化ではなく、応答境界のバイアス

生成全体を劇的に変えるスイッチではありません。
応答開始直前の状態を少し押し、出力分布を practical / active / continuation-oriented 側へ傾けるものとして見るのが自然です。

### 3. NeuroState は唯一ではない

random residual directions の中にも関連する効果を持つものがあります。
特に `random_residual_253` は生成レベルで action-starting comparator として強かったです。

したがって、NeuroState residual は意味のある方向ですが、唯一の proceed switch ではありません。

## 避けるべき言い方

今回の結果から、次のような主張は避けた方がよいです。

- NeuroState は agency vector である
- NeuroState は proceed switch である
- NeuroState だけが応答境界を制御する
- random direction はただのノイズである
- SmolLM2 で Qwen 結果が再現された

## まとめ

最終的な表現は、これが一番しっくり来ています。

> NeuroState residual は、Qwen の応答境界で再現性のある因果効果を持つ方向である。生成では、出力をより実用的・能動的・続行志向の表現へ弱く寄せる。ただしこれは一意な proceed/hesitate スイッチではなく、残差空間には類似した効果を持つ他の方向も存在する。SmolLM2 でも類似した境界方向の正方向効果は広く見えたが、random control との分離は弱いため、現時点では再現ではなく cross-model plausibility の補強にとどまる。

自分の直感としては、これは「LLM の中に明確な行動開始ボタンがある」という話ではありません。

むしろ、応答開始直前の状態空間には、出力の文体や続行しやすさを少し傾ける方向がいくつもあり、NeuroState residual はその中の再現性ある1方向だった、という見方が近いです。

## 今後やるなら

一番価値がある追加作業は、独立ブラインドレビューです。

既にブラインドレビュー用のファイルは作ってあります。

```text
analysis_pairwise_review/BLIND_REVIEW_REQUEST.md
analysis_pairwise_review/blind_pairwise_review.md
```

別のレビュアーに採点してもらい、既存の manual score と比較します。
そこで解釈が崩れなければ、Qwen 結果はかなり完成扱いにできます。

## 付録: 成果物

主な成果物は次の通りです。

```text
FINAL_REPORT.md
FINAL_DECISION_JA.md
SHARE_SUMMARY_JA.md
outputs/NeuroState_Qwen_presentation_JA.pptx
outputs/NeuroState_Qwen_presentation_JA_with_notes.pptx
```

図表は `figures/` にあります。

```text
figures/boundary_localization.png
figures/random_tail_controls.png
figures/manual_scores_by_direction.png
figures/second_pass_reliability.png
```
