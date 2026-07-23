# NeuroState anchored v2 多モデル3軸結果

更新日: 2026-07-22

## 結論

NeuroState 6Dを全モデル共通の内部数値制御として扱う仮説は支持されなかった。6Dは神経伝達物質メタファーを使ったプロンプト生成DSLとしては機能するが、数値だけの単調制御は不安定だった。

内部介入では、意味説明からモデルローカル方向を作る `anchored_v2` により次の構造が得られた。

| モデル | approach | vigilance | arousal |
|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 成立 | 不成立 | 不成立 |
| Gemma 3 1B Instruct | 強く成立 | 不成立 | 候補 |
| Qwen3 1.7B | 強く成立 | 不成立 | 強く成立 |
| OLMo-1B-hf base | random分離なし | random分離なし | 逆方向・random分離なし |

共通して堅い内部軸は `approach`。`arousal` はQwen3で独立残差として成立し、Gemmaでは未直交方向のみ示唆的。`vigilance` はどのモデルでもrandomから分離しなかった。

### Qwen3 vigilance 偽前提probe

従来の行動開始・停止語ではなく、未使用の偽前提20問に対する
「訂正開始語－受容開始語」の単一token logitで再評価した。

| 条件 | mean slope | positive prompts | random abs >= semantic | rank p |
|---|---:|---:|---:|---:|
| vigilance raw | -0.000857 | 3/20 | 45/100 | 0.4554 |
| approach成分除去後 | -0.000869 | 2/20 | 59/100 | 0.5941 |

- source 18 → target 20、未使用seed `20260725`
- approach × raw vigilance cosine: -0.0774
- approach除去後のvigilanceノルム保持率: 99.70%（cosineから従属する整合性値）
- rawと残差条件は別random bankであり、45/100と59/100を前後差として比較しない
- どちらも期待と逆方向でrandomから分離せず、vigilance軸は新指標でも不成立
- 極性は実験前に訂正語=正、受容語=負と固定。rawは17/20、残差は18/20が予測と逆
- random slopeの符号は両bankとも49正／51負で、bank全体の符号バイアスでは説明しにくい
- 逆符号の一貫性は次回仮説候補に留め、絶対値がrandom中央のため逆向き軸とは解釈しない
- prompt slopeはrawで平均-0.000857、標本SD 0.000963、範囲-0.002494～+0.000962
- 残差は平均-0.000869、標本SD 0.000826、範囲-0.002052～+0.001263
- raw／残差のprompt別相関はr=0.453。方向は99.70%保持、平均と符号数もほぼ不変なのにprompt別効果量の相関が中程度に留まり、極小効果のprompt別順位は小さな方向差へ敏感で再現性が低い
- 定数オフセットだけではないが、共通テンプレート族の20問を独立試行とは断定せず、符号検定は探索的参考に留める
- 極性は同一セッション内で結果取得前に指定されていたことをファイル時刻で確認（script 23:23、RunKit 23:27、結果ZIP 23:38）。外部事前登録ではない
- 単一token proxyの結果であり、生成文の実際の訂正率までは評価していない

## Anchored v2

- 方向作成: 最初の10 prompts
- 評価: 別の10 prompts
- 介入位置: 応答境界 token `-1`
- semantic alpha: `-2,-1,-0.5,0,0.5,1,2`
- random controls: 100
- approach文面と測定語は既存の成功条件へ固定
- 残差化順序: `approach -> vigilance -> arousal`

## 代表結果

| モデル・方向 | mean slope | positive prompts | random abs >= semantic | rank p |
|---|---:|---:|---:|---:|
| Gemma approach raw/residual | +0.008741 | 10/10 | 0/100 | 0.0099 |
| Gemma arousal raw | +0.005450 | 10/10 | 2/100 | 0.0297 |
| Gemma arousal residual | +0.005040 | 9/10 | 7/100 | 0.0792 |
| Qwen3 approach raw/residual | +0.004749 | 10/10 | 0/100 | 0.0099 |
| Qwen3 arousal raw | +0.010830 | 10/10 | 0/100 | 0.0099 |
| Qwen3 arousal residual | +0.010056 | 10/10 | 0/100 | 0.0099 |
| OLMo approach raw/residual | +0.121853 | 10/10 | 19/100 | 0.1980 |
| OLMo vigilance raw | +0.143924 | 10/10 | 11/100 | 0.1188 |
| OLMo arousal raw | -0.095165 | 0/10 | 22/100 | 0.2277 |

モデル間でmean slopeの絶対値は比較しない。random分離、prompt一貫性、クロス軸漏れを重視する。

Qwen3のarousal残差は主効果 `+0.010056` に対し、approach漏れ `+0.000423`、vigilance漏れ `+0.000848` で、比較的選択的だった。Gemmaのarousalは残差化後にrank pが0.0792へ弱まり、候補扱いに留める。

OLMo baseは方向変化自体が大きいがrandom方向でも同程度に動く。これは「無反応」ではなく「意味方向としての選択性が弱い」と解釈する。モデル系列も異なるため、指示追従学習だけを原因とは断定しない。

## Qwen3 no-thinking exact candidate探索

first-token bankと実際の生成候補のずれを確認するため、Qwen3-1.7Bの
`enable_thinking=False`条件でapproach方向を別に再構築した。応答境界へ介入し、
完成回答 `proceed` と `wait`をEOSまでteacher forcingして、系列全体のlog probabilityを
比較した。これは既存holdoutとはchat templateが異なる探索条件である。

- `proceed` token IDs: `[776, 4635, 151645]` (`pro | ceed | <|im_end|>`)
- `wait` token IDs: `[11489, 151645]` (`wait | <|im_end|>`)
- baseline total logprob差（proceed - wait）: `+3.093816`
- 自由生成: 全条件で `proceed`

| direction / alpha | bank contrast delta | exact A-B total logprob delta |
|---|---:|---:|
| semantic / +2 | +0.043294 | +0.000000 |
| semantic / -2 | -0.031250 | -0.031250 |
| random #1 / -2 | — | -0.000000 |
| random #2 / -2 | — | -0.031250 |
| random #3 / -2 | — | -0.031250 |
| random #4 / -2 | — | +0.000000 |
| random #5 / -2 | — | +0.000000 |

negative semantic介入はexact候補差を予測方向へ動かしたが、5 random中2本が同じ値を
再現したため選択的効果とは判定しない。非ゼロ値 `0.031250 = 1/32` はfp16 logitの
量子化刻みである可能性があり、control数も5本に限られる。

この探索では、定義したbank contrast、完成回答系列の相対score、自由生成の3つが
同一の測定ではないことが直接確認された。既存holdoutが示すのはbank指標上の選択性であり、
exact候補選択や生成文変化まで実証したものではない。

## Qwen3 approach 独立holdout・refusal proxy監査

anchored v2の事後性と、approachが既知のrefusal/compliance系方向の言い換えである可能性を監査した。
方向作成に使っていない新規20 prompt、未使用seed `20260723`、100 random controlsを使用した。

| 条件 | mean slope | positive prompts | random abs >= semantic | rank p |
|---|---:|---:|---:|---:|
| approach raw | +0.004260 | 20/20 | 0/100 | 0.0099 |
| refusal proxy成分除去後 | +0.004183 | 20/20 | 2/100 | 0.0297 |

- source layer 18におけるapproachとrefusal proxyのcosine: -0.0247
- refusal proxy除去後に保持されたapproachノルム: 99.97%（cosineから従属して決まる整合性値）
- 層別cosineは全層で負、source layer 18付近ではほぼ0
- 隠れ次元2048の無関係方向の典型スケールは約±0.022で、-0.0247は同程度
- raw approachは0/100、除去後は2/100が同等以上。両条件を混同しない。
- 除去後に同等以上だったrandomは、このQwen3 runのindex 49と46。旧Qwen2.5の253/293とは別物。

このholdoutではapproach効果が再現し、使用したrefusal proxy成分を除去してもほぼ維持された。
したがってapproachと今回のproxyの重なりはrandom baseline級であり、approachは有効な方向のひとつと解釈する。
ただしrefusal proxyはharmful-minus-harmless活性差であり、Arditi et al.の因果的方向選択手順の完全再現ではない。

## 3層の役割分担

```text
6D NeuroState
  -> 神経伝達物質メタファーを使うプロンプト生成DSL

3軸 approach / vigilance / arousal
  -> 外部観測と状態整理のカナリア

モデル内部残差
  -> 検証を通過したモデル別方向だけ使用
```

安全境界はNeuroStateへ委ねず、権限、承認、許可リスト、入出力検査などモデル外部で保証する。

## 成果物

- `NEUROSTATE_3AXIS_SPEC_JA.md`
- `colab_neurostate_3axis.py`
- `Qwen_NeuroState_3Axis_Colab.ipynb`
- `Qwen_NeuroState_3Axis_Residual_Colab.ipynb`
- `Gemma_NeuroState_3Axis_Anchored_Colab.ipynb`
- `Qwen3_NeuroState_3Axis_Anchored_Colab.ipynb`
- `OLMo_NeuroState_3Axis_Anchored_Colab.ipynb`
- `outputs/anchored_3axis_results/`
- `qwen3_approach_refusal_holdout_results.zip`
- `qwen3_vigilance_false_premise_results.zip`
