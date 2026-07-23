# Qwen3 Approach Steering & Token Trace Demo 使い方

## このデモで見られるもの

Qwen3-1.7Bの内部にあるapproach方向を操作し、生成中の
`proceed語群 - hesitate語群` のlogit差と、実際に選ばれたtokenを並べて観察します。
同じ強さのsemantic方向と5本の直交random方向を切り替え、単なる文章変化と
意味方向に選択的な効果を区別するところまでが、このデモの主目的です。

これはLLMの感情や意識を測るものではありません。また、安全装置でもありません。
「内部の出力傾向が変わっても、選ばれる言葉はすぐには変わらない」現象を触って見る研究デモです。

## 起動手順

1. Colabで `Qwen3_Approach_Steering_Demo_Colab.ipynb` を開く。
2. ランタイムをT4 GPUへ変更する。
3. セルを上から実行する。
4. アップロード画面で次の2ファイルを同時に選ぶ。
   - `colab_neurostate_3axis.py`
   - `colab_approach_steering_demo.py`
5. `demo ready` が出たら、下の操作パネルを使う。

介入層は監査済みの source 18 → target 20 に固定されています。他層はこのデモでは実行できません。

依存関係セルでは、ColabにプリインストールされたGradioを削除します。このデモはGradioを
使わず、固定した `transformers==4.53.2` とGradio 6系が要求する
`huggingface-hub` の世代衝突を避けるためです。削除はそのColabランタイム内だけに作用します。

## 3つのモード

### Observe only

介入せず、生成中の内部contrastだけ記録します。比較の基準です。

### Boundary: first token

最初の出力tokenだけへ介入します。現在の検証済み条件に最も近いモードです。

### Continuous: every token

生成中の全tokenへ継続介入します。文章差が見えやすい一方、未検証の探索条件です。
結果を検証済みapproach効果と混同しないでください。

## Thinking / no-thinking

`enable thinking (validated default)` は初期状態でオンです。この条件が既存のholdoutと
今回の4 prompt比較に対応します。

チェックを外すと、Qwen3のchat templateを `enable_thinking=False` に切り替えます。
response boundaryが変わるため、semantic方向とrandom 5本もno-thinking条件で新しく
構築してキャッシュします。初回実行だけ時間がかかります。

no-thinkingは探索条件です。既存のthinking条件と効果量を直接比較せず、no-thinking内で
semantic対randomを改めて比較してください。

## Extended alpha

既定範囲は検証に使用した `-2` から `+2` です。`extended alpha ±4 (exploratory)` を
オンにした場合だけ、スライダーを `-4` から `+4` へ拡張できます。

これは効果を強く見せるための成功判定ではなく、token順位の初回分岐、反復、文法崩壊、
終了tokenの消失などがどこで始まるかを観察する探索設定です。±2の結果と別枠で記録し、
semanticで変化した場合は同じalphaのrandom方向とも比較してください。

## 最初に試す入力

今回のデモで最初に推奨する入力です。

```text
I have ten minutes and a cluttered desk. What should I do first?
```

まずBoundaryモード・`direction = semantic`で、同じ入力を次の順番で実行します。

```text
alpha = -2, -1, 0, +1, +2
```

alphaを上げるほどdeltaが概ね増えれば、approach steeringが単調に働いています。
次にalphaを `+2`へ固定し、`random #1`〜`random #5`へ切り替えて比較してください。
randomでも文章は変わり得るため、文章差だけを成功判定には使いません。

## おすすめ入力集

### 小さな作業を始める

```text
I have been postponing a small household task. Help me begin.
```

```text
I need to back up an important folder. Give me the first concrete step.
```

```text
Write one short sentence about starting an unfinished project.
```

### 選択肢から一歩を決める

```text
I have several possible tasks and feel uncertain. Suggest one immediate action.
```

```text
I have fifteen minutes before a meeting. What useful thing should I do now?
```

### 検証済みholdoutに近い短文

```text
Write one short sentence about sorting photographs.
```

```text
Write one short sentence about checking a bicycle tire.
```

```text
Write one short sentence about drafting a shopping list.
```

### 内部と外部がずれる例を探す

```text
I know what I should do, but I keep waiting. Give me one practical first move.
```

```text
Describe the first step of checking tomorrow's calendar.
```

内部contrastが変わっても、首位tokenが同じなら生成文は同じ経路を進むことがあります。

## おすすめ比較実験

### 実験A: semantic対random

- 同じprompt、mode、tokens、alphaを固定
- directionだけsemantic、random #1〜#5へ切り替える
- 観察: semanticのdeltaがrandom群と比べて一貫して大きいか

文章が変わっただけでは意味方向の証拠になりません。randomでも変化することを先に体験します。

### 実験B: alphaの単調性

- mode: Boundary
- tokens: 40
- alpha: -2, -1, 0, +1, +2
- 観察: step 0のdeltaがalphaと同じ向きに動くか

### 実験C: 内部変化と文章の差

- 同じprompt、同じalphaを使う
- Observe、Boundary、Continuousを順番に実行
- contrastが変わり始める時点と、選択tokenが分岐する時点を比べる

### 実験D: 介入強度

- mode: Continuous
- alpha: +0.5, +1, +2
- 観察: 強くするほど文章が実用的になるか、不自然な反復や崩れが出るか

Continuousの文章変化は探索結果であり、1回の出力だけで意味を決めません。

### 実験E: 正負の非対称性

- mode: Continuous
- alpha: -2と+2
- 観察: +側の「進む」と-側の「ためらう」が鏡写しになるか

正負が対称とは限りません。LLMの生成は非線形です。

### 実験F: no-thinkingで日本語出力

- `enable thinking` のチェックを外す
- mode: Boundary
- tokens: 64
- alpha: +2
- direction: semantic、その後random #1〜#5
- prompt:

```text
会議まで15分あります。今からできる有益なことを一つ提案してください。必ず日本語で簡潔に回答してください。
```

最初のtokenが `<think>` 以外になるか、semanticがrandom群を上回るか、日本語の最終回答に
なるかを別々に記録します。回答言語の変化だけをapproach効果とは判定しません。

## 表とグラフの読み方

- `baseline`: その時点の生成履歴における、介入直前のcontrast。
- `steered`: そのtokenを選ぶ際に介入を加えたcontrast。
- `delta`: steered - baseline。
- `applied`: そのstepで介入したか。
- `token`: 実際に選ばれたtoken。
- `raw_token`: tokenizer語彙上の未加工token。日本語のbyte断片を確認できます。
- `cumulative`: そのstepまでのtokenをまとめてdecodeした文字列。
- `baseline_top`: 介入前の首位token。
- `choice_top`: 実際に選択した介入後の首位token。
- `argmax_changed`: 介入前後で首位tokenが入れ替わったか。
- `baseline_margin`: 介入前の1位logitと2位logitの差。
- `steered_margin`: 介入後の1位logitと2位logitの差。

内部値が変わってもtokenが変わらない場合があります。候補tokenの順位境界を越えるまで、
内部の連続変化は外部の文章へ現れません。

`alpha=0`では実効的なhookが入らないため、`applied`はFalseになります。Boundaryまたは
Continuousを選んでいても、ゼロ介入を適用済みとは表示しません。

## Exact candidate sequence scores

従来のcontrastは、先頭空白付きの英語bank語について最初のtoken IDを集計します。しかし
実際の回答では `proceed` が `pro | ceed` のように複数tokenへ分かれる場合があり、
first-token bankと生成候補は同一ではありません。

`candidate A` と `candidate B`へ完成回答を入力すると、各候補をteacher forcingでEOSまで
採点し、次を表示します。

- 候補を構成するtoken IDとraw token。
- 介入前後の候補全文の合計log probability。
- 1 tokenあたりの平均log probability。
- AとBの合計score差が介入でどれだけ動いたか。

合計scoreは候補系列そのものの尤度ですが、長い候補ほど負の値が加算されます。平均scoreは
長さをならした参考値で、別の判断基準です。両方を表示し、一方だけを結果後に選びません。
この採点は強制した候補の比較であり、モデルが自由生成でその回答を選ぶことを保証しません。

## 成功・不成功の見方

成功の目安:

- 正alphaでdeltaが正、負alphaでdeltaが負。
- alphaを増やすとdeltaが概ね単調に増える。
- 複数promptで同じ傾向が再現する。

成功とは限らないもの:

- 文章がたまたま行動的に見えた。
- 1 promptだけで大きく変わった。
- Continuousで派手な文章差が出た。

## 注意

- 無害なpromptだけで試してください。
- 削除、送信、購入、認証、外部ツール操作へ接続しないでください。
- approachはアクセル候補であり、安全な停止や確認行動を保証しません。
- 生成文の印象より、contrast・random control・複数promptでの再現を優先します。
- 内部表現の変化を、主観的な感情や意識の証拠とは解釈しません。
