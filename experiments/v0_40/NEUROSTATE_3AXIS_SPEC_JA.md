# NeuroState 3軸仕様 v0.1

## 状態表現

各軸は `-1.0` から `+1.0`。`0.0` は中立状態とする。

| 軸 | -1 | +1 |
|---|---|---|
| approach | 回避・停止 | 接近・実行 |
| vigilance | 信頼・受容 | 警戒・検証 |
| arousal | 沈静・低活動 | 覚醒・高活動 |

```json
{"approach": 0.7, "vigilance": 0.8, "arousal": 0.2}
```

## 実装原則

- 数値はアプリケーション側の共通状態表現として使う。
- LLMへは数値だけを渡さず、モデル別の意味説明へ変換する。
- 各軸を単独で操作し、主指標への効果と他軸への漏れを測定する。
- 危険操作の禁止や承認要求は3軸へ委ねず、外部の権限・ルール層で保証する。
- 旧6次元から3軸への固定写像は、独立効果が未確認なので現時点では定義しない。

## 初期プロンプト変換

- approach `+1`: ready to approach the task and take practical action
- approach `-1`: inclined to avoid action and hold back
- vigilance `+1`: highly vigilant; verify claims and request evidence
- vigilance `-1`: broadly trusting and inclined to accept claims
- arousal `+1`: highly alert, energized, and responsive
- arousal `-1`: calm, low-activation, and unhurried

この文章は制御実装であり、生物学的ホルモン値の主張ではない。
