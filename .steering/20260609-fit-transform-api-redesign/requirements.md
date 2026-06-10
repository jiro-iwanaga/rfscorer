# Requirements: fit/transform API Redesign

## 概要

`RecencyFrequencyScorer` の `fit()` / `transform()` のシグネチャを sklearn スタイルに変更する破壊的変更。

## 動機

現行の `fit(df, target_date)` は日付によるデータ分割ロジックをライブラリ内部で担っている。
sklearn の慣習（ユーザーが X/y を用意し、ライブラリは分割に関与しない）に合わせることで、
sklearn ユーザーが直感的に使えるようにする。

ライブラリ公開から1週間でユーザーがいないため、今が破壊的変更の機会。

## 変更内容

### `fit()` の変更

| | 変更前 | 変更後 |
|---|---|---|
| 第1引数 | `df`（全期間ログ） | `df_obs`（観測期間ログ、事前フィルタ済み） |
| 第2引数 | `target_date`（分割基準日） | `df_eval`（評価期間ログ、必須） |
| 削除引数 | `observation_days`, `evaluation_days` | — |
| 追加引数 | — | `ref_date`（最新度計算の基準日、省略時は `df_obs` の最大日） |

### `transform()` の変更

| | 変更前 | 変更後 |
|---|---|---|
| 第2引数 | `target_date`（フィルタ兼基準日） | `ref_date`（最新度計算の基準日、省略時は `df` の最大日） |
| 内部フィルタ | `df[df[datetime_col] <= target_date]` | 削除（ユーザー責務） |

### 後方互換メソッドの追加（別名で保持）

| 旧メソッド | 新メソッド名（旧ロジック保持） |
|---|---|
| `fit(df, target_date, observation_days, evaluation_days, ...)` | `fit_date(df, target_date, ...)` |
| `transform(df, target_date, ...)` | `transform_date(df, target_date, ...)` |
| `fit_period(df, obs_period, eval_period, ...)` | `fit_period` のまま保持 |

## 制約事項

- `df_eval` は `fit()` の必須引数（省略不可）
- `transform()` の `target_date` はなくなる。観測期間フィルタはユーザー責務
- `fit_period` は後方互換のため現行シグネチャのまま存続

## 受け入れ条件

- [ ] `scorer.fit(df_obs, df_eval)` が動作する
- [ ] `scorer.fit(df_obs, df_eval, ref_date="2024-01-07")` で ref_date 指定が動作する
- [ ] `scorer.transform(df_obs)` が動作する（target_date 不要）
- [ ] `scorer.fit_date(df, target_date, ...)` が旧 `fit` と同等の結果を返す
- [ ] `scorer.transform_date(df, target_date, ...)` が旧 `transform` と同等の結果を返す
- [ ] `scorer.fit_period(df, obs_period, eval_period, ...)` が引き続き動作する
- [ ] 全テストが通過する
