# Requirements: evaluate API Redesign

## 概要

`evaluate()` の第2引数を `UIrevisit: set` から `df_eval: pd.DataFrame` に変更する破壊的変更。
後方互換メソッドは残さない。

## 動機

`fit(df_obs, df_eval)` / `transform(df_obs)` の変更により、ユーザーは `df_eval` をすでに持っている。
`evaluate` に渡すために `{(user, item) for row in df_eval.itertuples()}` と変換させるのは不要な摩擦。

理想のフロー:
```python
scorer.fit(df_obs, df_eval)
df_rec = scorer.transform(df_obs)
scorer.evaluate(df_rec, df_eval)   # df_eval をそのまま渡す
```

ライブラリ公開から1週間、ユーザーなし。今が変更の機会。

## 変更内容

| | 変更前 | 変更後 |
|---|---|---|
| 第2引数 | `UIrevisit: set[tuple[str, str]]` | `df_eval: pd.DataFrame` |
| 後方互換 | — | なし（削除） |

`df_eval` からの `UIrevisit` 導出はライブラリ内部で行う。

## 制約事項

- `df_eval` には `user_col` と `item_col` が必要（`datetime_col` は不要）
- `user_col` / `item_col` は `evaluate` の引数（省略時は `__init__` の値）を `df_eval` にも適用する

## 受け入れ条件

- [ ] `scorer.evaluate(df_rec, df_eval)` が動作する
- [ ] `df_eval` が DataFrame でない場合 `TypeError` が出る
- [ ] `df_eval` に必要列が不足している場合 `ValueError` が出る
- [ ] 結果が旧 API（UIrevisit set 渡し）と一致する
- [ ] 全テストが通過する
