# Design: evaluate API Redesign

## 新しい公開 API

### `evaluate(df_rec, df_eval, order=1, user_col=None, item_col=None)`

```python
scorer.evaluate(df_rec, df_eval)
scorer.evaluate(df_rec, df_eval, order=10)
```

- `df_rec`: `transform()` の出力（`order` 列を持つ DataFrame）
- `df_eval`: 評価期間のインタラクションログ
- `user_col` / `item_col`: `df_rec` と `df_eval` 両方に適用

内部処理:
```python
UIrevisit = set(zip(df_eval[user_col].astype(str), df_eval[item_col].astype(str)))
# 以降は旧ロジックをそのまま使用
```

## バリデーション

```python
if not isinstance(df_eval, pd.DataFrame):
    raise TypeError("df_eval must be a pandas DataFrame.")
missing = [c for c in [user_col, item_col] if c not in df_eval.columns]
if missing:
    raise ValueError(f"Missing required columns in df_eval: {missing}")
```

`datetime_col` は不要（evaluate では使用しない）。

## テスト変更方針

| 旧テスト | 変更後 |
|---|---|
| `scorer_fitted.evaluate(df_rec, _UIREVISIT, ...)` | `scorer_fitted.evaluate(df_rec, df_eval, ...)` |
| `test_recall_norm_with_unseen_revisits` | `df_eval` に行を追加して拡張 |
| `test_uses_init_col_names_by_default` | `df_eval` DataFrame を渡すよう変更 |
| `test_invalid_uirevisit_type_raises` | `test_invalid_df_eval_type_raises` に変更 |

`df_eval` フィクスチャを追加:
```python
@pytest.fixture(scope="module")
def df_eval():
    df = _make_df()
    return df[pd.to_datetime(df["datetime"]) >= _EVAL_PERIOD[0]]
```
