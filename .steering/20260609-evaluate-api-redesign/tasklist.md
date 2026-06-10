# Tasklist: evaluate API Redesign

## ステアリングドキュメント

- [x] requirements.md
- [x] design.md
- [x] tasklist.md

## `scorer.py` 実装変更

- [x] `evaluate(df_rec, UIrevisit, ...)` → `evaluate(df_rec, df_eval, ...)` に変更
  - 第2引数を `df_eval: pd.DataFrame` に変更
  - バリデーション追加（TypeError / ValueError）
  - 内部で `UIrevisit` を導出
  - 旧 `UIrevisit` の型チェック・変換コードを削除

## `tests/test_scorer.py` 更新

- [x] `df_eval` フィクスチャを追加
- [x] `TestEvaluate` の全テストを `df_eval` DataFrame 渡しに変更
- [x] `test_recall_norm_with_unseen_revisits`: 行追加で拡張する形に変更
- [x] `test_uses_init_col_names_by_default`: `df_eval` DataFrame を渡すよう変更
- [x] `test_invalid_uirevisit_type_raises` → `test_invalid_df_eval_type_raises` に変更

## 品質チェック

- [x] `uv run pytest`
- [x] `uv run ruff check .`
- [x] `uv run ruff format --check .`
