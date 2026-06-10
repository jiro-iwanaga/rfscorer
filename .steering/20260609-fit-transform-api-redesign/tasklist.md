# Tasklist: fit/transform API Redesign

## ステアリングドキュメント

- [x] requirements.md
- [x] design.md
- [x] tasklist.md

## `scorer.py` 実装変更

- [x] 新 `fit(df_obs, df_eval, ref_date=None, recency_limit=None, frequency_limit=None)` を実装
  - df_obs / df_eval のバリデーションと内部形式への変換
  - ref_date のデフォルト計算（df_obs の最大日）
  - 日付属性を実データから設定
  - `_fit_impl` にコアロジックを抽出
- [x] 旧 `fit()` を `fit_date()` にリネーム、内部から `_fit_impl` を呼ぶよう整理
- [x] `fit_period()` を `_fit_impl` 呼び出しに整理（シグネチャ変更なし）
- [x] 新 `transform(df, ref_date=None, kind, ...)` を実装（`target_date` フィルタ削除）
- [x] 旧 `transform()` を `transform_date()` に追加、内部から新 `transform` を呼ぶよう整理
- [x] `_to_internal()` ヘルパーを追加（内部列名への変換を共通化）

## `tests/test_scorer.py` 更新

- [x] `df_rec` fixture を `transform_date` を使うよう更新
- [x] `TestFitValidation` → 新 `fit(df_obs, df_eval)` バリデーションテストに書き換え
- [x] `TestFitDateValidation` を新設（旧 `TestFitValidation` の内容を移動・fit_date に改名）
- [x] `TestFitResult` → 新 `fit(df_obs, df_eval)` 正常系テストに書き換え
- [x] `TestFitDateResult` を新設（旧 `TestFitResult` の内容を移動・fit_date に改名）
- [x] `TestTransform` を新 API (`transform(df)`) 向けに書き換え
- [x] `TestTransformDate` を新設（旧 `TestTransform` の内容を移動・transform_date に改名）

## 品質チェック

- [x] `uv run pytest` (285 passed)
- [x] `uv run ruff check .`
- [x] `uv run ruff format --check .`
