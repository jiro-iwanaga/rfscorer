# Tasklist: コンストラクタの軸対称化（`unit` → `recency_unit`）

## 実装タスク

### Phase 1: `__init__` の改名・並べ替え

- [x] **T01** `__init__` 署名を変更する（`scorer.py` 53–60 行付近、現状は複数行署名）
  - `unit=1` を削除し、末尾を `recency_mode="day"`, `recency_unit=1` の順にする
  - 既存の**複数行（1 引数 1 行）形式を維持**する（インライン1行は 100 桁超で `ruff` E501 になる）

- [x] **T02** バリデーションと属性を変更する（`scorer.py` 86–93 行付近）
  - `recency_mode` チェック → `recency_unit` チェックの順に並べる
  - `if recency_unit <= 0: raise ValueError(f"recency_unit must be a positive integer, got {recency_unit}.")`
  - `self.unit = unit` → `self.recency_unit = recency_unit`

- [x] **T03** `__init__` docstring を更新する
  - `unit` 段落 → `recency_unit` 段落（recency 軸ビン幅・式 `// recency_unit + 1`・view 時は無視）
  - `recency_mode` 段落の式・「`ref` and `unit` are ignored」→「`ref` and `recency_unit` are ignored」
  - 記述順を署名順（`recency_mode` → `recency_unit`）に合わせる
  - 将来メモ1行: `frequency_mode`（既定 `"view"`）/ `frequency_unit` を将来追加予定。既定非対称（recency=`"day"` / frequency=`"view"`）は古典的 RF（recency=時間ベース・frequency=回数ベース）に由来

### Phase 2: 内部参照の置換

- [x] **T04** `_recency.build_day_rf` の引数を改名する（`_recency.py`）
  - `def build_day_rf(df, user_col, item_col, seq_col, ref_int, recency_unit)`
  - 本体 `// unit + 1` → `// recency_unit + 1`
  - モジュール docstring 冒頭の `// unit + 1` も `// recency_unit + 1` に

- [x] **T05** ディスパッチャ `_build_ui_rf_df` を更新する（`scorer.py` 1645 行付近）
  - `build_day_rf(..., ref_int, self.recency_unit)` に変更
  - `_build_ui_rf_df` docstring の `// unit + 1` / 「`ref_int` and `unit` are ignored」を `recency_unit` に

- [x] **T06** `save_zip()` の metadata キーを変更する（`scorer.py` 1347 行付近）
  - `"unit": _to_python(self.unit)` → `"recency_unit": _to_python(self.recency_unit)`

- [x] **T07** `fit()` docstring の式を更新する（`scorer.py` 216 行付近）
  - `(ref - value) // unit + 1` → `(ref - value) // recency_unit + 1`

- [x] **T08** 残存 `unit` 参照の最終スイープ（`grep -rn "self\.unit\|// unit\|\bunit\b" src/rfscorer/`）
  - `recency_unit`・無関係な英語の "unit"（例: `utils.py` の "binning unit" 文）を切り分け、パラメータ由来のみ置換

### Phase 3: テストの更新

- [x] **T09** `tests/test_scorer.py` の `TestUnit` を更新・改名する（1700–1740 行付近）
  - クラス名 `TestUnit` → `TestRecencyUnit`
  - ヘルパー `_make_scorer_with_unit` → `_make_scorer_with_recency_unit`、引数名 `unit` → `recency_unit`
  - `RecencyFrequencyScorer(unit=...)` → `RecencyFrequencyScorer(recency_unit=...)`（通常・0・-1 の3箇所）
  - エラーテストの `match="unit must be a positive integer"` → `match="recency_unit must be a positive integer"`
  - テストメソッド名・docstring・コメントの `unit` を `recency_unit` に（例: `test_recency_unit_7_recency_is_floor_div_of_recency_unit_1`）

- [x] **T10** `tests/test_recency.py` のコメントを整える
  - `build_day_rf(..., 7)` は位置引数のため実コード変更不要。`# unit=7` コメントを `# recency_unit=7` に

- [x] **T11** 回帰確認: `uv run pytest -q`
  - 全テスト green（`unit` を渡すテストが残っていないこと）

### Phase 4: ドキュメント更新

- [x] **T12** `docs/functional-design.md` を更新する
  - コンストラクタ署名行（line 105）・パラメータ表（line 113 の `unit` 行 → `recency_unit` 行、`recency_mode` を前に）
  - `recency_mode` 補足（line 114）の式 `// unit + 1` → `// recency_unit + 1`
  - metadata 一覧（line 301）の `unit` → `recency_unit`
  - **将来拡張メモ（line 120）の修正**: 「ビン幅の軸別指定 `recency_unit` / `frequency_unit` を想定」から `recency_unit` を**外す**（本タスクで実現済み）。将来は `frequency_mode`（既定 `"view"`）/ `frequency_unit` のみとし、既定非対称の根拠を1行添える
  - `fit`/`transform` の `ref` 行は `unit` 記述なし（変更不要）

- [x] **T13** `docs/glossary.md` を更新する
  - day recency 式 `// unit + 1` → `// recency_unit + 1`
  - API 表「列名・粒度 `unit`」→「`recency_unit`」

- [x] **T14** `docs/product-requirements.md` を更新する
  - コンストラクタ入力の `unit` 箇条 → `recency_unit`（recency 軸ビン幅・view 時は無視）

### Phase 5: 品質チェック

- [x] **T15** リント: `uv run ruff check .`

- [x] **T16** フォーマット: `uv run ruff format --check .`

- [x] **T17** 全テスト: `uv run pytest`
  - 既存テスト＋改名後テストがすべて green

## 完了条件

すべてのタスクが完了し、以下が成立していること:

1. `uv run ruff check .` が `All checks passed!`
2. `uv run ruff format --check .` が `all files already formatted`
3. `uv run pytest` がすべて green
4. コンストラクタが `(user_col, item_col, time_col, recency_mode="day", recency_unit=1)` で利用可能
5. `recency_mode="day"` / `recency_unit=1` で現行（`unit=1`）と数値完全一致
6. `RecencyFrequencyScorer(unit=1)` が `TypeError`（パラメータ不在）になる
7. `recency_unit <= 0` で `ValueError`（`recency_unit must be a positive integer`）
8. `save_zip()` の metadata に `recency_unit` が含まれる（`unit` は無い）
9. docs（functional-design / glossary / product-requirements）が `recency_unit` に整合
10. テストが `TestRecencyUnit` に改名され、`unit=` 参照が残っていない
