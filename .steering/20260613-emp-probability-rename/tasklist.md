# Tasklist: `empirical_probability_*` を `emp_probability_*` にリネーム

## 実装タスク

### scorer.py の変更

- [ ] **T01** `__init__()` で属性名をリネームする
  - `self.empirical_probability_ = None` → `self.emp_probability_ = None`
  - `self.empirical_probability_table_ = None` → `self.emp_probability_table_ = None`
  - `self.empirical_probability_dict_ = None` → `self.emp_probability_dict_ = None`
  - 「# empirical」のコメントセクションヘッダは `# emp` または英語のまま維持（軽微）

- [ ] **T02** `_fit_impl()` で代入箇所をリネームする
  - `self.empirical_probability_dict_ = {...}` → `self.emp_probability_dict_ = {...}`
  - `self.empirical_probability_ = pd.DataFrame(...)` → `self.emp_probability_ = pd.DataFrame(...)`
  - `self.empirical_probability_table_ = self.empirical_probability_.pivot_table(...)` → `self.emp_probability_table_ = self.emp_probability_.pivot_table(...)`
  - `df_r = self.empirical_probability_.groupby(...)` → `df_r = self.emp_probability_.groupby(...)`
  - `df_f = self.empirical_probability_.groupby(...)` → `df_f = self.emp_probability_.groupby(...)`

- [ ] **T03** `predict()` の参照と事前チェックをリネームする
  - `if kind in ("emp", "er", "ef") and self.empirical_probability_dict_ is None:` → `... and self.emp_probability_dict_ is None:`
  - `prob = self.empirical_probability_dict_.get((r, f), 0.0)` → `prob = self.emp_probability_dict_.get((r, f), 0.0)`

- [ ] **T04** `transform()` の事前チェックをリネームする
  - `if self.empirical_probability_dict_ is None:` → `if self.emp_probability_dict_ is None:`

- [ ] **T05** `_probability_dict()` のデフォルト返却をリネームする
  - `return self.empirical_probability_dict_` → `return self.emp_probability_dict_`

- [ ] **T06** `plot_probability_surface()` の参照をリネームする
  - `if kind == "emp" and self.empirical_probability_table_ is None:` → `... and self.emp_probability_table_ is None:`
  - `table = self.empirical_probability_table_` → `table = self.emp_probability_table_`

- [ ] **T07** `show()` の参照と print ラベルをリネームする
  - `if self.empirical_probability_table_ is not None:` → `if self.emp_probability_table_ is not None:`
  - `print("empirical_probability_table_:")` → `print("emp_probability_table_:")`
  - `print(self.empirical_probability_table_.round(3)...)` → `print(self.emp_probability_table_.round(3)...)`

- [ ] **T08** `export_probability_csv()` の参照と CSV カラム名をリネームする
  - 事前チェック: `if kind in ("emp", "er", "ef", "all") and self.empirical_probability_ is None:` → `... and self.emp_probability_ is None:`
  - `kind="all"` 起点: `self.empirical_probability_.rename(columns={"probability": "empirical_probability"})` → `self.emp_probability_.rename(columns={"probability": "emp_probability"})`
  - `kind == "emp"` 分岐: `df = self.empirical_probability_` → `df = self.emp_probability_`

- [ ] **T09** 各メソッドの docstring Notes セクションをリネームする
  - `fit()` / `fit_date()` / `fit_period()` の Notes 内 `` ``empirical_probability_``, ``empirical_probability_table_``, ``empirical_probability_dict_`` `` → `` ``emp_probability_``, ``emp_probability_table_``, ``emp_probability_dict_`` ``

- [ ] **T10** `_KIND_ALIASES` dict が無傷であることを確認する（**変更しない**）
  - `"empirical": "emp"` のキー文字列が残っている
  - `"empirical_recency": "er"` のキー文字列が残っている
  - `"empirical_frequency": "ef"` のキー文字列が残っている

- [ ] **T11** `__main__` ブロックを確認する
  - `scorer.plot_probability_surface("empirical")` は kind alias を渡しているため変更不要を確認

### optimizer.py のチェック

- [ ] **T12** `src/rfscorer/optimizer.py` を確認する
  - `empirical_probability` のリテラル文字列があれば更新
  - 英語形容詞としての "empirical" は維持

### tests/test_scorer.py の変更

- [ ] **T13** 属性参照を全てリネームする
  - `scorer_fitted.empirical_probability_` → `scorer_fitted.emp_probability_`
  - `scorer_fitted.empirical_probability_table_` → `scorer_fitted.emp_probability_table_`
  - `scorer_fitted.empirical_probability_dict_` → `scorer_fitted.emp_probability_dict_`
  - `scorer.empirical_probability_` などの他のフィクスチャ経由参照も同様
  - 注意: テスト内の `kind="empirical"`（kind alias）は変更しない

- [ ] **T14** CSV 出力テストの期待カラム集合をリネームする
  - `assert set(df.columns) == {..., "empirical_probability", ...}` → `{..., "emp_probability", ...}`
  - 該当箇所: `test_all_output_columns` など

- [ ] **T15** テストメソッド名・コメントを確認する
  - `test_empirical_*` のような名前があれば検出し更新
  - コメント内の `empirical_probability_*` 参照があれば更新

### ドキュメントの変更

- [ ] **T16** `docs/product-requirements.md` を更新する
  - 出力表（L90 周辺）: `empirical_probability_` → `emp_probability_`
  - コード例（L129 周辺）: `df_emp = scorer.empirical_probability_` → `df_emp = scorer.emp_probability_`

- [ ] **T17** `docs/functional-design.md` を更新する
  - 属性表内の `empirical_probability_` / `_table_` / `_dict_` 行
  - データフロー図内の `empirical_probability_ / _table_ / _dict_` 表記
  - 各メソッドの Notes セクション内の属性名列挙

- [ ] **T18** `docs/glossary.md` を更新する
  - `empirical_probability_` / `empirical_probability_table_` / `empirical_probability_dict_` の3エントリを `emp_*` にリネーム

- [ ] **T19** `docs/development-guidelines.md` を確認・更新する
  - `empirical_probability` を grep して該当があれば更新

### 例・README の変更

- [ ] **T20** `examples/basic_usage.ipynb` を更新する
  - `scorer.empirical_probability_` の参照があれば `scorer.emp_probability_` に置換
  - 出力セルの再実行は別タスク（必要に応じて）

- [ ] **T21** `README.md` を確認・更新する
  - 英語の説明内に `empirical_probability_` の参照があれば更新

### 品質チェック

- [ ] **T22** 残存チェック: `empirical_probability` がコード/ドキュメントに残っていないことを確認する
  - 期待: 残存ゼロ（`.steering/` および `CHANGELOG.md` を除く）

- [ ] **T23** kind alias 保護の確認
  - `grep -n '"empirical":' src/rfscorer/scorer.py` → 1件残る
  - `grep -n '"empirical_recency":' src/rfscorer/scorer.py` → 1件残る
  - `grep -n '"empirical_frequency":' src/rfscorer/scorer.py` → 1件残る
  - `uv run pytest tests/test_scorer.py::TestKindAliases -v` → 全パス

- [ ] **T24** リントを実行する（`uv run ruff check .`）

- [ ] **T25** フォーマットを実行する（`uv run ruff format .`）

- [ ] **T26** テストを実行する（`uv run pytest`）

## 完了条件

すべてのタスクが完了し、以下が成立していること：

1. `uv run ruff check .` が `All checks passed!`
2. `uv run ruff format --check .` が `all files already formatted`
3. `uv run pytest` がすべて green
4. テスト件数の変動なし（純粋なリネームのため）
5. `grep -rn "empirical_probability" --include="*.py" --include="*.md" --include="*.ipynb" | grep -v "^\.steering\|^CHANGELOG"` が0件
6. `_KIND_ALIASES` の `"empirical"`, `"empirical_recency"`, `"empirical_frequency"` 各キーが scorer.py に保持されている
7. `TestKindAliases` のテストが全パス（kind alias 機能が無傷）

## 進め方の留意

- **T01-T11** を完了した時点で scorer.py 単体は self-consistent になる（属性の代入・参照のセット）
- **T13** 完了後に一度 `uv run pytest` を走らせると、scorer.py / test_scorer.py の整合性が確認できる
- **T22-T26** の最終チェックは順次実行
