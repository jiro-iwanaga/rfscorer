# Tasklist: er / ef を1次元周辺確率モデルにリファクタする

## 実装タスク

### scorer.py の変更

- [ ] **T01** `__init__()` から `_table_` 属性を削除する
  - `self.er_probability_table_ = None` を削除
  - `self.ef_probability_table_ = None` を削除

- [ ] **T02** `_marginal_dict()` ヘルパーメソッドを追加する
  - 引数: `kind` (`"mr"`, `"mf"`, `"er"`, `"ef"` のいずれか)
  - 戻り値: 対応する `_dict_` 属性
  - それ以外の `kind` は `ValueError`
  - 場所: 既存の `_probability_dict()` の近く

- [ ] **T03** `_fit_impl()` の er/ef 構築を1次元に変更する
  - `rows_er = [(r, f, self.R2Prob[r]) for r in self.R for f in self.F]` 形式の構築を削除
  - 代わりに以下を実装:
    ```python
    self.er_probability_dict_ = dict(self.R2Prob)
    self.er_probability_ = (
        pd.DataFrame(list(self.R2Prob.items()), columns=["recency", "probability"])
        .sort_values("recency").reset_index(drop=True)
    )
    ```
  - ef も同様（`F2Prob` → `frequency, probability`）
  - `_table_` 構築コードを削除

- [ ] **T04** `_probability_dict()` から er/ef を削除する
  - er / ef 分岐を削除（marginal 側に移動するため）
  - 残るのは emp, mono, mrc, mfc, mcc

- [ ] **T05** `predict()` で er/ef を 1次元処理に統合する
  - 1次元処理ブロックの条件を `kind == "mr"` → `kind in ("mr", "er")` に拡張
  - 同様に `kind == "mf"` → `kind in ("mf", "ef")`
  - `_marginal_dict(kind)` を用いて参照
  - 2次元処理ブロックから er / ef 分岐を削除
  - docstring の kind 説明と Notes を更新（er/ef も f / r を無視する旨）

- [ ] **T06** `transform()` で er/ef を 1次元 merge に統合する
  - `kind == "mr"` → `kind in ("mr", "er")` に拡張、`_marginal_dict(kind)` を使用
  - `kind == "mf"` → `kind in ("mf", "ef")` に拡張
  - 2D else 分岐から er/ef を除外（自動的に到達しなくなる）
  - docstring 更新

- [ ] **T07** `plot_probability_surface()` で er/ef を ValueError 拒否対象に追加する
  - `kind in ("mr", "mf")` → `kind in ("mr", "mf", "er", "ef")`
  - 有効 kind から er/ef を削除: `("emp", "er", "ef", "mono", "mrc", "mfc", "mcc")` → `("emp", "mono", "mrc", "mfc", "mcc")`
  - er/ef の事前チェックと table 取得分岐を削除
  - docstring の kind 説明から er/ef を削除、ValueError 説明に「1D marginal」として er/ef を追加

- [ ] **T08** `plot_marginal_probability()` に er/ef サポートを追加する
  - `valid_kinds = ("emp", "mr", "mf", "all")` → `("emp", "er", "ef", "mr", "mf", "all")`
  - 軸×kind の妥当性チェックを追加:
    - `axis == "recency" and kind in ("mf", "ef")` → ValueError
    - `axis == "frequency" and kind in ("mr", "er")` → ValueError
  - データソース選択: `kind == "er"` / `kind == "ef"` 時も `recency_probability_` / `frequency_probability_` を使用（emp と同等の周辺確率）
  - docstring の kind 説明を更新

- [ ] **T09** `export_probability_csv()` の er/ef を 1次元 merge に変更する
  - `kind == "all"` の merge チェーンで `er_probability_` を `on="recency"` に、`ef_probability_` を `on="frequency"` に変更
  - 単独 `kind="er"` / `kind="ef"` 時は新しい1次元 DataFrame をそのまま出力（コード変更不要、`_fit_impl()` で構造が変わる）
  - docstring の `kind="all"` 列説明（merge key）を更新

### tests/test_scorer.py の変更

- [ ] **T10** TestFitPeriodResult の er/ef 関連テストを更新する
  - `test_er_set_after_fit`: `er_probability_table_ is not None` 検証を削除
  - `test_ef_set_after_fit`: 同上
  - `test_er_constant_across_frequency` を削除
  - `test_ef_constant_across_recency` を削除
  - `test_er_matches_R2Prob` を `er_probability_dict_[r] == R2Prob[r]` 形式に修正
  - `test_ef_matches_F2Prob` を同様に修正
  - `test_er_table_shape` を削除
  - `test_ef_table_shape` を削除

- [ ] **T11** TestPredict の er/ef テストを更新・追加する
  - `test_er_kind`: `er_probability_dict_[(r,f)]` → `er_probability_dict_[r]` に修正
  - `test_ef_kind`: `ef_probability_dict_[(r,f)]` → `ef_probability_dict_[f]` に修正
  - 新規: `test_er_ignores_f`: `predict(1, 1, kind="er") == predict(1, 999, kind="er")`
  - 新規: `test_ef_ignores_r`: 同様
  - 新規: `test_clamps_r_to_recency_limit_er`: クランプ動作
  - 新規: `test_clamps_f_to_frequency_limit_ef`: 同上

- [ ] **T12** TestTransform に er/ef テストを新規追加する
  - `test_er_probability_matches_dict`: transform 結果の probability が `er_probability_dict_[recency_adj]` と一致
  - `test_ef_probability_matches_dict`: 同様

- [ ] **T13** TestPlotProbabilitySurface を er/ef ValueError 検証に変更する
  - `test_returns_figure_er` を削除
  - `test_returns_figure_ef` を削除
  - 新規: `test_er_raises_value_error`: `kind="er"` で `ValueError`（"1D marginal" マッチ）
  - 新規: `test_ef_raises_value_error`: 同様

- [ ] **T14** TestPlotMarginalProbability に er/ef テストを追加する
  - 新規: `test_returns_figure_er_recency`: `axis="recency", kind="er"` で Figure 返却
  - 新規: `test_returns_figure_ef_frequency`: `axis="frequency", kind="ef"` で Figure 返却
  - 新規: `test_ef_on_recency_axis_raises`: `axis="recency", kind="ef"` で `ValueError`
  - 新規: `test_er_on_frequency_axis_raises`: `axis="frequency", kind="er"` で `ValueError`

- [ ] **T15** TestExportProbabilityCsv の er/ef テストを更新する
  - `test_er_output_columns`: 期待カラムを `{"recency", "frequency", "probability"}` → `{"recency", "probability"}`
  - `test_ef_output_columns`: 期待カラムを `{"recency", "frequency", "probability"}` → `{"frequency", "probability"}`
  - `test_all_output_columns`: カラム集合は変わらない想定だが念のため確認
  - `test_all_row_count`: 行数は emp が支配的なので R × F のまま（変更なしを確認）

### ドキュメントの変更

- [ ] **T16** `docs/product-requirements.md` を更新する
  - L75 機能テーブル: er/ef 行の説明から「ブロードキャスト」を削除、「最新度 / 頻度の周辺確率（1次元）」と記述
  - L91, L92 出力テーブル: `er_probability_` / `ef_probability_` のカラム構成を1次元（`recency, probability` / `frequency, probability`）に修正

- [ ] **T17** `docs/functional-design.md` を更新する
  - 数式定義: er/ef を `$x_{r,f} = p_r$` 形式から 1次元 `$p_r$` / `$p_f$` に書き換え
  - 属性表: `er_probability_` / `ef_probability_` の構造説明を1次元に
  - 属性表: `er_probability_table_` / `ef_probability_table_` の行を削除
  - データフロー図: 「R2Prob を全 f にブロードキャスト」→「R2Prob を1次元 dict として保持」と修正
  - `predict()` / `transform()` / `plot_probability_surface()` / `plot_marginal_probability()` の docstring 説明箇所で er/ef の扱いを更新

- [ ] **T18** `docs/glossary.md` を更新する
  - `er_probability_` / `ef_probability_` のカラム構成を1次元に修正
  - `er_probability_dict_` / `ef_probability_dict_` のキー型を `(r,f)` → `r` / `f` に修正
  - `er_probability_table_` / `ef_probability_table_` の行を削除（mr/mf 同様）

- [ ] **T19** `examples/basic_usage.ipynb` を確認・修正する
  - `er` / `ef` の使用箇所を grep（特に `plot_probability_surface(kind="er")` 等）
  - 該当があれば `plot_marginal_probability(axis=..., kind="er"|"ef")` に変更、または削除
  - 出力セルが残っている場合は notebook を再実行して更新

### 品質チェック

- [ ] **T20** リントを実行する（`uv run ruff check .`）
- [ ] **T21** フォーマットを実行する（`uv run ruff format .`）
- [ ] **T22** テストを実行する（`uv run pytest`）

## 完了条件

すべてのタスクが完了し、T20〜T22 がエラーなく通過していること。

具体的に：
1. `uv run ruff check .` が `All checks passed!`
2. `uv run ruff format --check .` が `all files already formatted`
3. `uv run pytest` がすべて green（テスト件数は T10-T15 の追加・削除に応じて変動）
4. `docs/product-requirements.md` / `docs/functional-design.md` / `docs/glossary.md` で er/ef のブロードキャスト記述が完全に消えている
5. `er_probability_table_` / `ef_probability_table_` への参照がコード・テスト・ドキュメント全てで消えている
