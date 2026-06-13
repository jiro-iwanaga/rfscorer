# Requirements: er / ef を1次元周辺確率モデルとしてリファクタする

## 背景

現在、本パッケージの最適化モデル `mr` / `mf` は1次元周辺確率モデルとして実装されている（過去のリファクタ `refactor: implement 1D mr/mf models without 2D broadcast` で 2D ブロードキャストを廃止済み）。

一方、対応する経験的版 `er` / `ef`（Empirical Recency / Empirical Frequency）は、概念的には1次元の周辺確率（`R2Prob` / `F2Prob`）であるにもかかわらず、2次元グリッドにブロードキャストして保持するレガシー構造のまま残っている。

この設計の非対称性は以下の弊害を生んでいる：
- ドキュメントで「ブロードキャスト」概念を別途説明する必要がある
- mr ↔ er、mf ↔ ef のペアが対称になっていない
- 冗長なデータ保持（同じ値が R × F 個複製される）
- API の一貫性低下（`predict(kind="er")` が f を要求するが実質無視するなど）

## 目的

`er` / `ef` を `mr` / `mf` と同じ1次元構造に揃え、経験的・最適化・1次元・2次元の4象限を対称な設計にする。

## ユーザーストーリー

### US-1: 1次元 dict としての利用
- **As a** データサイエンティスト
- **I want** `er_probability_dict_` に最新度（int）をキーとして経験的周辺確率を引きたい
- **So that** mr モデル（最適化版）と同じ感覚で利用できる

### US-2: 1次元 DataFrame としての利用
- **As a** データサイエンティスト
- **I want** `er_probability_` を `recency` / `probability` の2列だけの DataFrame として受け取りたい
- **So that** 冗長な frequency 列が消え、1次元データとして扱える

### US-3: 折れ線グラフでの可視化
- **As a** データサイエンティスト
- **I want** `plot_marginal_probability(axis="recency", kind="er")` で経験的周辺確率を折れ線表示したい
- **So that** 「経験的な傾向」を確認できる（現状の `plot_marginal_probability(kind="emp")` と同じ目的だが、er kind で明示的に呼べる）

### US-4: 3次元サーフェスは不要
- **As a** データサイエンティスト
- **I want** `plot_probability_surface(kind="er")` は不可（ValueError）で構わない
- **So that** mr/mf と同じく「1次元モデルはサーフェス不可、折れ線で表示」というルールが一貫する

## 受け入れ条件

### AC-1: 出力構造の変更
- `er_probability_` は `pd.DataFrame`、カラム: `recency`, `probability`
- `ef_probability_` は `pd.DataFrame`、カラム: `frequency`, `probability`
- `er_probability_dict_` は `dict[int, float]`、キーは recency
- `ef_probability_dict_` は `dict[int, float]`、キーは frequency
- `er_probability_table_` / `ef_probability_table_` 属性は**削除**

### AC-2: `predict()` の挙動
- `predict(r, f, kind="er")`: r のみ参照、f は無視（mr と同じ）
- `predict(r, f, kind="ef")`: f のみ参照、r は無視（mf と同じ）
- 上限を超えた場合は `recency_limit` / `frequency_limit` にクランプ（既存仕様継承）

### AC-3: `transform()` の挙動
- `transform(df, kind="er")`: recency 列だけを用いて経験的周辺確率を割り当て（mr と同じ実装）
- `transform(df, kind="ef")`: frequency 列だけを用いて経験的周辺確率を割り当て（mf と同じ実装）

### AC-4: `plot_probability_surface()` の挙動
- `plot_probability_surface(kind="er")`: `ValueError` を送出（mr/mf と同じ扱い）
  - エラーメッセージで `plot_marginal_probability()` を案内する

### AC-5: `plot_marginal_probability()` の挙動
- `plot_marginal_probability(axis="recency", kind="er")`: er の1次元データを折れ線表示
- `plot_marginal_probability(axis="frequency", kind="ef")`: ef の1次元データを折れ線表示
- `kind="all"` の場合の重ね表示動作も維持（emp + er + mr など）

### AC-6: `export_probability_csv()` の挙動
- `export_probability_csv(kind="er")`: カラム `recency, probability` で出力
- `export_probability_csv(kind="ef")`: カラム `frequency, probability` で出力
- `export_probability_csv(kind="all")`: マージロジックを更新（er は recency マージ、ef は frequency マージ。mr / mf と同じ）

### AC-7: 内部辞書の整合
- `R2Prob` / `F2Prob` は引き続き `_fit_impl()` で計算され、er/ef の値はこれらと一致する
- 一致テストを追加・更新する

### AC-8: テストの更新
- TestFitPeriodResult の er/ef 関連テスト（行数、カラム構成、ブロードキャスト検証など）を1次元前提に修正
- TestPredict の er/ef テストを 1次元 dict 参照に修正
- TestTransform に er/ef の1次元 merge 動作テストを追加
- TestPlotProbabilitySurface の er/ef テストを ValueError 検証に変更
- TestPlotMarginalProbability に er/ef テストを追加
- TestExportProbabilityCsv の er/ef カラム数を更新

### AC-9: ドキュメントの更新
- `docs/product-requirements.md`: 機能・出力テーブルから er/ef のブロードキャスト記述を削除、1次元構造として記述
- `docs/functional-design.md`: er/ef の数式定義を `$x_{r,f} = p_r$` 形式から `$p_r$` (1次元) に変更。属性表からも `_table_` 削除。データフロー図も更新
- `docs/glossary.md`: er/ef の属性記述を 1次元前提に変更。`_table_` エントリは削除（存在しないため）

### AC-10: 例の整合
- `examples/basic_usage.ipynb` を確認し、er/ef の利用箇所があれば 1次元前提に修正（現状では明示的に er/ef を使う場面は薄いはず）

## 制約事項

### 破壊的変更であること
- これは public API の破壊的変更を含む（属性のカラム構成変更、`_table_` 削除、dict キー型変更）
- 現在 `feature/time-col-unit-support` ブランチで既に複数の破壊的変更を含んでいるため、**次期メジャー（0.4.0 もしくは 1.0.0）リリースにまとめる**前提で進める

### 互換シムは設けない
- ユーザー数が少ない（CLAUDE.md および過去判断より）ため、deprecated alias などの互換シムは作らず、クリーンに置き換える

### 内部カラム名 `cv`, `N` 等は不変
- 経験的計算で用いる内部データ構造の列名は維持する（破壊的変更を最小化）

### `R2N`, `R2Prob` 等の内部 dict は維持
- `_fit_impl()` 内で集約用に使われている辞書はそのまま残す（er/ef の出力構造のみ変更）

## 完了条件

1. `uv run ruff check .` がパス
2. `uv run ruff format --check .` がパス
3. `uv run pytest` がパス（テスト件数は er/ef テスト追加・修正に伴い変動）
4. 3つの docs ファイル（product-requirements / functional-design / glossary）が更新済み
5. `examples/basic_usage.ipynb` の整合性が確認済み
