# Requirements: `fit_rolling()` ローリング集計による経験的商品選択確率の推定

## 背景

現在の `fit(df_obs, df_gt, ref=None, ...)` は、**単一の基準点**で経験的商品選択確率 $p_{r,f} = n_{r,f} / N_{r,f}$ を推定する。観測期間で $(r, f)$ だった (user, item) ペアを集計し、正解期間 `df_gt` で対象イベントが発生した回数を数えて確率を求める。つまり **1つの基準日に対する1スナップショット**での集計である。

単一スナップショット集計には次の課題がある。

1. **サンプル数が限られ、経験的確率がノイズを含みやすい**: 特に大きい $r$ や大きい $f$ のセルは観測サンプルが少なく、確率が不安定になる。
2. **特定基準日の偏り（曜日バイアス等）を受けやすい**: 1つの分割点のみに依存するため、その日固有の季節性・曜日性が確率に乗りやすい。

これに対し、分割点（基準日）を1日ずつ過去にずらしながら複数基準日で集計を積み増す「**ローリング集計**」を行えば、サンプル数を増やして経験的確率を安定化でき、基準日依存の偏りも平滑化できる。本タスクはこのローリング集計を行う新メソッド `fit_rolling()` を追加する。

### `df_obs` と `df_gt` は別イベントであり得る

重要な前提として、`df_obs`（観測：閲覧）と `df_gt`（正解：対象イベント）は**別種のイベント・別テーブルであり得る**。`df_gt` の対象イベントは再閲覧だけでなく、購買・CV なども指定できる仕様に拡張済みである。

現行 `fit()` は `df_gt` を「正解期間で対象イベントを起こした (user, item) の集合」として受け取るだけで、`df_gt[time_col]` は参照しない（呼び出し側が正解期間でフィルタ済みである前提）。

```python
# 現行 fit() 内（time_col を見ていない）
gt_log = df_gt[[self.user_col, self.item_col]].copy()
```

`fit_rolling()` では基準日 `td_k` ごとに正解窓で `df_gt` を切り出す必要があるため、**`df_gt` にも `time_col` が必須**になる（現行 `fit()` との差分）。また、観測ログと正解ログを**それぞれ別の期間で独立に時刻フィルタ**するため、単一 df を分割する `split_by_date()` はそのままでは利用できない（観測窓・正解窓の独立フィルタに置き換える）。

## 目的

1. `RecencyFrequencyScorer` に、`df_obs` / `df_gt` から複数基準日のローリング集計で経験的商品選択確率を推定する新メソッド `fit_rolling()` を追加する。
2. `df_gt` が購買・CV など `df_obs` と別イベントであっても正しく扱えるよう、観測ログ・正解ログを別々に受け取り、ロールごとに独立フィルタする。
3. `fit_rolling()` は既存 `fit()` と同じ属性（`emp_probability_` 等）を生成し、後続の `predict()` / `transform()` / `optimize()` / `plot_*()` をそのまま利用できるようにする。
4. データ最終日（`end_date`）の指定だけで、内部分割点・ローリング範囲を自動算出し、実務的に扱いやすい高レベル API を提供する。

## ユーザーストーリー

### US-1: サンプル増による経験的確率の安定化
- **As a** データサイエンティスト
- **I want** `fit_rolling(df_obs, df_gt, observation_days, gt_days, roll_days=30)` で複数基準日の集計を積み増したい
- **So that** サンプル数を増やし、ノイズの少ない経験的商品選択確率を得られる

### US-2: 購買・CV を正解イベントとするローリング
- **As a** リピート購買・CV をモデリングしたい分析者
- **I want** `df_obs`（閲覧ログ）と `df_gt`（購買・CV ログ）を別テーブルで渡し、ロールごとに観測窓・正解窓を独立に切り出したい
- **So that** 別イベント種別を正解としたローリング集計ができる

### US-3: データ最終日指定による直感的なセットアップ
- **As a** マーケティング分析者・ビジネスアナリスト
- **I want** 分割点を自分で計算せず「正解データの最終日（`end_date`）」だけ指定したい
- **So that** `gt_days` の引き算を意識せず、データ範囲の感覚そのままで学習できる

### US-4: 既存ワークフローの非破壊
- **As a** 既存利用者
- **I want** 現行 `fit(df_obs, df_gt)` をそのまま使い続けたい
- **So that** 既存コードを変更せずに新機能を選択的に導入できる

## 受け入れ条件

### AC-1: `fit_rolling()` メソッドの新規追加

`RecencyFrequencyScorer` に `fit_rolling()` を新規追加する。シグネチャは以下で確定する。

```python
def fit_rolling(
    self,
    df_obs,
    df_gt,
    observation_days,
    gt_days,
    roll_days=1,
    end_date=None,
    recency_limit=None,
    frequency_limit=None,
    time_col=None,
):
    ...
    return self
```

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `df_obs` | `pd.DataFrame` | — | 観測イベントの行動履歴（閲覧）。`time_col` 必須 |
| `df_gt` | `pd.DataFrame` | — | 正解イベントの履歴（再閲覧・購買・CV など）。**`time_col` 必須**（ロールごとに正解窓で切るため） |
| `observation_days` | `int` | — | 観測期間の長さ（必須）。各ロールで `df_obs` を切り出す窓幅 |
| `gt_days` | `int` | — | 正解期間の長さ（必須）。各ロールで `df_gt` を切り出す窓幅 |
| `roll_days` | `int` | `1` | ローリング回数。`1` で単一スナップショット。`N` で起点〜起点$-(N-1)$ の N 基準日を集計 |
| `end_date` | `str \| datetime \| int \| None` | `None` | 使用する正解データの最終日。`None` の場合は `df_gt[time_col]` の最大値を使用 |
| `recency_limit` | `int \| None` | `None` | 最大最新度。`None` の場合、**全ロール集計後のプール**から自動決定 |
| `frequency_limit` | `int \| None` | `None` | 最大頻度。`None` の場合、**全ロール集計後のプール**から自動決定 |
| `time_col` | `str \| None` | `None` | 時点カラム名。省略時は `__init__` の値。`df_obs`・`df_gt` 双方に適用 |

戻り値: `self`

### AC-2: 起点（anchor）の自動算出

正解データの最終日を `end_int` とし、最新ロールの分割点（anchor）を次式で算出する。

```
end_int = normalize_sequence_col(df_gt[time_col]).max()  if end_date is None
          else normalize_ref(end_date)
anchor  = end_int - gt_days
```

これにより、最新ロールの正解期間は `[anchor + 1, anchor + gt_days] = [end_int - gt_days + 1, end_int]` となり、**指定した正解データ最終日 `end_int` でちょうど終わる**。

### AC-3: 過去方向へのローリングと独立ウィンドウフィルタ

`k = 0, 1, …, roll_days-1` について分割点を `td_k = anchor - k`（過去方向）とし、各 `td_k` で観測ログ・正解ログを**独立に**時刻フィルタする。

- 観測窓: `df_obs` を `[td_k - observation_days + 1, td_k]` でフィルタ
- 正解窓: `df_gt` を `[td_k + 1, td_k + gt_days]` でフィルタ

観測窓の (user, item) から `ref = td_k` で最新度・頻度を算出し、正解窓に出現した (user, item) を cv=1 と判定する。`split_by_date()`（単一 df の分割）は使わず、各ログの期間フィルタで実装する。

### AC-4: 集計の積み増し（独立サンプル加算）

各ロールの `(user, item)` ペアは独立サンプルとして加算する。全ロール集計後に `N_{r,f}` と `CV_{r,f}` を合算し、`p_{r,f} = CV_{r,f} / N_{r,f}` を算出する。同一 `(user, item)` ペアが複数ロールに現れた場合、その分 `N` が増える（これがサンプル増による安定化の仕組み）。

`recency_limit` / `frequency_limit` の自動決定は、**全ロールを結合したプール**の累積 cv 分布に対して行う（個別ロールごとには行わない）。

### AC-5: 集計開始前の境界バリデーション（fail-fast）

集計（ロールごとのウィンドウフィルタ）に入る**前**に、全ロールが完全な観測窓を確保できるか検証する。過去側境界は観測ログ基準（`obs_min = df_obs[time_col] 正規化済み最小値`）とする。最古ロールの分割点 `anchor - (roll_days - 1)` の観測窓開始が `obs_min` を下回る場合は **`ValueError` を送出**して即停止する。

- 完全な観測窓を確保できる条件:
  `anchor - (roll_days - 1) - observation_days + 1 >= obs_min`
- 違反時のエラーメッセージには、**確保可能な最大 `roll_days`** を含める。
  最大 `roll_days = anchor - obs_min - observation_days + 2`
- このチェックはウィンドウフィルタ・確率集計より前に行い、途中まで集計してから失敗することがないようにする（fail-fast）。

### AC-6: `end_date` 明示時の範囲バリデーション

`end_date` がユーザーから明示指定された場合、最新ロールが成立することを集計開始前に検証する。

- `end_int > df_gt[time_col] の最大値`（最新ロールの正解期間が正解データを超える）の場合は **`ValueError`**。
- `anchor`（= `end_int - gt_days`）の観測窓が `obs_min` を下回るなど、最新ロール自体が成立しない場合も **`ValueError`**。

### AC-7: 既存 `fit()` 属性との互換

`fit_rolling()` 成功後、`fit()` と同一の属性が利用可能になること。

- `emp_probability_`, `emp_probability_table_`, `emp_probability_dict_`
- `er_probability_`, `er_probability_dict_`, `ef_probability_`, `ef_probability_dict_`
- `recency_limit`, `frequency_limit`
- 相関診断（`recency_corr_` 等）
- これらを生成する集計部ロジックは `fit()` と共有し、挙動の一貫性を保つ（重複実装しない）。

### AC-8: 診断統計の定義

`fit_rolling()` 後の診断統計を以下で定義する。

- `observation_end_` = `anchor`（最新ロールの分割点）
- `observation_start_` = 最古ロールの観測開始日（`anchor - (roll_days-1) - observation_days + 1`、`obs_min` でクランプ）
- `record_num_*` / `total_cv*` は**全ロール集計後の合算値**
- `show()` が `fit_rolling()` 後にもエラーなく動作すること

### AC-9: `roll_days=1` の等価性

`roll_days=1` のとき、`fit_rolling(df_obs, df_gt, observation_days, gt_days, end_date=D)` の結果が、`df_obs` を `[anchor - observation_days + 1, anchor]`、`df_gt` を `[anchor + 1, anchor + gt_days]` で手動フィルタして `fit(df_obs_f, df_gt_f, ref=anchor)` を呼んだ結果と一致すること（同一の `emp_probability_` を生成）。`anchor = D - gt_days`。

### AC-10: テストの追加

`tests/test_scorer.py`（または適切なテストファイル）に `fit_rolling()` のテストクラスを追加し、以下を検証する。

- `roll_days=1` で手動フィルタ + `fit` と結果が一致する（AC-9）
- `roll_days>1` で `N` が単調に増える（サンプル積み増しの確認）
- `df_gt` が `df_obs` と別イベント（別 user/item 集合）でも正しく cv 判定される
- `end_date=None` のとき `df_gt[time_col].max()` を最終日として採用する
- `end_date` 明示指定が正しく反映される
- 境界突破時に `ValueError`（メッセージに最大 `roll_days` を含む）（AC-5）
- `end_date > df_gt max` での `ValueError`（AC-6）
- 整数 time_col・日付 time_col の双方で動作する
- `time_col` 引数による列名上書きが効く
- `fit_rolling()` 後に `predict` / `transform` / `optimize` / `show` が動作する

### AC-11: ドキュメント更新

- `docs/functional-design.md`: `fit_rolling()` のメソッド仕様を追加。データフロー図に経路を追記。
- `docs/glossary.md`: API 簡潔版テーブルに `fit_rolling()` を追加。必要なら「ローリング集計」の用語を追加。
- `README.md` / `examples/`: 該当があれば追記（任意）。

## 制約事項

### 非破壊的変更であること

- `fit_rolling()` は**新規追加**であり、既存 `fit(df_obs, df_gt, ...)` のシグネチャ・挙動は一切変更しない。
- 既存テストはすべてそのままパスする。

### `df_gt` の time_col 必須化は fit_rolling 内に閉じる

- 現行 `fit()` は `df_gt` の `time_col` を要求しない。`fit_rolling()` のみロールごとの正解窓フィルタのために `df_gt[time_col]` を要求する。
- `df_gt` に `time_col` が無い場合は明確な `ValueError` を出す。

### ウィンドウフィルタは split_by_date と別実装

- `df_obs` と `df_gt` を別期間で独立にフィルタするため、単一 df を分割する `split_by_date()` はそのまま使わない。
- 時刻ウィンドウフィルタ（`[start, end]` の範囲抽出）を共有ヘルパーとして抽出し、`split_by_date()` からも再利用するか、`fit_rolling()` 内に閉じるかは `design.md` で確定する。

### 集計部ロジックの共有

- `fit()` の集計部（limit 自動決定・`N`/`CV` カウント・各属性生成、現行 `_fit_impl` の 319 行目以降相当）を `fit_rolling()` と共有する。
- 共有の具体的な分離方法（`_fit_impl` の2段分離、または集計部ヘルパーの抽出）は `design.md` で確定する。

### ローリング方向は過去のみ

- 未来方向（`anchor + k`）は正解期間を確保できないため行わない。ローリングは過去方向（`anchor - k`）のみ。

### 時刻正規化の一貫性

- `end_date` の正規化は `normalize_ref()`、`df[time_col]` の正規化は `normalize_sequence_col()` を用い、既存 `fit()` / `split_by_date()` と同一の正規化を使う。
- `observation_days` / `gt_days` の単位は time_col の型に依存するタイムステップ（日付型なら日、整数型なら整数ステップ）であり、`unit`（最新度の粒度）とは独立。

## 完了条件

1. `uv run ruff check .` がパス
2. `uv run ruff format --check .` がパス
3. `uv run pytest` が全テスト green（既存テスト＋新規 `fit_rolling` テスト）
4. `RecencyFrequencyScorer.fit_rolling()` が AC-1 のシグネチャで利用可能
5. `df_gt` が `df_obs` と別イベントでも正しく集計される
6. `roll_days=1` で手動フィルタ + `fit` と結果一致（AC-9）
7. 境界バリデーションが集計開始前に fail-fast で動作（AC-5 / AC-6）
8. `fit_rolling()` 後に `predict` / `transform` / `optimize` / `show` が動作
9. `docs/functional-design.md` / `docs/glossary.md` が `fit_rolling()` に整合
