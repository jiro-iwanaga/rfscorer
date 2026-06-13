# Requirements: `fit()` のみ公開 + `split_by_date()` ユーティリティ提供

## 背景

`RecencyFrequencyScorer` は現在3つの fit メソッドを公開している：

| メソッド | 役割 |
|---------|------|
| `fit(df_obs, df_eval, ref=None, ...)` | scikit-learn スタイルの主要 fit |
| `fit_date(df, target_date, observation_days=28, evaluation_days=7, ...)` | 単一 DataFrame と分割日から自動分割 |
| `fit_period(df, observation_period, evaluation_period, ...)` | 観測・評価期間を tuple で明示指定 |

加えて transform にも同様の対称メソッドが存在：

| メソッド | 役割 |
|---------|------|
| `transform(df, ref=None, kind="emp", ...)` | scikit-learn スタイルの主要 transform |
| `transform_date(df, target_date, kind="emp", ...)` | `target_date` 以前の行にフィルタしてから transform |

これらの「`_date` / `_period` 派生メソッド」は data preparation を Scorer 内部に取り込んだ便宜的なラッパーである。しかし以下の問題がある：

1. **API surface が大きく、初学者が「どれを使うべきか」迷う**
2. **データ準備とモデル fit の関心の分離が壊れる**: scikit-learn の標準は「データ準備は外、モデルは fit のみ」
3. **将来の ローリング workflow（曜日バイアス除去等の研究的用途）に対応できない**:
   - `for target_date in date_range: ...` で target_date を1日ずつずらしながら経験値を蓄積し、集約後に `RecencyFrequencyOptimizer` に投入するワークフローを将来想定
   - この場合 `RecencyFrequencyScorer` を経由しないため、`fit_date()` ではなく**データ分割のユーティリティ関数**が必要

`RecencyFrequencyOptimizer` を公開 API へ昇格する方針と整合性を取り、本パッケージを「**高レベル API（簡易）× 低レベル部品（柔軟）の2層構造のライブラリ**」として整理する。

## 目的

1. `RecencyFrequencyScorer` から `fit_date()` / `fit_period()` / `transform_date()` を削除し、公開 API を `fit()` / `transform()` などの主要メソッドに絞る
2. データ分割の汎用ユーティリティ `split_by_date()` を新規追加し、公開 API として提供する
3. 上記により、scikit-learn 風の「データ準備は外、モデルは fit のみ」原則と、研究的 workflow への拡張性の両立を実現する

## ユーザーストーリー

### US-1: scikit-learn 風の標準ワークフロー
- **As a** データサイエンティスト・ML エンジニア
- **I want** `fit(df_obs, df_eval)` で経験値を推定したい
- **So that** sklearn 互換の感覚で本パッケージを使える

### US-2: 分割日からのワンライナーセットアップ
- **As a** マーケティング分析者・ビジネスアナリスト
- **I want** `split_by_date(df, target_date)` で観測・評価ログに分割してから `fit()` に渡したい
- **So that** 「分割日基準のワークフロー」を最小限のコードで実現できる

### US-3: 研究的ローリング workflow
- **As a** 研究者
- **I want** `for target_date in date_range: split_by_date(df, target_date, ...)` のループで複数 target_date での経験値を集約したい
- **So that** 曜日バイアス除去等の独自集約を組み合わせて `RecencyFrequencyOptimizer` に投入できる

### US-4: 期間明示の独自フィルタ
- **As a** 任意の期間制御を必要とするユーザー
- **I want** 標準 pandas フィルタで `df[mask_obs]` / `df[mask_eval]` を作って `fit()` に渡したい
- **So that** `fit_period()` を介さずに完全な柔軟性を得られる

## 受け入れ条件

### AC-1: `fit_date()` を完全削除

`RecencyFrequencyScorer.fit_date()` メソッドを削除する。互換シム（プロパティ alias 等）は設けない。

### AC-2: `fit_period()` を完全削除

`RecencyFrequencyScorer.fit_period()` メソッドを削除する。互換シム不要。

### AC-3: `transform_date()` を完全削除

`RecencyFrequencyScorer.transform_date()` メソッドを削除する。互換シム不要。
ユーザーは `df_filtered = df[df["date"] <= target_date]` でフィルタしてから `scorer.transform(df_filtered, ref=target_date)` を呼ぶ。

### AC-4: `split_by_date()` 関数の新規追加

`rfscorer.utils.split_by_date()` 関数を新規追加する。シグネチャ：

```python
def split_by_date(
    df: pd.DataFrame,
    target_date,
    observation_days: int | None = 28,
    evaluation_days: int | None = 7,
    time_col: str = "datetime",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """target_date を基準に df を観測ログと評価ログに分割する。

    Returns
    -------
    df_obs, df_eval : tuple[pd.DataFrame, pd.DataFrame]
        観測ログと評価ログ。両方とも元の df のサブセット。
    """
```

挙動：
- 観測期間: `max(df の time_col 最小値, target_date - observation_days 時間単位)` 〜 `target_date`（含む）
- 評価期間: `target_date の翌時点` 〜 `min(df の time_col 最大値, target_date + evaluation_days 時間単位)`
- `observation_days=None` の場合は df の先頭から
- `evaluation_days=None` の場合は df の末尾まで
- `target_date` は日付（`str`・`datetime`）・整数のいずれも受け付ける
- 内部で datetime↔ordinal の正規化を行い、整数列にも対応

### AC-5: 公開エクスポート

`split_by_date()` は `from rfscorer import split_by_date` でインポートできるよう `__init__.py` から公開する。
（`RecencyFrequencyOptimizer` の公開は別タスクのため、本タスクの対象外）

### AC-6: 既存テストの整理

- `test_scorer.py` 内の `TestFitDateValidation` / `TestFitDateResult` / `TestFitPeriodValidation` / `TestFitPeriodResult` / `TestTransformDate` を削除
- これらに含まれていた検証は、必要に応じて `test_scorer.py::TestFitResult` や新規の `tests/test_utils.py::TestSplitByDate` で代替する

### AC-7: `split_by_date()` の新規テスト追加

`tests/test_utils.py` を新規作成し、以下を検証：
- 日付入力での分割（observation_days / evaluation_days デフォルト）
- 日付入力での分割（observation_days / evaluation_days 明示指定）
- `observation_days=None` / `evaluation_days=None` での「全範囲」挙動
- 整数列入力での分割
- 不正な `target_date` 型での `ValueError`
- 必須カラム（`time_col`）欠落での `ValueError`
- 戻り値が tuple[DataFrame, DataFrame] であること
- 元の df を mutate しないこと

### AC-8: ドキュメントの更新

以下のファイルで `fit_date` / `fit_period` / `transform_date` への言及を削除し、`split_by_date` の使い方を追記する：

- `docs/product-requirements.md`: 入力セクション、機能テーブル、コード例、glossary 連携
- `docs/functional-design.md`: メソッド仕様、データフロー図
- `docs/glossary.md`: API エントリ
- `README.md`: 該当があれば
- `examples/basic_usage.ipynb`: 該当があれば（現状 `fit_date()` は使われていないため修正不要の見込み）

### AC-9: scorer.py docstring の整合性

`fit()` 等の主要メソッドの docstring から「他の fit_X() については fit_date(), fit_period() を参照」等の旧言及を削除。

### AC-10: 既存テストの全パス

リファクタの結果、`fit_date` / `fit_period` / `transform_date` 関連テストは削除されるため、テスト件数は減少する。残った全テストと新規追加テスト全パスが完了条件。

## 制約事項

### 破壊的変更であること

- 公開メソッド3つ（`fit_date`, `fit_period`, `transform_date`）の削除は明確な API 破壊
- 互換シムは設けない（過去判断方針に準拠）
- 現ブランチに集約中の他破壊的変更と共に次期メジャーリリースへ

### 内部の `_fit_impl()` は維持

`fit()` の内部実装ヘルパーであり、削除対象ではない。

### `split_by_date()` の位置づけ

- 単純な pandas フィルタの汎化ユーティリティ
- 統計的判断やモデルロジックは含まない
- 将来 `split_by_value()` のような兄弟関数を追加する余地を残すため `utils.py` 内に配置

### scorer.py 内部の時刻正規化ロジックとの関係

`_normalize_ref()` / `_normalize_sequence_col()` を `split_by_date()` から再利用する必要がある。重複を避けるため、これらを共有モジュール（例: `rfscorer/_time_utils.py`）へ抽出するか、`RecencyFrequencyScorer` の static / classmethod として公開するかは design.md で確定する。

### `RecencyFrequencyOptimizer` の `__init__.py` 公開は別タスク

本タスクは `split_by_date` の公開のみを対象とする。`RecencyFrequencyOptimizer` の export 追加は将来の別ステアリングで扱う。

## 完了条件

1. `uv run ruff check .` がパス
2. `uv run ruff format --check .` がパス
3. `uv run pytest` が全テスト green
4. `RecencyFrequencyScorer` から `fit_date` / `fit_period` / `transform_date` が消えている
5. `from rfscorer import split_by_date` が成功する
6. `tests/test_utils.py` が新規作成され、`TestSplitByDate` の全 case パス
7. 4つの docs（product-requirements / functional-design / glossary / README）が新方針に整合
