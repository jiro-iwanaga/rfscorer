# Requirements: コンストラクタの軸対称化（`unit` → `recency_unit`）

## 背景

`RecencyFrequencyScorer` のコンストラクタは現在以下である。

```python
RecencyFrequencyScorer(user_col="user", item_col="item", time_col="datetime", unit=1, recency_mode="day")
```

`unit` は recency 軸のビン幅だが、名前から軸が読み取れない。また将来 frequency 軸のビン化（`frequency_unit`）や算出方式（`frequency_mode`）を追加する際、`recency_mode`/`recency_unit` と `frequency_mode`/`frequency_unit` が**軸対称**に並ぶ設計が自然である。本タスクでは recency 軸を対称形に揃える。

```python
# 本タスクの到達点
RecencyFrequencyScorer(user_col="user", item_col="item", time_col="datetime", recency_mode="day", recency_unit=1)

# 将来の到達点（本タスクでは実装しない）
RecencyFrequencyScorer(user_col="user", item_col="item", time_col="datetime",
                       recency_mode="day", recency_unit=1,
                       frequency_mode="view", frequency_unit=1)
```

## 決定事項（ユーザー確認済み）

1. **破壊的変更を許容する**: `unit` を利用しているユーザーは皆無（チュートリアルにも登場しない）と判断し、非推奨エイリアスを設けず**ハード改名**する。
2. **`frequency_mode` の既定は `"view"`**（＝現状の view freq = 閲覧イベント数）。本タスクでは `frequency_mode`/`frequency_unit` は**追加しない**が、将来の既定として記録する。
3. **`"day"`/`"view"` 語彙を踏襲する**（軸固有名にはしない）。既定が非対称（recency=`"day"` / frequency=`"view"`）になることを docstring・docs に明記する。
4. **キーワード専用化はしない**（必要なら将来）。
5. **`recency_unit` の view ランクへの適用は将来検討**（本タスクでは現状どおり view モードで `recency_unit` を無視）。

## 目的

1. コンストラクタを `(user_col, item_col, time_col, recency_mode="day", recency_unit=1)` に変更する（`unit`→`recency_unit` 改名、`recency_mode` を `recency_unit` の前に配置）。
2. 内部・ドキュメント・テストの `unit` 参照を `recency_unit` に統一する。
3. 将来の `frequency_mode`/`frequency_unit` 追加が軸対称に行えるよう、設計方針とメモを整える。

## ユーザーストーリー

### US-1: 軸が明確なビン幅指定
- **As a** 利用者
- **I want** `RecencyFrequencyScorer(recency_mode="day", recency_unit=7)` のように軸を明示してビン幅を指定したい
- **So that** どの軸の粒度かが一目で分かり、将来 `frequency_unit` と並べても自然である

## 受け入れ条件

### AC-1: コンストラクタ署名の変更

```python
def __init__(self, user_col="user", item_col="item", time_col="datetime",
             recency_mode="day", recency_unit=1):
```

- `unit` を削除し `recency_unit` を追加する（非推奨エイリアスは設けない）。
- 並び順は `... time_col, recency_mode, recency_unit`（`recency_mode` を前に）。
- `recency_unit <= 0` で `ValueError`（メッセージは `recency_unit must be a positive integer, got {recency_unit}.`）。
- `recency_unit` は `recency_mode="view"` 時は recency 計算に使用しない（現状の `unit` と同じ扱い）。

### AC-2: 内部参照の置換

- `self.unit` → `self.recency_unit`。
- `_recency.build_day_rf(..., unit)` の引数名を `recency_unit` に改名し、`_build_ui_rf_df` ディスパッチャは `self.recency_unit` を渡す。
- `save_zip()` の metadata キー `"unit"` → `"recency_unit"`。

### AC-3: docstring の更新

- `__init__` の `unit` パラメータ説明を `recency_unit` に書き換え（recency 軸ビン幅・view 時は無視）。
- `fit()` / `transform()` / `_build_ui_rf_df` / `_recency.py` 内の `unit` 言及を `recency_unit` に統一。

### AC-4: 既存テストの更新と非回帰

- `tests/test_scorer.py` の `unit=` キーワード使用を `recency_unit=` に置換（エラーテストの期待メッセージ・`match` を含む）。
- `tests/test_recency.py` の `build_day_rf` 呼び出し（位置引数）は引数名変更の影響を受けないが、必要に応じてコメントを整える。
- 改名後、`uv run pytest` が全 green。

### AC-5: ドキュメント更新

- `docs/functional-design.md`: コンストラクタ表・`recency_mode` 補足・metadata 一覧・recency 式中の `unit` を `recency_unit` に。
- `docs/glossary.md`: day recency の式・API 表の `unit` を `recency_unit` に。
- `docs/product-requirements.md`: コンストラクタ入力の `unit` を `recency_unit` に。
- 将来メモを更新: `frequency_mode` 既定=`"view"`・既定非対称・`frequency_unit` は将来・`view`/`day` 語彙踏襲。

### AC-6: 将来拡張の整合

- `frequency_mode`/`frequency_unit` は**本タスクでは追加しない**が、追加時に軸対称で載るよう設計メモを残す。
- 既定非対称（`recency_mode="day"` / `frequency_mode="view"`）の根拠（古典的 RF: recency は時間ベース・frequency は回数ベース）を明記する。

## 制約事項

- 破壊的変更を許容するが、変更は recency 軸のみに限定する（frequency 系は将来）。
- `recency_mode="day"`（既定）かつ `recency_unit=1`（既定）で、現行 `unit=1` と数値的に完全一致すること。
- 公開 API（`from rfscorer import RecencyFrequencyScorer`）の import 経路は変更しない。

## 完了条件

1. `uv run ruff check .` がパス
2. `uv run ruff format --check .` がパス
3. `uv run pytest` が全テスト green
4. コンストラクタが `(user_col, item_col, time_col, recency_mode="day", recency_unit=1)` で利用可能
5. `recency_mode="day"` / `recency_unit=1` で現行挙動と数値完全一致
6. `unit` を渡すと `TypeError`（パラメータが存在しない）になる
7. docs（functional-design / glossary / product-requirements）と将来メモが `recency_unit` に整合
