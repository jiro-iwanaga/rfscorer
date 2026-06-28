# Requirements: view_recency の追加

## 背景

現在の `recency` は、最終閲覧日から基準日（`ref`）までの経過日数をビン分割した **day recency** である。

```
recency = (ref_ordinal - last_view_ordinal) // unit + 1
```

この設計では、閲覧の「日付の差」を重視しているが、「ユーザーが直近に見た順番」という観点においては、日付の絶対差よりも単純な閲覧順位の方が自然な指標となるケースがある。本タスクでは、閲覧順位ベースの新指標 **view recency** を追加する。

また、将来的に `recency_mode="session"`（セッション単位の閲覧順位）や `frequency_mode="day"`（日付粒度の閲覧頻度）の追加が見込まれる。本タスクでは view recency の実装に加え、これらの将来拡張を想定した設計方針を定める。

### 用語整理

| 用語 | 定義 | mode |
|------|------|------|
| **day recency**（現状） | `(ref_ordinal - last_view_ordinal) // unit + 1`（経過日数ビン） | `recency_mode="day"` |
| **view recency**（新規） | ユーザー内で最終閲覧 timestamp が新しい順に振る 1 起算の連番（1 = 最新） | `recency_mode="view"` |
| **session recency**（将来） | ユーザー内でセッション単位の閲覧順位（1 = 最新セッション） | `recency_mode="session"` |
| **view freq**（現状の freq） | 観測期間内の閲覧イベント数 | `frequency_mode="view"`（将来のデフォルト） |
| **day freq**（将来） | 観測期間内に閲覧した日数 | `frequency_mode="day"` |

### 将来拡張の想定

**`recency_mode="session"`**（本タスクでは実装しない）
- 連続する閲覧イベントをセッションとしてまとめ、セッション単位で閲覧順位を定義する
- セッション境界の定義（例: 一定時間の無操作）は追加時に確定する

**`frequency_mode="day"`**（本タスクでは実装しない）
- 現状の `frequency` は閲覧イベント数（view freq）
- day freq は閲覧した日数（ユニーク日付数）を frequency として使う指標
- 本タスクで `recency_mode` の設計パターンを確立し、`frequency_mode` 追加時も同じパターンで実装できるようにする

## 目的

1. `RecencyFrequencyScorer` に `recency_mode` パラメータを追加し、day recency / view recency を切り替えられるようにする。
2. 将来の `recency_mode="session"` および `frequency_mode` 追加を見越した拡張しやすい設計とする。
3. `fit()` のシグネチャは変更しない。
4. 既存コードへの影響を最小限に抑え、非破壊的な変更とする。

## ユーザーストーリー

### US-1: view recency での学習
- **As a** データサイエンティスト
- **I want** `RecencyFrequencyScorer(recency_mode="view")` で view recency を使って `fit()` したい
- **So that** 日付差に依存しない閲覧順位ベースの商品選択確率を推定できる

### US-2: 既存ワークフローの非破壊
- **As a** 既存利用者
- **I want** デフォルト（`recency_mode="day"`）で現行と全く同じ挙動を保ちたい
- **So that** 既存コードを変更せずに新機能を選択的に導入できる

## view recency の定義

- 各 (user, item) の代表時刻 = 観測期間内の**最終閲覧 timestamp**
- ユーザーごとに代表時刻の**降順**でランク付けし、**1 起算のユニークな連番**を付与
- 同一商品が複数回閲覧された場合は最新の閲覧時刻を採用

## 例

あるユーザーの行動履歴:

| timestamp | 商品 |
|-----------|------|
| 2026-06-28 17:00 | A |
| 2026-06-28 16:50 | B |
| 2026-06-28 16:40 | A（重複） |
| 2025-06-27 18:00 | C |

各商品の最終閲覧時刻:
- 商品 A → 2026-06-28 17:00
- 商品 B → 2026-06-28 16:50
- 商品 C → 2025-06-27 18:00

view recency:
- `view_recency = 1` → 商品 A
- `view_recency = 2` → 商品 B
- `view_recency = 3` → 商品 C

## 受け入れ条件

### AC-1: `recency_mode` パラメータの追加

`RecencyFrequencyScorer.__init__()` に `recency_mode: str = "day"` を追加する。

- `"day"`: 現行の day recency（デフォルト、既存挙動を維持）
- `"view"`: view recency（閲覧順位ベース）
- 未知の値が渡された場合は `ValueError` を送出する

型アノテーションは `str` とし、将来の mode 追加時に型定義の変更が不要になるようにする。
docstring に現在サポートする値（`"day"`, `"view"`）を明記し、将来追加予定の値（`"session"` 等）は「予約済み」として記載しない（追加時に記載する）。

### AC-2: `fit()` シグネチャの維持

`fit()` のシグネチャは変更しない。`recency_mode="view"` 時は `ref` および `unit` を無視する。

### AC-3: view recency の計算

`recency_mode="view"` 時、各 (user, item) の recency を次のルールで計算する。

1. `(user, item)` ごとに最終閲覧 timestamp を集計
2. ユーザーごとに以下の優先順位で並べ、1 起算の連番（重複なし）を付与する
   - **第1キー**: 最終閲覧 timestamp の降順（新しいほど小さい recency）
   - **第2キー（タイブレーク）**: 入力データ上の初出行番号の昇順（先に出てきた商品の方が小さい recency）

上記の「例」と一致する結果が得られること。

#### 時刻解像度の契約（重要）

view recency は **timestamp の完全な解像度**で順位付けする（同一日内の時・分・秒も区別する）。

- 現状の内部表現（`normalize_sequence_col`）は datetime を**日単位の序数に切り捨てる**ため、そのままでは同一日内の時刻順を区別できない。view recency では別途、時刻を保持した高解像度キーで順位付けする（詳細は design 参照）。
- **datetime / 文字列日時**: ナノ秒解像度まで保持して順位付けする。
- **整数 time_col**: 値をそのまま順位キーに用いる（ユーザーが秒・連番などの解像度で与えた粒度に従う）。
- 「第1キーが同値」とは、この高解像度キーが**完全一致**することを指す（例: 全く同一の timestamp）。同値時のみ第2キー（初出行番号）でタイブレークする。
- **day recency（`recency_mode="day"`）の挙動は一切変更しない**。高解像度キーは view モードでのみ生成・使用する。

### AC-4: `unit` の扱い

`recency_mode="view"` 時、`unit` は recency 計算に使用しない。

### AC-5: 拡張しやすい内部設計

将来の `recency_mode="session"` および `frequency_mode` 追加を見越し、以下を満たす設計とする。

- recency の計算ロジックと frequency の計算ロジックを**明確に分離**する
- `recency_mode` の分岐は 1 箇所に集約し、新 mode 追加時に変更箇所が最小になるようにする
- 将来 `frequency_mode` パラメータを追加する際、`recency_mode` と同じパターンで追加できること
  （本タスクでは `frequency_mode` パラメータは追加しない）

### AC-6: テストの追加

- `recency_mode="view"` で下記テスト例と一致する recency が得られること
- `recency_mode="day"` でデフォルト挙動が変わらないこと
- `recency_mode="view"` で `fit()` / `predict()` / `transform()` が動作すること
- 未知の `recency_mode` で `ValueError` が送出されること

#### テスト例 1: 基本（同一商品の重複閲覧と順位付け）

あるユーザーの行動履歴（入力順）:

| 行番号 | timestamp | 商品 |
|--------|-----------|------|
| 0 | 2026-06-28 17:00 | A |
| 1 | 2026-06-28 16:50 | B |
| 2 | 2026-06-28 16:40 | A（重複） |
| 3 | 2025-06-27 18:00 | C |

期待する結果（各商品の最終閲覧 timestamp で順位付け）:

| 商品 | 最終閲覧 | recency |
|------|----------|---------|
| A | 2026-06-28 17:00 | 1 |
| B | 2026-06-28 16:50 | 2 |
| C | 2025-06-27 18:00 | 3 |

#### テスト例 2: タイブレーク（最終閲覧 timestamp が同一）

あるユーザーの行動履歴（入力順）:

| 行番号 | timestamp | 商品 |
|--------|-----------|------|
| 0 | 2026-06-28 17:00 | A（先に出現） |
| 1 | 2026-06-28 17:00 | B（後に出現、同じ timestamp） |
| 2 | 2026-06-28 16:00 | C |

期待する結果（timestamp 同値は初出行番号が小さい方を優先）:

| 商品 | 最終閲覧 | 初出行番号 | recency |
|------|----------|-----------|---------|
| A | 2026-06-28 17:00 | 0 | 1 |
| B | 2026-06-28 17:00 | 1 | 2 |
| C | 2026-06-28 16:00 | 2 | 3 |

#### テスト例 3: 複数ユーザー（ランクはユーザーごとに独立）

行動履歴（入力順）:

| 行番号 | user | timestamp | 商品 |
|--------|------|-----------|------|
| 0 | U1 | 2026-06-28 17:00 | P |
| 1 | U1 | 2026-06-28 16:00 | Q |
| 2 | U2 | 2026-06-28 16:00 | P |
| 3 | U2 | 2026-06-28 17:00 | Q |

期待する結果（各ユーザーで 1 から振り直し）:

| user | 商品 | recency |
|------|------|---------|
| U1 | P | 1 |
| U1 | Q | 2 |
| U2 | Q | 1 |
| U2 | P | 2 |

### AC-7: ドキュメント更新

- `RecencyFrequencyScorer.__init__()` の docstring に `recency_mode` パラメータを追記
- `docs/functional-design.md` の該当箇所を更新

## 制約事項

- 非破壊的変更：既存テストがすべてパスすること
- `fit()` のシグネチャは変更しない
- `recency_mode="day"` がデフォルトであり、デフォルト時は現行と完全に同一の挙動を保つ
- `frequency_mode` は本タスクでは実装しない（設計方針の整理のみ）
- `recency_mode="session"` は本タスクでは実装しない（将来の拡張ポイントとして設計に反映する）

## 完了条件

1. `uv run ruff check .` がパス
2. `uv run ruff format --check .` がパス
3. `uv run pytest` が全テスト green（既存テスト＋新規テスト）
4. `recency_mode="view"` で view recency が正しく計算される
5. `recency_mode="day"`（デフォルト）で既存挙動が変わらない
6. `fit()` / `predict()` / `transform()` が `recency_mode="view"` でも動作する
7. 未知の `recency_mode` で `ValueError` が送出される
