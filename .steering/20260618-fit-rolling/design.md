# Design: `fit_rolling()` ローリング集計

本書は `requirements.md` の受け入れ条件を満たす実装設計を定義する。

## 1. 全体方針

`fit()` の内部処理を **(A) ロール単位の (user,item)→(recency,frequency,cv) 生成** と **(B) 集計部（limit 決定・N/CV カウント・各属性生成・相関診断）** の2段に分離し、`fit_rolling()` と共有する。`fit_rolling()` は (A) を `roll_days` 回まわして結合し、(B) を1回呼ぶ。これにより `fit()` / `fit_rolling()` の集計挙動の一貫性を保証する（AC-7）。

```
fit()         : Part A ×1  → Part B
fit_rolling() : Part A ×roll_days → concat → Part B
```

## 2. 現行コードの分解と再利用

`scorer.py` の現行 `_fit_impl(obs_log, gt_log, ref_int, recency_limit, frequency_limit)`（252–416 行）を以下に分解する。

| 区分 | 現行行 | 内容 | 抽出後 |
|------|--------|------|--------|
| stats | 254–255 | `record_num_obs` / `record_num_gt` 設定 | `_fit_impl` に残す（fit 用） |
| **Part A** | 257–264 | `UIcv` 構築・`_build_ui_rf_df`・cv 付与 → `df_ui2frc` | 新ヘルパー `_build_ui_rf_cv()` に抽出 |
| **Part B** | 265–416 | limit 決定・`_RF2N`/`_RF2CV` カウント・`emp/er/ef` 各属性・相関診断 | 新ヘルパー `_aggregate_empirical()` に抽出 |

### 2.1 新ヘルパー `_build_ui_rf_cv(obs_log, gt_log, ref_int)`

内部列（`_USER_COL`, `_ITEM_COL`, `_SEQUENCE_COL`）を持つ観測ログ・正解ログと `ref_int` を受け取り、`df_ui2frc`（列: `user`, `item`, `recency`, `frequency`, `cv`）を返す。現行 257–264 行をそのまま移設。

```python
def _build_ui_rf_cv(self, obs_log, gt_log, ref_int):
    UIcv = {(row.user, row.item) for row in gt_log.itertuples()}
    df = self._build_ui_rf_df(obs_log, ref_int)
    df["cv"] = (
        pd.MultiIndex.from_frame(df[[self._USER_COL, self._ITEM_COL]])
        .isin(UIcv)
        .astype(int)
    )
    return df
```

### 2.2 新ヘルパー `_aggregate_empirical(df_ui2frc, recency_limit, frequency_limit)`

結合済み `df_ui2frc`（列に `recency`, `frequency`, `cv` を含む）を受け取り、現行 265–416 行の処理を実行して全属性を設定する。`obs_log` / `gt_log` / `ref_int` には依存しない（確認済み）。`record_num_target_org = len(df_ui2frc)` から始まる。

### 2.3 `_fit_impl` の改修後

```python
def _fit_impl(self, obs_log, gt_log, ref_int, recency_limit, frequency_limit):
    self.record_num_obs = len(obs_log)
    self.record_num_gt = len(gt_log)
    df_ui2frc = self._build_ui_rf_cv(obs_log, gt_log, ref_int)
    self._aggregate_empirical(df_ui2frc, recency_limit, frequency_limit)
```

`fit()` 本体（146–239 行）は無変更。これにより既存挙動・既存テストは完全保存される（AC-非破壊）。

## 3. `_to_internal()` の time_col 引数追加

現行 `_to_internal(self, df)`（241–250 行）は `self.time_col` 固定。`fit_rolling()` の `time_col` 上書きと、`df_gt` の内部化に再利用するため、任意引数を追加する（非破壊）。

```python
def _to_internal(self, df, time_col=None):
    tc = time_col if time_col is not None else self.time_col
    result = df[[self.user_col, self.item_col, tc]].copy()
    result.columns = [self._USER_COL, self._ITEM_COL, self._SEQUENCE_COL]
    # 以降は現行どおり（str キャスト・normalize_sequence_col）
    ...
```

`df_gt` は従来 user/item のみ要求だったが、`fit_rolling()` では `time_col` も必要なため、`_to_internal()` を `df_gt` にも適用して内部化（正規化済み整数の時刻列を得る）。

## 4. `fit_rolling()` の実装

### 4.1 シグネチャ

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
    return self
```

### 4.2 入力検証

1. `df_obs` / `df_gt` が `pd.DataFrame`（でなければ `TypeError`）。
2. `tc = time_col if time_col is not None else self.time_col`。
3. `df_obs` に `[user_col, item_col, tc]`、`df_gt` に `[user_col, item_col, tc]` が存在（欠落で `ValueError`）。**`df_gt` の `tc` 欠落もここで検出**（AC: df_gt time_col 必須）。
4. `observation_days >= 1`, `gt_days >= 1`, `roll_days >= 1`（でなければ `ValueError`）。

### 4.3 内部化と境界量の算出

```python
obs_internal = self._to_internal(df_obs, time_col=tc)
gt_internal  = self._to_internal(df_gt,  time_col=tc)

obs_seq = obs_internal[self._SEQUENCE_COL]
gt_seq  = gt_internal[self._SEQUENCE_COL]
obs_min = int(obs_seq.min())
gt_max  = int(gt_seq.max())

end_int = gt_max if end_date is None else normalize_ref(end_date)
anchor  = end_int - gt_days
```

### 4.4 集計開始前バリデーション（fail-fast / AC-5・AC-6）

`split_by_date`・ウィンドウフィルタ・集計に入る前に実施する。

```python
# AC-6: end_date 明示時の範囲チェック
if end_date is not None and end_int > gt_max:
    raise ValueError(
        f"end_date ({end_int}) exceeds the latest ground-truth date "
        f"({gt_max}); the most recent ground-truth window would extend "
        f"beyond available df_gt data."
    )

# AC-5: 最古ロールの観測窓が obs データに収まるか
oldest_obs_start = anchor - (roll_days - 1) - observation_days + 1
if oldest_obs_start < obs_min:
    max_roll_days = anchor - obs_min - observation_days + 2
    if max_roll_days < 1:
        raise ValueError(
            f"Data range is too short for observation_days={observation_days} "
            f"and gt_days={gt_days}: even roll_days=1 cannot secure a full "
            f"observation window (anchor={anchor}, obs_min={obs_min})."
        )
    raise ValueError(
        f"roll_days={roll_days} exceeds available observation history. "
        f"Maximum feasible roll_days is {max_roll_days} "
        f"(anchor={anchor}, obs_min={obs_min}, observation_days={observation_days})."
    )
```

> `anchor` 自体の観測窓不足（`roll_days=1` でも成立しない）は `oldest_obs_start`（k=0 を含む）の判定に内包される。

### 4.5 ローリング集計（過去方向）

```python
frames = []
total_obs_rows = 0
total_gt_rows = 0
for k in range(roll_days):
    td = anchor - k
    obs_w = obs_internal[
        (obs_seq >= td - observation_days + 1) & (obs_seq <= td)
    ]
    gt_w = gt_internal[
        (gt_seq >= td + 1) & (gt_seq <= td + gt_days)
    ]
    total_obs_rows += len(obs_w)
    total_gt_rows += len(gt_w)
    # cv 付与後 user/item は集計に不要。3 列に絞って concat メモリを抑える
    frames.append(self._build_ui_rf_cv(obs_w, gt_w, td)[["recency", "frequency", "cv"]])

combined = pd.concat(frames, ignore_index=True)
```

`_build_ui_rf_df` は `ref_int=td` で `recency=(td - seq)//unit + 1` を計算。窓により `td - observation_days + 1 <= seq <= td` のため `1 <= recency` が保証される。

> **集計部の入力契約**: `_aggregate_empirical()` は `recency` / `frequency` / `cv` 列のみ参照する（`record_num_target_org = len(df_ui2frc)` を含め user/item には非依存）。よって `fit()` 経路はフル列の `df_ui2frc` を、`fit_rolling()` 経路は 3 列に絞った `combined` を渡せ、どちらも同一の集計部で処理できる。

### 4.6 stats 設定と集計部呼び出し

```python
self.record_num_obs = total_obs_rows
self.record_num_gt = total_gt_rows
self.record_num = total_obs_rows + total_gt_rows
self.observation_end_ = anchor
self.observation_start_ = max(obs_min, oldest_obs_start)

self._aggregate_empirical(combined, recency_limit, frequency_limit)
return self
```

`record_num_target_org` / `record_num_target` / `total_cv_org` / `total_cv` は `_aggregate_empirical` 内で `combined` から算出され、全ロール合算値となる（AC-8）。

## 5. 等価性（AC-9）

`roll_days=1` のとき `frames` は1要素で、`combined` は単一ロール `td=anchor` の `df_ui2frc`。これは `df_obs` を `[anchor-observation_days+1, anchor]`、`df_gt` を `[anchor+1, anchor+gt_days]` で手動フィルタし `fit(df_obs_f, df_gt_f, ref=anchor)` を呼んだときの `df_ui2frc` と同一窓・同一 `ref` であるため、`emp_probability_` も一致する。テストで数値一致を検証する。

> 注: `fit()` は `df_gt` の time_col を見ないため、手動フィルタ側で `df_gt` を正解窓に絞ってから渡す必要がある。`fit_rolling()` 側は内部で正解窓に絞るので、両者が同じ (user,item) 集合を見る。

## 6. 正解側カバレッジの扱い（持ち越し論点 a/b → a を採用）

過去側 fail-fast は観測窓（`obs_min`）基準のみとする（決定A）。古いロールでは正解窓が `df_gt` のデータ範囲より前に外れ、その窓の cv が 0 になり得るが、これは **ハードエラーにしない**。理由:

- `end_date` を正解データ末尾に固定し過去へ遡る設計上、正解窓は観測窓より後方にあり、通常は観測窓が先に `obs_min` に達する（観測窓が律速）。
- 観測ログと正解ログがほぼ同一期間にわたる一般的ケースでは問題が生じない。

`df_gt` が `df_obs` より明確に遅く始まるデータでは、利用者が `roll_days` / `end_date` を調整する想定とし、`design` 段階では警告を追加しない（将来の拡張余地として残す）。本判断は `requirements.md` 制約事項の (a) に一致。

## 7. メソッド配置

`scorer.py` の以下に配置する。

- `_build_ui_rf_cv()` / `_aggregate_empirical()`: `_fit_impl()` の直後。
- `fit_rolling()`: `fit()` と `_fit_impl()` のあと、`transform()` の前（Fitting セクション内）。
- `_to_internal()`: 既存箇所で引数追加のみ。

## 8. テスト設計（AC-10）

`tests/test_scorer.py` に `TestFitRolling` を追加。

| テスト | 検証内容 | 対応 AC |
|--------|----------|---------|
| `test_roll_days_1_equiv_manual_fit` | `roll_days=1` が手動フィルタ+`fit` と `emp_probability_` 一致 | AC-9 |
| `test_rolling_accumulates_N` | `roll_days` 増で `emp_probability_["N"].sum()` 増加 | AC-4 |
| `test_gt_distinct_event` | `df_gt` が `df_obs` と別 user/item 集合でも cv 判定が正しい | 目的-2 |
| `test_end_date_default_uses_gt_max` | `end_date=None` で `observation_end_ == gt_max - gt_days` | AC-2 |
| `test_end_date_explicit` | `end_date` 明示が `anchor` に反映 | AC-2 |
| `test_roll_days_too_large_raises` | 境界突破で `ValueError`、メッセージに最大 `roll_days` | AC-5 |
| `test_end_date_beyond_gt_max_raises` | `end_date > gt_max` で `ValueError` | AC-6 |
| `test_integer_time_col` | 整数 time_col で動作 | — |
| `test_datetime_time_col` | 日付 time_col で動作 | — |
| `test_time_col_override` | `time_col` 引数で列名上書き | AC-1 |
| `test_same_log_df_df` | `fit_rolling(df, df, ...)`（再閲覧）で動作 | 目的-2 |
| `test_downstream_after_rolling` | 後続 `predict`/`transform`/`optimize`/`show` 動作 | AC-7 |
| `test_df_gt_missing_time_col_raises` | `df_gt` に time_col 無で `ValueError` | 制約 |

既存テストは無変更で全パスすること（リファクタ後の `_fit_impl` が `fit()` 経由で同一結果を返すことを既存テストが担保）。

## 9. ドキュメント更新（AC-11）

- `docs/functional-design.md`:
  - 「クラス仕様 > メソッド」に `fit_rolling(...)` 節を追加（引数表・挙動・anchor 算出・バリデーション）。
  - データフロー図に `df_obs`/`df_gt` →（ローリング窓フィルタ）→ `fit_rolling` → 経験的確率 の経路を追記。
- `docs/glossary.md`:
  - API 簡潔版テーブルに `fit_rolling(df_obs, df_gt, observation_days, gt_days, roll_days, end_date)` 行を追加。
  - 「ローリング集計」用語を「アルゴリズム」節に追加（任意）。
- `README.md` / `examples/`: 該当あれば追記（任意）。

## 10. 影響範囲まとめ

| ファイル | 変更 |
|----------|------|
| `src/rfscorer/scorer.py` | `_to_internal` 引数追加、`_fit_impl` 分解、`_build_ui_rf_cv`/`_aggregate_empirical`/`fit_rolling` 追加 |
| `tests/test_scorer.py` | `TestFitRolling` 追加 |
| `docs/functional-design.md` | `fit_rolling` 仕様・データフロー追記 |
| `docs/glossary.md` | API テーブル・用語追記 |

`optimizer.py` / `_plotting.py` / `utils.py` / `_time_utils.py` は変更不要（`split_by_date` も無変更）。

## 11. 性能設計（ホットスポットと最適化シーム）

ローリング集計は `roll_days` 倍の計算量になる重い処理である。本タスクの**初手は可読性を優先**するが、後から局所的に高速化できるよう、計算量が `roll_days` に比例して効くホットスポットを明示し、各々の将来最適化を「**集計シーム**」の内側に閉じ込める。

### 11.1 設計上の最適化シーム

`fit_rolling()` 本体（ループ + concat、design 4.5）は薄く保ち、重い処理を**2つの共有ヘルパーに局所化**する。

- `_build_ui_rf_cv()` … ロール単位の (user,item)→(r,f,cv) 生成
- `_aggregate_empirical()` … 結合フレーム → 全属性の集計

両ヘルパーは `fit()` と共有されるため、**ここを最適化すれば `fit()` / `fit_rolling()` 双方が同時に速くなり、`fit_rolling()` 本体のコードは変更不要**である。`fit_rolling()` が「ロール結果を tidy フレームに積んで集計部へ渡す」という契約（4.5 の集計部入力契約）を保つ限り、内部実装は差し替え自由とする。

### 11.2 ホットスポット一覧（計算量と将来最適化）

| ID | 箇所 | 初手（可読版） | コスト | 将来最適化 |
|----|------|----------------|--------|-----------|
| H1 | `_aggregate_empirical` のカウント（現 319–322 のitertuples ループ） | 現行ロジックをそのまま移設（`fit()` と同一） | `O(roll_days × pairs)` の Python 行ループ。**ローリング時の主ボトルネック** | `combined.groupby(["recency","frequency"]).agg(N=("cv","size"), cv=("cv","sum"))` をフルグリッドに reindex(fill 0)。`fit()` も同時に高速化。整数カウントゆえ数値完全一致 |
| H2 | `_build_ui_rf_cv` の `UIcv` 集合構築 | `{(u,i) for ... in gt_w.itertuples()}` | ロールごと `O(gt_rows)` の Python ループ | `df` と `gt_w` の (user,item) を merge indicator / `isin(MultiIndex)` でベクトル化 |
| H3 | ウィンドウ抽出のブール マスク（4.5） | `(seq >= a) & (seq <= b)` を毎ロール全行スキャン | `O(roll_days × N)` | 時刻で一度ソートし `searchsorted` で連続区間をスライス（`O(log N)`/ロール） |
| H4 | `pd.concat(frames)` とピーク メモリ | 3 列に絞って concat（4.5 で対応済み） | `O(roll_days × pairs)` のメモリ | ロールごとに (r,f)→(N,cv) へ事前集約し**カウント表を加算**（pairs を実体化しない）。最大の構造的削減 |

### 11.3 初手のスタンスと段階導入

- **初手（本タスク）**: H1 は `fit()` と同一の現行ロジックを「純粋移設」する（Phase 1 が挙動不変・既存テスト全パスであることを担保）。H2・H3 も可読な素朴形。H4 のうち列スリム化のみ採用（4.5）。
- **将来タスク（本タスク対象外）**: H1 のベクトル化（最優先・効果大）、続いて H4 の事前集約、H3 の searchsorted、H2 のベクトル化。いずれも 11.1 のシームの内側で完結し、`fit_rolling()` 本体・公開 API・テスト契約を変えずに差し替え可能。

> H1 のベクトル化は可読性も損なわず効果が大きいため、初手に含めることも選択肢。その場合も Phase 1 の純粋移設（既存テスト緑）を先に確定し、その後シーム内で差し替えて再度テスト緑を確認する2段で行い、リスクを切り分ける。
