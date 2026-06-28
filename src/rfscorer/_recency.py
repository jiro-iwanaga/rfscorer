"""Builders that turn an event log into per-(user, item) recency/frequency rows.

Separated from the scorer so each recency_mode is a small, independently testable
unit. Both builders return columns: user, item, recency, frequency.

- ``build_day_rf``  consumes the day-ordinal time column (``seq_col``); recency is the
  elapsed-days bin ``(ref_int - last_view) // recency_unit + 1`` (1-indexed, 1 = most recent).
- ``build_view_rf`` consumes a high-resolution time key (``key_col``) so that sub-day
  order is preserved; recency is a 1-indexed rank within each user by most-recent view
  (newest = 1), ties broken by first appearance in the input.

``frequency`` is the event count (view freq) in both. The frequency aggregation is kept
to a single named-agg entry so a future ``frequency_mode`` (e.g. day freq = nunique of
the day ordinal) can swap it without adding another groupby.
"""


def build_day_rf(df, user_col, item_col, seq_col, ref_int, recency_unit):
    """Build (user, item) -> (recency, frequency) for day recency.

    Recency is ``(ref_int - last_view) // recency_unit + 1`` where last_view is the max
    day ordinal of the pair. Equivalent to taking the per-row recency minimum.
    """
    ui = (
        df.groupby([user_col, item_col], sort=False)[seq_col]
        .agg(last_ts="max", frequency="count")
        .reset_index()
    )
    ui["recency"] = (ref_int - ui["last_ts"]) // recency_unit + 1
    return ui[[user_col, item_col, "recency", "frequency"]]


def build_view_rf(df, user_col, item_col, key_col, seq_col):
    """Build (user, item) -> (recency, frequency) for view recency.

    Recency is a 1-indexed rank within each user, ordered by the pair's most-recent view
    key (``key_col``, high resolution) descending, ties broken by first appearance in the
    input (smaller row index first). frequency is the event count.
    """
    df = df.reset_index(drop=True)
    ui = (
        df.assign(_row_idx=df.index)
        .groupby([user_col, item_col], sort=False)
        .agg(
            last_key=(key_col, "max"),  # 最新閲覧（高解像度）
            frequency=(seq_col, "count"),  # view freq（件数）
            first_idx=("_row_idx", "min"),  # 同値タイブレーク用（初出行）
        )
        .reset_index()
    )
    # user 内で「最新閲覧 降順, 初出行 昇順」に並べ、1 起算ランクを付与
    ui = ui.sort_values([user_col, "last_key", "first_idx"], ascending=[True, False, True])
    ui["recency"] = ui.groupby(user_col, sort=False).cumcount() + 1
    return ui[[user_col, item_col, "recency", "frequency"]]
