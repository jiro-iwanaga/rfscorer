"""Unit tests for the recency/frequency builders in rfscorer._recency.

These exercise the pure builder functions directly (no scorer setup), validating the
view-recency examples from the design (.steering/20260628-view-recency) plus edge cases,
column-name parametrization, and the day/view equivalence guarantees.
"""

import pandas as pd

from rfscorer._recency import build_day_rf, build_view_rf

USER = "user"
ITEM = "item"
SEQ = "datetime"
KEY = "_view_key"


def _view_df(rows):
    """rows: list of (user, item, key, seq). Builds an internal-style frame."""
    return pd.DataFrame(rows, columns=[USER, ITEM, KEY, SEQ])


def _rank(out):
    """Return {(user, item): recency} for easy assertions."""
    return {(r.user, r.item): r.recency for r in out.itertuples()}


def _freq(out):
    return {(r.user, r.item): r.frequency for r in out.itertuples()}


class TestBuildViewRf:
    # --- canonical examples (design .steering/20260628-view-recency) -------------

    def test_example1_basic(self):
        # A@17:00, B@16:50, A@16:40(dup), C(prev day). Keys are monotonic with time.
        df = _view_df(
            [
                ("u", "A", 1700, 10),
                ("u", "B", 1650, 10),
                ("u", "A", 1640, 10),
                ("u", "C", 1000, 9),
            ]
        )
        out = build_view_rf(df, USER, ITEM, KEY, SEQ)
        assert _rank(out) == {("u", "A"): 1, ("u", "B"): 2, ("u", "C"): 3}
        # A viewed twice -> frequency 2
        assert _freq(out) == {("u", "A"): 2, ("u", "B"): 1, ("u", "C"): 1}

    def test_example2_tiebreak_equal_key(self):
        # A and B share the exact same key; A appears first in the input -> smaller recency.
        df = _view_df(
            [
                ("u", "A", 1700, 10),
                ("u", "B", 1700, 10),
                ("u", "C", 1600, 10),
            ]
        )
        out = build_view_rf(df, USER, ITEM, KEY, SEQ)
        assert _rank(out) == {("u", "A"): 1, ("u", "B"): 2, ("u", "C"): 3}

    def test_example3_multi_user_intraday(self):
        # Same-day different times. Ranking is per user, by key desc.
        # Regression guard against day-truncation: U2 must rank Q (newer) first.
        df = _view_df(
            [
                ("U1", "P", 1700, 10),
                ("U1", "Q", 1600, 10),
                ("U2", "P", 1600, 10),
                ("U2", "Q", 1700, 10),
            ]
        )
        out = build_view_rf(df, USER, ITEM, KEY, SEQ)
        assert _rank(out) == {
            ("U1", "P"): 1,
            ("U1", "Q"): 2,
            ("U2", "Q"): 1,
            ("U2", "P"): 2,
        }

    # --- ranking semantics ------------------------------------------------------

    def test_integer_key_orders_by_value(self):
        df = _view_df([("u", "A", 5, 5), ("u", "B", 30, 30), ("u", "C", 12, 12)])
        out = build_view_rf(df, USER, ITEM, KEY, SEQ)
        assert _rank(out) == {("u", "B"): 1, ("u", "C"): 2, ("u", "A"): 3}

    def test_ranking_uses_key_not_seq(self):
        # key ascending A<B<C, seq descending A>B>C. Recency must follow key (desc),
        # proving the rank is driven by key_col and not by seq_col.
        df = _view_df([("u", "A", 1, 100), ("u", "B", 2, 50), ("u", "C", 3, 10)])
        out = build_view_rf(df, USER, ITEM, KEY, SEQ)
        assert _rank(out) == {("u", "C"): 1, ("u", "B"): 2, ("u", "A"): 3}

    def test_three_way_tie_break_by_first_appearance(self):
        df = _view_df([("u", "A", 7, 1), ("u", "B", 7, 1), ("u", "C", 7, 1)])
        out = build_view_rf(df, USER, ITEM, KEY, SEQ)
        assert _rank(out) == {("u", "A"): 1, ("u", "B"): 2, ("u", "C"): 3}

    def test_tiebreak_uses_first_not_last_appearance(self):
        # A appears at rows 0 and 2, B at row 1; all share key 5. first_idx(A)=0 < first_idx(B)=1,
        # so A ranks first even though A's last occurrence (row 2) is after B.
        df = _view_df([("u", "A", 5, 1), ("u", "B", 5, 1), ("u", "A", 5, 1)])
        out = build_view_rf(df, USER, ITEM, KEY, SEQ)
        assert _rank(out) == {("u", "A"): 1, ("u", "B"): 2}

    def test_last_key_is_max_over_duplicates(self):
        # A's newest view (key 200, row 2) decides its rank, beating B (key 150).
        df = _view_df([("u", "A", 50, 1), ("u", "B", 150, 1), ("u", "A", 200, 1)])
        out = build_view_rf(df, USER, ITEM, KEY, SEQ)
        assert _rank(out) == {("u", "A"): 1, ("u", "B"): 2}

    def test_ranks_are_permutation_per_user(self):
        rows = [("u", chr(65 + i), i * 10, 1) for i in range(6)]  # A..F distinct keys
        out = build_view_rf(_view_df(rows), USER, ITEM, KEY, SEQ)
        ranks = sorted(r.recency for r in out.itertuples())
        assert ranks == [1, 2, 3, 4, 5, 6]  # contiguous 1..n, no gaps/dupes

    def test_multi_user_independent_counts(self):
        df = _view_df(
            [
                ("u1", "A", 3, 1),
                ("u1", "B", 2, 1),
                ("u1", "C", 1, 1),
                ("u2", "X", 9, 1),
            ]
        )
        out = build_view_rf(df, USER, ITEM, KEY, SEQ)
        assert _rank(out) == {
            ("u1", "A"): 1,
            ("u1", "B"): 2,
            ("u1", "C"): 3,
            ("u2", "X"): 1,
        }

    # --- frequency semantics ----------------------------------------------------

    def test_frequency_is_event_count_not_unique_days(self):
        # Three events on the same seq value -> frequency 3 (count, not nunique).
        df = _view_df([("u", "A", 1, 10), ("u", "A", 2, 10), ("u", "A", 3, 10)])
        out = build_view_rf(df, USER, ITEM, KEY, SEQ)
        assert _freq(out) == {("u", "A"): 3}

    # --- structure / robustness -------------------------------------------------

    def test_columns_and_no_leak(self):
        out = build_view_rf(_view_df([("u", "A", 1700, 10)]), USER, ITEM, KEY, SEQ)
        assert list(out.columns) == [USER, ITEM, "recency", "frequency"]

    def test_single_pair(self):
        out = build_view_rf(_view_df([("u", "A", 1, 1)]), USER, ITEM, KEY, SEQ)
        assert _rank(out) == {("u", "A"): 1}
        assert _freq(out) == {("u", "A"): 1}

    def test_empty_input(self):
        out = build_view_rf(_view_df([]), USER, ITEM, KEY, SEQ)
        assert list(out.columns) == [USER, ITEM, "recency", "frequency"]
        assert len(out) == 0

    def test_non_contiguous_index(self):
        # Simulate a fit_rolling window slice: a boolean-filtered frame with a gappy index.
        full = _view_df(
            [
                ("u", "Z", 0, 1),  # filtered out
                ("u", "A", 30, 1),
                ("u", "B", 20, 1),
                ("u", "Z", 0, 1),  # filtered out
                ("u", "C", 10, 1),
            ]
        )
        sliced = full[full[ITEM] != "Z"]  # index [1, 2, 4]
        out = build_view_rf(sliced, USER, ITEM, KEY, SEQ)
        assert _rank(out) == {("u", "A"): 1, ("u", "B"): 2, ("u", "C"): 3}

    def test_custom_column_names(self):
        df = pd.DataFrame(
            {
                "uid": ["u", "u"],
                "sku": ["A", "B"],
                "k": [200, 100],
                "ts": [5, 5],
            }
        )
        out = build_view_rf(df, "uid", "sku", "k", "ts")
        assert list(out.columns) == ["uid", "sku", "recency", "frequency"]
        ranks = {(r.uid, r.sku): r.recency for r in out.itertuples()}
        assert ranks == {("u", "A"): 1, ("u", "B"): 2}

    def test_does_not_mutate_input(self):
        df = _view_df([("u", "A", 2, 1), ("u", "B", 1, 1)])
        before = df.copy()
        build_view_rf(df, USER, ITEM, KEY, SEQ)
        pd.testing.assert_frame_equal(df, before)


class TestBuildDayRf:
    def test_matches_legacy_formula(self):
        # Legacy: recency = min over rows of (ref - seq)//unit + 1 == (ref - max(seq))//unit + 1
        df = pd.DataFrame(
            {
                USER: ["u", "u", "u", "v"],
                ITEM: ["A", "A", "B", "A"],
                SEQ: [100, 104, 102, 100],
            }
        )
        ref_int, unit = 110, 2
        out = build_day_rf(df, USER, ITEM, SEQ, ref_int, unit)
        rank = _rank(out)
        freq = _freq(out)
        # (u, A): last=104 -> (110-104)//2+1 = 4 ; viewed twice
        assert rank[("u", "A")] == 4
        assert freq[("u", "A")] == 2
        # (u, B): last=102 -> (110-102)//2+1 = 5
        assert rank[("u", "B")] == 5
        assert freq[("u", "B")] == 1
        # (v, A): last=100 -> (110-100)//2+1 = 6
        assert rank[("v", "A")] == 6

    def test_recency_at_ref_is_one(self):
        df = pd.DataFrame({USER: ["u"], ITEM: ["A"], SEQ: [110]})
        out = build_day_rf(df, USER, ITEM, SEQ, 110, 1)
        assert _rank(out) == {("u", "A"): 1}

    def test_unit_one_simple(self):
        df = pd.DataFrame({USER: ["u", "u"], ITEM: ["A", "B"], SEQ: [109, 105]})
        out = build_day_rf(df, USER, ITEM, SEQ, 110, 1)
        # A: (110-109)//1+1 = 2 ; B: (110-105)//1+1 = 6
        assert _rank(out) == {("u", "A"): 2, ("u", "B"): 6}

    def test_unit_binning_collapses_same_bin(self):
        # unit=7: last_ts 104 and 105 both fall in the same recency bin as ref 110.
        df = pd.DataFrame({USER: ["u", "u"], ITEM: ["A", "B"], SEQ: [104, 105]})
        out = build_day_rf(df, USER, ITEM, SEQ, 110, 7)
        # (110-104)//7+1 = 1 ; (110-105)//7+1 = 1
        assert _rank(out) == {("u", "A"): 1, ("u", "B"): 1}

    def test_unit_binning_separates_different_bins(self):
        df = pd.DataFrame({USER: ["u", "u"], ITEM: ["A", "B"], SEQ: [109, 100]})
        out = build_day_rf(df, USER, ITEM, SEQ, 110, 7)
        # (110-109)//7+1 = 1 ; (110-100)//7+1 = 2
        assert _rank(out) == {("u", "A"): 1, ("u", "B"): 2}

    def test_frequency_counts_all_events(self):
        df = pd.DataFrame({USER: ["u"] * 3, ITEM: ["A"] * 3, SEQ: [100, 101, 102]})
        out = build_day_rf(df, USER, ITEM, SEQ, 110, 1)
        assert _freq(out) == {("u", "A"): 3}

    def test_multi_user(self):
        df = pd.DataFrame({USER: ["u1", "u2"], ITEM: ["A", "A"], SEQ: [108, 100]})
        out = build_day_rf(df, USER, ITEM, SEQ, 110, 1)
        assert _rank(out) == {("u1", "A"): 3, ("u2", "A"): 11}

    def test_columns(self):
        df = pd.DataFrame({USER: ["u"], ITEM: ["A"], SEQ: [100]})
        out = build_day_rf(df, USER, ITEM, SEQ, 110, 1)
        assert list(out.columns) == [USER, ITEM, "recency", "frequency"]

    def test_empty_input(self):
        df = pd.DataFrame({USER: [], ITEM: [], SEQ: []})
        out = build_day_rf(df, USER, ITEM, SEQ, 110, 1)
        assert list(out.columns) == [USER, ITEM, "recency", "frequency"]
        assert len(out) == 0

    def test_custom_column_names(self):
        df = pd.DataFrame({"uid": ["u"], "sku": ["A"], "ts": [105]})
        out = build_day_rf(df, "uid", "sku", "ts", 110, 1)
        assert list(out.columns) == ["uid", "sku", "recency", "frequency"]
        assert out.iloc[0]["recency"] == 6

    def test_non_contiguous_index(self):
        full = pd.DataFrame(
            {
                USER: ["u", "u", "u", "u"],
                ITEM: ["Z", "A", "Z", "B"],
                SEQ: [1, 108, 1, 100],
            }
        )
        sliced = full[full[ITEM] != "Z"]  # index [1, 3]
        out = build_day_rf(sliced, USER, ITEM, SEQ, 110, 1)
        assert _rank(out) == {("u", "A"): 3, ("u", "B"): 11}

    def test_does_not_mutate_input(self):
        df = pd.DataFrame({USER: ["u", "u"], ITEM: ["A", "A"], SEQ: [100, 104]})
        before = df.copy()
        build_day_rf(df, USER, ITEM, SEQ, 110, 2)
        pd.testing.assert_frame_equal(df, before)

    def test_matches_legacy_pipeline_on_larger_frame(self):
        # Independent legacy computation: per-row recency then groupby min/count.
        rng = list(range(50))
        df = pd.DataFrame(
            {
                USER: [f"u{i % 4}" for i in rng],
                ITEM: [f"i{i % 5}" for i in rng],
                SEQ: [100 + (i * 7) % 23 for i in rng],
            }
        )
        ref_int, unit = 130, 3

        tmp = df.copy()
        tmp["recency"] = (ref_int - tmp[SEQ]) // unit + 1
        legacy = (
            tmp.groupby([USER, ITEM], sort=False)
            .agg(recency=("recency", "min"), frequency=("recency", "count"))
            .reset_index()
        )

        out = build_day_rf(df, USER, ITEM, SEQ, ref_int, unit)

        legacy_sorted = legacy.sort_values([USER, ITEM]).reset_index(drop=True)
        out_sorted = out.sort_values([USER, ITEM]).reset_index(drop=True)
        pd.testing.assert_frame_equal(out_sorted, legacy_sorted)
