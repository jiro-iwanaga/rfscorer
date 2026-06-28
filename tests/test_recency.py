"""Unit tests for the recency/frequency builders in rfscorer._recency.

These exercise the pure builder functions directly (no scorer setup), validating the
view-recency examples from the design (.steering/20260628-view-recency).
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

    def test_integer_key_orders_by_value(self):
        df = _view_df(
            [
                ("u", "A", 5, 5),
                ("u", "B", 30, 30),
                ("u", "C", 12, 12),
            ]
        )
        out = build_view_rf(df, USER, ITEM, KEY, SEQ)
        assert _rank(out) == {("u", "B"): 1, ("u", "C"): 2, ("u", "A"): 3}

    def test_columns_and_no_leak(self):
        df = _view_df([("u", "A", 1700, 10)])
        out = build_view_rf(df, USER, ITEM, KEY, SEQ)
        assert list(out.columns) == [USER, ITEM, "recency", "frequency"]


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

    def test_columns(self):
        df = pd.DataFrame({USER: ["u"], ITEM: ["A"], SEQ: [100]})
        out = build_day_rf(df, USER, ITEM, SEQ, 110, 1)
        assert list(out.columns) == [USER, ITEM, "recency", "frequency"]
