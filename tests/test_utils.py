import datetime as _datetime

import numpy as np
import pandas as pd
import pytest

from rfscorer import RecencyFrequencyScorer, split_by_date
from rfscorer._time_utils import normalize_ref, normalize_sequence_col


# ---------------------------------------------------------------------------
# normalize_ref
# ---------------------------------------------------------------------------
class TestNormalizeRef:
    def test_string_date(self):
        expected = pd.Timestamp("2024-01-01").toordinal()
        assert normalize_ref("2024-01-01") == expected

    def test_timestamp(self):
        ts = pd.Timestamp("2024-01-01")
        assert normalize_ref(ts) == ts.toordinal()

    def test_int(self):
        assert normalize_ref(100) == 100

    def test_float(self):
        assert normalize_ref(100.9) == 100

    def test_numpy_int(self):
        assert normalize_ref(np.int64(42)) == 42

    def test_numpy_float(self):
        assert normalize_ref(np.float64(7.0)) == 7

    def test_python_datetime(self):
        dt = _datetime.datetime(2024, 1, 1)
        expected = pd.Timestamp("2024-01-01").toordinal()
        assert normalize_ref(dt) == expected

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="time value could not be normalized"):
            normalize_ref(object())

    def test_invalid_string_date_raises(self):
        with pytest.raises(ValueError, match="time value could not be normalized"):
            normalize_ref("not a real date")


# ---------------------------------------------------------------------------
# normalize_sequence_col
# ---------------------------------------------------------------------------
class TestNormalizeSequenceCol:
    def test_datetime64_col(self):
        s = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-07"]))
        result = normalize_sequence_col(s)
        assert list(result) == [
            pd.Timestamp("2024-01-01").toordinal(),
            pd.Timestamp("2024-01-07").toordinal(),
        ]

    def test_string_date_col(self):
        s = pd.Series(["2024-01-01", "2024-01-07"])
        result = normalize_sequence_col(s)
        assert list(result) == [
            pd.Timestamp("2024-01-01").toordinal(),
            pd.Timestamp("2024-01-07").toordinal(),
        ]

    def test_int_col(self):
        s = pd.Series([1, 7, 100])
        result = normalize_sequence_col(s)
        assert list(result) == [1, 7, 100]

    def test_float_col(self):
        s = pd.Series([1.0, 7.5, 100.9])
        result = normalize_sequence_col(s)
        assert list(result) == [1, 7, 100]

    def test_invalid_dtype_raises(self):
        s = pd.Series([object(), object()])
        with pytest.raises(ValueError, match="time_col must be datetime or integer type"):
            normalize_sequence_col(s)


# ---------------------------------------------------------------------------
# split_by_date
# ---------------------------------------------------------------------------
def _make_df():
    rows = [
        ("u1", "i1", "2024-01-01"),
        ("u1", "i2", "2024-01-03"),
        ("u2", "i1", "2024-01-05"),
        ("u2", "i2", "2024-01-07"),
        ("u3", "i1", "2024-01-09"),
        ("u3", "i2", "2024-01-12"),
        ("u4", "i1", "2024-01-15"),
    ]
    return pd.DataFrame(rows, columns=["user", "item", "datetime"])


class TestSplitByDate:
    def test_basic_split_with_string_date(self):
        df = _make_df()
        df_obs, df_eval = split_by_date(df, "2024-01-07", observation_days=28, evaluation_days=7)
        # obs: Jan01 - Jan07 (28日遡り→全て対象)、eval: Jan08 - Jan14
        assert set(df_obs["datetime"]) == {"2024-01-01", "2024-01-03", "2024-01-05", "2024-01-07"}
        assert set(df_eval["datetime"]) == {"2024-01-09", "2024-01-12"}

    def test_basic_split_with_int_target(self):
        df = _make_df().copy()
        df["seq"] = pd.to_datetime(df["datetime"]).map(lambda x: x.toordinal())
        df = df.drop(columns="datetime")
        target_ord = pd.Timestamp("2024-01-07").toordinal()
        df_obs, df_eval = split_by_date(
            df, target_ord, observation_days=28, evaluation_days=7, time_col="seq"
        )
        obs_dates = ["2024-01-01", "2024-01-03", "2024-01-05", "2024-01-07"]
        eval_dates = ["2024-01-09", "2024-01-12"]
        obs_expected = {pd.Timestamp(d).toordinal() for d in obs_dates}
        eval_expected = {pd.Timestamp(d).toordinal() for d in eval_dates}
        assert set(df_obs["seq"]) == obs_expected
        assert set(df_eval["seq"]) == eval_expected

    def test_observation_days_none_uses_df_start(self):
        df = _make_df()
        df_obs, df_eval = split_by_date(df, "2024-01-07", observation_days=None, evaluation_days=7)
        # observation_days=None で df 先頭まで使う
        assert "2024-01-01" in set(df_obs["datetime"])

    def test_evaluation_days_none_uses_df_end(self):
        df = _make_df()
        df_obs, df_eval = split_by_date(df, "2024-01-07", observation_days=7, evaluation_days=None)
        # evaluation_days=None で df 末尾まで使う
        assert "2024-01-15" in set(df_eval["datetime"])

    def test_observation_days_caps_at_df_start(self):
        df = _make_df()
        df_obs, _ = split_by_date(df, "2024-01-07", observation_days=2, evaluation_days=7)
        # target - 2 + 1 = Jan06、観測期間: Jan06 - Jan07 (2日間)
        assert set(df_obs["datetime"]) == {"2024-01-07"}

    def test_evaluation_days_caps_at_df_end(self):
        df = _make_df()
        _, df_eval = split_by_date(df, "2024-01-07", observation_days=7, evaluation_days=3)
        # target + 3 = Jan10、評価期間: Jan08 - Jan10
        assert set(df_eval["datetime"]) == {"2024-01-09"}

    def test_target_date_inclusive_in_obs(self):
        df = _make_df()
        df_obs, _ = split_by_date(df, "2024-01-07", observation_days=7, evaluation_days=7)
        assert "2024-01-07" in set(df_obs["datetime"])

    def test_target_date_exclusive_in_eval(self):
        df = _make_df()
        _, df_eval = split_by_date(df, "2024-01-07", observation_days=7, evaluation_days=7)
        assert "2024-01-07" not in set(df_eval["datetime"])

    def test_returns_tuple_of_dataframes(self):
        df = _make_df()
        result = split_by_date(df, "2024-01-07")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], pd.DataFrame)
        assert isinstance(result[1], pd.DataFrame)

    def test_does_not_mutate_input_df(self):
        df = _make_df()
        original = df.copy()
        split_by_date(df, "2024-01-07")
        pd.testing.assert_frame_equal(df, original)

    def test_preserves_original_columns(self):
        df = _make_df()
        df_obs, df_eval = split_by_date(df, "2024-01-07")
        assert list(df_obs.columns) == list(df.columns)
        assert list(df_eval.columns) == list(df.columns)

    def test_preserves_original_time_col_type(self):
        df = _make_df()
        df_obs, _ = split_by_date(df, "2024-01-07")
        # 元の datetime 列が string 型のまま保持される
        assert df_obs["datetime"].dtype == df["datetime"].dtype

    def test_not_dataframe_raises(self):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            split_by_date("not_a_df", "2024-01-07")

    def test_missing_time_col_raises(self):
        df = _make_df().drop(columns="datetime")
        with pytest.raises(ValueError, match="Missing required column"):
            split_by_date(df, "2024-01-07")

    def test_custom_time_col(self):
        df = _make_df().rename(columns={"datetime": "date"})
        df_obs, df_eval = split_by_date(df, "2024-01-07", time_col="date")
        assert "2024-01-07" in set(df_obs["date"])

    def test_integer_time_col(self):
        df = _make_df().copy()
        df["seq"] = pd.to_datetime(df["datetime"]).map(lambda x: x.toordinal())
        df = df.drop(columns="datetime")
        target_ord = pd.Timestamp("2024-01-07").toordinal()
        df_obs, df_eval = split_by_date(df, target_ord, time_col="seq")
        assert (df_obs["seq"] <= target_ord).all()
        assert (df_eval["seq"] > target_ord).all()

    def test_invalid_target_date_raises(self):
        df = _make_df()
        with pytest.raises(ValueError, match="time value could not be normalized"):
            split_by_date(df, object())

    def test_chained_with_fit(self):
        df = _make_df()
        df_obs, df_eval = split_by_date(df, "2024-01-07", observation_days=28, evaluation_days=7)
        scorer = RecencyFrequencyScorer()
        scorer.fit(df_obs, df_eval, recency_limit=7, frequency_limit=3)
        # 例外なく fit が完了し、属性が設定されている
        assert scorer.emp_probability_dict_ is not None
