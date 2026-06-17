import math

import matplotlib.figure
import pandas as pd
import pytest

from rfscorer import RecencyFrequencyScorer

# ---------------------------------------------------------------------------
# テストデータ
#
# 観測期間: 2024-01-01 〜 2024-01-07 (obs_end = Jan07)
# 正解期間: 2024-01-08 〜 2024-01-14
#
# obs_end 基準の Recency (unit=1) = (ordinal(obs_end) - ordinal(datetime)) + 1
#
# u1-item1: Jan01(7), Jan03(5), Jan05(3) → recency=min=3, freq=3, cv=1
# u1-item2: Jan02(6)                     → recency=6, freq=1, cv=0
# u2-item1: Jan04(4)                     → recency=4, freq=1, cv=0
# u2-item2: Jan06(2), Jan07(1)           → recency=1, freq=2, cv=1
# ---------------------------------------------------------------------------
_OBS_PERIOD = ("2024-01-01", "2024-01-07")
_GT_PERIOD = ("2024-01-08", "2024-01-14")

# 明示的な上限値: 全ペアが範囲内に収まる
_RECENCY_LIMIT = 7
_FREQUENCY_LIMIT = 3

# 自動決定される上限値:
# recency CV累積 → limit=3, frequency CV累積 → limit=3
_AUTO_RECENCY_LIMIT = 3
_AUTO_FREQUENCY_LIMIT = 3

# 分割基準日: obs=Jan01-07, gt=Jan08-10
_FIT_TARGET_DATE = "2024-01-07"


def _make_df():
    rows = [
        ("u1", "item1", "2024-01-01"),
        ("u1", "item1", "2024-01-03"),
        ("u1", "item1", "2024-01-05"),
        ("u1", "item2", "2024-01-02"),
        ("u2", "item1", "2024-01-04"),
        ("u2", "item2", "2024-01-06"),
        ("u2", "item2", "2024-01-07"),
        # 正解期間: u1→item1, u2→item2 で対象イベント発生
        ("u1", "item1", "2024-01-09"),
        ("u2", "item2", "2024-01-10"),
    ]
    return pd.DataFrame(rows, columns=["user", "item", "datetime"])


def _split_by_period(df, obs_period, gt_period, time_col="datetime"):
    """Filter df into observation and ground truth data by string-date period tuples."""
    obs_mask = (df[time_col] >= obs_period[0]) & (df[time_col] <= obs_period[1])
    gt_mask = (df[time_col] >= gt_period[0]) & (df[time_col] <= gt_period[1])
    return df[obs_mask], df[gt_mask]


@pytest.fixture
def scorer():
    return RecencyFrequencyScorer()


@pytest.fixture
def df():
    return _make_df()


@pytest.fixture(scope="module")
def scorer_fitted():
    s = RecencyFrequencyScorer()
    s.fit(
        *_split_by_period(_make_df(), _OBS_PERIOD, _GT_PERIOD),
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    return s


@pytest.fixture(scope="module")
def scorer_optimized_mono():
    s = RecencyFrequencyScorer()
    s.fit(
        *_split_by_period(_make_df(), _OBS_PERIOD, _GT_PERIOD),
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    s.optimize(kind="mono")
    return s


@pytest.fixture(scope="module")
def scorer_optimized_mr():
    s = RecencyFrequencyScorer()
    s.fit(
        *_split_by_period(_make_df(), _OBS_PERIOD, _GT_PERIOD),
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    s.optimize(kind="mr")
    return s


@pytest.fixture(scope="module")
def scorer_optimized_mf():
    s = RecencyFrequencyScorer()
    s.fit(
        *_split_by_period(_make_df(), _OBS_PERIOD, _GT_PERIOD),
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    s.optimize(kind="mf")
    return s


@pytest.fixture(scope="module")
def scorer_optimized_mrc():
    s = RecencyFrequencyScorer()
    s.fit(
        *_split_by_period(_make_df(), _OBS_PERIOD, _GT_PERIOD),
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    s.optimize(kind="mrc")
    return s


@pytest.fixture(scope="module")
def scorer_optimized_mfc():
    s = RecencyFrequencyScorer()
    s.fit(
        *_split_by_period(_make_df(), _OBS_PERIOD, _GT_PERIOD),
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    s.optimize(kind="mfc")
    return s


@pytest.fixture(scope="module")
def scorer_optimized_mcc():
    s = RecencyFrequencyScorer()
    s.fit(
        *_split_by_period(_make_df(), _OBS_PERIOD, _GT_PERIOD),
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    s.optimize(kind="mcc")
    return s


@pytest.fixture(scope="module")
def scorer_all_optimized():
    s = RecencyFrequencyScorer()
    s.fit(
        *_split_by_period(_make_df(), _OBS_PERIOD, _GT_PERIOD),
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    for kind in ("mono", "mr", "mf", "mrc", "mfc", "mcc"):
        s.optimize(kind=kind)
    return s


@pytest.fixture(scope="module")
def df_rec(scorer_fitted):
    df = _make_df()
    df_filtered = df[df["datetime"] <= _FIT_TARGET_DATE]
    return scorer_fitted.transform(df_filtered, ref=_FIT_TARGET_DATE)


@pytest.fixture(scope="module")
def df_gt():
    df = _make_df()
    return df[pd.to_datetime(df["datetime"]) >= _GT_PERIOD[0]]


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
class TestInit:
    def test_default_column_names(self, scorer):
        assert scorer.user_col == "user"
        assert scorer.item_col == "item"
        assert scorer.time_col == "datetime"

    def test_custom_column_names(self):
        s = RecencyFrequencyScorer(user_col="uid", item_col="iid", time_col="ts")
        assert s.user_col == "uid"
        assert s.item_col == "iid"
        assert s.time_col == "ts"

    def test_initial_state(self, scorer):
        assert scorer._R == []
        assert scorer._F == []
        assert scorer._RF2N == {}
        assert scorer.emp_probability_ is None
        assert scorer.emp_probability_dict_ is None
        assert scorer.record_num is None
        assert scorer.recency_limit is None
        assert scorer.frequency_limit is None
        assert scorer.mr_probability_dict_ is None
        assert scorer.mf_probability_dict_ is None


# ---------------------------------------------------------------------------
# kind エイリアス
# ---------------------------------------------------------------------------
class TestKindAliases:
    @pytest.mark.parametrize(
        "alias,canonical",
        [
            ("empirical", "emp"),
            ("empirical_recency", "er"),
            ("empirical_frequency", "ef"),
            ("monotonic", "mono"),
            ("monotonic_recency", "mr"),
            ("monotonic_frequency", "mf"),
            ("monotonic_recency_convex", "mrc"),
            ("monotonic_frequency_concave", "mfc"),
            ("monotonic_convex_concave", "mcc"),
        ],
    )
    def test_predict_alias(
        self,
        alias,
        canonical,
        scorer_fitted,
        scorer_optimized_mono,
        scorer_optimized_mr,
        scorer_optimized_mf,
        scorer_optimized_mrc,
        scorer_optimized_mfc,
        scorer_optimized_mcc,
    ):
        scorers = {
            "emp": scorer_fitted,
            "er": scorer_fitted,
            "ef": scorer_fitted,
            "mono": scorer_optimized_mono,
            "mr": scorer_optimized_mr,
            "mf": scorer_optimized_mf,
            "mrc": scorer_optimized_mrc,
            "mfc": scorer_optimized_mfc,
            "mcc": scorer_optimized_mcc,
        }
        s = scorers[canonical]
        assert s.predict(1, 1, kind=alias) == s.predict(1, 1, kind=canonical)


# ---------------------------------------------------------------------------
# fit — バリデーション (新 API: df_obs, df_gt)
# ---------------------------------------------------------------------------
class TestFitValidation:
    def test_obs_not_dataframe_raises(self, scorer, df):
        df_gt = df[pd.to_datetime(df["datetime"]) > "2024-01-07"]
        with pytest.raises(TypeError, match="pandas DataFrame"):
            scorer.fit("not_a_df", df_gt)

    def test_gt_not_dataframe_raises(self, scorer, df):
        df_obs = df[pd.to_datetime(df["datetime"]) <= "2024-01-07"]
        with pytest.raises(TypeError, match="pandas DataFrame"):
            scorer.fit(df_obs, "not_a_df")

    def test_missing_col_in_obs_raises(self, scorer, df):
        df_obs = df[pd.to_datetime(df["datetime"]) <= "2024-01-07"].drop(columns="item")
        df_gt = df[pd.to_datetime(df["datetime"]) > "2024-01-07"]
        with pytest.raises(ValueError, match="df_obs"):
            scorer.fit(df_obs, df_gt)

    def test_missing_col_in_gt_raises(self, scorer, df):
        df_obs = df[pd.to_datetime(df["datetime"]) <= "2024-01-07"]
        df_gt = df[pd.to_datetime(df["datetime"]) > "2024-01-07"].drop(columns="item")
        with pytest.raises(ValueError, match="df_gt"):
            scorer.fit(df_obs, df_gt)

    def test_invalid_ref_raises(self, scorer, df):
        df_obs = df[pd.to_datetime(df["datetime"]) <= "2024-01-07"]
        df_gt = df[pd.to_datetime(df["datetime"]) > "2024-01-07"]
        with pytest.raises(ValueError, match="time value could not be normalized"):
            scorer.fit(df_obs, df_gt, ref=object())

    def test_no_cv_events_raises(self, scorer):
        # df_gt の user-item ペアが df_obs と完全に不一致 → total_cv=0 で自動上限計算不能
        df_obs = pd.DataFrame(
            [("u1", "item1", "2024-01-01")],
            columns=["user", "item", "datetime"],
        )
        df_gt = pd.DataFrame(
            [("u99", "item99", "2024-01-09")],
            columns=["user", "item", "datetime"],
        )
        with pytest.raises(ValueError, match="No events observed in ground truth period"):
            scorer.fit(df_obs, df_gt)


# ---------------------------------------------------------------------------
# fit — 正常系 (新 API: df_obs, df_gt)
# ---------------------------------------------------------------------------
class TestFitResult:
    def _make_obs(self):
        df = _make_df()
        return df[pd.to_datetime(df["datetime"]) <= _OBS_PERIOD[1]]

    def _make_gt(self):
        df = _make_df()
        return df[pd.to_datetime(df["datetime"]) >= _GT_PERIOD[0]]

    def test_returns_self(self, scorer):
        result = scorer.fit(
            self._make_obs(),
            self._make_gt(),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert result is scorer

    def test_same_probabilities_as_split_helper(self):
        # 事前分割した (df_obs, df_gt) と _split_by_period 経由が同一結果
        s1 = RecencyFrequencyScorer()
        s1.fit(
            self._make_obs(),
            self._make_gt(),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        s2 = RecencyFrequencyScorer()
        s2.fit(
            *_split_by_period(_make_df(), _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s1.emp_probability_dict_ == s2.emp_probability_dict_

    def test_ref_date_default_is_obs_max(self):
        # ref=None → df_obs の最大日 (2024-01-07) が observation_end_ に ordinal で格納される
        s = RecencyFrequencyScorer()
        s.fit(
            self._make_obs(),
            self._make_gt(),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.observation_end_ == pd.Timestamp("2024-01-07").toordinal()

    def test_ref_date_explicit(self):
        s = RecencyFrequencyScorer()
        s.fit(
            self._make_obs(),
            self._make_gt(),
            ref="2024-01-07",
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.observation_end_ == pd.Timestamp("2024-01-07").toordinal()

    def test_observation_start_from_data(self):
        s = RecencyFrequencyScorer()
        s.fit(
            self._make_obs(),
            self._make_gt(),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.observation_start_ == pd.Timestamp("2024-01-01").toordinal()

    def test_emp_probability_dict_populated(self):
        s = RecencyFrequencyScorer()
        s.fit(
            self._make_obs(),
            self._make_gt(),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.emp_probability_dict_ is not None

    def test_auto_recency_limit(self):
        s = RecencyFrequencyScorer()
        s.fit(self._make_obs(), self._make_gt())
        assert s.recency_limit == _AUTO_RECENCY_LIMIT

    def test_auto_frequency_limit(self):
        s = RecencyFrequencyScorer()
        s.fit(self._make_obs(), self._make_gt())
        assert s.frequency_limit == _AUTO_FREQUENCY_LIMIT


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------
class TestOptimize:
    def test_before_fit_raises(self, scorer):
        with pytest.raises(RuntimeError, match="fit"):
            scorer.optimize(kind="mono")

    def test_invalid_kind_raises(self, scorer, df):
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(ValueError, match="kind"):
            scorer.optimize(kind="invalid")

    def test_optimize_mono_returns_none(self, scorer, df):
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        result = scorer.optimize(kind="mono")
        assert result is None

    def test_mono_sets_probability(self, scorer_optimized_mono):
        assert scorer_optimized_mono.mono_probability_ is not None
        assert isinstance(scorer_optimized_mono.mono_probability_, pd.DataFrame)
        assert set(scorer_optimized_mono.mono_probability_.columns) == {
            "recency",
            "frequency",
            "probability",
        }

    def test_mono_sets_probability_dict(self, scorer_optimized_mono):
        d = scorer_optimized_mono.mono_probability_dict_
        assert d is not None
        expected = {(r, f) for r in scorer_optimized_mono._R for f in scorer_optimized_mono._F}
        assert set(d.keys()) == expected

    def test_mono_sets_probability_table(self, scorer_optimized_mono):
        tbl = scorer_optimized_mono.mono_probability_table_
        assert tbl is not None
        assert tbl.shape == (_RECENCY_LIMIT, _FREQUENCY_LIMIT)

    def test_mrc_sets_probability_table(self, scorer_optimized_mrc):
        tbl = scorer_optimized_mrc.mrc_probability_table_
        assert tbl is not None
        assert tbl.shape == (_RECENCY_LIMIT, _FREQUENCY_LIMIT)

    def test_mfc_sets_probability_table(self, scorer_optimized_mfc):
        tbl = scorer_optimized_mfc.mfc_probability_table_
        assert tbl is not None
        assert tbl.shape == (_RECENCY_LIMIT, _FREQUENCY_LIMIT)

    def test_mcc_sets_probability_table(self, scorer_optimized_mcc):
        tbl = scorer_optimized_mcc.mcc_probability_table_
        assert tbl is not None
        assert tbl.shape == (_RECENCY_LIMIT, _FREQUENCY_LIMIT)

    def test_mono_probability_values_in_bounds(self, scorer_optimized_mono):
        tol = 1e-6
        for val in scorer_optimized_mono.mono_probability_dict_.values():
            assert -tol <= val <= 1 + tol

    def test_mr_sets_probability(self, scorer_optimized_mr):
        assert scorer_optimized_mr.mr_probability_ is not None
        assert isinstance(scorer_optimized_mr.mr_probability_, pd.DataFrame)
        assert set(scorer_optimized_mr.mr_probability_.columns) == {"recency", "probability"}

    def test_mr_sets_probability_dict(self, scorer_optimized_mr):
        d = scorer_optimized_mr.mr_probability_dict_
        assert d is not None
        expected = set(scorer_optimized_mr._R)
        assert set(d.keys()) == expected

    def test_mr_probability_values_in_bounds(self, scorer_optimized_mr):
        tol = 1e-6
        for val in scorer_optimized_mr.mr_probability_dict_.values():
            assert -tol <= val <= 1 + tol

    def test_mf_sets_probability(self, scorer_optimized_mf):
        assert scorer_optimized_mf.mf_probability_ is not None
        assert isinstance(scorer_optimized_mf.mf_probability_, pd.DataFrame)
        assert set(scorer_optimized_mf.mf_probability_.columns) == {"frequency", "probability"}

    def test_mf_sets_probability_dict(self, scorer_optimized_mf):
        d = scorer_optimized_mf.mf_probability_dict_
        assert d is not None
        expected = set(scorer_optimized_mf._F)
        assert set(d.keys()) == expected

    def test_mf_probability_values_in_bounds(self, scorer_optimized_mf):
        tol = 1e-6
        for val in scorer_optimized_mf.mf_probability_dict_.values():
            assert -tol <= val <= 1 + tol

    def test_mrc_sets_probability(self, scorer_optimized_mrc):
        assert scorer_optimized_mrc.mrc_probability_ is not None
        assert isinstance(scorer_optimized_mrc.mrc_probability_, pd.DataFrame)
        assert set(scorer_optimized_mrc.mrc_probability_.columns) == {
            "recency",
            "frequency",
            "probability",
        }

    def test_mrc_sets_probability_dict(self, scorer_optimized_mrc):
        d = scorer_optimized_mrc.mrc_probability_dict_
        assert d is not None
        expected = {(r, f) for r in scorer_optimized_mrc._R for f in scorer_optimized_mrc._F}
        assert set(d.keys()) == expected

    def test_mrc_probability_values_in_bounds(self, scorer_optimized_mrc):
        tol = 1e-6
        for val in scorer_optimized_mrc.mrc_probability_dict_.values():
            assert -tol <= val <= 1 + tol

    def test_mfc_sets_probability(self, scorer_optimized_mfc):
        assert scorer_optimized_mfc.mfc_probability_ is not None
        assert isinstance(scorer_optimized_mfc.mfc_probability_, pd.DataFrame)
        assert set(scorer_optimized_mfc.mfc_probability_.columns) == {
            "recency",
            "frequency",
            "probability",
        }

    def test_mfc_sets_probability_dict(self, scorer_optimized_mfc):
        d = scorer_optimized_mfc.mfc_probability_dict_
        assert d is not None
        expected = {(r, f) for r in scorer_optimized_mfc._R for f in scorer_optimized_mfc._F}
        assert set(d.keys()) == expected

    def test_mfc_probability_values_in_bounds(self, scorer_optimized_mfc):
        tol = 1e-6
        for val in scorer_optimized_mfc.mfc_probability_dict_.values():
            assert -tol <= val <= 1 + tol

    def test_mcc_sets_probability(self, scorer_optimized_mcc):
        assert scorer_optimized_mcc.mcc_probability_ is not None
        assert isinstance(scorer_optimized_mcc.mcc_probability_, pd.DataFrame)
        assert set(scorer_optimized_mcc.mcc_probability_.columns) == {
            "recency",
            "frequency",
            "probability",
        }

    def test_mcc_sets_probability_dict(self, scorer_optimized_mcc):
        d = scorer_optimized_mcc.mcc_probability_dict_
        assert d is not None
        expected = {(r, f) for r in scorer_optimized_mcc._R for f in scorer_optimized_mcc._F}
        assert set(d.keys()) == expected

    def test_mcc_probability_values_in_bounds(self, scorer_optimized_mcc):
        tol = 1e-6
        for val in scorer_optimized_mcc.mcc_probability_dict_.values():
            assert -tol <= val <= 1 + tol

    # --- eps ---

    def test_optimize_eps_negative_raises(self, scorer, df):
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(ValueError, match="eps"):
            scorer.optimize(kind="mono", eps=-1e-6)

    def test_optimize_eps_too_large_mono_raises(self, scorer, df):
        # eps_max(2D) = min(p_max/(nr-1), p_max/(nf-1)) = min(1.0/6, 1.0/2) = 1.0/6 ≈ 0.167
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(ValueError, match="eps"):
            scorer.optimize(kind="mono", eps=1.0 / 6 + 1e-9)

    def test_optimize_eps_too_large_mr_raises(self, scorer, df):
        # eps_max(mr) = max(R2Prob) / (nr - 1) = 1.0 / 6 ≈ 0.1667
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(ValueError, match="eps"):
            scorer.optimize(kind="mr", eps=1.0 / 6 + 1e-9)

    def test_optimize_eps_too_large_mf_raises(self, scorer, df):
        # eps_max(mf) = max(F2Prob) / (nf - 1) = 1.0 / 2 = 0.5
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(ValueError, match="eps"):
            scorer.optimize(kind="mf", eps=0.5 + 1e-9)

    def test_optimize_eps_too_large_mrc_raises(self, scorer, df):
        # 2D eps 上限は mono と共通: min(p_max/(nr-1), p_max/(nf-1)) = min(1.0/6, 1.0/2) = 1.0/6
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(ValueError, match="eps"):
            scorer.optimize(kind="mrc", eps=1.0 / 6 + 1e-9)

    def test_optimize_eps_too_large_mfc_raises(self, scorer, df):
        # 2D eps 上限は mono と共通: min(p_max/(nr-1), p_max/(nf-1)) = min(1.0/6, 1.0/2) = 1.0/6
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(ValueError, match="eps"):
            scorer.optimize(kind="mfc", eps=1.0 / 6 + 1e-9)

    def test_optimize_eps_too_large_mcc_raises(self, scorer, df):
        # 2D eps 上限は mono と共通: min(p_max/(nr-1), p_max/(nf-1)) = min(1.0/6, 1.0/2) = 1.0/6
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(ValueError, match="eps"):
            scorer.optimize(kind="mcc", eps=1.0 / 6 + 1e-9)

    def test_optimize_with_eps_mono_produces_results(self, scorer, df):
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        scorer.optimize(kind="mono", eps=1e-4)
        assert scorer.mono_probability_dict_ is not None

    def test_optimize_with_eps_mr_produces_results(self, scorer, df):
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        scorer.optimize(kind="mr", eps=1e-4)
        assert scorer.mr_probability_dict_ is not None

    def test_optimize_with_eps_mf_produces_results(self, scorer, df):
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        scorer.optimize(kind="mf", eps=1e-4)
        assert scorer.mf_probability_dict_ is not None


# ---------------------------------------------------------------------------
# optimize — kind エイリアス
# ---------------------------------------------------------------------------
class TestOptimizeAliases:
    @pytest.mark.parametrize(
        "alias,canonical",
        [
            ("monotonic", "mono"),
            ("monotonic_recency", "mr"),
            ("monotonic_frequency", "mf"),
            ("monotonic_recency_convex", "mrc"),
            ("monotonic_frequency_concave", "mfc"),
            ("monotonic_convex_concave", "mcc"),
        ],
    )
    def test_optimize_alias_sets_probability(self, alias, canonical, df):
        s = RecencyFrequencyScorer()
        s.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        s.optimize(kind=alias)
        assert getattr(s, f"{canonical}_probability_dict_") is not None


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------
class TestPredict:
    def test_before_fit_raises(self, scorer):
        with pytest.raises(RuntimeError, match="fit"):
            scorer.predict(1, 1, kind="emp")

    def test_before_optimize_mono_raises(self, scorer, df):
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(RuntimeError, match="optimize"):
            scorer.predict(1, 1, kind="mono")

    def test_before_optimize_mrc_raises(self, scorer, df):
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(RuntimeError, match="optimize"):
            scorer.predict(1, 1, kind="mrc")

    def test_before_optimize_mfc_raises(self, scorer, df):
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(RuntimeError, match="optimize"):
            scorer.predict(1, 1, kind="mfc")

    def test_before_optimize_mcc_raises(self, scorer, df):
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(RuntimeError, match="optimize"):
            scorer.predict(1, 1, kind="mcc")

    def test_before_optimize_mr_raises(self, scorer, df):
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(RuntimeError, match="optimize"):
            scorer.predict(1, 1, kind="mr")

    def test_before_optimize_mf_raises(self, scorer, df):
        scorer.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(RuntimeError, match="optimize"):
            scorer.predict(1, 1, kind="mf")

    def test_invalid_kind_raises(self, scorer_fitted):
        with pytest.raises(ValueError, match="kind"):
            scorer_fitted.predict(1, 1, kind="invalid")

    def test_r_not_int_raises(self, scorer_fitted):
        with pytest.raises(TypeError, match="r"):
            scorer_fitted.predict(1.0, 1)

    def test_r_zero_raises(self, scorer_fitted):
        with pytest.raises(TypeError, match="r"):
            scorer_fitted.predict(0, 1)

    def test_f_not_int_raises(self, scorer_fitted):
        with pytest.raises(TypeError, match="f"):
            scorer_fitted.predict(1, "1")

    def test_f_zero_raises(self, scorer_fitted):
        with pytest.raises(TypeError, match="f"):
            scorer_fitted.predict(1, 0)

    def test_r_bool_treated_as_int(self, scorer_fitted):
        # bool は int のサブクラスのため型チェックを通過し、True=1 として扱われる
        assert scorer_fitted.predict(True, 1) == scorer_fitted.predict(1, 1)

    def test_f_bool_treated_as_int(self, scorer_fitted):
        assert scorer_fitted.predict(1, True) == scorer_fitted.predict(1, 1)

    def test_empirical_known_value(self, scorer_fitted):
        # (1,2) u2-item2: prob=1.0
        assert scorer_fitted.predict(1, 2, kind="emp") == pytest.approx(1.0)

    def test_empirical_returns_float(self, scorer_fitted):
        assert isinstance(scorer_fitted.predict(1, 1), float)

    def test_clamps_r_to_recency_limit(self, scorer_fitted):
        # dict と直接比較してクランプが機能していることを検証
        # (predict同士の比較だと両辺 0.0 で vacuous になるため)
        r_over = _RECENCY_LIMIT + 10
        for f in scorer_fitted._F:
            expected = scorer_fitted.emp_probability_dict_[_RECENCY_LIMIT, f]
            assert scorer_fitted.predict(r_over, f) == pytest.approx(expected)

    def test_clamps_f_to_frequency_limit(self, scorer_fitted):
        # dict と直接比較してクランプが機能していることを検証
        f_over = _FREQUENCY_LIMIT + 10
        for r in scorer_fitted._R:
            expected = scorer_fitted.emp_probability_dict_[r, _FREQUENCY_LIMIT]
            assert scorer_fitted.predict(r, f_over) == pytest.approx(expected)

    def test_clamps_r_to_recency_limit_mr(self, scorer_optimized_mr):
        # mr: r > recency_limit のとき recency_limit にクランプされる
        r_over = _RECENCY_LIMIT + 10
        expected = scorer_optimized_mr.mr_probability_dict_[_RECENCY_LIMIT]
        assert scorer_optimized_mr.predict(r_over, 1, kind="mr") == pytest.approx(expected)

    def test_clamps_f_to_frequency_limit_mf(self, scorer_optimized_mf):
        # mf: f > frequency_limit のとき frequency_limit にクランプされる
        f_over = _FREQUENCY_LIMIT + 10
        expected = scorer_optimized_mf.mf_probability_dict_[_FREQUENCY_LIMIT]
        assert scorer_optimized_mf.predict(1, f_over, kind="mf") == pytest.approx(expected)

    def test_mono_kind(self, scorer_optimized_mono):
        # 型・範囲に加えて、mono_probability_dict_ から値を引いていることを確認
        prob = scorer_optimized_mono.predict(1, 1, kind="mono")
        assert isinstance(prob, float)
        assert 0.0 - 1e-6 <= prob <= 1.0 + 1e-6
        assert prob == pytest.approx(scorer_optimized_mono.mono_probability_dict_[1, 1])

    def test_mr_kind(self, scorer_optimized_mr):
        prob = scorer_optimized_mr.predict(1, 1, kind="mr")
        assert isinstance(prob, float)
        assert 0.0 - 1e-6 <= prob <= 1.0 + 1e-6
        # mr は 1D: recency=1 のみで参照
        assert prob == pytest.approx(scorer_optimized_mr.mr_probability_dict_[1])

    def test_mf_kind(self, scorer_optimized_mf):
        prob = scorer_optimized_mf.predict(1, 1, kind="mf")
        assert isinstance(prob, float)
        assert 0.0 - 1e-6 <= prob <= 1.0 + 1e-6
        # mf は 1D: frequency=1 のみで参照
        assert prob == pytest.approx(scorer_optimized_mf.mf_probability_dict_[1])

    def test_mr_ignores_f(self, scorer_optimized_mr):
        # mr は 1D モデル: f の値が変わっても確率は変わらない
        assert scorer_optimized_mr.predict(1, 1, kind="mr") == scorer_optimized_mr.predict(
            1, 999, kind="mr"
        )

    def test_mf_ignores_r(self, scorer_optimized_mf):
        # mf は 1D モデル: r の値が変わっても確率は変わらない
        assert scorer_optimized_mf.predict(1, 1, kind="mf") == scorer_optimized_mf.predict(
            999, 1, kind="mf"
        )

    def test_mrc_kind(self, scorer_optimized_mrc):
        # 型・範囲に加えて、mrc_probability_dict_ から値を引いていることを確認
        prob = scorer_optimized_mrc.predict(1, 1, kind="mrc")
        assert isinstance(prob, float)
        assert 0.0 - 1e-6 <= prob <= 1.0 + 1e-6
        assert prob == pytest.approx(scorer_optimized_mrc.mrc_probability_dict_[1, 1])

    def test_mfc_kind(self, scorer_optimized_mfc):
        # 型・範囲に加えて、mfc_probability_dict_ から値を引いていることを確認
        prob = scorer_optimized_mfc.predict(1, 1, kind="mfc")
        assert isinstance(prob, float)
        assert 0.0 - 1e-6 <= prob <= 1.0 + 1e-6
        assert prob == pytest.approx(scorer_optimized_mfc.mfc_probability_dict_[1, 1])

    def test_mcc_kind(self, scorer_optimized_mcc):
        # 型・範囲に加えて、mcc_probability_dict_ から値を引いていることを確認
        prob = scorer_optimized_mcc.predict(1, 1, kind="mcc")
        assert isinstance(prob, float)
        assert 0.0 - 1e-6 <= prob <= 1.0 + 1e-6
        assert prob == pytest.approx(scorer_optimized_mcc.mcc_probability_dict_[1, 1])

    def test_er_kind(self, scorer_fitted):
        prob = scorer_fitted.predict(1, 1, kind="er")
        assert isinstance(prob, float)
        # er は 1D: recency=1 のみで参照
        assert prob == pytest.approx(scorer_fitted.er_probability_dict_[1])

    def test_ef_kind(self, scorer_fitted):
        prob = scorer_fitted.predict(1, 1, kind="ef")
        assert isinstance(prob, float)
        # ef は 1D: frequency=1 のみで参照
        assert prob == pytest.approx(scorer_fitted.ef_probability_dict_[1])

    def test_er_ignores_f(self, scorer_fitted):
        # er は 1D モデル: f の値が変わっても確率は変わらない
        assert scorer_fitted.predict(1, 1, kind="er") == scorer_fitted.predict(1, 999, kind="er")

    def test_ef_ignores_r(self, scorer_fitted):
        # ef は 1D モデル: r の値が変わっても確率は変わらない
        assert scorer_fitted.predict(1, 1, kind="ef") == scorer_fitted.predict(999, 1, kind="ef")

    def test_clamps_r_to_recency_limit_er(self, scorer_fitted):
        # er: r > recency_limit のとき recency_limit にクランプされる
        r_over = _RECENCY_LIMIT + 10
        expected = scorer_fitted.er_probability_dict_[_RECENCY_LIMIT]
        assert scorer_fitted.predict(r_over, 1, kind="er") == pytest.approx(expected)

    def test_clamps_f_to_frequency_limit_ef(self, scorer_fitted):
        # ef: f > frequency_limit のとき frequency_limit にクランプされる
        f_over = _FREQUENCY_LIMIT + 10
        expected = scorer_fitted.ef_probability_dict_[_FREQUENCY_LIMIT]
        assert scorer_fitted.predict(1, f_over, kind="ef") == pytest.approx(expected)

    def test_emp_alias_equals_empirical(self, scorer_fitted):
        prob_emp = scorer_fitted.predict(1, 1, kind="emp")
        prob_empirical = scorer_fitted.predict(1, 1, kind="empirical")
        assert prob_emp == prob_empirical


# ---------------------------------------------------------------------------
# transform — 新 API (ref, 事前フィルタ済み df)
# ---------------------------------------------------------------------------
class TestTransform:
    def _make_obs(self):
        df = _make_df()
        return df[pd.to_datetime(df["datetime"]) <= _OBS_PERIOD[1]]

    def test_before_fit_raises(self, scorer, df):
        with pytest.raises(RuntimeError, match="fit"):
            scorer.transform(self._make_obs())

    def test_returns_dataframe_with_expected_columns(self, scorer_fitted):
        result = scorer_fitted.transform(self._make_obs())
        assert set(result.columns) == {
            "user",
            "item",
            "recency",
            "frequency",
            "probability",
            "order",
        }

    def test_sorted_by_user_and_probability(self, scorer_fitted):
        result = scorer_fitted.transform(self._make_obs())
        for user, grp in result.groupby("user"):
            assert list(grp["probability"]) == sorted(grp["probability"], reverse=True)
            assert list(grp["order"]) == list(range(1, len(grp) + 1))

    def test_ref_none_uses_data_max(self, scorer_fitted):
        # ref=None → df_obs の最大日 (2024-01-07) が基準になることを確認
        # u2-item2 は Jan07 に閲覧 → ref=Jan07 なら recency=1
        result = scorer_fitted.transform(self._make_obs())
        u2_item2 = result[(result["user"] == "u2") & (result["item"] == "item2")].iloc[0]
        assert u2_item2["recency"] == 1

    def test_ref_explicit(self, scorer_fitted):
        # ref="2024-01-08" → u2-item2 (最終閲覧 Jan07) の recency = 2
        result = scorer_fitted.transform(self._make_obs(), ref="2024-01-08")
        u2_item2 = result[(result["user"] == "u2") & (result["item"] == "item2")].iloc[0]
        assert u2_item2["recency"] == 2

    def test_explicit_col_names_override_init(self, scorer_fitted):
        df_b = _make_df().rename(columns={"user": "uid", "item": "iid", "datetime": "ts"})
        df_b_obs = df_b[pd.to_datetime(df_b["ts"]) <= _OBS_PERIOD[1]]
        result = scorer_fitted.transform(
            df_b_obs, ref=_FIT_TARGET_DATE, user_col="uid", item_col="iid", time_col="ts"
        )
        assert "uid" in result.columns
        assert "iid" in result.columns

    def test_emp_alias_equals_empirical(self, scorer_fitted):
        result_emp = scorer_fitted.transform(self._make_obs(), kind="emp")
        result_empirical = scorer_fitted.transform(self._make_obs(), kind="empirical")
        assert list(result_emp["probability"]) == list(result_empirical["probability"])

    def test_before_optimize_mr_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.transform(self._make_obs(), kind="mr")

    def test_before_optimize_mf_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.transform(self._make_obs(), kind="mf")

    def test_mr_probability_matches_dict(self, scorer_optimized_mr):
        result = scorer_optimized_mr.transform(self._make_obs(), ref=_FIT_TARGET_DATE, kind="mr")
        for _, row in result.iterrows():
            r_adj = min(int(row["recency"]), _RECENCY_LIMIT)
            expected = scorer_optimized_mr.mr_probability_dict_[r_adj]
            assert row["probability"] == pytest.approx(expected)

    def test_mf_probability_matches_dict(self, scorer_optimized_mf):
        result = scorer_optimized_mf.transform(self._make_obs(), ref=_FIT_TARGET_DATE, kind="mf")
        for _, row in result.iterrows():
            f_adj = min(int(row["frequency"]), _FREQUENCY_LIMIT)
            expected = scorer_optimized_mf.mf_probability_dict_[f_adj]
            assert row["probability"] == pytest.approx(expected)

    def test_er_probability_matches_dict(self, scorer_fitted):
        result = scorer_fitted.transform(self._make_obs(), ref=_FIT_TARGET_DATE, kind="er")
        for _, row in result.iterrows():
            r_adj = min(int(row["recency"]), _RECENCY_LIMIT)
            expected = scorer_fitted.er_probability_dict_[r_adj]
            assert row["probability"] == pytest.approx(expected)

    def test_ef_probability_matches_dict(self, scorer_fitted):
        result = scorer_fitted.transform(self._make_obs(), ref=_FIT_TARGET_DATE, kind="ef")
        for _, row in result.iterrows():
            f_adj = min(int(row["frequency"]), _FREQUENCY_LIMIT)
            expected = scorer_fitted.ef_probability_dict_[f_adj]
            assert row["probability"] == pytest.approx(expected)

    def test_before_optimize_mono_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.transform(self._make_obs(), kind="mono")

    def test_before_optimize_mrc_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.transform(self._make_obs(), kind="mrc")

    def test_before_optimize_mfc_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.transform(self._make_obs(), kind="mfc")

    def test_before_optimize_mcc_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.transform(self._make_obs(), kind="mcc")

    @pytest.mark.parametrize("kind", ["mono", "mrc", "mfc", "mcc"])
    def test_2d_probability_matches_dict(
        self,
        kind,
        scorer_optimized_mono,
        scorer_optimized_mrc,
        scorer_optimized_mfc,
        scorer_optimized_mcc,
    ):
        # 2次元最適化モデルは (r_adj, f_adj) を {kind}_probability_dict_ で参照する
        scorers = {
            "mono": scorer_optimized_mono,
            "mrc": scorer_optimized_mrc,
            "mfc": scorer_optimized_mfc,
            "mcc": scorer_optimized_mcc,
        }
        s = scorers[kind]
        prob_dict = getattr(s, f"{kind}_probability_dict_")
        result = s.transform(self._make_obs(), ref=_FIT_TARGET_DATE, kind=kind)
        for _, row in result.iterrows():
            r_adj = min(int(row["recency"]), _RECENCY_LIMIT)
            f_adj = min(int(row["frequency"]), _FREQUENCY_LIMIT)
            expected = prob_dict[r_adj, f_adj]
            assert row["probability"] == pytest.approx(expected)

    def test_clamps_recency_above_limit(self, scorer_fitted):
        # recency > recency_limit の行は limit にクランプして確率を引く
        # 2023-12-30 → ref_date Jan07 との差 = 8日 → recency = 9 > limit=7
        extra = pd.DataFrame([("u3", "item3", "2023-12-30")], columns=["user", "item", "datetime"])
        df_obs = pd.concat([self._make_obs(), extra], ignore_index=True)
        result = scorer_fitted.transform(df_obs, ref=_FIT_TARGET_DATE)
        u3_row = result[(result["user"] == "u3") & (result["item"] == "item3")].iloc[0]
        assert u3_row["recency"] == 9  # 出力の recency は元値を保持
        expected = scorer_fitted.emp_probability_dict_[_RECENCY_LIMIT, 1]
        assert u3_row["probability"] == pytest.approx(expected)

    def test_clamps_frequency_above_limit(self, scorer_fitted):
        # frequency > frequency_limit の行は limit にクランプして確率を引く
        # Jan04〜Jan07 の4回閲覧 → frequency=4 > limit=3
        extra_rows = [("u3", "item3", f"2024-01-0{d}") for d in range(4, 8)]
        extra = pd.DataFrame(extra_rows, columns=["user", "item", "datetime"])
        df_obs = pd.concat([self._make_obs(), extra], ignore_index=True)
        result = scorer_fitted.transform(df_obs, ref=_FIT_TARGET_DATE)
        u3_row = result[(result["user"] == "u3") & (result["item"] == "item3")].iloc[0]
        assert u3_row["frequency"] == 4  # 出力の frequency は元値を保持
        # min recency = Jan07→recency=1, frequency_adj = min(4, 3) = 3
        expected = scorer_fitted.emp_probability_dict_[1, _FREQUENCY_LIMIT]
        assert u3_row["probability"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
# テストデータの期待値 (scorer_fitted: recency_limit=7, frequency_limit=3)
#
# transform(df_obs, ref="2024-01-07") の結果:
#   u1: item1(r=3,f=3,prob=1.0,order=1), item2(r=6,f=1,prob=0.0,order=2)
#   u2: item2(r=1,f=2,prob=1.0,order=1), item1(r=4,f=1,prob=0.0,order=2)
#
# df_gt: u1-item1(Jan09), u2-item2(Jan10) → 対象イベント発生ペア数=2
#
# order=1: UIrec={(u1,item1),(u2,item2)}, n_hit=2, n_rec=2
#   precision=1.0, recall=1.0, f1=1.0, recall_norm=1.0, f1_norm=1.0
# order=2 (=order_max): UIrec=all 4 pairs, n_hit=2, n_rec=4
#   precision=0.5, recall=1.0, f1≈0.667
# ---------------------------------------------------------------------------
class TestEvaluate:
    def test_returns_dataframe(self, scorer_fitted, df_rec, df_gt):
        result = scorer_fitted.evaluate(df_rec, df_gt, order=1)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, scorer_fitted, df_rec, df_gt):
        result = scorer_fitted.evaluate(df_rec, df_gt, order=1)
        assert set(result.columns) == {
            "order",
            "n_recommended",
            "n_hit",
            "precision",
            "recall",
            "f1",
            "recall_norm",
            "f1_norm",
        }

    def test_row_count_with_order_1(self, scorer_fitted, df_rec, df_gt):
        # order=1 でも order_max(=2) の行が追加されるので 2 行
        result = scorer_fitted.evaluate(df_rec, df_gt, order=1)
        assert len(result) == 2

    def test_order_max_always_included(self, scorer_fitted, df_rec, df_gt):
        result = scorer_fitted.evaluate(df_rec, df_gt, order=1)
        assert df_rec["order"].max() in result["order"].values

    def test_n_recommended_at_order1(self, scorer_fitted, df_rec, df_gt):
        result = scorer_fitted.evaluate(df_rec, df_gt, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["n_recommended"] == 2  # 2ユーザー × 1推薦

    def test_n_hit_at_order1(self, scorer_fitted, df_rec, df_gt):
        result = scorer_fitted.evaluate(df_rec, df_gt, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["n_hit"] == 2

    def test_precision_at_order1(self, scorer_fitted, df_rec, df_gt):
        result = scorer_fitted.evaluate(df_rec, df_gt, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["precision"] == pytest.approx(1.0)

    def test_recall_at_order1(self, scorer_fitted, df_rec, df_gt):
        result = scorer_fitted.evaluate(df_rec, df_gt, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["recall"] == pytest.approx(1.0)

    def test_f1_at_order1(self, scorer_fitted, df_rec, df_gt):
        result = scorer_fitted.evaluate(df_rec, df_gt, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["f1"] == pytest.approx(1.0)

    def test_precision_at_order_max(self, scorer_fitted, df_rec, df_gt):
        # order=2: 4推薦中2ヒット → precision=0.5
        result = scorer_fitted.evaluate(df_rec, df_gt, order=2)
        row = result[result["order"] == 2].iloc[0]
        assert row["precision"] == pytest.approx(0.5)

    def test_recall_norm_with_unseen_events(self, scorer_fitted, df_rec, df_gt):
        # df_gt に存在しないペアを行追加すると recall < recall_norm
        extra = pd.DataFrame([("u3", "item3", "2024-01-11")], columns=["user", "item", "datetime"])
        df_gt_extended = pd.concat([df_gt, extra], ignore_index=True)
        result = scorer_fitted.evaluate(df_rec, df_gt_extended, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["recall"] == pytest.approx(2 / 3)
        assert row["recall_norm"] == pytest.approx(1.0)

    def test_f1_norm_with_unseen_events(self, scorer_fitted, df_rec, df_gt):
        extra = pd.DataFrame([("u3", "item3", "2024-01-11")], columns=["user", "item", "datetime"])
        df_gt_extended = pd.concat([df_gt, extra], ignore_index=True)
        result = scorer_fitted.evaluate(df_rec, df_gt_extended, order=1)
        row = result[result["order"] == 1].iloc[0]
        # recall_norm=1.0, precision=1.0 → f1_norm=1.0 > f1
        assert row["f1_norm"] == pytest.approx(1.0)
        assert row["f1"] < row["f1_norm"]

    def test_uses_init_col_names_by_default(self):
        df_custom = _make_df().rename(columns={"user": "uid", "item": "iid", "datetime": "ts"})
        s = RecencyFrequencyScorer(user_col="uid", item_col="iid", time_col="ts")
        s.fit(
            *_split_by_period(df_custom, _OBS_PERIOD, _GT_PERIOD, time_col="ts"),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        df_custom_obs = df_custom[df_custom["ts"] <= _FIT_TARGET_DATE]
        df_rec_custom = s.transform(df_custom_obs, ref=_FIT_TARGET_DATE)
        df_gt_custom = df_custom[pd.to_datetime(df_custom["ts"]) >= _GT_PERIOD[0]]
        result = s.evaluate(df_rec_custom, df_gt_custom)  # user_col/item_col 省略
        assert isinstance(result, pd.DataFrame)

    def test_explicit_col_override(self, scorer_fitted, df_rec, df_gt):
        df_rec_renamed = df_rec.rename(columns={"user": "uid", "item": "iid"})
        df_gt_renamed = df_gt.rename(columns={"user": "uid", "item": "iid"})
        result = scorer_fitted.evaluate(
            df_rec_renamed, df_gt_renamed, order=1, user_col="uid", item_col="iid"
        )
        assert isinstance(result, pd.DataFrame)

    def test_invalid_df_gt_type_raises(self, scorer_fitted, df_rec):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            scorer_fitted.evaluate(df_rec, 12345, order=1)

    def test_missing_col_in_df_gt_raises(self, scorer_fitted, df_rec, df_gt):
        with pytest.raises(ValueError, match="df_gt"):
            scorer_fitted.evaluate(df_rec, df_gt.drop(columns="item"), order=1)


# ---------------------------------------------------------------------------
# plot_probability_surface
# ---------------------------------------------------------------------------
class TestPlotProbabilitySurface:
    @pytest.fixture(autouse=True)
    def close_figures(self):
        import matplotlib.pyplot as plt

        yield
        plt.close("all")

    def test_before_fit_raises(self, scorer):
        with pytest.raises(RuntimeError, match="fit"):
            scorer.plot_probability_surface()

    def test_invalid_kind_raises(self, scorer_fitted):
        with pytest.raises(ValueError, match="kind"):
            scorer_fitted.plot_probability_surface(kind="invalid")

    def test_before_optimize_mono_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.plot_probability_surface(kind="mono")

    def test_mr_raises_value_error(self, scorer_fitted):
        with pytest.raises(ValueError, match="1D marginal"):
            scorer_fitted.plot_probability_surface(kind="mr")

    def test_mf_raises_value_error(self, scorer_fitted):
        with pytest.raises(ValueError, match="1D marginal"):
            scorer_fitted.plot_probability_surface(kind="mf")

    def test_before_optimize_mrc_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.plot_probability_surface(kind="mrc")

    def test_before_optimize_mfc_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.plot_probability_surface(kind="mfc")

    def test_before_optimize_mcc_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.plot_probability_surface(kind="mcc")

    def test_returns_figure_empirical(self, scorer_fitted):
        fig = scorer_fitted.plot_probability_surface(kind="emp")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_mono(self, scorer_optimized_mono):
        fig = scorer_optimized_mono.plot_probability_surface(kind="mono")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_mr_always_raises(self, scorer_optimized_mr):
        # mr は 1D モデルのためサーフェス描画不可
        with pytest.raises(ValueError, match="1D marginal"):
            scorer_optimized_mr.plot_probability_surface(kind="mr")

    def test_mf_always_raises(self, scorer_optimized_mf):
        # mf は 1D モデルのためサーフェス描画不可
        with pytest.raises(ValueError, match="1D marginal"):
            scorer_optimized_mf.plot_probability_surface(kind="mf")

    def test_returns_figure_mrc(self, scorer_optimized_mrc):
        fig = scorer_optimized_mrc.plot_probability_surface(kind="mrc")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_mfc(self, scorer_optimized_mfc):
        fig = scorer_optimized_mfc.plot_probability_surface(kind="mfc")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_mcc(self, scorer_optimized_mcc):
        fig = scorer_optimized_mcc.plot_probability_surface(kind="mcc")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_er_raises_value_error(self, scorer_fitted):
        # er は 1D 周辺モデルのためサーフェス描画不可
        with pytest.raises(ValueError, match="1D marginal"):
            scorer_fitted.plot_probability_surface(kind="er")

    def test_ef_raises_value_error(self, scorer_fitted):
        # ef は 1D 周辺モデルのためサーフェス描画不可
        with pytest.raises(ValueError, match="1D marginal"):
            scorer_fitted.plot_probability_surface(kind="ef")

    def test_figsize_applied(self, scorer_fitted):
        fig = scorer_fitted.plot_probability_surface(figsize=(4, 3))
        assert tuple(fig.get_size_inches()) == (4, 3)

    def test_title_shown_when_set(self, scorer_fitted):
        fig = scorer_fitted.plot_probability_surface(title="My Title")
        assert fig.axes[0].get_title() == "My Title"

    def test_title_default_when_none(self, scorer_fitted):
        fig = scorer_fitted.plot_probability_surface(title=None)
        assert fig.axes[0].get_title() == "Empirical"

    def test_title_suppressed_when_empty_string(self, scorer_fitted):
        fig = scorer_fitted.plot_probability_surface(title="")
        assert fig.axes[0].get_title() == ""

    def test_fontsize_applied_to_labels(self, scorer_fitted):
        fig = scorer_fitted.plot_probability_surface(fontsize=16)
        ax = fig.axes[0]
        assert ax.xaxis.label.get_size() == 16
        assert ax.yaxis.label.get_size() == 16

    def test_recency_label_applied(self, scorer_fitted):
        fig = scorer_fitted.plot_probability_surface(recency_label="custom_r")
        assert fig.axes[0].get_xlabel() == "custom_r"

    def test_frequency_label_applied(self, scorer_fitted):
        fig = scorer_fitted.plot_probability_surface(frequency_label="custom_f")
        assert fig.axes[0].get_ylabel() == "custom_f"

    def test_probability_label_applied(self, scorer_fitted):
        fig = scorer_fitted.plot_probability_surface(probability_label="custom_p")
        assert fig.axes[0].get_zlabel() == "custom_p"

    def test_path_directory_saves_default_name(self, scorer_fitted, tmp_path):
        # ディレクトリを渡すと surface_{kind}_probability.png として保存される
        scorer_fitted.plot_probability_surface(kind="emp", path=str(tmp_path))
        assert (tmp_path / "surface_emp_probability.png").exists()

    def test_path_file_saves_with_given_name(self, scorer_fitted, tmp_path):
        out = tmp_path / "my_surface.png"
        scorer_fitted.plot_probability_surface(kind="emp", path=str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
# 周辺確率属性 (R2N / R2CV / R2Prob / F2N / F2CV / F2Prob)
#
# テストデータ (recency_limit=7, frequency_limit=3) での期待値:
#   u1-item1: r=3, f=3, cv=1  u1-item2: r=6, f=1, cv=0
#   u2-item1: r=4, f=1, cv=0  u2-item2: r=1, f=2, cv=1
#
#   R2N:  {1:1, 2:0, 3:1, 4:1, 5:0, 6:1, 7:0}
#   R2CV: {1:1, 2:0, 3:1, 4:0, 5:0, 6:0, 7:0}
#   R2Prob: {1:1.0, 3:1.0, 4:0.0, 6:0.0, 2/5/7:0.0}
#   F2N:  {1:2, 2:1, 3:1}
#   F2CV: {1:0, 2:1, 3:1}
#   F2Prob: {1:0.0, 2:1.0, 3:1.0}
# ---------------------------------------------------------------------------
class TestMarginalProbabilityAttributes:
    def test_r2n_known_values(self, scorer_fitted):
        assert scorer_fitted._R2N[1] == pytest.approx(1.0)
        assert scorer_fitted._R2N[3] == pytest.approx(1.0)
        assert scorer_fitted._R2N[4] == pytest.approx(1.0)
        assert scorer_fitted._R2N[6] == pytest.approx(1.0)
        assert scorer_fitted._R2N[2] == pytest.approx(0.0)

    def test_r2cv_known_values(self, scorer_fitted):
        assert scorer_fitted._R2CV[1] == pytest.approx(1.0)
        assert scorer_fitted._R2CV[3] == pytest.approx(1.0)
        assert scorer_fitted._R2CV[4] == pytest.approx(0.0)
        assert scorer_fitted._R2CV[6] == pytest.approx(0.0)

    def test_r2prob_known_values(self, scorer_fitted):
        assert scorer_fitted._R2Prob[1] == pytest.approx(1.0)
        assert scorer_fitted._R2Prob[3] == pytest.approx(1.0)
        assert scorer_fitted._R2Prob[4] == pytest.approx(0.0)
        assert scorer_fitted._R2Prob[6] == pytest.approx(0.0)

    def test_r2n_keys_match_r(self, scorer_fitted):
        assert set(scorer_fitted._R2N.keys()) == set(scorer_fitted._R)

    def test_f2n_known_values(self, scorer_fitted):
        assert scorer_fitted._F2N[1] == pytest.approx(2.0)
        assert scorer_fitted._F2N[2] == pytest.approx(1.0)
        assert scorer_fitted._F2N[3] == pytest.approx(1.0)

    def test_f2cv_known_values(self, scorer_fitted):
        assert scorer_fitted._F2CV[1] == pytest.approx(0.0)
        assert scorer_fitted._F2CV[2] == pytest.approx(1.0)
        assert scorer_fitted._F2CV[3] == pytest.approx(1.0)

    def test_f2prob_known_values(self, scorer_fitted):
        assert scorer_fitted._F2Prob[1] == pytest.approx(0.0)
        assert scorer_fitted._F2Prob[2] == pytest.approx(1.0)
        assert scorer_fitted._F2Prob[3] == pytest.approx(1.0)

    def test_f2n_keys_match_f(self, scorer_fitted):
        assert set(scorer_fitted._F2N.keys()) == set(scorer_fitted._F)

    def test_r2n_f2n_sum_equals_record_num_target(self, scorer_fitted):
        # R2N の合計 == F2N の合計 == 分析対象レコード数
        assert sum(scorer_fitted._R2N.values()) == pytest.approx(scorer_fitted.record_num_target)
        assert sum(scorer_fitted._F2N.values()) == pytest.approx(scorer_fitted.record_num_target)


# ---------------------------------------------------------------------------
# plot_marginal_probability
# ---------------------------------------------------------------------------
class TestPlotMarginalProbability:
    @pytest.fixture(autouse=True)
    def close_figures(self):
        import matplotlib.pyplot as plt

        yield
        plt.close("all")

    def test_before_fit_raises(self, scorer):
        with pytest.raises(RuntimeError, match="fit"):
            scorer.plot_marginal_probability()

    def test_invalid_kind_raises(self, scorer_fitted):
        with pytest.raises(ValueError, match="kind"):
            scorer_fitted.plot_marginal_probability(kind="invalid")

    def test_mr_before_optimize_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.plot_marginal_probability(kind="mr")

    def test_mf_before_optimize_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.plot_marginal_probability(kind="mf")

    def test_rboth_before_optimize_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.plot_marginal_probability(kind="rboth")

    def test_fboth_before_optimize_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.plot_marginal_probability(kind="fboth")

    def test_returns_figure_er(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(kind="er")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_ef(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(kind="ef")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_mr(self, scorer_optimized_mr):
        fig = scorer_optimized_mr.plot_marginal_probability(kind="mr")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_mf(self, scorer_optimized_mf):
        fig = scorer_optimized_mf.plot_marginal_probability(kind="mf")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_rboth_returns_figure_with_legend(self, scorer_optimized_mr):
        fig = scorer_optimized_mr.plot_marginal_probability(kind="rboth")
        assert isinstance(fig, matplotlib.figure.Figure)
        assert fig.axes[0].get_legend() is not None

    def test_rboth_draws_er_and_mr_lines(self, scorer_optimized_mr):
        fig = scorer_optimized_mr.plot_marginal_probability(kind="rboth")
        ax = fig.axes[0]
        assert len(ax.lines) == 2
        assert {line.get_label() for line in ax.lines} == {"er", "mr"}

    def test_fboth_returns_figure_with_legend(self, scorer_optimized_mf):
        fig = scorer_optimized_mf.plot_marginal_probability(kind="fboth")
        assert isinstance(fig, matplotlib.figure.Figure)
        assert fig.axes[0].get_legend() is not None

    def test_fboth_draws_ef_and_mf_lines(self, scorer_optimized_mf):
        fig = scorer_optimized_mf.plot_marginal_probability(kind="fboth")
        ax = fig.axes[0]
        assert len(ax.lines) == 2
        assert {line.get_label() for line in ax.lines} == {"ef", "mf"}

    def test_er_has_no_legend(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(kind="er")
        assert fig.axes[0].get_legend() is None

    def test_ef_has_no_legend(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(kind="ef")
        assert fig.axes[0].get_legend() is None

    def test_mr_has_no_legend(self, scorer_optimized_mr):
        fig = scorer_optimized_mr.plot_marginal_probability(kind="mr")
        assert fig.axes[0].get_legend() is None

    def test_mf_has_no_legend(self, scorer_optimized_mf):
        fig = scorer_optimized_mf.plot_marginal_probability(kind="mf")
        assert fig.axes[0].get_legend() is None

    def test_figsize_applied(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(figsize=(4, 3))
        assert tuple(fig.get_size_inches()) == (4, 3)

    def test_title_shown_when_set(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(title="My Title")
        assert fig.axes[0].get_title() == "My Title"

    def test_title_default_when_none(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(kind="er", title=None)
        assert fig.axes[0].get_title() == "Empirical Recency"

    def test_title_suppressed_when_empty_string(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(title="")
        assert fig.axes[0].get_title() == ""

    def test_fontsize_applied_to_labels(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(fontsize=16)
        ax = fig.axes[0]
        assert ax.xaxis.label.get_size() == 16
        assert ax.yaxis.label.get_size() == 16

    def test_default_axis_label_recency(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(kind="er")
        assert fig.axes[0].get_xlabel() == "recency"

    def test_default_axis_label_frequency(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(kind="ef")
        assert fig.axes[0].get_xlabel() == "frequency"

    def test_axis_label_applied(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(kind="er", axis_label="custom")
        assert fig.axes[0].get_xlabel() == "custom"

    def test_probability_label_applied(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(probability_label="custom_p")
        assert fig.axes[0].get_ylabel() == "custom_p"

    def test_path_directory_saves_default_name(self, scorer_fitted, tmp_path):
        # ディレクトリを渡すと marginal_{kind}_probability.png として保存される
        scorer_fitted.plot_marginal_probability(kind="er", path=str(tmp_path))
        assert (tmp_path / "marginal_er_probability.png").exists()

    def test_path_file_saves_with_given_name(self, scorer_fitted, tmp_path):
        out = tmp_path / "my_marginal.png"
        scorer_fitted.plot_marginal_probability(kind="er", path=str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
# export_probability_csv
# ---------------------------------------------------------------------------
class TestExportProbabilityCsv:
    def test_before_fit_raises(self, scorer):
        with pytest.raises(RuntimeError, match="fit"):
            scorer.export_probability_csv()

    def test_invalid_kind_raises(self, scorer_fitted):
        with pytest.raises(ValueError, match="kind"):
            scorer_fitted.export_probability_csv(kind="invalid")

    def test_before_optimize_mono_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.export_probability_csv(kind="mono")

    def test_before_optimize_mr_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.export_probability_csv(kind="mr")

    def test_before_optimize_mf_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.export_probability_csv(kind="mf")

    def test_returns_none(self, scorer_fitted, tmp_path):
        result = scorer_fitted.export_probability_csv(kind="emp", path=tmp_path / "out.csv")
        assert result is None

    def test_default_path_creates_file(self, scorer_fitted, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        scorer_fitted.export_probability_csv(kind="emp")
        assert (tmp_path / "probability_emp.csv").exists()

    def test_explicit_file_path(self, scorer_fitted, tmp_path):
        out = tmp_path / "my_output.csv"
        scorer_fitted.export_probability_csv(kind="emp", path=str(out))
        assert out.exists()

    def test_directory_path_creates_file_inside(self, scorer_fitted, tmp_path):
        scorer_fitted.export_probability_csv(kind="emp", path=str(tmp_path))
        assert (tmp_path / "probability_emp.csv").exists()

    def test_emp_output_columns(self, scorer_fitted, tmp_path):
        out = tmp_path / "emp.csv"
        scorer_fitted.export_probability_csv(kind="emp", path=str(out))
        df = pd.read_csv(out)
        assert set(df.columns) == {"recency", "frequency", "N", "cv", "probability"}

    def test_er_output_columns(self, scorer_fitted, tmp_path):
        out = tmp_path / "er.csv"
        scorer_fitted.export_probability_csv(kind="er", path=str(out))
        df = pd.read_csv(out)
        assert set(df.columns) == {"recency", "probability"}

    def test_ef_output_columns(self, scorer_fitted, tmp_path):
        out = tmp_path / "ef.csv"
        scorer_fitted.export_probability_csv(kind="ef", path=str(out))
        df = pd.read_csv(out)
        assert set(df.columns) == {"frequency", "probability"}

    def test_mono_output_columns(self, scorer_optimized_mono, tmp_path):
        out = tmp_path / "mono.csv"
        scorer_optimized_mono.export_probability_csv(kind="mono", path=str(out))
        df = pd.read_csv(out)
        assert set(df.columns) == {"recency", "frequency", "probability"}

    def test_mr_output_columns(self, scorer_optimized_mr, tmp_path):
        out = tmp_path / "mr.csv"
        scorer_optimized_mr.export_probability_csv(kind="mr", path=str(out))
        df = pd.read_csv(out)
        assert set(df.columns) == {"recency", "probability"}

    def test_mf_output_columns(self, scorer_optimized_mf, tmp_path):
        out = tmp_path / "mf.csv"
        scorer_optimized_mf.export_probability_csv(kind="mf", path=str(out))
        df = pd.read_csv(out)
        assert set(df.columns) == {"frequency", "probability"}

    def test_mrc_output_columns(self, scorer_optimized_mrc, tmp_path):
        out = tmp_path / "mrc.csv"
        scorer_optimized_mrc.export_probability_csv(kind="mrc", path=str(out))
        df = pd.read_csv(out)
        assert set(df.columns) == {"recency", "frequency", "probability"}

    def test_mfc_output_columns(self, scorer_optimized_mfc, tmp_path):
        out = tmp_path / "mfc.csv"
        scorer_optimized_mfc.export_probability_csv(kind="mfc", path=str(out))
        df = pd.read_csv(out)
        assert set(df.columns) == {"recency", "frequency", "probability"}

    def test_mcc_output_columns(self, scorer_optimized_mcc, tmp_path):
        out = tmp_path / "mcc.csv"
        scorer_optimized_mcc.export_probability_csv(kind="mcc", path=str(out))
        df = pd.read_csv(out)
        assert set(df.columns) == {"recency", "frequency", "probability"}

    def test_emp_row_count(self, scorer_fitted, tmp_path):
        out = tmp_path / "emp.csv"
        scorer_fitted.export_probability_csv(kind="emp", path=str(out))
        df = pd.read_csv(out)
        assert len(df) == _RECENCY_LIMIT * _FREQUENCY_LIMIT

    def test_all_output_columns(self, scorer_all_optimized, tmp_path):
        out = tmp_path / "all.csv"
        scorer_all_optimized.export_probability_csv(kind="all", path=str(out))
        df = pd.read_csv(out)
        assert set(df.columns) == {
            "recency",
            "frequency",
            "N",
            "cv",
            "emp_probability",
            "er_probability",
            "ef_probability",
            "mono_probability",
            "mr_probability",
            "mf_probability",
            "mrc_probability",
            "mfc_probability",
            "mcc_probability",
        }

    def test_all_row_count(self, scorer_all_optimized, tmp_path):
        out = tmp_path / "all.csv"
        scorer_all_optimized.export_probability_csv(kind="all", path=str(out))
        df = pd.read_csv(out)
        assert len(df) == _RECENCY_LIMIT * _FREQUENCY_LIMIT

    def test_empirical_alias(self, scorer_fitted, tmp_path):
        out = tmp_path / "out.csv"
        scorer_fitted.export_probability_csv(kind="empirical", path=str(out))
        assert out.exists()


# ---------------------------------------------------------------------------
# 整数入力での fit — T16
# ---------------------------------------------------------------------------
class TestIntegerTimeCol:
    """整数列を time_col として渡したとき正常に動作することを確認する。"""

    def _make_int_df(self):
        """_make_df() の datetime を ordinal 整数に変換したデータ。"""
        df = _make_df().copy()
        df["seq"] = pd.to_datetime(df["datetime"]).map(lambda x: x.toordinal())
        return df.drop(columns="datetime")

    def test_fit_with_integer_col(self):
        df = self._make_int_df()
        target_ord = pd.Timestamp(_FIT_TARGET_DATE).toordinal()
        obs = df[df["seq"] <= target_ord]
        evl = df[df["seq"] > target_ord]
        s = RecencyFrequencyScorer(time_col="seq")
        s.fit(obs, evl, recency_limit=_RECENCY_LIMIT, frequency_limit=_FREQUENCY_LIMIT)
        assert isinstance(s.emp_probability_dict_, dict)

    def test_integer_and_datetime_produce_same_rf_distribution(self):
        """整数入力と日付入力で同一の RF 分布が得られること。"""
        df_date = _make_df()
        df_int = self._make_int_df()

        s_date = RecencyFrequencyScorer()
        s_date.fit(
            *_split_by_period(df_date, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )

        obs_start = pd.Timestamp(_OBS_PERIOD[0]).toordinal()
        obs_end = pd.Timestamp(_OBS_PERIOD[1]).toordinal()
        gt_start = pd.Timestamp(_GT_PERIOD[0]).toordinal()
        gt_end = pd.Timestamp(_GT_PERIOD[1]).toordinal()
        s_int = RecencyFrequencyScorer(time_col="seq")
        obs_mask = (obs_start <= df_int["seq"]) & (df_int["seq"] <= obs_end)
        gt_mask = (gt_start <= df_int["seq"]) & (df_int["seq"] <= gt_end)
        s_int.fit(
            df_int[obs_mask],
            df_int[gt_mask],
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )

        assert s_date.emp_probability_dict_ == s_int.emp_probability_dict_


# ---------------------------------------------------------------------------
# unit パラメータ — T17
# ---------------------------------------------------------------------------
class TestUnit:
    def _make_scorer_with_unit(self, unit):
        df = _make_df()
        s = RecencyFrequencyScorer(unit=unit)
        s.fit(
            *_split_by_period(df, _OBS_PERIOD, _GT_PERIOD),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        return s

    def test_unit_zero_raises(self):
        with pytest.raises(ValueError, match="unit must be a positive integer"):
            RecencyFrequencyScorer(unit=0)

    def test_unit_negative_raises(self):
        with pytest.raises(ValueError, match="unit must be a positive integer"):
            RecencyFrequencyScorer(unit=-1)

    def test_unit_7_recency_is_floor_div_of_unit_1(self):
        """unit=7 の Recency が unit=1 の Recency の // 7 になること。"""
        df = _make_df()
        obs = df[pd.to_datetime(df["datetime"]) <= _OBS_PERIOD[1]]

        s1 = self._make_scorer_with_unit(1)
        s7 = self._make_scorer_with_unit(7)

        result1 = s1.transform(obs)
        result7 = s7.transform(obs)

        merged = result1.merge(
            result7,
            on=["user", "item"],
            suffixes=("_1", "_7"),
        )
        for _, row in merged.iterrows():
            assert row["recency_7"] == (row["recency_1"] - 1) // 7 + 1

    def test_unit_1_default(self):
        s = RecencyFrequencyScorer()
        assert s.unit == 1


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------
class TestShow:
    def test_show_does_not_raise(self, scorer_fitted):
        scorer_fitted.show()

    def test_show_not_fitted(self, scorer, capsys):
        scorer.show()
        out = capsys.readouterr().out
        assert "not fitted" in out

    def test_show_outputs_header(self, scorer_fitted, capsys):
        scorer_fitted.show()
        out = capsys.readouterr().out
        assert "RecencyFrequencyScorer" in out

    def test_show_outputs_sections(self, scorer_fitted, capsys):
        scorer_fitted.show()
        out = capsys.readouterr().out
        assert "Data" in out
        assert "Model" in out
        assert "Correlation" in out
        assert "Empirical Probability Table" in out

    def test_show_outputs_period_info(self, scorer_fitted, capsys):
        scorer_fitted.show()
        out = capsys.readouterr().out
        assert "observation" in out
        assert "2024-01-01" in out  # ordinal が日付文字列に変換されている

    def test_show_outputs_limits(self, scorer_fitted, capsys):
        scorer_fitted.show()
        out = capsys.readouterr().out
        assert "recency_limit" in out
        assert "frequency_limit" in out

    def test_show_outputs_corr(self, scorer_fitted, capsys):
        scorer_fitted.show()
        out = capsys.readouterr().out
        assert "recency" in out
        assert "frequency" in out
        assert "p=" in out
        assert "weighted" in out

    def test_show_outputs_slice_corr(self, scorer_fitted, capsys):
        scorer_fitted.show()
        out = capsys.readouterr().out
        assert "Slice" in out
        assert "r=" in out
        assert "f=" in out


# ---------------------------------------------------------------------------
# Spearman correlations
# ---------------------------------------------------------------------------
class TestCorrelation:
    def test_recency_corr_none_before_fit(self, scorer):
        assert scorer.recency_corr_ is None

    def test_frequency_corr_none_before_fit(self, scorer):
        assert scorer.frequency_corr_ is None

    def test_recency_corr_pvalue_none_before_fit(self, scorer):
        assert scorer.recency_corr_pvalue_ is None

    def test_frequency_corr_pvalue_none_before_fit(self, scorer):
        assert scorer.frequency_corr_pvalue_ is None

    def test_recency_corr_weighted_none_before_fit(self, scorer):
        assert scorer.recency_corr_weighted_ is None

    def test_frequency_corr_weighted_none_before_fit(self, scorer):
        assert scorer.frequency_corr_weighted_ is None

    def test_recency_corr_is_float(self, scorer_fitted):
        assert isinstance(scorer_fitted.recency_corr_, float)

    def test_frequency_corr_is_float(self, scorer_fitted):
        assert isinstance(scorer_fitted.frequency_corr_, float)

    def test_recency_corr_pvalue_is_float(self, scorer_fitted):
        assert isinstance(scorer_fitted.recency_corr_pvalue_, float)

    def test_frequency_corr_pvalue_is_float(self, scorer_fitted):
        assert isinstance(scorer_fitted.frequency_corr_pvalue_, float)

    def test_recency_corr_pvalue_in_range(self, scorer_fitted):
        assert 0.0 <= scorer_fitted.recency_corr_pvalue_ <= 1.0

    def test_frequency_corr_pvalue_in_range(self, scorer_fitted):
        assert 0.0 <= scorer_fitted.frequency_corr_pvalue_ <= 1.0

    def test_recency_corr_weighted_is_float(self, scorer_fitted):
        assert isinstance(scorer_fitted.recency_corr_weighted_, float)

    def test_frequency_corr_weighted_is_float(self, scorer_fitted):
        assert isinstance(scorer_fitted.frequency_corr_weighted_, float)

    def test_recency_corr_in_range(self, scorer_fitted):
        assert -1.0 <= scorer_fitted.recency_corr_ <= 1.0

    def test_frequency_corr_in_range(self, scorer_fitted):
        assert -1.0 <= scorer_fitted.frequency_corr_ <= 1.0

    def test_recency_corr_weighted_in_range(self, scorer_fitted):
        assert -1.0 <= scorer_fitted.recency_corr_weighted_ <= 1.0

    def test_frequency_corr_weighted_in_range(self, scorer_fitted):
        assert -1.0 <= scorer_fitted.frequency_corr_weighted_ <= 1.0

    def test_recency_corr_is_negative(self, scorer_fitted):
        # r=1 が最直近・最高確率なので、r と P(r) は負の相関
        assert scorer_fitted.recency_corr_ < 0

    def test_frequency_corr_is_positive(self, scorer_fitted):
        # 高頻度ほど高確率なので、f と P(f) は正の相関
        assert scorer_fitted.frequency_corr_ > 0

    def test_recency_corr_weighted_is_negative(self, scorer_fitted):
        assert scorer_fitted.recency_corr_weighted_ < 0

    def test_frequency_corr_weighted_is_positive(self, scorer_fitted):
        assert scorer_fitted.frequency_corr_weighted_ > 0

    def test_equal_n_gives_same_weighted_unweighted(self, scorer_fitted):
        # テストデータは各 r の N_r がすべて 1 なので等重み＝重み付き
        assert scorer_fitted.recency_corr_ == pytest.approx(
            scorer_fitted.recency_corr_weighted_, abs=1e-9
        )


# ---------------------------------------------------------------------------
# Slice-wise Spearman correlations
# ---------------------------------------------------------------------------
class TestSliceCorrelation:
    """Tests for recency_slice_corr_ and frequency_slice_corr_."""

    @pytest.fixture(scope="class")
    def scorer_slice(self):
        """Denser dataset with ≥2 non-zero cells per slice for both r and f."""
        # r=1,f=1: cv=0  r=1,f=2: cv=1  → recency_slice_corr_[1] > 0
        # r=2,f=1: cv=0  r=2,f=2: cv=1  → recency_slice_corr_[2] > 0
        # f=1: r=1 cv=0, r=2 cv=0       → frequency_slice_corr_[1] = nan (all prob=0 → tied)
        # f=2: r=1 cv=1, r=2 cv=1       → frequency_slice_corr_[2] = nan (all prob=1 → tied)
        # Richer dataset where f=1 and f=2 have clear r-vs-prob slope:
        rows = [
            # r=1, f=1 (最直近・低頻度): cv=0
            ("u1", "itemA", "2024-01-07"),
            # r=1, f=2 (最直近・高頻度): cv=1
            ("u2", "itemB", "2024-01-06"),
            ("u2", "itemB", "2024-01-07"),
            # r=2, f=1 (やや古・低頻度): cv=0
            ("u3", "itemC", "2024-01-06"),
            # r=2, f=2 (やや古・高頻度): cv=1
            ("u4", "itemD", "2024-01-05"),
            ("u4", "itemD", "2024-01-06"),
            # r=3, f=1 (古・低頻度): cv=0
            ("u5", "itemE", "2024-01-05"),
            # r=3, f=2 (古・高頻度): cv=0
            ("u6", "itemF", "2024-01-04"),
            ("u6", "itemF", "2024-01-05"),
            # ground truth
            ("u2", "itemB", "2024-01-09"),
            ("u4", "itemD", "2024-01-09"),
        ]
        df = pd.DataFrame(rows, columns=["user", "item", "datetime"])
        df_obs = df[df["datetime"] <= "2024-01-07"]
        df_gt = df[df["datetime"] >= "2024-01-08"]
        s = RecencyFrequencyScorer()
        s.fit(df_obs, df_gt, recency_limit=3, frequency_limit=2)
        return s

    def test_recency_slice_corr_none_before_fit(self, scorer):
        assert scorer.recency_slice_corr_ is None

    def test_frequency_slice_corr_none_before_fit(self, scorer):
        assert scorer.frequency_slice_corr_ is None

    def test_recency_slice_corr_is_dict(self, scorer_fitted):
        assert isinstance(scorer_fitted.recency_slice_corr_, dict)

    def test_frequency_slice_corr_is_dict(self, scorer_fitted):
        assert isinstance(scorer_fitted.frequency_slice_corr_, dict)

    def test_recency_slice_corr_keys_are_observed_r(self, scorer_fitted):
        # キーは N_r > 0 の r 値のみ（r=1,3,4,6）
        assert set(scorer_fitted.recency_slice_corr_.keys()) == {1, 3, 4, 6}

    def test_frequency_slice_corr_keys_are_observed_f(self, scorer_fitted):
        assert set(scorer_fitted.frequency_slice_corr_.keys()) == {1, 2, 3}

    def test_recency_slice_corr_values_are_float(self, scorer_fitted):
        for v in scorer_fitted.recency_slice_corr_.values():
            assert isinstance(v, float)

    def test_frequency_slice_corr_values_are_float(self, scorer_fitted):
        for v in scorer_fitted.frequency_slice_corr_.values():
            assert isinstance(v, float)

    def test_recency_slice_corr_sparse_data_all_nan(self, scorer_fitted):
        # デフォルトテストデータは各 r スライスに有効セルが1つ以下 → 全 NaN
        assert all(math.isnan(v) for v in scorer_fitted.recency_slice_corr_.values())

    def test_recency_slice_corr_positive_for_dense_data(self, scorer_slice):
        # r=1,2: f=1 → cv=0, f=2 → cv=1 → f と P(r,f) は正の相関
        non_nan = {k: v for k, v in scorer_slice.recency_slice_corr_.items() if not math.isnan(v)}
        assert len(non_nan) > 0
        assert all(v > 0 for v in non_nan.values())

    def test_frequency_slice_corr_negative_for_dense_data(self, scorer_slice):
        # f=2: r=1 cv=1, r=2 cv=1, r=3 cv=0 → r と P(r,f) は負の相関
        non_nan = {k: v for k, v in scorer_slice.frequency_slice_corr_.items() if not math.isnan(v)}
        assert len(non_nan) > 0
        assert all(v < 0 for v in non_nan.values())
