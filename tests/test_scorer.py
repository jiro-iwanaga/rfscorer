import pandas as pd
import pytest

from rfscorer import RecencyFrequencyScorer

# ---------------------------------------------------------------------------
# テストデータ
#
# 観測期間: 2024-01-01 〜 2024-01-07 (obs_end = Jan07)
# 評価期間: 2024-01-08 〜 2024-01-14
#
# obs_end 基準の Recency (unit=1) = (ordinal(obs_end) - ordinal(datetime)) + 1
#
# u1-item1: Jan01(7), Jan03(5), Jan05(3) → recency=min=3, freq=3, cv=1
# u1-item2: Jan02(6)                     → recency=6, freq=1, cv=0
# u2-item1: Jan04(4)                     → recency=4, freq=1, cv=0
# u2-item2: Jan06(2), Jan07(1)           → recency=1, freq=2, cv=1
# ---------------------------------------------------------------------------
_OBS_PERIOD = ("2024-01-01", "2024-01-07")
_EVAL_PERIOD = ("2024-01-08", "2024-01-14")

# 明示的な上限値: 全ペアが範囲内に収まる
_RECENCY_LIMIT = 7
_FREQUENCY_LIMIT = 3

# 自動決定される上限値:
# recency CV累積 → limit=3, frequency CV累積 → limit=3
_AUTO_RECENCY_LIMIT = 3
_AUTO_FREQUENCY_LIMIT = 3

# fit_date のテスト基準日: obs=Jan01-07, eval=Jan08-10 (fit_period の全期間指定と等価)
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
        # 評価期間: u1→item1, u2→item2 で対象イベント発生
        ("u1", "item1", "2024-01-09"),
        ("u2", "item2", "2024-01-10"),
    ]
    return pd.DataFrame(rows, columns=["user", "item", "datetime"])


@pytest.fixture
def scorer():
    return RecencyFrequencyScorer()


@pytest.fixture
def df():
    return _make_df()


@pytest.fixture(scope="module")
def scorer_fitted():
    s = RecencyFrequencyScorer()
    s.fit_period(
        _make_df(),
        _OBS_PERIOD,
        _EVAL_PERIOD,
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    return s


@pytest.fixture(scope="module")
def scorer_optimized_mono():
    s = RecencyFrequencyScorer()
    s.fit_period(
        _make_df(),
        _OBS_PERIOD,
        _EVAL_PERIOD,
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    s.optimize(kind="mono")
    return s


@pytest.fixture(scope="module")
def scorer_optimized_mr():
    s = RecencyFrequencyScorer()
    s.fit_period(
        _make_df(),
        _OBS_PERIOD,
        _EVAL_PERIOD,
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    s.optimize(kind="mr")
    return s


@pytest.fixture(scope="module")
def scorer_optimized_mf():
    s = RecencyFrequencyScorer()
    s.fit_period(
        _make_df(),
        _OBS_PERIOD,
        _EVAL_PERIOD,
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    s.optimize(kind="mf")
    return s


@pytest.fixture(scope="module")
def scorer_optimized_mrc():
    s = RecencyFrequencyScorer()
    s.fit_period(
        _make_df(),
        _OBS_PERIOD,
        _EVAL_PERIOD,
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    s.optimize(kind="mrc")
    return s


@pytest.fixture(scope="module")
def scorer_optimized_mfc():
    s = RecencyFrequencyScorer()
    s.fit_period(
        _make_df(),
        _OBS_PERIOD,
        _EVAL_PERIOD,
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    s.optimize(kind="mfc")
    return s


@pytest.fixture(scope="module")
def scorer_optimized_mcc():
    s = RecencyFrequencyScorer()
    s.fit_period(
        _make_df(),
        _OBS_PERIOD,
        _EVAL_PERIOD,
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    s.optimize(kind="mcc")
    return s


@pytest.fixture(scope="module")
def scorer_all_optimized():
    s = RecencyFrequencyScorer()
    s.fit_period(
        _make_df(),
        _OBS_PERIOD,
        _EVAL_PERIOD,
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    for kind in ("mono", "mr", "mf", "mrc", "mfc", "mcc"):
        s.optimize(kind=kind)
    return s


@pytest.fixture(scope="module")
def df_rec(scorer_fitted):
    return scorer_fitted.transform_date(_make_df(), _FIT_TARGET_DATE)


@pytest.fixture(scope="module")
def df_eval():
    df = _make_df()
    return df[pd.to_datetime(df["datetime"]) >= _EVAL_PERIOD[0]]


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
        assert scorer.R == []
        assert scorer.F == []
        assert scorer.RF2N == {}
        assert scorer.empirical_probability_ is None
        assert scorer.empirical_probability_dict_ is None
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
# fit_period — バリデーション
# ---------------------------------------------------------------------------
class TestFitPeriodValidation:
    def test_not_dataframe_raises(self, scorer):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            scorer.fit_period("not_a_df", _OBS_PERIOD, _EVAL_PERIOD)

    def test_missing_column_raises(self, scorer, df):
        with pytest.raises(ValueError, match="Missing required columns"):
            scorer.fit_period(df.drop(columns="item"), _OBS_PERIOD, _EVAL_PERIOD)

    def test_custom_col_missing_raises(self, df):
        s = RecencyFrequencyScorer(user_col="uid")
        with pytest.raises(ValueError, match="Missing required columns"):
            s.fit_period(df, _OBS_PERIOD, _EVAL_PERIOD)

    def test_observation_period_wrong_length_raises(self, scorer, df):
        with pytest.raises(ValueError, match="observation_period"):
            scorer.fit_period(df, ("2024-01-01",), _EVAL_PERIOD)

    def test_evaluation_period_wrong_length_raises(self, scorer, df):
        with pytest.raises(ValueError, match="evaluation_period"):
            scorer.fit_period(df, _OBS_PERIOD, ("2024-01-08",))

    def test_observation_period_reversed_raises(self, scorer, df):
        with pytest.raises(ValueError, match="observation_period must be ordered"):
            scorer.fit_period(df, ("2024-01-07", "2024-01-01"), _EVAL_PERIOD)

    def test_evaluation_period_reversed_raises(self, scorer, df):
        with pytest.raises(ValueError, match="evaluation_period must be ordered"):
            scorer.fit_period(df, _OBS_PERIOD, ("2024-01-14", "2024-01-08"))

    def test_overlapping_periods_raises(self, scorer, df):
        with pytest.raises(ValueError, match="observation_period must end before"):
            scorer.fit_period(df, ("2024-01-01", "2024-01-09"), ("2024-01-08", "2024-01-14"))

    def test_no_cv_auto_limit_raises(self, scorer):
        # 評価期間に対象イベントなし → 自動上限計算で ValueError
        rows = [
            ("u1", "item1", "2024-01-03"),
            ("u1", "item1", "2024-01-04"),  # 評価期間に閲覧なし
        ]
        df_no_cv = pd.DataFrame(rows, columns=["user", "item", "datetime"])
        with pytest.raises(ValueError, match="No events"):
            scorer.fit_period(df_no_cv, _OBS_PERIOD, _EVAL_PERIOD)


# ---------------------------------------------------------------------------
# fit_period — 正常系
# ---------------------------------------------------------------------------
class TestFitPeriodResult:
    def test_returns_self(self, scorer, df):
        result = scorer.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert result is scorer

    def test_record_num(self, scorer_fitted):
        # 観測+評価期間の全レコード数
        assert scorer_fitted.record_num == len(_make_df())

    def test_periods_stored(self, scorer_fitted):
        assert scorer_fitted.observation_start_ == pd.Timestamp("2024-01-01").toordinal()
        assert scorer_fitted.observation_end_ == pd.Timestamp("2024-01-07").toordinal()
        assert scorer_fitted.evaluation_start_ == pd.Timestamp("2024-01-08").toordinal()
        assert scorer_fitted.evaluation_end_ == pd.Timestamp("2024-01-14").toordinal()

    def test_explicit_limits_respected(self, scorer_fitted):
        assert scorer_fitted.recency_limit == _RECENCY_LIMIT
        assert scorer_fitted.frequency_limit == _FREQUENCY_LIMIT

    def test_R_range(self, scorer_fitted):
        assert scorer_fitted.R == list(range(1, _RECENCY_LIMIT + 1))

    def test_F_range(self, scorer_fitted):
        assert scorer_fitted.F == list(range(1, _FREQUENCY_LIMIT + 1))

    def test_RF2N_keys(self, scorer_fitted):
        expected = {(r, f) for r in scorer_fitted.R for f in scorer_fitted.F}
        assert set(scorer_fitted.RF2N.keys()) == expected

    def test_known_RF2N_values(self, scorer_fitted):
        # 観測期間の集計: 各 (user,item) ペアが1レコードとしてカウントされる
        assert scorer_fitted.RF2N[3, 3] == 1  # u1-item1
        assert scorer_fitted.RF2N[6, 1] == 1  # u1-item2
        assert scorer_fitted.RF2N[4, 1] == 1  # u2-item1
        assert scorer_fitted.RF2N[1, 2] == 1  # u2-item2

    def test_known_probabilities(self, scorer_fitted):
        # (3,3) u1-item1: cv=1/N=1 → prob=1.0
        assert scorer_fitted.RF2Prob[3, 3] == pytest.approx(1.0)
        # (1,2) u2-item2: cv=1/N=1 → prob=1.0
        assert scorer_fitted.RF2Prob[1, 2] == pytest.approx(1.0)
        # (6,1) u1-item2: cv=0/N=1 → prob=0.0
        assert scorer_fitted.RF2Prob[6, 1] == pytest.approx(0.0)
        # (4,1) u2-item1: cv=0/N=1 → prob=0.0
        assert scorer_fitted.RF2Prob[4, 1] == pytest.approx(0.0)

    def test_empirical_probability_is_dataframe(self, scorer_fitted):
        assert isinstance(scorer_fitted.empirical_probability_, pd.DataFrame)
        assert set(scorer_fitted.empirical_probability_.columns) == {
            "recency",
            "frequency",
            "N",
            "cv",
            "probability",
        }

    def test_empirical_probability_row_count(self, scorer_fitted):
        expected_rows = _RECENCY_LIMIT * _FREQUENCY_LIMIT
        assert len(scorer_fitted.empirical_probability_) == expected_rows

    def test_empirical_probability_dict_keys(self, scorer_fitted):
        expected = {(r, f) for r in scorer_fitted.R for f in scorer_fitted.F}
        assert set(scorer_fitted.empirical_probability_dict_.keys()) == expected

    def test_empirical_probability_table_shape(self, scorer_fitted):
        tbl = scorer_fitted.empirical_probability_table_
        assert tbl.shape == (_RECENCY_LIMIT, _FREQUENCY_LIMIT)

    def test_auto_limits(self, df):
        # 自動決定: recency_limit=3, frequency_limit=3
        s = RecencyFrequencyScorer()
        s.fit_period(df, _OBS_PERIOD, _EVAL_PERIOD)
        assert s.recency_limit == _AUTO_RECENCY_LIMIT
        assert s.frequency_limit == _AUTO_FREQUENCY_LIMIT
        # 上限が実際に適用されていること (limit 超えのキーが存在しないこと)
        for r, f in s.RF2N:
            assert r <= _AUTO_RECENCY_LIMIT
            assert f <= _AUTO_FREQUENCY_LIMIT

    def test_custom_column_names(self):
        df_custom = _make_df().rename(columns={"user": "uid", "item": "iid", "datetime": "ts"})
        s = RecencyFrequencyScorer(user_col="uid", item_col="iid", time_col="ts")
        s.fit_period(
            df_custom,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.record_num == len(df_custom)
        assert s.empirical_probability_dict_ is not None

    def test_does_not_mutate_input(self, df):
        original_columns = list(df.columns)
        original_values = df.values.copy()
        s = RecencyFrequencyScorer()
        s.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert list(df.columns) == original_columns
        assert (df.values == original_values).all()

    def test_er_set_after_fit(self, scorer_fitted):
        assert scorer_fitted.er_probability_ is not None
        assert scorer_fitted.er_probability_dict_ is not None
        assert scorer_fitted.er_probability_table_ is not None

    def test_ef_set_after_fit(self, scorer_fitted):
        assert scorer_fitted.ef_probability_ is not None
        assert scorer_fitted.ef_probability_dict_ is not None
        assert scorer_fitted.ef_probability_table_ is not None

    def test_mr_probability_dict_none_after_fit(self, scorer_fitted):
        # optimize() 前は 1D 属性が None のまま (1D 分離設計の invariant)
        assert scorer_fitted.mr_probability_dict_ is None

    def test_mf_probability_dict_none_after_fit(self, scorer_fitted):
        assert scorer_fitted.mf_probability_dict_ is None

    def test_er_constant_across_frequency(self, scorer_fitted):
        tol = 1e-9
        for r in scorer_fitted.R:
            vals = [scorer_fitted.er_probability_dict_[r, f] for f in scorer_fitted.F]
            assert all(abs(v - vals[0]) < tol for v in vals)

    def test_ef_constant_across_recency(self, scorer_fitted):
        tol = 1e-9
        for f in scorer_fitted.F:
            vals = [scorer_fitted.ef_probability_dict_[r, f] for r in scorer_fitted.R]
            assert all(abs(v - vals[0]) < tol for v in vals)

    def test_er_matches_R2Prob(self, scorer_fitted):
        for r in scorer_fitted.R:
            expected = scorer_fitted.R2Prob[r]
            for f in scorer_fitted.F:
                assert scorer_fitted.er_probability_dict_[r, f] == pytest.approx(expected)

    def test_ef_matches_F2Prob(self, scorer_fitted):
        for f in scorer_fitted.F:
            expected = scorer_fitted.F2Prob[f]
            for r in scorer_fitted.R:
                assert scorer_fitted.ef_probability_dict_[r, f] == pytest.approx(expected)

    def test_er_table_shape(self, scorer_fitted):
        assert scorer_fitted.er_probability_table_.shape == (_RECENCY_LIMIT, _FREQUENCY_LIMIT)

    def test_ef_table_shape(self, scorer_fitted):
        assert scorer_fitted.ef_probability_table_.shape == (_RECENCY_LIMIT, _FREQUENCY_LIMIT)


# ---------------------------------------------------------------------------
# fit_date — バリデーション
# ---------------------------------------------------------------------------
class TestFitDateValidation:
    def test_not_dataframe_raises(self, scorer):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            scorer.fit_date("not_a_df", "2024-01-07")

    def test_missing_time_col_raises(self, scorer, df):
        with pytest.raises(ValueError, match="Missing required columns"):
            scorer.fit_date(df.drop(columns="datetime"), "2024-01-07")

    def test_invalid_target_date_raises(self, scorer, df):
        with pytest.raises(ValueError, match="time value could not be normalized"):
            scorer.fit_date(df, object())


# ---------------------------------------------------------------------------
# fit — バリデーション (新 API: df_obs, df_eval)
# ---------------------------------------------------------------------------
class TestFitValidation:
    def test_obs_not_dataframe_raises(self, scorer, df):
        df_eval = df[pd.to_datetime(df["datetime"]) > "2024-01-07"]
        with pytest.raises(TypeError, match="pandas DataFrame"):
            scorer.fit("not_a_df", df_eval)

    def test_eval_not_dataframe_raises(self, scorer, df):
        df_obs = df[pd.to_datetime(df["datetime"]) <= "2024-01-07"]
        with pytest.raises(TypeError, match="pandas DataFrame"):
            scorer.fit(df_obs, "not_a_df")

    def test_missing_col_in_obs_raises(self, scorer, df):
        df_obs = df[pd.to_datetime(df["datetime"]) <= "2024-01-07"].drop(columns="item")
        df_eval = df[pd.to_datetime(df["datetime"]) > "2024-01-07"]
        with pytest.raises(ValueError, match="df_obs"):
            scorer.fit(df_obs, df_eval)

    def test_missing_col_in_eval_raises(self, scorer, df):
        df_obs = df[pd.to_datetime(df["datetime"]) <= "2024-01-07"]
        df_eval = df[pd.to_datetime(df["datetime"]) > "2024-01-07"].drop(columns="item")
        with pytest.raises(ValueError, match="df_eval"):
            scorer.fit(df_obs, df_eval)

    def test_invalid_ref_raises(self, scorer, df):
        df_obs = df[pd.to_datetime(df["datetime"]) <= "2024-01-07"]
        df_eval = df[pd.to_datetime(df["datetime"]) > "2024-01-07"]
        with pytest.raises(ValueError, match="time value could not be normalized"):
            scorer.fit(df_obs, df_eval, ref=object())


# ---------------------------------------------------------------------------
# fit_date — 正常系
# ---------------------------------------------------------------------------
class TestFitDateResult:
    def test_returns_self(self, scorer, df):
        result = scorer.fit_date(
            df, _FIT_TARGET_DATE, recency_limit=_RECENCY_LIMIT, frequency_limit=_FREQUENCY_LIMIT
        )
        assert result is scorer

    def test_periods_match_target_date(self, df):
        s = RecencyFrequencyScorer()
        s.fit_date(
            df, _FIT_TARGET_DATE, recency_limit=_RECENCY_LIMIT, frequency_limit=_FREQUENCY_LIMIT
        )
        assert s.observation_end_ == pd.Timestamp(_FIT_TARGET_DATE).toordinal()
        assert s.evaluation_start_ == pd.Timestamp("2024-01-08").toordinal()

    def test_observation_start_bounded_by_observation_days(self, df):
        # observation_days=3 → obs_start = max(Jan01, Jan07-3d) = Jan04
        s = RecencyFrequencyScorer()
        s.fit_date(
            df,
            _FIT_TARGET_DATE,
            observation_days=3,
            evaluation_days=None,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.observation_start_ == pd.Timestamp("2024-01-04").toordinal()

    def test_observation_days_none_uses_df_min(self, df):
        s = RecencyFrequencyScorer()
        s.fit_date(
            df,
            _FIT_TARGET_DATE,
            observation_days=None,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.observation_start_ == pd.Timestamp("2024-01-01").toordinal()

    def test_evaluation_end_bounded_by_evaluation_days(self, df):
        # evaluation_days=2 → eval_end = min(Jan10, Jan07+2d) = Jan09
        s = RecencyFrequencyScorer()
        s.fit_date(
            df,
            _FIT_TARGET_DATE,
            evaluation_days=2,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.evaluation_end_ == pd.Timestamp("2024-01-09").toordinal()

    def test_evaluation_days_none_uses_df_max(self, df):
        s = RecencyFrequencyScorer()
        s.fit_date(
            df,
            _FIT_TARGET_DATE,
            evaluation_days=None,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.evaluation_end_ == pd.Timestamp("2024-01-10").toordinal()

    def test_same_result_as_fit_period(self, df):
        # fit_date(target_date, obs_days=None, eval_days=None) == fit_period(full range)
        s1 = RecencyFrequencyScorer()
        s1.fit_date(
            df,
            _FIT_TARGET_DATE,
            observation_days=None,
            evaluation_days=None,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        s2 = RecencyFrequencyScorer()
        s2.fit_period(
            df,
            _OBS_PERIOD,
            ("2024-01-08", "2024-01-10"),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s1.empirical_probability_dict_ == s2.empirical_probability_dict_

    def test_empirical_probability_dict_populated(self, df):
        s = RecencyFrequencyScorer()
        s.fit_date(
            df, _FIT_TARGET_DATE, recency_limit=_RECENCY_LIMIT, frequency_limit=_FREQUENCY_LIMIT
        )
        assert s.empirical_probability_dict_ is not None


# ---------------------------------------------------------------------------
# fit — 正常系 (新 API: df_obs, df_eval)
# ---------------------------------------------------------------------------
class TestFitResult:
    def _make_obs(self):
        df = _make_df()
        return df[pd.to_datetime(df["datetime"]) <= _OBS_PERIOD[1]]

    def _make_eval(self):
        df = _make_df()
        return df[pd.to_datetime(df["datetime"]) >= _EVAL_PERIOD[0]]

    def test_returns_self(self, scorer):
        result = scorer.fit(
            self._make_obs(),
            self._make_eval(),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert result is scorer

    def test_same_probabilities_as_fit_period(self):
        # fit(df_obs, df_eval) == fit_period(df, obs_period, eval_period)
        s1 = RecencyFrequencyScorer()
        s1.fit(
            self._make_obs(),
            self._make_eval(),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        s2 = RecencyFrequencyScorer()
        s2.fit_period(
            _make_df(),
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s1.empirical_probability_dict_ == s2.empirical_probability_dict_

    def test_ref_date_default_is_obs_max(self):
        # ref=None → df_obs の最大日 (2024-01-07) が observation_end_ に ordinal で格納される
        s = RecencyFrequencyScorer()
        s.fit(
            self._make_obs(),
            self._make_eval(),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.observation_end_ == pd.Timestamp("2024-01-07").toordinal()

    def test_ref_date_explicit(self):
        s = RecencyFrequencyScorer()
        s.fit(
            self._make_obs(),
            self._make_eval(),
            ref="2024-01-07",
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.observation_end_ == pd.Timestamp("2024-01-07").toordinal()

    def test_observation_start_from_data(self):
        s = RecencyFrequencyScorer()
        s.fit(
            self._make_obs(),
            self._make_eval(),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.observation_start_ == pd.Timestamp("2024-01-01").toordinal()

    def test_evaluation_dates_from_data(self):
        s = RecencyFrequencyScorer()
        s.fit(
            self._make_obs(),
            self._make_eval(),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.evaluation_start_ == pd.Timestamp("2024-01-09").toordinal()
        assert s.evaluation_end_ == pd.Timestamp("2024-01-10").toordinal()

    def test_empirical_probability_dict_populated(self):
        s = RecencyFrequencyScorer()
        s.fit(
            self._make_obs(),
            self._make_eval(),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.empirical_probability_dict_ is not None


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------
class TestOptimize:
    def test_before_fit_raises(self, scorer):
        with pytest.raises(RuntimeError, match="fit"):
            scorer.optimize(kind="mono")

    def test_invalid_kind_raises(self, scorer, df):
        scorer.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(ValueError, match="kind"):
            scorer.optimize(kind="invalid")

    def test_optimize_mono_returns_self(self, scorer, df):
        scorer.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        result = scorer.optimize(kind="mono")
        assert result is scorer

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
        expected = {(r, f) for r in scorer_optimized_mono.R for f in scorer_optimized_mono.F}
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
        expected = set(scorer_optimized_mr.R)
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
        expected = set(scorer_optimized_mf.F)
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
        expected = {(r, f) for r in scorer_optimized_mrc.R for f in scorer_optimized_mrc.F}
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
        expected = {(r, f) for r in scorer_optimized_mfc.R for f in scorer_optimized_mfc.F}
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
        expected = {(r, f) for r in scorer_optimized_mcc.R for f in scorer_optimized_mcc.F}
        assert set(d.keys()) == expected

    def test_mcc_probability_values_in_bounds(self, scorer_optimized_mcc):
        tol = 1e-6
        for val in scorer_optimized_mcc.mcc_probability_dict_.values():
            assert -tol <= val <= 1 + tol

    # --- eps ---

    def test_optimize_eps_negative_raises(self, scorer, df):
        scorer.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(ValueError, match="eps"):
            scorer.optimize(kind="mono", eps=-1e-6)

    def test_optimize_eps_too_large_mono_raises(self, scorer, df):
        # eps_max(2D) = max(RF2Prob) / min(nr-1, nf-1) = 1.0 / min(6, 2) = 0.5
        scorer.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(ValueError, match="eps"):
            scorer.optimize(kind="mono", eps=0.5 + 1e-9)

    def test_optimize_eps_too_large_mr_raises(self, scorer, df):
        # eps_max(mr) = max(R2Prob) / (nr - 1) = 1.0 / 6 ≈ 0.1667
        scorer.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(ValueError, match="eps"):
            scorer.optimize(kind="mr", eps=1.0 / 6 + 1e-9)

    def test_optimize_eps_too_large_mf_raises(self, scorer, df):
        # eps_max(mf) = max(F2Prob) / (nf - 1) = 1.0 / 2 = 0.5
        scorer.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(ValueError, match="eps"):
            scorer.optimize(kind="mf", eps=0.5 + 1e-9)

    def test_optimize_eps_too_large_mrc_raises(self, scorer, df):
        # 2D eps 上限は mono と共通: min(p_max/(nr-1), p_max/(nf-1)) = min(1.0/6, 1.0/2) = 1.0/6
        scorer.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(ValueError, match="eps"):
            scorer.optimize(kind="mrc", eps=0.5 + 1e-9)

    def test_optimize_with_eps_mono_produces_results(self, scorer, df):
        scorer.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        scorer.optimize(kind="mono", eps=1e-4)
        assert scorer.mono_probability_dict_ is not None

    def test_optimize_with_eps_mr_produces_results(self, scorer, df):
        scorer.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        scorer.optimize(kind="mr", eps=1e-4)
        assert scorer.mr_probability_dict_ is not None

    def test_optimize_with_eps_mf_produces_results(self, scorer, df):
        scorer.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
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
        s.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
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
        scorer.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(RuntimeError, match="optimize"):
            scorer.predict(1, 1, kind="mono")

    def test_before_optimize_mrc_raises(self, scorer, df):
        scorer.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(RuntimeError, match="optimize"):
            scorer.predict(1, 1, kind="mrc")

    def test_before_optimize_mfc_raises(self, scorer, df):
        scorer.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(RuntimeError, match="optimize"):
            scorer.predict(1, 1, kind="mfc")

    def test_before_optimize_mcc_raises(self, scorer, df):
        scorer.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(RuntimeError, match="optimize"):
            scorer.predict(1, 1, kind="mcc")

    def test_before_optimize_mr_raises(self, scorer, df):
        scorer.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(RuntimeError, match="optimize"):
            scorer.predict(1, 1, kind="mr")

    def test_before_optimize_mf_raises(self, scorer, df):
        scorer.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
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
        for f in scorer_fitted.F:
            expected = scorer_fitted.empirical_probability_dict_[_RECENCY_LIMIT, f]
            assert scorer_fitted.predict(r_over, f) == pytest.approx(expected)

    def test_clamps_f_to_frequency_limit(self, scorer_fitted):
        # dict と直接比較してクランプが機能していることを検証
        f_over = _FREQUENCY_LIMIT + 10
        for r in scorer_fitted.R:
            expected = scorer_fitted.empirical_probability_dict_[r, _FREQUENCY_LIMIT]
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
        assert prob == pytest.approx(scorer_fitted.er_probability_dict_[1, 1])

    def test_ef_kind(self, scorer_fitted):
        prob = scorer_fitted.predict(1, 1, kind="ef")
        assert isinstance(prob, float)
        assert prob == pytest.approx(scorer_fitted.ef_probability_dict_[1, 1])

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

    def test_same_result_as_transform_date(self, scorer_fitted):
        # transform(df_obs, ref) と transform_date(df, target_date) が一致
        df = _make_df()
        result_new = scorer_fitted.transform(self._make_obs(), ref=_FIT_TARGET_DATE)
        result_old = scorer_fitted.transform_date(df, _FIT_TARGET_DATE)
        assert list(result_new.sort_values(["user", "item"])["probability"]) == pytest.approx(
            list(result_old.sort_values(["user", "item"])["probability"])
        )

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

    def test_clamps_recency_above_limit(self, scorer_fitted):
        # recency > recency_limit の行は limit にクランプして確率を引く
        # 2023-12-30 → ref_date Jan07 との差 = 8日 → recency = 9 > limit=7
        extra = pd.DataFrame([("u3", "item3", "2023-12-30")], columns=["user", "item", "datetime"])
        df_obs = pd.concat([self._make_obs(), extra], ignore_index=True)
        result = scorer_fitted.transform(df_obs, ref=_FIT_TARGET_DATE)
        u3_row = result[(result["user"] == "u3") & (result["item"] == "item3")].iloc[0]
        assert u3_row["recency"] == 9  # 出力の recency は元値を保持
        expected = scorer_fitted.empirical_probability_dict_[_RECENCY_LIMIT, 1]
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
        expected = scorer_fitted.empirical_probability_dict_[1, _FREQUENCY_LIMIT]
        assert u3_row["probability"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# transform_date — 後方互換 API (target_date でフィルタ)
# ---------------------------------------------------------------------------
class TestTransformDate:
    def test_before_fit_raises(self, scorer, df):
        with pytest.raises(RuntimeError, match="fit"):
            scorer.transform_date(df, "2024-01-07")

    def test_uses_init_col_names_by_default(self):
        df_custom = _make_df().rename(columns={"user": "uid", "item": "iid", "datetime": "ts"})
        s = RecencyFrequencyScorer(user_col="uid", item_col="iid", time_col="ts")
        s.fit_period(
            df_custom,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        result = s.transform_date(df_custom, "2024-01-07")
        assert "uid" in result.columns
        assert "iid" in result.columns

    def test_default_col_names_work_without_args(self, scorer_fitted, df):
        result = scorer_fitted.transform_date(df, "2024-01-07")
        assert "user" in result.columns
        assert "item" in result.columns

    def test_explicit_col_names_override_init(self):
        df_a = _make_df()
        df_b = _make_df().rename(columns={"user": "uid", "item": "iid", "datetime": "ts"})
        s = RecencyFrequencyScorer()
        s.fit_period(
            df_a,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        result = s.transform_date(df_b, "2024-01-07", user_col="uid", item_col="iid", time_col="ts")
        assert "uid" in result.columns
        assert "iid" in result.columns

    def test_returns_dataframe_with_expected_columns(self, scorer_fitted, df):
        result = scorer_fitted.transform_date(df, "2024-01-07")
        assert set(result.columns) == {
            "user",
            "item",
            "recency",
            "frequency",
            "probability",
            "order",
        }

    def test_sorted_by_user_and_probability(self, scorer_fitted, df):
        result = scorer_fitted.transform_date(df, "2024-01-07")
        for user, grp in result.groupby("user"):
            assert list(grp["probability"]) == sorted(grp["probability"], reverse=True)
            assert list(grp["order"]) == list(range(1, len(grp) + 1))

    def test_emp_alias_equals_empirical(self, scorer_fitted, df):
        result_emp = scorer_fitted.transform_date(df, "2024-01-07", kind="emp")
        result_empirical = scorer_fitted.transform_date(df, "2024-01-07", kind="empirical")
        assert list(result_emp["probability"]) == list(result_empirical["probability"])

    def test_target_date_is_inclusive(self, scorer_fitted, df):
        # target_date 当日の行が含まれること
        # u2-item2 は Jan06 と Jan07 に閲覧 → target_date="2024-01-07" なら frequency=2
        result = scorer_fitted.transform_date(df, "2024-01-07")
        u2_item2 = result[(result["user"] == "u2") & (result["item"] == "item2")].iloc[0]
        assert u2_item2["frequency"] == 2

    def test_invalid_target_date_raises(self, scorer_fitted, df):
        with pytest.raises((ValueError, Exception)):
            scorer_fitted.transform_date(df, object())


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
# テストデータの期待値 (scorer_fitted: recency_limit=7, frequency_limit=3)
#
# transform_date(df, "2024-01-07") の結果:
#   u1: item1(r=3,f=3,prob=1.0,order=1), item2(r=6,f=1,prob=0.0,order=2)
#   u2: item2(r=1,f=2,prob=1.0,order=1), item1(r=4,f=1,prob=0.0,order=2)
#
# df_eval: u1-item1(Jan09), u2-item2(Jan10) → 対象イベント発生ペア数=2
#
# order=1: UIrec={(u1,item1),(u2,item2)}, n_hit=2, n_rec=2
#   precision=1.0, recall=1.0, f1=1.0, recall_norm=1.0, f1_norm=1.0
# order=2 (=order_max): UIrec=all 4 pairs, n_hit=2, n_rec=4
#   precision=0.5, recall=1.0, f1≈0.667
# ---------------------------------------------------------------------------
class TestEvaluate:
    def test_returns_dataframe(self, scorer_fitted, df_rec, df_eval):
        result = scorer_fitted.evaluate(df_rec, df_eval, order=1)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, scorer_fitted, df_rec, df_eval):
        result = scorer_fitted.evaluate(df_rec, df_eval, order=1)
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

    def test_row_count_with_order_1(self, scorer_fitted, df_rec, df_eval):
        # order=1 でも order_max(=2) の行が追加されるので 2 行
        result = scorer_fitted.evaluate(df_rec, df_eval, order=1)
        assert len(result) == 2

    def test_order_max_always_included(self, scorer_fitted, df_rec, df_eval):
        result = scorer_fitted.evaluate(df_rec, df_eval, order=1)
        assert df_rec["order"].max() in result["order"].values

    def test_n_recommended_at_order1(self, scorer_fitted, df_rec, df_eval):
        result = scorer_fitted.evaluate(df_rec, df_eval, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["n_recommended"] == 2  # 2ユーザー × 1推薦

    def test_n_hit_at_order1(self, scorer_fitted, df_rec, df_eval):
        result = scorer_fitted.evaluate(df_rec, df_eval, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["n_hit"] == 2

    def test_precision_at_order1(self, scorer_fitted, df_rec, df_eval):
        result = scorer_fitted.evaluate(df_rec, df_eval, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["precision"] == pytest.approx(1.0)

    def test_recall_at_order1(self, scorer_fitted, df_rec, df_eval):
        result = scorer_fitted.evaluate(df_rec, df_eval, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["recall"] == pytest.approx(1.0)

    def test_f1_at_order1(self, scorer_fitted, df_rec, df_eval):
        result = scorer_fitted.evaluate(df_rec, df_eval, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["f1"] == pytest.approx(1.0)

    def test_precision_at_order_max(self, scorer_fitted, df_rec, df_eval):
        # order=2: 4推薦中2ヒット → precision=0.5
        result = scorer_fitted.evaluate(df_rec, df_eval, order=2)
        row = result[result["order"] == 2].iloc[0]
        assert row["precision"] == pytest.approx(0.5)

    def test_recall_norm_with_unseen_events(self, scorer_fitted, df_rec, df_eval):
        # df_eval に存在しないペアを行追加すると recall < recall_norm
        extra = pd.DataFrame([("u3", "item3", "2024-01-11")], columns=["user", "item", "datetime"])
        df_eval_extended = pd.concat([df_eval, extra], ignore_index=True)
        result = scorer_fitted.evaluate(df_rec, df_eval_extended, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["recall"] == pytest.approx(2 / 3)
        assert row["recall_norm"] == pytest.approx(1.0)

    def test_f1_norm_with_unseen_events(self, scorer_fitted, df_rec, df_eval):
        extra = pd.DataFrame([("u3", "item3", "2024-01-11")], columns=["user", "item", "datetime"])
        df_eval_extended = pd.concat([df_eval, extra], ignore_index=True)
        result = scorer_fitted.evaluate(df_rec, df_eval_extended, order=1)
        row = result[result["order"] == 1].iloc[0]
        # recall_norm=1.0, precision=1.0 → f1_norm=1.0 > f1
        assert row["f1_norm"] == pytest.approx(1.0)
        assert row["f1"] < row["f1_norm"]

    def test_uses_init_col_names_by_default(self):
        df_custom = _make_df().rename(columns={"user": "uid", "item": "iid", "datetime": "ts"})
        s = RecencyFrequencyScorer(user_col="uid", item_col="iid", time_col="ts")
        s.fit_period(
            df_custom,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        df_rec_custom = s.transform_date(df_custom, _FIT_TARGET_DATE)
        df_eval_custom = df_custom[pd.to_datetime(df_custom["ts"]) >= _EVAL_PERIOD[0]]
        result = s.evaluate(df_rec_custom, df_eval_custom)  # user_col/item_col 省略
        assert isinstance(result, pd.DataFrame)

    def test_explicit_col_override(self, scorer_fitted, df_rec, df_eval):
        df_rec_renamed = df_rec.rename(columns={"user": "uid", "item": "iid"})
        df_eval_renamed = df_eval.rename(columns={"user": "uid", "item": "iid"})
        result = scorer_fitted.evaluate(
            df_rec_renamed, df_eval_renamed, order=1, user_col="uid", item_col="iid"
        )
        assert isinstance(result, pd.DataFrame)

    def test_invalid_df_eval_type_raises(self, scorer_fitted, df_rec):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            scorer_fitted.evaluate(df_rec, 12345, order=1)

    def test_missing_col_in_df_eval_raises(self, scorer_fitted, df_rec, df_eval):
        with pytest.raises(ValueError, match="df_eval"):
            scorer_fitted.evaluate(df_rec, df_eval.drop(columns="item"), order=1)


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
        import matplotlib.figure

        fig = scorer_fitted.plot_probability_surface(kind="emp")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_mono(self, scorer_optimized_mono):
        import matplotlib.figure

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
        import matplotlib.figure

        fig = scorer_optimized_mrc.plot_probability_surface(kind="mrc")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_mfc(self, scorer_optimized_mfc):
        import matplotlib.figure

        fig = scorer_optimized_mfc.plot_probability_surface(kind="mfc")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_mcc(self, scorer_optimized_mcc):
        import matplotlib.figure

        fig = scorer_optimized_mcc.plot_probability_surface(kind="mcc")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_er(self, scorer_fitted):
        import matplotlib.figure

        fig = scorer_fitted.plot_probability_surface(kind="er")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_ef(self, scorer_fitted):
        import matplotlib.figure

        fig = scorer_fitted.plot_probability_surface(kind="ef")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figsize_applied(self, scorer_fitted):
        fig = scorer_fitted.plot_probability_surface(figsize=(4, 3))
        assert tuple(fig.get_size_inches()) == (4, 3)

    def test_title_shown_when_set(self, scorer_fitted):
        fig = scorer_fitted.plot_probability_surface(title="My Title")
        ax = fig.axes[0]
        assert ax.get_title() == "My Title"

    def test_title_empty_when_none(self, scorer_fitted):
        fig = scorer_fitted.plot_probability_surface(title=None)
        ax = fig.axes[0]
        assert ax.get_title() == ""

    def test_fontsize_applied_to_labels(self, scorer_fitted):
        fig = scorer_fitted.plot_probability_surface(fontsize=16)
        ax = fig.axes[0]
        assert ax.xaxis.label.get_size() == 16
        assert ax.yaxis.label.get_size() == 16


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
        assert scorer_fitted.R2N[1] == pytest.approx(1.0)
        assert scorer_fitted.R2N[3] == pytest.approx(1.0)
        assert scorer_fitted.R2N[4] == pytest.approx(1.0)
        assert scorer_fitted.R2N[6] == pytest.approx(1.0)
        assert scorer_fitted.R2N[2] == pytest.approx(0.0)

    def test_r2cv_known_values(self, scorer_fitted):
        assert scorer_fitted.R2CV[1] == pytest.approx(1.0)
        assert scorer_fitted.R2CV[3] == pytest.approx(1.0)
        assert scorer_fitted.R2CV[4] == pytest.approx(0.0)
        assert scorer_fitted.R2CV[6] == pytest.approx(0.0)

    def test_r2prob_known_values(self, scorer_fitted):
        assert scorer_fitted.R2Prob[1] == pytest.approx(1.0)
        assert scorer_fitted.R2Prob[3] == pytest.approx(1.0)
        assert scorer_fitted.R2Prob[4] == pytest.approx(0.0)
        assert scorer_fitted.R2Prob[6] == pytest.approx(0.0)

    def test_r2n_keys_match_r(self, scorer_fitted):
        assert set(scorer_fitted.R2N.keys()) == set(scorer_fitted.R)

    def test_f2n_known_values(self, scorer_fitted):
        assert scorer_fitted.F2N[1] == pytest.approx(2.0)
        assert scorer_fitted.F2N[2] == pytest.approx(1.0)
        assert scorer_fitted.F2N[3] == pytest.approx(1.0)

    def test_f2cv_known_values(self, scorer_fitted):
        assert scorer_fitted.F2CV[1] == pytest.approx(0.0)
        assert scorer_fitted.F2CV[2] == pytest.approx(1.0)
        assert scorer_fitted.F2CV[3] == pytest.approx(1.0)

    def test_f2prob_known_values(self, scorer_fitted):
        assert scorer_fitted.F2Prob[1] == pytest.approx(0.0)
        assert scorer_fitted.F2Prob[2] == pytest.approx(1.0)
        assert scorer_fitted.F2Prob[3] == pytest.approx(1.0)

    def test_f2n_keys_match_f(self, scorer_fitted):
        assert set(scorer_fitted.F2N.keys()) == set(scorer_fitted.F)

    def test_recency_probability_is_dataframe(self, scorer_fitted):
        df = scorer_fitted.recency_probability_
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"recency", "N", "cv", "probability"}

    def test_recency_probability_row_count(self, scorer_fitted):
        assert len(scorer_fitted.recency_probability_) == _RECENCY_LIMIT

    def test_frequency_probability_is_dataframe(self, scorer_fitted):
        df = scorer_fitted.frequency_probability_
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"frequency", "N", "cv", "probability"}

    def test_frequency_probability_row_count(self, scorer_fitted):
        assert len(scorer_fitted.frequency_probability_) == _FREQUENCY_LIMIT

    def test_r2n_f2n_sum_equals_record_num_target(self, scorer_fitted):
        # R2N の合計 == F2N の合計 == 分析対象レコード数
        assert sum(scorer_fitted.R2N.values()) == pytest.approx(scorer_fitted.record_num_target)
        assert sum(scorer_fitted.F2N.values()) == pytest.approx(scorer_fitted.record_num_target)


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

    def test_invalid_axis_raises(self, scorer_fitted):
        with pytest.raises(ValueError, match="axis"):
            scorer_fitted.plot_marginal_probability(axis="invalid")

    def test_returns_figure_recency(self, scorer_fitted):
        import matplotlib.figure

        fig = scorer_fitted.plot_marginal_probability(axis="recency")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_frequency(self, scorer_fitted):
        import matplotlib.figure

        fig = scorer_fitted.plot_marginal_probability(axis="frequency")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figsize_applied(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(figsize=(4, 3))
        assert tuple(fig.get_size_inches()) == (4, 3)

    def test_title_shown_when_set(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(title="My Title")
        ax = fig.axes[0]
        assert ax.get_title() == "My Title"

    def test_title_empty_when_none(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(title=None)
        ax = fig.axes[0]
        assert ax.get_title() == ""

    def test_fontsize_applied_to_labels(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(fontsize=16)
        ax = fig.axes[0]
        assert ax.xaxis.label.get_size() == 16
        assert ax.yaxis.label.get_size() == 16

    def test_invalid_kind_raises(self, scorer_fitted):
        with pytest.raises(ValueError, match="kind"):
            scorer_fitted.plot_marginal_probability(axis="recency", kind="invalid")

    def test_mf_on_recency_axis_raises(self, scorer_fitted):
        with pytest.raises(ValueError, match="kind='mf'"):
            scorer_fitted.plot_marginal_probability(axis="recency", kind="mf")

    def test_mr_on_frequency_axis_raises(self, scorer_fitted):
        with pytest.raises(ValueError, match="kind='mr'"):
            scorer_fitted.plot_marginal_probability(axis="frequency", kind="mr")

    def test_mr_before_optimize_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.plot_marginal_probability(axis="recency", kind="mr")

    def test_mf_before_optimize_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.plot_marginal_probability(axis="frequency", kind="mf")

    def test_mr_returns_figure(self, scorer_optimized_mr):
        import matplotlib.figure

        fig = scorer_optimized_mr.plot_marginal_probability(axis="recency", kind="mr")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_mf_returns_figure(self, scorer_optimized_mf):
        import matplotlib.figure

        fig = scorer_optimized_mf.plot_marginal_probability(axis="frequency", kind="mf")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_all_recency_returns_figure_with_legend(self, scorer_optimized_mr):
        import matplotlib.figure

        fig = scorer_optimized_mr.plot_marginal_probability(axis="recency", kind="all")
        assert isinstance(fig, matplotlib.figure.Figure)
        ax = fig.axes[0]
        assert ax.get_legend() is not None

    def test_all_frequency_returns_figure_with_legend(self, scorer_optimized_mf):
        import matplotlib.figure

        fig = scorer_optimized_mf.plot_marginal_probability(axis="frequency", kind="all")
        assert isinstance(fig, matplotlib.figure.Figure)
        ax = fig.axes[0]
        assert ax.get_legend() is not None

    def test_emp_has_no_legend(self, scorer_fitted):
        fig = scorer_fitted.plot_marginal_probability(axis="recency", kind="emp")
        ax = fig.axes[0]
        assert ax.get_legend() is None


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
        assert (tmp_path / "emp_probability.csv").exists()

    def test_explicit_file_path(self, scorer_fitted, tmp_path):
        out = tmp_path / "my_output.csv"
        scorer_fitted.export_probability_csv(kind="emp", path=str(out))
        assert out.exists()

    def test_directory_path_creates_file_inside(self, scorer_fitted, tmp_path):
        scorer_fitted.export_probability_csv(kind="emp", path=str(tmp_path))
        assert (tmp_path / "emp_probability.csv").exists()

    def test_emp_output_columns(self, scorer_fitted, tmp_path):
        out = tmp_path / "emp.csv"
        scorer_fitted.export_probability_csv(kind="emp", path=str(out))
        df = pd.read_csv(out)
        assert set(df.columns) == {"recency", "frequency", "N", "cv", "probability"}

    def test_er_output_columns(self, scorer_fitted, tmp_path):
        out = tmp_path / "er.csv"
        scorer_fitted.export_probability_csv(kind="er", path=str(out))
        df = pd.read_csv(out)
        assert set(df.columns) == {"recency", "frequency", "probability"}

    def test_ef_output_columns(self, scorer_fitted, tmp_path):
        out = tmp_path / "ef.csv"
        scorer_fitted.export_probability_csv(kind="ef", path=str(out))
        df = pd.read_csv(out)
        assert set(df.columns) == {"recency", "frequency", "probability"}

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
            "empirical_probability",
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
# _normalize_ref — T14
# ---------------------------------------------------------------------------
class TestNormalizeRef:
    @pytest.fixture
    def scorer(self):
        return RecencyFrequencyScorer()

    def test_string_date(self, scorer):
        expected = pd.Timestamp("2024-01-01").toordinal()
        assert scorer._normalize_ref("2024-01-01") == expected

    def test_timestamp(self, scorer):
        ts = pd.Timestamp("2024-01-01")
        assert scorer._normalize_ref(ts) == ts.toordinal()

    def test_int(self, scorer):
        assert scorer._normalize_ref(100) == 100

    def test_float(self, scorer):
        assert scorer._normalize_ref(100.9) == 100

    def test_numpy_int(self, scorer):
        import numpy as np

        assert scorer._normalize_ref(np.int64(42)) == 42

    def test_numpy_float(self, scorer):
        import numpy as np

        assert scorer._normalize_ref(np.float64(7.0)) == 7

    def test_python_datetime(self, scorer):
        import datetime

        dt = datetime.datetime(2024, 1, 1)
        expected = pd.Timestamp("2024-01-01").toordinal()
        assert scorer._normalize_ref(dt) == expected

    def test_invalid_type_raises(self, scorer):
        with pytest.raises(ValueError, match="time value could not be normalized"):
            scorer._normalize_ref(object())


# ---------------------------------------------------------------------------
# _normalize_sequence_col — T15
# ---------------------------------------------------------------------------
class TestNormalizeSequenceCol:
    @pytest.fixture
    def scorer(self):
        return RecencyFrequencyScorer()

    def test_datetime64_col(self, scorer):
        s = pd.Series(pd.to_datetime(["2024-01-01", "2024-01-07"]))
        result = scorer._normalize_sequence_col(s)
        assert list(result) == [
            pd.Timestamp("2024-01-01").toordinal(),
            pd.Timestamp("2024-01-07").toordinal(),
        ]

    def test_string_date_col(self, scorer):
        s = pd.Series(["2024-01-01", "2024-01-07"])
        result = scorer._normalize_sequence_col(s)
        assert list(result) == [
            pd.Timestamp("2024-01-01").toordinal(),
            pd.Timestamp("2024-01-07").toordinal(),
        ]

    def test_int_col(self, scorer):
        s = pd.Series([1, 7, 100])
        result = scorer._normalize_sequence_col(s)
        assert list(result) == [1, 7, 100]

    def test_float_col(self, scorer):
        s = pd.Series([1.0, 7.5, 100.9])
        result = scorer._normalize_sequence_col(s)
        assert list(result) == [1, 7, 100]

    def test_invalid_dtype_raises(self, scorer):
        s = pd.Series([object(), object()])
        with pytest.raises(ValueError, match="time_col must be datetime or integer type"):
            scorer._normalize_sequence_col(s)


# ---------------------------------------------------------------------------
# 整数入力での fit / fit_date / fit_period — T16
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
        assert isinstance(s.empirical_probability_dict_, dict)

    def test_fit_date_with_integer_col(self):
        df = self._make_int_df()
        target_ord = pd.Timestamp(_FIT_TARGET_DATE).toordinal()
        s = RecencyFrequencyScorer(time_col="seq")
        s.fit_date(
            df,
            target_ord,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert isinstance(s.empirical_probability_dict_, dict)

    def test_fit_period_with_integer_col(self):
        df = self._make_int_df()
        obs_start = pd.Timestamp(_OBS_PERIOD[0]).toordinal()
        obs_end = pd.Timestamp(_OBS_PERIOD[1]).toordinal()
        eval_start = pd.Timestamp(_EVAL_PERIOD[0]).toordinal()
        eval_end = pd.Timestamp(_EVAL_PERIOD[1]).toordinal()
        s = RecencyFrequencyScorer(time_col="seq")
        s.fit_period(
            df,
            (obs_start, obs_end),
            (eval_start, eval_end),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert isinstance(s.empirical_probability_dict_, dict)

    def test_integer_and_datetime_produce_same_rf_distribution(self):
        """整数入力と日付入力で同一の RF 分布が得られること。"""
        df_date = _make_df()
        df_int = self._make_int_df()

        s_date = RecencyFrequencyScorer()
        s_date.fit_period(
            df_date,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )

        obs_start = pd.Timestamp(_OBS_PERIOD[0]).toordinal()
        obs_end = pd.Timestamp(_OBS_PERIOD[1]).toordinal()
        eval_start = pd.Timestamp(_EVAL_PERIOD[0]).toordinal()
        eval_end = pd.Timestamp(_EVAL_PERIOD[1]).toordinal()
        s_int = RecencyFrequencyScorer(time_col="seq")
        s_int.fit_period(
            df_int,
            (obs_start, obs_end),
            (eval_start, eval_end),
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )

        assert s_date.empirical_probability_dict_ == s_int.empirical_probability_dict_


# ---------------------------------------------------------------------------
# unit パラメータ — T17
# ---------------------------------------------------------------------------
class TestUnit:
    def _make_scorer_with_unit(self, unit):
        df = _make_df()
        s = RecencyFrequencyScorer(unit=unit)
        s.fit_period(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
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
    def test_show_does_not_raise(self, scorer_fitted, capsys):
        scorer_fitted.show()

    def test_show_outputs_profiling(self, scorer_fitted, capsys):
        scorer_fitted.show()
        out = capsys.readouterr().out
        assert "profiling" in out

    def test_show_outputs_period_info(self, scorer_fitted, capsys):
        scorer_fitted.show()
        out = capsys.readouterr().out
        assert "observation" in out
        assert "evaluation" in out

    def test_show_outputs_limits(self, scorer_fitted, capsys):
        scorer_fitted.show()
        out = capsys.readouterr().out
        assert "recency_limit" in out
        assert "frequency_limit" in out
