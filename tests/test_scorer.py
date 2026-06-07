import pandas as pd
import pytest

from rfscorer import RecencyFrequencyScorer

# ---------------------------------------------------------------------------
# テストデータ
#
# 観測期間: 2024-01-01 〜 2024-01-07 (obs_end = Jan07)
# 評価期間: 2024-01-08 〜 2024-01-14
#
# obs_end 基準の Recency = (obs_end - datetime).days + 1
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

_UIREVISIT = {("u1", "item1"), ("u2", "item2")}  # 評価期間に再閲覧したペア


def _make_df():
    rows = [
        ("u1", "item1", "2024-01-01"),
        ("u1", "item1", "2024-01-03"),
        ("u1", "item1", "2024-01-05"),
        ("u1", "item2", "2024-01-02"),
        ("u2", "item1", "2024-01-04"),
        ("u2", "item2", "2024-01-06"),
        ("u2", "item2", "2024-01-07"),
        # 評価期間: u1→item1, u2→item2 を再閲覧
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
def df_rec(scorer_fitted):
    return scorer_fitted.transform(_make_df(), _FIT_TARGET_DATE)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
class TestInit:
    def test_default_column_names(self, scorer):
        assert scorer.user_col == "user"
        assert scorer.item_col == "item"
        assert scorer.datetime_col == "datetime"

    def test_custom_column_names(self):
        s = RecencyFrequencyScorer(user_col="uid", item_col="iid", datetime_col="ts")
        assert s.user_col == "uid"
        assert s.item_col == "iid"
        assert s.datetime_col == "ts"

    def test_initial_state(self, scorer):
        assert scorer.R == []
        assert scorer.F == []
        assert scorer.RF2N == {}
        assert scorer.empirical_probability_ is None
        assert scorer.empirical_probability_dict_ is None
        assert scorer.record_num is None


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
        # 評価期間に再閲覧なし → 自動上限計算で ValueError
        rows = [
            ("u1", "item1", "2024-01-03"),
            ("u1", "item1", "2024-01-04"),  # 評価期間に閲覧なし
        ]
        df_no_cv = pd.DataFrame(rows, columns=["user", "item", "datetime"])
        with pytest.raises(ValueError, match="No revisits"):
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
        assert scorer_fitted.observation_start_date_ == pd.Timestamp("2024-01-01")
        assert scorer_fitted.observation_end_date_ == pd.Timestamp("2024-01-07")
        assert scorer_fitted.evaluation_start_date_ == pd.Timestamp("2024-01-08")
        assert scorer_fitted.evaluation_end_date_ == pd.Timestamp("2024-01-14")

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
        s = RecencyFrequencyScorer(user_col="uid", item_col="iid", datetime_col="ts")
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


# ---------------------------------------------------------------------------
# fit — バリデーション
# ---------------------------------------------------------------------------
class TestFitValidation:
    def test_not_dataframe_raises(self, scorer):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            scorer.fit("not_a_df", "2024-01-07")

    def test_missing_datetime_col_raises(self, scorer, df):
        with pytest.raises(ValueError, match="Missing required columns"):
            scorer.fit(df.drop(columns="datetime"), "2024-01-07")

    def test_invalid_target_date_raises(self, scorer, df):
        with pytest.raises(ValueError, match="target_date could not be parsed"):
            scorer.fit(df, "not-a-date")


# ---------------------------------------------------------------------------
# fit — 正常系
# ---------------------------------------------------------------------------
# テストデータ (2024-01-01 〜 2024-01-10) に対して target_date="2024-01-07" を使用:
#   obs: 2024-01-01 〜 2024-01-07 (observation_days=28 → df_min=Jan01 が floor)
#   eval: 2024-01-08 〜 2024-01-10 (evaluation_days=7 → df_max=Jan10 が ceil)
# fit_period(_OBS_PERIOD, ("2024-01-08","2024-01-10")) と等価
_FIT_TARGET_DATE = "2024-01-07"


class TestFitResult:
    def test_returns_self(self, scorer, df):
        result = scorer.fit(
            df, _FIT_TARGET_DATE, recency_limit=_RECENCY_LIMIT, frequency_limit=_FREQUENCY_LIMIT
        )
        assert result is scorer

    def test_periods_match_target_date(self, df):
        s = RecencyFrequencyScorer()
        s.fit(df, _FIT_TARGET_DATE, recency_limit=_RECENCY_LIMIT, frequency_limit=_FREQUENCY_LIMIT)
        assert s.observation_end_date_ == pd.Timestamp(_FIT_TARGET_DATE)
        assert s.evaluation_start_date_ == pd.Timestamp("2024-01-08")

    def test_observation_start_bounded_by_observation_days(self, df):
        # observation_days=3 → obs_start = max(Jan01, Jan07-3d) = Jan04
        s = RecencyFrequencyScorer()
        s.fit(
            df,
            _FIT_TARGET_DATE,
            observation_days=3,
            evaluation_days=None,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.observation_start_date_ == pd.Timestamp("2024-01-04")

    def test_observation_days_none_uses_df_min(self, df):
        s = RecencyFrequencyScorer()
        s.fit(
            df,
            _FIT_TARGET_DATE,
            observation_days=None,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.observation_start_date_ == pd.Timestamp("2024-01-01")

    def test_evaluation_end_bounded_by_evaluation_days(self, df):
        # evaluation_days=2 → eval_end = min(Jan10, Jan07+2d) = Jan09
        s = RecencyFrequencyScorer()
        s.fit(
            df,
            _FIT_TARGET_DATE,
            evaluation_days=2,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.evaluation_end_date_ == pd.Timestamp("2024-01-09")

    def test_evaluation_days_none_uses_df_max(self, df):
        s = RecencyFrequencyScorer()
        s.fit(
            df,
            _FIT_TARGET_DATE,
            evaluation_days=None,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert s.evaluation_end_date_ == pd.Timestamp("2024-01-10")

    def test_same_result_as_fit_period(self, df):
        # fit(target_date, obs_days=None, eval_days=None) == fit_period(full range)
        s1 = RecencyFrequencyScorer()
        s1.fit(
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
        s.fit(df, _FIT_TARGET_DATE, recency_limit=_RECENCY_LIMIT, frequency_limit=_FREQUENCY_LIMIT)
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

    def test_mono_probability_values_in_bounds(self, scorer_optimized_mono):
        tol = 1e-6
        for val in scorer_optimized_mono.mono_probability_dict_.values():
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


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------
class TestPredict:
    def test_before_fit_raises(self, scorer):
        with pytest.raises(RuntimeError, match="fit"):
            scorer.predict(1, 1, kind="empirical")

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
        assert scorer_fitted.predict(1, 2, kind="empirical") == pytest.approx(1.0)

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

    def test_mono_kind(self, scorer_optimized_mono):
        # 型・範囲に加えて、mono_probability_dict_ から値を引いていることを確認
        prob = scorer_optimized_mono.predict(1, 1, kind="mono")
        assert isinstance(prob, float)
        assert 0.0 - 1e-6 <= prob <= 1.0 + 1e-6
        assert prob == pytest.approx(scorer_optimized_mono.mono_probability_dict_[1, 1])

    def test_mcc_kind(self, scorer_optimized_mcc):
        # 型・範囲に加えて、mcc_probability_dict_ から値を引いていることを確認
        prob = scorer_optimized_mcc.predict(1, 1, kind="mcc")
        assert isinstance(prob, float)
        assert 0.0 - 1e-6 <= prob <= 1.0 + 1e-6
        assert prob == pytest.approx(scorer_optimized_mcc.mcc_probability_dict_[1, 1])


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------
class TestTransform:
    def test_before_fit_raises(self, scorer, df):
        with pytest.raises(RuntimeError, match="fit"):
            scorer.transform(df, "2024-01-07")

    def test_uses_init_col_names_by_default(self):
        # カスタムカラム名で初期化 → transform に渡さなくても動作する
        df_custom = _make_df().rename(columns={"user": "uid", "item": "iid", "datetime": "ts"})
        s = RecencyFrequencyScorer(user_col="uid", item_col="iid", datetime_col="ts")
        s.fit_period(
            df_custom,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        result = s.transform(df_custom, "2024-01-07")
        assert "uid" in result.columns
        assert "iid" in result.columns

    def test_default_col_names_work_without_args(self, scorer_fitted, df):
        # デフォルトカラム名 (user/item/datetime) の場合も引数なしで動作する
        result = scorer_fitted.transform(df, "2024-01-07")
        assert "user" in result.columns
        assert "item" in result.columns

    def test_explicit_col_names_override_init(self):
        # __init__ と異なるカラム名を持つ DataFrame も明示指定で動作する
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
        result = s.transform(df_b, "2024-01-07", user_col="uid", item_col="iid", datetime_col="ts")
        assert "uid" in result.columns
        assert "iid" in result.columns

    def test_returns_dataframe_with_expected_columns(self, scorer_fitted, df):
        result = scorer_fitted.transform(df, "2024-01-07")
        assert set(result.columns) == {
            "user",
            "item",
            "recency",
            "frequency",
            "probability",
            "order",
        }

    def test_sorted_by_user_and_probability(self, scorer_fitted, df):
        result = scorer_fitted.transform(df, "2024-01-07")
        for user, grp in result.groupby("user"):
            assert list(grp["probability"]) == sorted(grp["probability"], reverse=True)
            assert list(grp["order"]) == list(range(1, len(grp) + 1))


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
# テストデータの期待値 (scorer_fitted: recency_limit=7, frequency_limit=3)
#
# transform(df, "2024-01-07") の結果:
#   u1: item1(r=3,f=3,prob=1.0,order=1), item2(r=6,f=1,prob=0.0,order=2)
#   u2: item2(r=1,f=2,prob=1.0,order=1), item1(r=4,f=1,prob=0.0,order=2)
#
# _UIREVISIT = {(u1,item1),(u2,item2)} → len=2
#
# order=1: UIrec={(u1,item1),(u2,item2)}, n_hit=2, n_rec=2
#   precision=1.0, recall=1.0, f1=1.0, recall_norm=1.0, f1_norm=1.0
# order=2 (=order_max): UIrec=all 4 pairs, n_hit=2, n_rec=4
#   precision=0.5, recall=1.0, f1≈0.667
# ---------------------------------------------------------------------------
class TestEvaluate:
    def test_returns_dataframe(self, scorer_fitted, df_rec):
        result = scorer_fitted.evaluate(df_rec, _UIREVISIT, order=1)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self, scorer_fitted, df_rec):
        result = scorer_fitted.evaluate(df_rec, _UIREVISIT, order=1)
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

    def test_row_count_with_order_1(self, scorer_fitted, df_rec):
        # order=1 でも order_max(=2) の行が追加されるので 2 行
        result = scorer_fitted.evaluate(df_rec, _UIREVISIT, order=1)
        assert len(result) == 2

    def test_order_max_always_included(self, scorer_fitted, df_rec):
        result = scorer_fitted.evaluate(df_rec, _UIREVISIT, order=1)
        assert df_rec["order"].max() in result["order"].values

    def test_n_recommended_at_order1(self, scorer_fitted, df_rec):
        result = scorer_fitted.evaluate(df_rec, _UIREVISIT, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["n_recommended"] == 2  # 2ユーザー × 1推薦

    def test_n_hit_at_order1(self, scorer_fitted, df_rec):
        result = scorer_fitted.evaluate(df_rec, _UIREVISIT, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["n_hit"] == 2

    def test_precision_at_order1(self, scorer_fitted, df_rec):
        result = scorer_fitted.evaluate(df_rec, _UIREVISIT, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["precision"] == pytest.approx(1.0)

    def test_recall_at_order1(self, scorer_fitted, df_rec):
        result = scorer_fitted.evaluate(df_rec, _UIREVISIT, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["recall"] == pytest.approx(1.0)

    def test_f1_at_order1(self, scorer_fitted, df_rec):
        result = scorer_fitted.evaluate(df_rec, _UIREVISIT, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["f1"] == pytest.approx(1.0)

    def test_precision_at_order_max(self, scorer_fitted, df_rec):
        # order=2: 4推薦中2ヒット → precision=0.5
        result = scorer_fitted.evaluate(df_rec, _UIREVISIT, order=2)
        row = result[result["order"] == 2].iloc[0]
        assert row["precision"] == pytest.approx(0.5)

    def test_recall_norm_with_unseen_revisits(self, scorer_fitted, df_rec):
        # UIrevisit に df_rec に存在しないペアを追加すると recall < recall_norm
        extended = _UIREVISIT | {("u3", "item3")}  # len=3
        result = scorer_fitted.evaluate(df_rec, extended, order=1)
        row = result[result["order"] == 1].iloc[0]
        assert row["recall"] == pytest.approx(2 / 3)
        assert row["recall_norm"] == pytest.approx(1.0)

    def test_f1_norm_with_unseen_revisits(self, scorer_fitted, df_rec):
        extended = _UIREVISIT | {("u3", "item3")}
        result = scorer_fitted.evaluate(df_rec, extended, order=1)
        row = result[result["order"] == 1].iloc[0]
        # recall_norm=1.0, precision=1.0 → f1_norm=1.0 > f1
        assert row["f1_norm"] == pytest.approx(1.0)
        assert row["f1"] < row["f1_norm"]

    def test_uses_init_col_names_by_default(self):
        df_custom = _make_df().rename(columns={"user": "uid", "item": "iid", "datetime": "ts"})
        s = RecencyFrequencyScorer(user_col="uid", item_col="iid", datetime_col="ts")
        s.fit_period(
            df_custom,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        df_rec_custom = s.transform(df_custom, _FIT_TARGET_DATE)
        uirevisit = {("u1", "item1"), ("u2", "item2")}
        result = s.evaluate(df_rec_custom, uirevisit)  # user_col/item_col 省略
        assert isinstance(result, pd.DataFrame)

    def test_explicit_col_override(self, scorer_fitted, df_rec):
        df_rec_renamed = df_rec.rename(columns={"user": "uid", "item": "iid"})
        result = scorer_fitted.evaluate(
            df_rec_renamed, _UIREVISIT, order=1, user_col="uid", item_col="iid"
        )
        assert isinstance(result, pd.DataFrame)

    def test_invalid_uirevisit_type_raises(self, scorer_fitted, df_rec):
        with pytest.raises(ValueError, match="UIrevisit"):
            scorer_fitted.evaluate(df_rec, 12345, order=1)


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

    def test_before_optimize_mcc_raises(self, scorer_fitted):
        with pytest.raises(RuntimeError, match="optimize"):
            scorer_fitted.plot_probability_surface(kind="mcc")

    def test_returns_figure_empirical(self, scorer_fitted):
        import matplotlib.figure

        fig = scorer_fitted.plot_probability_surface(kind="empirical")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_mono(self, scorer_optimized_mono):
        import matplotlib.figure

        fig = scorer_optimized_mono.plot_probability_surface(kind="mono")
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_returns_figure_mcc(self, scorer_optimized_mcc):
        import matplotlib.figure

        fig = scorer_optimized_mcc.plot_probability_surface(kind="mcc")
        assert isinstance(fig, matplotlib.figure.Figure)


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
