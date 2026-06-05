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
    s.fit(
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
    s.fit(
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
    s.fit(
        _make_df(),
        _OBS_PERIOD,
        _EVAL_PERIOD,
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    s.optimize(kind="mcc")
    return s


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
# fit — バリデーション
# ---------------------------------------------------------------------------
class TestFitValidation:
    def test_not_dataframe_raises(self, scorer):
        with pytest.raises(TypeError, match="pandas DataFrame"):
            scorer.fit("not_a_df", _OBS_PERIOD, _EVAL_PERIOD)

    def test_missing_column_raises(self, scorer, df):
        with pytest.raises(ValueError, match="Missing required columns"):
            scorer.fit(df.drop(columns="item"), _OBS_PERIOD, _EVAL_PERIOD)

    def test_custom_col_missing_raises(self, df):
        s = RecencyFrequencyScorer(user_col="uid")
        with pytest.raises(ValueError, match="Missing required columns"):
            s.fit(df, _OBS_PERIOD, _EVAL_PERIOD)

    def test_observation_period_wrong_length_raises(self, scorer, df):
        with pytest.raises(ValueError, match="observation_period"):
            scorer.fit(df, ("2024-01-01",), _EVAL_PERIOD)

    def test_evaluation_period_wrong_length_raises(self, scorer, df):
        with pytest.raises(ValueError, match="evaluation_period"):
            scorer.fit(df, _OBS_PERIOD, ("2024-01-08",))

    def test_observation_period_reversed_raises(self, scorer, df):
        with pytest.raises(ValueError, match="observation_period must be ordered"):
            scorer.fit(df, ("2024-01-07", "2024-01-01"), _EVAL_PERIOD)

    def test_evaluation_period_reversed_raises(self, scorer, df):
        with pytest.raises(ValueError, match="evaluation_period must be ordered"):
            scorer.fit(df, _OBS_PERIOD, ("2024-01-14", "2024-01-08"))

    def test_overlapping_periods_raises(self, scorer, df):
        with pytest.raises(ValueError, match="observation_period must end before"):
            scorer.fit(df, ("2024-01-01", "2024-01-09"), ("2024-01-08", "2024-01-14"))

    def test_no_cv_auto_limit_raises(self, scorer):
        # 評価期間に再閲覧なし → 自動上限計算で ValueError
        rows = [
            ("u1", "item1", "2024-01-03"),
            ("u1", "item1", "2024-01-04"),  # 評価期間に閲覧なし
        ]
        df_no_cv = pd.DataFrame(rows, columns=["user", "item", "datetime"])
        with pytest.raises(ValueError, match="No revisits"):
            scorer.fit(df_no_cv, _OBS_PERIOD, _EVAL_PERIOD)


# ---------------------------------------------------------------------------
# fit — 正常系
# ---------------------------------------------------------------------------
class TestFitResult:
    def test_returns_self(self, scorer, df):
        result = scorer.fit(
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
        s.fit(df, _OBS_PERIOD, _EVAL_PERIOD)
        assert s.recency_limit == _AUTO_RECENCY_LIMIT
        assert s.frequency_limit == _AUTO_FREQUENCY_LIMIT
        # 上限が実際に適用されていること (limit 超えのキーが存在しないこと)
        for r, f in s.RF2N:
            assert r <= _AUTO_RECENCY_LIMIT
            assert f <= _AUTO_FREQUENCY_LIMIT

    def test_custom_column_names(self):
        df_custom = _make_df().rename(columns={"user": "uid", "item": "iid", "datetime": "ts"})
        s = RecencyFrequencyScorer(user_col="uid", item_col="iid", datetime_col="ts")
        s.fit(
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
        s.fit(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        assert list(df.columns) == original_columns
        assert (df.values == original_values).all()


# ---------------------------------------------------------------------------
# optimize
# ---------------------------------------------------------------------------
class TestOptimize:
    def test_before_fit_raises(self, scorer):
        with pytest.raises(RuntimeError, match="fit"):
            scorer.optimize(kind="mono")

    def test_invalid_kind_raises(self, scorer, df):
        scorer.fit(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(ValueError, match="kind"):
            scorer.optimize(kind="invalid")

    def test_optimize_mono_returns_self(self, scorer, df):
        scorer.fit(
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
        scorer.fit(
            df,
            _OBS_PERIOD,
            _EVAL_PERIOD,
            recency_limit=_RECENCY_LIMIT,
            frequency_limit=_FREQUENCY_LIMIT,
        )
        with pytest.raises(RuntimeError, match="optimize"):
            scorer.predict(1, 1, kind="mono")

    def test_before_optimize_mcc_raises(self, scorer, df):
        scorer.fit(
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
