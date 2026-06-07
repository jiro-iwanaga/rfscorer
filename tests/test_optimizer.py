import math

import pytest

from rfscorer.optimizer import RFOptimizer

# 3×3 テストデータ（頻度単調性を意図的に違反させ、最適化に実作業を与える）
_R = [1, 2, 3]
_F = [1, 2, 3]
_RF2N = {(r, f): 100 for r in _R for f in _F}
_RF2Prob = {
    (1, 1): 0.80,
    (1, 2): 0.75,
    (1, 3): 0.90,
    (2, 1): 0.70,
    (2, 2): 0.65,
    (2, 3): 0.60,
    (3, 1): 0.50,
    (3, 2): 0.55,
    (3, 3): 0.40,
}
_R2N = {1: 300, 2: 300, 3: 300}
_R2Prob = {1: 0.82, 2: 0.65, 3: 0.48}
_F2N = {1: 300, 2: 300, 3: 300}
_F2Prob = {1: 0.55, 2: 0.70, 3: 0.65}  # 意図的に単調性違反あり
_TOL = 1e-4


@pytest.fixture
def opt():
    return RFOptimizer()


@pytest.fixture
def opt_with_data():
    o = RFOptimizer()
    o.set_data(_R, _F, _RF2N, _RF2Prob)
    return o


@pytest.fixture
def opt_with_marginal_data():
    o = RFOptimizer()
    o.set_data(_R, _F, _RF2N, _RF2Prob)
    o.set_marginal_data(_R2N, _R2Prob, _F2N, _F2Prob)
    return o


@pytest.fixture(scope="module")
def opt_solved_mr():
    o = RFOptimizer()
    o.set_data(_R, _F, _RF2N, _RF2Prob)
    o.set_marginal_data(_R2N, _R2Prob, _F2N, _F2Prob)
    o.build_model(kind="mr")
    o.solve()
    o.postprocess()
    return o


@pytest.fixture(scope="module")
def opt_solved_mf():
    o = RFOptimizer()
    o.set_data(_R, _F, _RF2N, _RF2Prob)
    o.set_marginal_data(_R2N, _R2Prob, _F2N, _F2Prob)
    o.build_model(kind="mf")
    o.solve()
    o.postprocess()
    return o


@pytest.fixture(scope="module")
def opt_after_solve():
    o = RFOptimizer()
    o.set_data(_R, _F, _RF2N, _RF2Prob)
    o.build_model(kind="mono")
    o.solve()
    return o


@pytest.fixture(scope="module")
def opt_solved_mono():
    o = RFOptimizer()
    o.set_data(_R, _F, _RF2N, _RF2Prob)
    o.build_model(kind="mono")
    o.solve()
    o.postprocess()
    return o


@pytest.fixture(scope="module")
def opt_solved_mrc():
    o = RFOptimizer()
    o.set_data(_R, _F, _RF2N, _RF2Prob)
    o.build_model(kind="mrc")
    o.solve()
    o.postprocess()
    return o


@pytest.fixture(scope="module")
def opt_solved_mfc():
    o = RFOptimizer()
    o.set_data(_R, _F, _RF2N, _RF2Prob)
    o.build_model(kind="mfc")
    o.solve()
    o.postprocess()
    return o


@pytest.fixture(scope="module")
def opt_solved_mcc():
    o = RFOptimizer()
    o.set_data(_R, _F, _RF2N, _RF2Prob)
    o.build_model(kind="mcc")
    o.solve()
    o.postprocess()
    return o


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------
class TestInit:
    def test_initial_state(self, opt):
        assert opt.R == []
        assert opt.F == []
        assert opt.RF2N == {}
        assert opt.RF2Prob == {}
        assert opt.kind is None
        assert opt.x is None
        assert opt.problem is None
        assert opt.status is None
        assert opt.RF2X == {}


# ---------------------------------------------------------------------------
# set_data
# ---------------------------------------------------------------------------
class TestSetData:
    def test_valid(self, opt):
        opt.set_data(_R, _F, _RF2N, _RF2Prob)
        assert opt.R == _R
        assert opt.F == _F
        assert opt.RF2N == _RF2N
        assert opt.RF2Prob == _RF2Prob

    def test_copies_inputs(self, opt):
        R = [1, 2]
        F = [1, 2]
        RF2N = {(1, 1): 10, (1, 2): 10, (2, 1): 10, (2, 2): 10}
        RF2Prob = {(1, 1): 0.8, (1, 2): 0.9, (2, 1): 0.6, (2, 2): 0.7}
        opt.set_data(R, F, RF2N, RF2Prob)
        R.append(3)
        F.append(4)
        RF2N[(1, 1)] = 999
        RF2Prob[(1, 1)] = 0.0
        assert opt.R == [1, 2]
        assert opt.F == [1, 2]
        assert opt.RF2N[(1, 1)] == 10
        assert opt.RF2Prob[(1, 1)] == 0.8

    def test_resets_model_state(self, opt_with_data):
        opt_with_data.build_model()
        opt_with_data.set_data(_R, _F, _RF2N, _RF2Prob)
        assert opt_with_data.kind is None
        assert opt_with_data.problem is None
        assert opt_with_data.status is None
        assert opt_with_data.RF2X == {}

    def test_empty_R_raises(self, opt):
        with pytest.raises(ValueError, match="R must not be empty"):
            opt.set_data([], _F, _RF2N, _RF2Prob)

    def test_empty_F_raises(self, opt):
        with pytest.raises(ValueError, match="F must not be empty"):
            opt.set_data(_R, [], _RF2N, _RF2Prob)

    def test_duplicate_R_raises(self, opt):
        with pytest.raises(ValueError, match="R must not contain duplicate"):
            opt.set_data([1, 1, 2], _F, _RF2N, _RF2Prob)

    def test_duplicate_F_raises(self, opt):
        with pytest.raises(ValueError, match="F must not contain duplicate"):
            opt.set_data(_R, [1, 1, 2], _RF2N, _RF2Prob)

    def test_missing_RF2N_key_raises(self, opt):
        RF2N_incomplete = dict(_RF2N)
        del RF2N_incomplete[(1, 1)]
        with pytest.raises(ValueError, match="RF2N is missing 1"):
            opt.set_data(_R, _F, RF2N_incomplete, _RF2Prob)

    def test_missing_RF2N_truncates_long_list(self, opt):
        # r=1 の 3 キーと (2,1) を除去して 4 件欠損させ "..." が付くことを確認
        RF2N_incomplete = {
            (r, f): 100 for r in _R for f in _F if not (r == 1 or (r == 2 and f == 1))
        }
        with pytest.raises(ValueError, match=r"\.\.\.$"):
            opt.set_data(_R, _F, RF2N_incomplete, _RF2Prob)

    def test_missing_RF2Prob_key_raises(self, opt):
        RF2Prob_incomplete = dict(_RF2Prob)
        del RF2Prob_incomplete[(2, 3)]
        with pytest.raises(ValueError, match="RF2Prob is missing 1"):
            opt.set_data(_R, _F, _RF2N, RF2Prob_incomplete)


# ---------------------------------------------------------------------------
# build_model
# ---------------------------------------------------------------------------
class TestBuildModel:
    def test_invalid_kind_raises(self, opt_with_data):
        with pytest.raises(ValueError, match="kind must be 'mono'"):
            opt_with_data.build_model(kind="invalid")

    def test_before_set_data_raises(self, opt):
        with pytest.raises(RuntimeError, match="set_data"):
            opt.build_model()

    def test_sets_kind_mono(self, opt_with_data):
        opt_with_data.build_model(kind="mono")
        assert opt_with_data.kind == "mono"

    def test_sets_kind_mr(self, opt_with_marginal_data):
        opt_with_marginal_data.build_model(kind="mr")
        assert opt_with_marginal_data.kind == "mr"

    def test_sets_kind_mf(self, opt_with_marginal_data):
        opt_with_marginal_data.build_model(kind="mf")
        assert opt_with_marginal_data.kind == "mf"

    def test_sets_kind_mrc(self, opt_with_data):
        opt_with_data.build_model(kind="mrc")
        assert opt_with_data.kind == "mrc"

    def test_sets_kind_mfc(self, opt_with_data):
        opt_with_data.build_model(kind="mfc")
        assert opt_with_data.kind == "mfc"

    def test_sets_kind_mcc(self, opt_with_data):
        opt_with_data.build_model(kind="mcc")
        assert opt_with_data.kind == "mcc"

    def test_mr_requires_marginal_data(self, opt_with_data):
        with pytest.raises(RuntimeError, match="set_marginal_data"):
            opt_with_data.build_model(kind="mr")

    def test_mf_requires_marginal_data(self, opt_with_data):
        with pytest.raises(RuntimeError, match="set_marginal_data"):
            opt_with_data.build_model(kind="mf")

    def test_num_variables(self, opt_with_data):
        opt_with_data.build_model()
        # 3×3 格子 → 9 変数
        assert opt_with_data.num_variables == 9

    def test_num_variables_mr(self, opt_with_marginal_data):
        opt_with_marginal_data.build_model(kind="mr")
        # 1D: |R| = 3
        assert opt_with_marginal_data.num_variables == 3

    def test_num_variables_mf(self, opt_with_marginal_data):
        opt_with_marginal_data.build_model(kind="mf")
        # 1D: |F| = 3
        assert opt_with_marginal_data.num_variables == 3

    def test_num_constraints_mr(self, opt_with_marginal_data):
        opt_with_marginal_data.build_model(kind="mr")
        # 範囲: 3+3=6、単調性: (3-1)=2、凸性: (3-2)=1 → 合計 9
        assert opt_with_marginal_data.num_constraints == 9

    def test_num_constraints_mf(self, opt_with_marginal_data):
        opt_with_marginal_data.build_model(kind="mf")
        # 範囲: 3+3=6、単調性: (3-1)=2、凹性: (3-2)=1 → 合計 9
        assert opt_with_marginal_data.num_constraints == 9

    def test_num_constraints_mono(self, opt_with_data):
        opt_with_data.build_model(kind="mono")
        # 範囲: 9+9=18、Recency: (3-1)*3=6、Frequency: 3*(3-1)=6 → 合計 30
        assert opt_with_data.num_constraints == 30

    def test_num_constraints_mrc(self, opt_with_data):
        opt_with_data.build_model(kind="mrc")
        # mono 30 + Recency凸性: (3-2)*3=3 → 合計 33
        assert opt_with_data.num_constraints == 33

    def test_num_constraints_mfc(self, opt_with_data):
        opt_with_data.build_model(kind="mfc")
        # mono 30 + Frequency凹性: 3*(3-2)=3 → 合計 33
        assert opt_with_data.num_constraints == 33

    def test_num_constraints_mcc(self, opt_with_data):
        opt_with_data.build_model(kind="mcc")
        # mono 30 + Recency凸性: (3-2)*3=3 + Frequency凹性: 3*(3-2)=3 → 合計 36
        assert opt_with_data.num_constraints == 36

    def test_resets_solve_state(self, opt_with_data):
        opt_with_data.build_model()
        opt_with_data.solve()
        opt_with_data.postprocess()
        opt_with_data.build_model()  # 再ビルドでソルバー結果がリセットされることを確認
        assert opt_with_data.status is None
        assert opt_with_data.objective_value is None
        assert opt_with_data.RF2X == {}

    def test_eps_default_is_zero(self, opt_with_data):
        opt_with_data.build_model()
        assert opt_with_data.eps == 0.0

    def test_eps_stored(self, opt_with_data):
        opt_with_data.build_model(eps=1e-4)
        assert opt_with_data.eps == 1e-4

    def test_negative_eps_raises(self, opt_with_data):
        with pytest.raises(ValueError, match="eps"):
            opt_with_data.build_model(eps=-1e-6)

    def test_eps_exceeds_max_2d_raises(self, opt_with_data):
        # eps_max = max(RF2Prob) / (nr - 1) = 0.90 / 2 = 0.45
        with pytest.raises(ValueError, match="eps"):
            opt_with_data.build_model(kind="mono", eps=0.45 + 1e-9)

    def test_eps_at_max_2d_ok(self, opt_with_data):
        opt_with_data.build_model(kind="mono", eps=0.45)  # 上界ちょうどは許容

    def test_eps_exceeds_max_mr_raises(self, opt_with_marginal_data):
        # eps_max = max(R2Prob) / (nr - 1) = 0.82 / 2 = 0.41
        with pytest.raises(ValueError, match="eps"):
            opt_with_marginal_data.build_model(kind="mr", eps=0.41 + 1e-9)

    def test_eps_at_max_mr_ok(self, opt_with_marginal_data):
        opt_with_marginal_data.build_model(kind="mr", eps=0.41)  # 上界ちょうどは許容

    def test_eps_exceeds_max_mf_raises(self, opt_with_marginal_data):
        # eps_max = max(F2Prob) / (nf - 1) = 0.70 / 2 = 0.35
        with pytest.raises(ValueError, match="eps"):
            opt_with_marginal_data.build_model(kind="mf", eps=0.35 + 1e-9)

    def test_eps_at_max_mf_ok(self, opt_with_marginal_data):
        opt_with_marginal_data.build_model(kind="mf", eps=0.35)  # 上界ちょうどは許容

    def test_strict_mono_recency_enforced(self, opt_with_marginal_data):
        """eps > 0 の場合、最適化後の recency 隣接値の差が eps 以上になること。"""
        eps = 1e-4
        opt_with_marginal_data.build_model(kind="mr", eps=eps)
        opt_with_marginal_data.solve()
        opt_with_marginal_data.postprocess()
        vals = [opt_with_marginal_data.x.value[i] for i in range(len(opt_with_marginal_data.R))]
        for i in range(len(vals) - 1):
            assert vals[i] >= vals[i + 1] + eps - 1e-9

    def test_strict_mono_frequency_enforced(self, opt_with_marginal_data):
        """eps > 0 の場合、最適化後の frequency 隣接値の差が eps 以上になること。"""
        eps = 1e-4
        opt_with_marginal_data.build_model(kind="mf", eps=eps)
        opt_with_marginal_data.solve()
        opt_with_marginal_data.postprocess()
        vals = [opt_with_marginal_data.x.value[i] for i in range(len(opt_with_marginal_data.F))]
        for i in range(len(vals) - 1):
            assert vals[i + 1] >= vals[i] + eps - 1e-9


# ---------------------------------------------------------------------------
# solve
# ---------------------------------------------------------------------------
class TestSolve:
    def test_before_build_model_raises(self, opt_with_data):
        with pytest.raises(RuntimeError, match="build_model"):
            opt_with_data.solve()

    def test_sets_status(self, opt_after_solve):
        assert opt_after_solve.status is not None

    def test_optimal_status(self, opt_after_solve):
        assert opt_after_solve.status == "optimal"

    def test_elapsed_time_positive(self, opt_after_solve):
        assert opt_after_solve.elapsed_time > 0

    def test_objective_value_finite(self, opt_after_solve):
        assert opt_after_solve.objective_value is not None
        assert math.isfinite(opt_after_solve.objective_value)

    def test_objective_value_nonnegative(self, opt_after_solve):
        assert opt_after_solve.objective_value >= 0


# ---------------------------------------------------------------------------
# postprocess
# ---------------------------------------------------------------------------
class TestPostprocess:
    def test_before_solve_raises(self, opt_with_data):
        opt_with_data.build_model()
        with pytest.raises(RuntimeError, match="solve"):
            opt_with_data.postprocess()

    def test_RF2X_populated(self, opt_solved_mono):
        assert len(opt_solved_mono.RF2X) == len(_R) * len(_F)

    def test_RF2X_keys(self, opt_solved_mono):
        expected = {(r, f) for r in _R for f in _F}
        assert set(opt_solved_mono.RF2X.keys()) == expected

    def test_RF2X_values_are_float(self, opt_solved_mono):
        for val in opt_solved_mono.RF2X.values():
            assert isinstance(val, float)

    def test_RF2X_values_in_bounds(self, opt_solved_mono):
        tol = 1e-6
        for val in opt_solved_mono.RF2X.values():
            assert -tol <= val <= 1 + tol


# ---------------------------------------------------------------------------
# 制約充足: mono
# ---------------------------------------------------------------------------
class TestMonoConstraints:
    def test_recency_monotonicity(self, opt_solved_mono):
        for f in _F:
            for i in range(len(_R) - 1):
                r, r_next = _R[i], _R[i + 1]
                assert opt_solved_mono.RF2X[r, f] >= opt_solved_mono.RF2X[r_next, f] - _TOL

    def test_frequency_monotonicity(self, opt_solved_mono):
        for r in _R:
            for j in range(len(_F) - 1):
                f, f_next = _F[j], _F[j + 1]
                assert opt_solved_mono.RF2X[r, f] <= opt_solved_mono.RF2X[r, f_next] + _TOL


# ---------------------------------------------------------------------------
# 制約充足: mrc
# ---------------------------------------------------------------------------
class TestMRCConstraints:
    def test_recency_monotonicity(self, opt_solved_mrc):
        for f in _F:
            for i in range(len(_R) - 1):
                r, r_next = _R[i], _R[i + 1]
                assert opt_solved_mrc.RF2X[r, f] >= opt_solved_mrc.RF2X[r_next, f] - _TOL

    def test_frequency_monotonicity(self, opt_solved_mrc):
        for r in _R:
            for j in range(len(_F) - 1):
                f, f_next = _F[j], _F[j + 1]
                assert opt_solved_mrc.RF2X[r, f] <= opt_solved_mrc.RF2X[r, f_next] + _TOL

    def test_recency_convexity(self, opt_solved_mrc):
        for f in _F:
            for i in range(len(_R) - 2):
                r0, r1, r2 = _R[i], _R[i + 1], _R[i + 2]
                second_diff = (
                    opt_solved_mrc.RF2X[r0, f]
                    - 2 * opt_solved_mrc.RF2X[r1, f]
                    + opt_solved_mrc.RF2X[r2, f]
                )
                assert second_diff >= -_TOL


# ---------------------------------------------------------------------------
# 制約充足: mfc
# ---------------------------------------------------------------------------
class TestMFCConstraints:
    def test_recency_monotonicity(self, opt_solved_mfc):
        for f in _F:
            for i in range(len(_R) - 1):
                r, r_next = _R[i], _R[i + 1]
                assert opt_solved_mfc.RF2X[r, f] >= opt_solved_mfc.RF2X[r_next, f] - _TOL

    def test_frequency_monotonicity(self, opt_solved_mfc):
        for r in _R:
            for j in range(len(_F) - 1):
                f, f_next = _F[j], _F[j + 1]
                assert opt_solved_mfc.RF2X[r, f] <= opt_solved_mfc.RF2X[r, f_next] + _TOL

    def test_frequency_concavity(self, opt_solved_mfc):
        for r in _R:
            for j in range(len(_F) - 2):
                f0, f1, f2 = _F[j], _F[j + 1], _F[j + 2]
                second_diff = (
                    opt_solved_mfc.RF2X[r, f0]
                    - 2 * opt_solved_mfc.RF2X[r, f1]
                    + opt_solved_mfc.RF2X[r, f2]
                )
                assert second_diff <= _TOL


# ---------------------------------------------------------------------------
# 制約充足: mr
# ---------------------------------------------------------------------------
class TestMRConstraints:
    def test_recency_monotonicity(self, opt_solved_mr):
        for i in range(len(_R) - 1):
            r, r_next = _R[i], _R[i + 1]
            for f in _F:
                assert opt_solved_mr.RF2X[r, f] >= opt_solved_mr.RF2X[r_next, f] - _TOL

    def test_recency_convexity(self, opt_solved_mr):
        for i in range(len(_R) - 2):
            r0, r1, r2 = _R[i], _R[i + 1], _R[i + 2]
            for f in _F:
                second_diff = (
                    opt_solved_mr.RF2X[r0, f]
                    - 2 * opt_solved_mr.RF2X[r1, f]
                    + opt_solved_mr.RF2X[r2, f]
                )
                assert second_diff >= -_TOL

    def test_constant_across_frequency(self, opt_solved_mr):
        # 1D モデルなので f によらず同じ値
        for r in _R:
            vals = [opt_solved_mr.RF2X[r, f] for f in _F]
            assert all(abs(v - vals[0]) < _TOL for v in vals)

    def test_RF2X_covers_all_pairs(self, opt_solved_mr):
        assert set(opt_solved_mr.RF2X.keys()) == {(r, f) for r in _R for f in _F}


# ---------------------------------------------------------------------------
# 制約充足: mf
# ---------------------------------------------------------------------------
class TestMFConstraints:
    def test_frequency_monotonicity(self, opt_solved_mf):
        for j in range(len(_F) - 1):
            f, f_next = _F[j], _F[j + 1]
            for r in _R:
                assert opt_solved_mf.RF2X[r, f] <= opt_solved_mf.RF2X[r, f_next] + _TOL

    def test_frequency_concavity(self, opt_solved_mf):
        for j in range(len(_F) - 2):
            f0, f1, f2 = _F[j], _F[j + 1], _F[j + 2]
            for r in _R:
                second_diff = (
                    opt_solved_mf.RF2X[r, f0]
                    - 2 * opt_solved_mf.RF2X[r, f1]
                    + opt_solved_mf.RF2X[r, f2]
                )
                assert second_diff <= _TOL

    def test_constant_across_recency(self, opt_solved_mf):
        # 1D モデルなので r によらず同じ値
        for f in _F:
            vals = [opt_solved_mf.RF2X[r, f] for r in _R]
            assert all(abs(v - vals[0]) < _TOL for v in vals)

    def test_RF2X_covers_all_pairs(self, opt_solved_mf):
        assert set(opt_solved_mf.RF2X.keys()) == {(r, f) for r in _R for f in _F}


# ---------------------------------------------------------------------------
# 制約充足: mcc
# ---------------------------------------------------------------------------
class TestMCCConstraints:
    def test_recency_monotonicity(self, opt_solved_mcc):
        for f in _F:
            for i in range(len(_R) - 1):
                r, r_next = _R[i], _R[i + 1]
                assert opt_solved_mcc.RF2X[r, f] >= opt_solved_mcc.RF2X[r_next, f] - _TOL

    def test_frequency_monotonicity(self, opt_solved_mcc):
        for r in _R:
            for j in range(len(_F) - 1):
                f, f_next = _F[j], _F[j + 1]
                assert opt_solved_mcc.RF2X[r, f] <= opt_solved_mcc.RF2X[r, f_next] + _TOL

    def test_recency_convexity(self, opt_solved_mcc):
        for f in _F:
            for i in range(len(_R) - 2):
                r0, r1, r2 = _R[i], _R[i + 1], _R[i + 2]
                second_diff = (
                    opt_solved_mcc.RF2X[r0, f]
                    - 2 * opt_solved_mcc.RF2X[r1, f]
                    + opt_solved_mcc.RF2X[r2, f]
                )
                assert second_diff >= -_TOL

    def test_frequency_concavity(self, opt_solved_mcc):
        for r in _R:
            for j in range(len(_F) - 2):
                f0, f1, f2 = _F[j], _F[j + 1], _F[j + 2]
                second_diff = (
                    opt_solved_mcc.RF2X[r, f0]
                    - 2 * opt_solved_mcc.RF2X[r, f1]
                    + opt_solved_mcc.RF2X[r, f2]
                )
                assert second_diff <= _TOL


# ---------------------------------------------------------------------------
# show メソッド
# ---------------------------------------------------------------------------
class TestShowMethods:
    def test_show_input(self, opt_with_data, capsys):
        opt_with_data.show_input()
        out = capsys.readouterr().out
        assert "show input" in out
        assert "RF2N" in out
        assert "RF2Prob" in out

    def test_show_input_before_set_data_raises(self, opt):
        with pytest.raises(RuntimeError, match="set_data"):
            opt.show_input()

    def test_show_input_with_int_N(self, opt):
        # RF2N が Python int でも ValueError にならないことを確認
        RF2N_int = {(r, f): r * 10 + f for r in _R for f in _F}
        opt.set_data(_R, _F, RF2N_int, _RF2Prob)
        opt.show_input()  # 例外なし

    def test_show_solve_info(self, opt_after_solve, capsys):
        opt_after_solve.show_solve_info()
        out = capsys.readouterr().out
        assert "show solve info" in out
        assert "status" in out
        assert "elapsed_time" in out

    def test_show_solve_info_before_solve_raises(self, opt_with_data):
        opt_with_data.build_model()
        with pytest.raises(RuntimeError, match="solve"):
            opt_with_data.show_solve_info()

    def test_show_result(self, opt_solved_mono, capsys):
        opt_solved_mono.show_result()
        out = capsys.readouterr().out
        assert "show result" in out

    def test_show_result_before_postprocess_raises(self, opt_after_solve):
        with pytest.raises(RuntimeError, match="postprocess"):
            opt_after_solve.show_result()
