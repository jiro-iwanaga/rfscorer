import math
import time

import cvxpy as cp


class RecencyFrequencyOptimizer:
    """Convex quadratic optimizer for RF monotonicity-constrained product-choice probabilities.

    Minimizes weighted least-squares deviation from empirical probabilities
    over a recency-frequency grid under RF constraints. Intended to be called
    from RecencyFrequencyScorer.optimize().

    Typical call sequence for 2D models::

        optimizer = RecencyFrequencyOptimizer()
        optimizer.set_data(R, F, RF2N, RF2Prob)
        optimizer.build_model(kind="mono")
        optimizer.solve()
        optimizer.show_solve_info()  # optional
        optimizer.postprocess()
        # optimizer.RF2X holds the optimized probabilities

    Typical call sequence for 1D marginal models::

        optimizer = RecencyFrequencyOptimizer()
        optimizer.set_data(R, F, RF2N, RF2Prob)
        optimizer.set_marginal_data(R2N, R2Prob, F2N, F2Prob)
        optimizer.build_marginal_model(axis="r")
        optimizer.solve()
        optimizer.postprocess()
        # optimizer.R2X holds the optimized recency probabilities
    """

    # Permissively licensed (MIT / Apache-2.0 / BSD) cvxpy solvers that support
    # this QP formulation. CLARABEL/SCS/OSQP are bundled with cvxpy; the others
    # require their own package (e.g. `daqp`, `piqp`, `proxqp`).
    _ALLOWED_SOLVERS = ("CLARABEL", "SCS", "OSQP", "DAQP", "PIQP", "PROXQP")

    def __init__(self):
        # 集合と定数
        self.R = []
        self.F = []
        self.RF2N = {}
        self.RF2Prob = {}
        self.R2N = {}
        self.R2Prob = {}
        self.F2N = {}
        self.F2Prob = {}

        # 数理最適化モデルとソルバー
        self.kind = None
        self.axis = None  # None=2D, "r"=recency marginal, "f"=frequency marginal
        self.eps = 0.0
        self.x = None
        self.constraints = []
        self.objectives = []
        self.problem = None
        self.num_variables = None
        self.num_constraints = None

        # 最適化計算結果
        self.elapsed_time = None
        self.status = None
        self.objective_value = None
        self.RF2X = {}
        self.R2X = {}
        self.F2X = {}

    def set_data(self, R, F, RF2N, RF2Prob):
        """Load recency-frequency data before building the model.

        Must be called before build_model() or build_marginal_model().
        Calling this method resets any previously built model and solver state.

        Parameters
        ----------
        R : list[int]
            Unique recency values (1 = most recent).
        F : list[int]
            Unique frequency values.
        RF2N : dict[tuple[int, int], float]
            Number of observations for each (recency, frequency) pair.
        RF2Prob : dict[tuple[int, int], float]
            Empirical product-choice probability for each (recency, frequency) pair.
        """
        if len(R) == 0:
            raise ValueError("R must not be empty")
        if len(set(R)) != len(R):
            raise ValueError("R must not contain duplicate values")
        if len(F) == 0:
            raise ValueError("F must not be empty")
        if len(set(F)) != len(F):
            raise ValueError("F must not contain duplicate values")
        missing_N = [(r, f) for r in R for f in F if (r, f) not in RF2N]
        if missing_N:
            suffix = "..." if len(missing_N) > 3 else ""
            raise ValueError(f"RF2N is missing {len(missing_N)} key(s): {missing_N[:3]}{suffix}")
        missing_Prob = [(r, f) for r in R for f in F if (r, f) not in RF2Prob]
        if missing_Prob:
            suffix = "..." if len(missing_Prob) > 3 else ""
            raise ValueError(
                f"RF2Prob is missing {len(missing_Prob)} key(s): {missing_Prob[:3]}{suffix}"
            )

        self.R = list(R)
        self.F = list(F)
        self.RF2N = dict(RF2N)
        self.RF2Prob = dict(RF2Prob)
        self.R2N = {}
        self.R2Prob = {}
        self.F2N = {}
        self.F2Prob = {}

        self.kind = None
        self.axis = None
        self.eps = 0.0
        self.x = None
        self.constraints = []
        self.objectives = []
        self.problem = None
        self.num_variables = None
        self.num_constraints = None
        self.elapsed_time = None
        self.status = None
        self.objective_value = None
        self.RF2X = {}
        self.R2X = {}
        self.F2X = {}

    def set_marginal_data(self, R2N, R2Prob, F2N, F2Prob):
        """Load marginal recency and frequency data for 1-D optimization models.

        Must be called after set_data() and before build_marginal_model().
        Calling this method resets any previously built model and solver state.

        Parameters
        ----------
        R2N : dict[int, float]
            Number of observations for each recency value.
        R2Prob : dict[int, float]
            Empirical product-choice probability for each recency value.
        F2N : dict[int, float]
            Number of observations for each frequency value.
        F2Prob : dict[int, float]
            Empirical product-choice probability for each frequency value.
        """
        if len(self.R) == 0 or len(self.F) == 0:
            raise RuntimeError("set_data() must be called before set_marginal_data()")
        missing_R2N = [r for r in self.R if r not in R2N]
        if missing_R2N:
            suffix = "..." if len(missing_R2N) > 3 else ""
            raise ValueError(f"R2N is missing {len(missing_R2N)} key(s): {missing_R2N[:3]}{suffix}")
        missing_R2Prob = [r for r in self.R if r not in R2Prob]
        if missing_R2Prob:
            suffix = "..." if len(missing_R2Prob) > 3 else ""
            raise ValueError(
                f"R2Prob is missing {len(missing_R2Prob)} key(s): {missing_R2Prob[:3]}{suffix}"
            )
        missing_F2N = [f for f in self.F if f not in F2N]
        if missing_F2N:
            suffix = "..." if len(missing_F2N) > 3 else ""
            raise ValueError(f"F2N is missing {len(missing_F2N)} key(s): {missing_F2N[:3]}{suffix}")
        missing_F2Prob = [f for f in self.F if f not in F2Prob]
        if missing_F2Prob:
            suffix = "..." if len(missing_F2Prob) > 3 else ""
            raise ValueError(
                f"F2Prob is missing {len(missing_F2Prob)} key(s): {missing_F2Prob[:3]}{suffix}"
            )
        self.R2N = dict(R2N)
        self.R2Prob = dict(R2Prob)
        self.F2N = dict(F2N)
        self.F2Prob = dict(F2Prob)

        self.kind = None
        self.axis = None
        self.eps = 0.0
        self.x = None
        self.constraints = []
        self.objectives = []
        self.problem = None
        self.num_variables = None
        self.num_constraints = None
        self.elapsed_time = None
        self.status = None
        self.objective_value = None
        self.RF2X = {}
        self.R2X = {}
        self.F2X = {}

    def build_model(self, kind="mono", eps=0.0):
        """Build the 2D joint optimization model.

        Fits against joint RF probabilities (RF2Prob). Must be called after
        set_data(). Resets any previous solve and postprocess state.

        Parameters
        ----------
        kind : {"mono", "mrc", "mfc", "mcc"}, default "mono"
            "mono" applies monotonicity constraints only.
            "mrc" additionally applies convexity in recency: the rate of
            probability decrease slows as recency grows
            (second difference >= 0 along the recency axis).
            "mfc" additionally applies concavity in frequency: diminishing
            marginal returns as frequency grows
            (second difference <= 0 along the frequency axis).
            "mcc" applies both recency convexity and frequency concavity.
        eps : float, default 0.0
            Minimum gap enforced between adjacent values in monotonicity
            constraints.  When 0.0 (default), weak monotonicity is used
            (``x[r,f] >= x[r+1,f]`` and ``x[r,f] <= x[r,f+1]``).  When
            positive, strict monotonicity is enforced
            (``x[r,f] >= x[r+1,f] + eps`` and ``x[r,f] + eps <= x[r,f+1]``),
            preventing ties between adjacent recency or frequency levels.

        Raises
        ------
        ValueError
            If kind is not one of the accepted values, eps is negative, or
            eps exceeds the maximum feasible gap given the data.
        RuntimeError
            If set_data() has not been called.
        """
        if kind not in ("mono", "mrc", "mfc", "mcc"):
            raise ValueError(f"kind must be 'mono', 'mrc', 'mfc', or 'mcc', got {kind!r}")
        if len(self.R) == 0 or len(self.F) == 0:
            raise RuntimeError("set_data() must be called before build_model()")
        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps!r}")

        self.kind = kind
        self.axis = None
        self.eps = eps

        nr = len(self.R)
        nf = len(self.F)

        if eps > 0:
            p_max = max(self.RF2Prob.values())
            candidates = []
            if nr > 1:
                candidates.append(p_max / (nr - 1))
            if nf > 1:
                candidates.append(p_max / (nf - 1))
            eps_max = min(candidates) if candidates else math.inf
            if eps > eps_max:
                raise ValueError(f"eps={eps!r} exceeds the maximum feasible value {eps_max:.6g}")

        self.constraints = []
        self.objectives = []

        self.x = cp.Variable((nr, nf))
        self.constraints.append(self.x >= 0)
        self.constraints.append(self.x <= 1)
        # Recency 単調性: r < r' => x[r,f] >= x[r',f] + eps
        for r_idx in range(nr - 1):
            for f_idx in range(nf):
                self.constraints.append(self.x[r_idx, f_idx] >= self.x[r_idx + 1, f_idx] + eps)
        # Frequency 単調性: f < f' => x[r,f] + eps <= x[r,f']
        for r_idx in range(nr):
            for f_idx in range(nf - 1):
                self.constraints.append(self.x[r_idx, f_idx] + eps <= self.x[r_idx, f_idx + 1])
        if kind in ("mrc", "mcc"):
            # Recency 凸性（二階差分 >= 0）
            for r_idx in range(nr - 2):
                for f_idx in range(nf):
                    self.constraints.append(
                        self.x[r_idx, f_idx]
                        - 2 * self.x[r_idx + 1, f_idx]
                        + self.x[r_idx + 2, f_idx]
                        >= 0
                    )
        if kind in ("mfc", "mcc"):
            # Frequency 凹性（二階差分 <= 0）
            for r_idx in range(nr):
                for f_idx in range(nf - 2):
                    self.constraints.append(
                        self.x[r_idx, f_idx]
                        - 2 * self.x[r_idx, f_idx + 1]
                        + self.x[r_idx, f_idx + 2]
                        <= 0
                    )
        for r_idx, r in enumerate(self.R):
            for f_idx, f in enumerate(self.F):
                N = self.RF2N[r, f]
                p = self.RF2Prob[r, f]
                self.objectives.append(N * (self.x[r_idx, f_idx] - p) ** 2)

        self.problem = cp.Problem(cp.Minimize(cp.sum(self.objectives)), self.constraints)
        self.num_variables = sum(v.size for v in self.problem.variables())
        self.num_constraints = sum(c.size for c in self.constraints)

        self.elapsed_time = None
        self.status = None
        self.objective_value = None
        self.RF2X = {}
        self.R2X = {}
        self.F2X = {}

    def build_marginal_model(self, axis="r", eps=0.0):
        """Build the 1D marginal optimization model.

        Fits against marginal probabilities (R2Prob when axis='r', F2Prob when
        axis='f'), not the joint RF2Prob. Must be called after set_data() and
        set_marginal_data(). Resets any previous solve and postprocess state.

        Parameters
        ----------
        axis : {"r", "f"}, default "r"
            "r" fits a recency-only model with monotonicity (``x[r] >= x[r+1]``)
            and convexity (second difference >= 0): the rate of probability
            decrease slows as recency grows.
            "f" fits a frequency-only model with monotonicity
            (``x[f] <= x[f+1]``) and concavity (second difference <= 0):
            diminishing marginal returns as frequency grows.
        eps : float, default 0.0
            Minimum gap enforced between adjacent values in monotonicity
            constraints.  When 0.0 (default), weak monotonicity is used
            (``x[r] >= x[r+1]`` or ``x[f] <= x[f+1]``).  When positive,
            strict monotonicity is enforced
            (``x[r] >= x[r+1] + eps`` or ``x[f] + eps <= x[f+1]``),
            preventing ties between adjacent levels.

        Raises
        ------
        ValueError
            If axis is not 'r' or 'f', eps is negative, or eps exceeds the
            maximum feasible gap given the marginal data.
        RuntimeError
            If set_data() or set_marginal_data() has not been called.
        """
        if axis not in ("r", "f"):
            raise ValueError(f"axis must be 'r' or 'f', got {axis!r}")
        if len(self.R) == 0 or len(self.F) == 0:
            raise RuntimeError("set_data() must be called before build_marginal_model()")
        if not self.R2N:
            raise RuntimeError("set_marginal_data() must be called before build_marginal_model()")
        if eps < 0:
            raise ValueError(f"eps must be non-negative, got {eps!r}")

        self.kind = None
        self.axis = axis
        self.eps = eps

        nr = len(self.R)
        nf = len(self.F)

        if eps > 0:
            if axis == "r":
                p_max = max(self.R2Prob.values())
                eps_max = p_max / (nr - 1) if nr > 1 else math.inf
            else:
                p_max = max(self.F2Prob.values())
                eps_max = p_max / (nf - 1) if nf > 1 else math.inf
            if eps > eps_max:
                raise ValueError(f"eps={eps!r} exceeds the maximum feasible value {eps_max:.6g}")

        self.constraints = []
        self.objectives = []

        if axis == "r":
            self.x = cp.Variable(nr)
            self.constraints.append(self.x >= 0)
            self.constraints.append(self.x <= 1)
            # Recency 単調性: r < r' => x[r] >= x[r'] + eps
            for r_idx in range(nr - 1):
                self.constraints.append(self.x[r_idx] >= self.x[r_idx + 1] + eps)
            # Recency 凸性（二階差分 >= 0）
            for r_idx in range(nr - 2):
                self.constraints.append(
                    self.x[r_idx] - 2 * self.x[r_idx + 1] + self.x[r_idx + 2] >= 0
                )
            for r_idx, r in enumerate(self.R):
                N = self.R2N[r]
                p = self.R2Prob[r]
                self.objectives.append(N * (self.x[r_idx] - p) ** 2)

        else:  # axis == "f"
            self.x = cp.Variable(nf)
            self.constraints.append(self.x >= 0)
            self.constraints.append(self.x <= 1)
            # Frequency 単調性: f < f' => x[f] + eps <= x[f']
            for f_idx in range(nf - 1):
                self.constraints.append(self.x[f_idx] + eps <= self.x[f_idx + 1])
            # Frequency 凹性（二階差分 <= 0）
            for f_idx in range(nf - 2):
                self.constraints.append(
                    self.x[f_idx] - 2 * self.x[f_idx + 1] + self.x[f_idx + 2] <= 0
                )
            for f_idx, f in enumerate(self.F):
                N = self.F2N[f]
                p = self.F2Prob[f]
                self.objectives.append(N * (self.x[f_idx] - p) ** 2)

        self.problem = cp.Problem(cp.Minimize(cp.sum(self.objectives)), self.constraints)
        self.num_variables = sum(v.size for v in self.problem.variables())
        self.num_constraints = sum(c.size for c in self.constraints)

        self.elapsed_time = None
        self.status = None
        self.objective_value = None
        self.RF2X = {}
        self.R2X = {}
        self.F2X = {}

    def solve(self, solver="CLARABEL"):
        """Solve the optimization problem built by build_model() or build_marginal_model().

        Sets status, objective_value, and elapsed_time.
        Solver failures are not raised here; call postprocess() to detect them.

        Parameters
        ----------
        solver : {"CLARABEL", "SCS", "OSQP", "DAQP", "PIQP", "PROXQP"}, default "CLARABEL"
            Convex solver to use. Restricted to permissively licensed
            (MIT / Apache-2.0 / BSD) open-source solvers that support the
            QP formulation used here.
            - "CLARABEL" (Apache-2.0): modern general convex solver, the
              default solver in cvxpy 1.5+. Bundled with cvxpy.
            - "SCS" (MIT): general convex solver, bundled with cvxpy.
            - "OSQP" (Apache-2.0): classic QP solver, bundled with cvxpy.
            - "DAQP" (MIT): dense QP-specialized solver. Requires the
              `daqp` package.
            - "PIQP" (BSD-2-Clause): proximal interior point QP solver.
              Requires the `piqp` package.
            - "PROXQP" (BSD-2-Clause): proximal QP solver from Inria.
              Requires the `proxsuite` package.

        Raises
        ------
        ValueError
            If solver is not one of the allowed permissively licensed solvers.
        RuntimeError
            If neither build_model() nor build_marginal_model() has been called.
        """
        if solver not in self._ALLOWED_SOLVERS:
            raise ValueError(f"solver must be one of {self._ALLOWED_SOLVERS}, got {solver!r}")
        if self.problem is None:
            raise RuntimeError(
                "build_model() or build_marginal_model() must be called before solve()"
            )

        start_time = time.time()
        self.problem.solve(solver=solver)
        self.elapsed_time = time.time() - start_time

        self.status = self.problem.status
        self.objective_value = self.problem.value

    def postprocess(self):
        """Extract optimized probabilities from the solver solution.

        For 2D models (build_model): populates RF2X.
        For 1D marginal models (build_marginal_model):
          axis="r" populates R2X, axis="f" populates F2X.
        No broadcast is performed.

        Must be called after solve(). Raises RuntimeError if the solver did
        not find a feasible solution.
        """
        if self.status is None:
            raise RuntimeError("solve() must be called before postprocess()")
        if self.x.value is None:
            raise RuntimeError(f"Cannot postprocess: solver status is {self.status!r}")

        if self.axis == "r":
            self.R2X = {r: float(self.x.value[r_idx]) for r_idx, r in enumerate(self.R)}
        elif self.axis == "f":
            self.F2X = {f: float(self.x.value[f_idx]) for f_idx, f in enumerate(self.F)}
        else:
            self.RF2X = {
                (r, f): float(self.x.value[r_idx, f_idx])
                for r_idx, r in enumerate(self.R)
                for f_idx, f in enumerate(self.F)
            }

    def _print_pivot(self, RF2Val, fmt="g"):
        row_w = max(9, max(len(str(r)) for r in self.R) + 1)
        col_w = max(9, max(len(str(f)) for f in self.F) + 1)
        print(" " * row_w + "".join(f"{f:>{col_w}}" for f in self.F))
        for r in self.R:
            cells = "".join(format(float(RF2Val[r, f]), f">{col_w}{fmt}") for f in self.F)
            print(f"{r:>{row_w}}{cells}")

    def show_input(self):
        """Print R, F, RF2N, and RF2Prob to stdout."""
        if len(self.R) == 0 or len(self.F) == 0:
            raise RuntimeError("set_data() must be called before show_input()")
        print("=== show input ===")

        print("--- R (recency) ---")
        print(self.R)

        print("--- F (frequency) ---")
        print(self.F)

        print("--- RF2N ---")
        self._print_pivot(self.RF2N, fmt="g")

        print("--- RF2Prob ---")
        self._print_pivot(self.RF2Prob, fmt=".4f")

    def show_solve_info(self):
        """Print solver status, objective value, elapsed time, and problem size to stdout."""
        if self.status is None:
            raise RuntimeError("solve() must be called before show_solve_info()")
        obj_val = self.objective_value
        obj_str = f"{obj_val:.4f}" if (obj_val is not None and math.isfinite(obj_val)) else "N/A"
        print("=== show solve info ===")
        if self.axis is not None:
            print(f"axis: {self.axis}")
        else:
            print(f"kind: {self.kind}")
        print(f"eps: {self.eps}")
        print(f"status: {self.status}")
        print(f"objective_value: {obj_str}")
        print(f"elapsed_time: {self.elapsed_time:.2f}[s]")
        print(f"num_variables: {self.num_variables}")
        print(f"num_constraints: {self.num_constraints}")

    def show_result(self):
        """Print the optimized probability table to stdout."""
        if self.axis == "r":
            if not self.R2X:
                raise RuntimeError("postprocess() must be called before show_result()")
            print("=== show result (recency marginal) ===")
            col_w = 10
            print(f"{'recency':>{col_w}}{'probability':>{col_w}}")
            for r in self.R:
                print(f"{r:>{col_w}}{self.R2X[r]:>{col_w}.4f}")
        elif self.axis == "f":
            if not self.F2X:
                raise RuntimeError("postprocess() must be called before show_result()")
            print("=== show result (frequency marginal) ===")
            col_w = 10
            print(f"{'frequency':>{col_w}}{'probability':>{col_w}}")
            for f in self.F:
                print(f"{f:>{col_w}}{self.F2X[f]:>{col_w}.4f}")
        else:
            if not self.RF2X:
                raise RuntimeError("postprocess() must be called before show_result()")
            print("=== show result ===")
            self._print_pivot(self.RF2X, fmt=".4f")


if __name__ == "__main__":
    import pandas as pd

    # データの取得
    df = pd.read_csv("all_probability.csv")
    R = df.recency.unique()
    F = df.frequency.unique()
    RF2N = {(row.recency, row.frequency): row.N for row in df.itertuples()}
    RF2Prob = {(row.recency, row.frequency): row.emp_probability for row in df.itertuples()}

    # 数理モデルのインスタンスの作成とデータのセット
    optimizer = RecencyFrequencyOptimizer()
    optimizer.set_data(R, F, RF2N, RF2Prob)
    optimizer.show_input()

    # モデルの構築と求解
    optimizer.build_model(kind="mcc")
    optimizer.solve()
    optimizer.show_solve_info()

    # 後処理
    optimizer.postprocess()
    optimizer.show_result()
