import math
import time

import cvxpy as cp


class RFOptimizer:
    """Convex quadratic optimizer for RF monotonicity-constrained revisit probabilities.

    Minimizes weighted least-squares deviation from empirical probabilities
    over a recency-frequency grid under RF constraints. Intended to be called
    from RecencyFrequencyScorer.optimize().

    Typical call sequence::

        optimizer = RFOptimizer()
        optimizer.set_data(R, F, RF2N, RF2Prob)
        optimizer.build_model(kind="mono")
        optimizer.solve()
        optimizer.show_solve_info()  # optional
        optimizer.postprocess()
        # optimizer.RF2X holds the optimized probabilities
    """

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
        self.x = None
        self.constraints = None
        self.objectives = None
        self.problem = None
        self.num_variables = None
        self.num_constraints = None

        # 最適化計算結果
        self.elapsed_time = None
        self.status = None
        self.objective_value = None
        self.RF2X = {}

    def set_data(self, R, F, RF2N, RF2Prob):
        """Load recency-frequency data before building the model.

        Parameters
        ----------
        R : list[int]
            Unique recency values (1 = most recent).
        F : list[int]
            Unique frequency values.
        RF2N : dict[tuple[int, int], float]
            Number of observations for each (recency, frequency) pair.
        RF2Prob : dict[tuple[int, int], float]
            Empirical revisit probability for each (recency, frequency) pair.
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
            n = len(missing_Prob)
            suffix = "..." if n > 3 else ""
            raise ValueError(f"RF2Prob is missing {n} key(s): {missing_Prob[:3]}{suffix}")

        self.R = list(R)
        self.F = list(F)
        self.RF2N = dict(RF2N)
        self.RF2Prob = dict(RF2Prob)
        self.R2N = {}
        self.R2Prob = {}
        self.F2N = {}
        self.F2Prob = {}

        # 初期化
        self.kind = None
        self.x = None
        self.constraints = None
        self.objectives = None
        self.problem = None
        self.num_variables = None
        self.num_constraints = None
        self.elapsed_time = None
        self.status = None
        self.objective_value = None
        self.RF2X = {}

    def set_marginal_data(self, R2N, R2Prob, F2N, F2Prob):
        """Load marginal recency and frequency data for 1-D optimization models.

        Must be called after set_data() and before build_model() with kind='mr' or 'mf'.

        Parameters
        ----------
        R2N : dict[int, float]
            Number of observations for each recency value.
        R2Prob : dict[int, float]
            Empirical revisit probability for each recency value.
        F2N : dict[int, float]
            Number of observations for each frequency value.
        F2Prob : dict[int, float]
            Empirical revisit probability for each frequency value.
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

    def build_model(self, kind="mono"):
        """Build the optimization model.

        Parameters
        ----------
        kind : {"mono", "mrc", "mfc", "mcc"}, default "mono"
            "mono" applies monotonicity constraints only.
            "mrc" additionally applies convexity in recency (diminishing
            marginal penalty as recency grows).
            "mfc" additionally applies concavity in frequency (diminishing
            marginal returns as frequency grows).
            "mcc" applies both recency convexity and frequency concavity.
        """
        if kind not in ("mono", "mr", "mf", "mrc", "mfc", "mcc"):
            raise ValueError(
                f"kind must be 'mono', 'mr', 'mf', 'mrc', 'mfc', or 'mcc', got {kind!r}"
            )
        if len(self.R) == 0 or len(self.F) == 0:
            raise RuntimeError("set_data() must be called before build_model()")
        if kind in ("mr", "mf") and not self.R2N:
            raise RuntimeError(
                "set_marginal_data() must be called before build_model() with kind='mr' or 'mf'"
            )

        self.kind = kind

        nr = len(self.R)
        nf = len(self.F)

        self.constraints = []
        self.objectives = []

        if kind == "mr":
            self.x = cp.Variable(nr)
            self.constraints.append(self.x >= 0)
            self.constraints.append(self.x <= 1)
            # Recency 単調性: r < r' => x[r] >= x[r']
            for r_idx in range(nr - 1):
                self.constraints.append(self.x[r_idx] >= self.x[r_idx + 1])
            # Recency 凸性（二階差分 >= 0）
            for r_idx in range(nr - 2):
                self.constraints.append(
                    self.x[r_idx] - 2 * self.x[r_idx + 1] + self.x[r_idx + 2] >= 0
                )
            for r_idx, r in enumerate(self.R):
                N = self.R2N[r]
                p = self.R2Prob[r]
                self.objectives.append(N * (self.x[r_idx] - p) ** 2)

        elif kind == "mf":
            self.x = cp.Variable(nf)
            self.constraints.append(self.x >= 0)
            self.constraints.append(self.x <= 1)
            # Frequency 単調性: f < f' => x[f] <= x[f']
            for f_idx in range(nf - 1):
                self.constraints.append(self.x[f_idx] <= self.x[f_idx + 1])
            # Frequency 凹性（二階差分 <= 0）
            for f_idx in range(nf - 2):
                self.constraints.append(
                    self.x[f_idx] - 2 * self.x[f_idx + 1] + self.x[f_idx + 2] <= 0
                )
            for f_idx, f in enumerate(self.F):
                N = self.F2N[f]
                p = self.F2Prob[f]
                self.objectives.append(N * (self.x[f_idx] - p) ** 2)

        else:
            self.x = cp.Variable((nr, nf))
            self.constraints.append(self.x >= 0)
            self.constraints.append(self.x <= 1)
            # Recency 単調性: r < r' => x[r,f] >= x[r',f]
            for r_idx in range(nr - 1):
                for f_idx in range(nf):
                    self.constraints.append(self.x[r_idx, f_idx] >= self.x[r_idx + 1, f_idx])
            # Frequency 単調性: f < f' => x[r,f] <= x[r,f']
            for r_idx in range(nr):
                for f_idx in range(nf - 1):
                    self.constraints.append(self.x[r_idx, f_idx] <= self.x[r_idx, f_idx + 1])
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

        # 初期化
        self.elapsed_time = None
        self.status = None
        self.objective_value = None
        self.RF2X = {}

    def solve(self):
        """Solve the optimization problem built by build_model().

        Sets status, objective_value, and elapsed_time.
        Raises RuntimeError if build_model() has not been called.
        Solver failures are not raised here; call postprocess() to detect them.
        """
        if self.problem is None:
            raise RuntimeError("build_model() must be called before solve()")

        start_time = time.time()
        self.problem.solve()
        self.elapsed_time = time.time() - start_time

        self.status = self.problem.status
        self.objective_value = self.problem.value

    def postprocess(self):
        """Extract optimized probabilities from the solver solution into RF2X.

        Must be called after solve(). Raises RuntimeError if the solver did
        not find a feasible solution.
        """
        if self.status is None:
            raise RuntimeError("solve() must be called before postprocess()")
        if self.x.value is None:
            raise RuntimeError(f"Cannot postprocess: solver status is {self.status!r}")
        self.RF2X = {}
        if self.kind == "mr":
            for r_idx, r in enumerate(self.R):
                val = float(self.x.value[r_idx])
                for f in self.F:
                    self.RF2X[r, f] = val
        elif self.kind == "mf":
            for f_idx, f in enumerate(self.F):
                val = float(self.x.value[f_idx])
                for r in self.R:
                    self.RF2X[r, f] = val
        else:
            for r_idx, r in enumerate(self.R):
                for f_idx, f in enumerate(self.F):
                    self.RF2X[r, f] = float(self.x.value[r_idx, f_idx])

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
        print(f"kind: {self.kind}")
        print(f"status: {self.status}")
        print(f"objective_value: {obj_str}")
        print(f"elapsed_time: {self.elapsed_time:.2f}[s]")
        print(f"num_variables: {self.num_variables}")
        print(f"num_constraints: {self.num_constraints}")

    def show_result(self):
        """Print the optimized probability table (RF2X) to stdout."""
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
    RF2Prob = {(row.recency, row.frequency): row.empirical_probability for row in df.itertuples()}

    # 数理モデルのインスタンスの作成とデータのセット
    optimizer = RFOptimizer()
    optimizer.set_data(R, F, RF2N, RF2Prob)
    optimizer.show_input()

    # モデルの構築と探索
    optimizer.build_model(kind="mcc")
    optimizer.solve()
    optimizer.show_solve_info()

    # 後処理
    optimizer.postprocess()
    optimizer.show_result()
