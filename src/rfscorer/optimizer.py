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
        optimizer.build_model(kind='mono')
        optimizer.solve()
        optimizer.postprocess()
        # optimizer.RF2X holds the optimized probabilities
    """

    def __init__(self):
        # 集合と定数
        self.R = []
        self.F = []
        self.RF2N = {}
        self.RF2Prob = {}

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
            raise ValueError(
                f"RF2N is missing {len(missing_N)} key(s): {missing_N[:3]}{suffix}"
            )
        missing_Prob = [(r, f) for r in R for f in F if (r, f) not in RF2Prob]
        if missing_Prob:
            n = len(missing_Prob)
            suffix = "..." if n > 3 else ""
            raise ValueError(
                f"RF2Prob is missing {n} key(s): {missing_Prob[:3]}{suffix}"
            )

        self.R = list(R)
        self.F = list(F)
        self.RF2N = dict(RF2N)
        self.RF2Prob = dict(RF2Prob)

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

    def build_model(self, kind="mono"):
        """Build the optimization model.

        Parameters
        ----------
        kind : {'mono', 'mcc'}, default 'mono'
            'mono' applies monotonicity constraints only.
            'mcc' additionally applies convexity in recency and concavity in
            frequency (diminishing marginal returns).
        """
        if kind not in ("mono", "mcc"):
            raise ValueError(f"kind must be 'mono' or 'mcc', got '{kind}'")
        if len(self.R) == 0 or len(self.F) == 0:
            raise RuntimeError("set_data() must be called before build_model()")

        self.kind = kind

        nr = len(self.R)
        nf = len(self.F)

        self.x = cp.Variable((nr, nf))

        self.constraints = []

        # 確率の範囲制約: 0 <= x <= 1
        self.constraints.append(self.x >= 0)
        self.constraints.append(self.x <= 1)

        # Recency 単調性: r < r' => x[r,f] >= x[r',f]
        for r_idx in range(nr - 1):
            for f_idx in range(nf):
                self.constraints.append(
                    self.x[r_idx, f_idx] >= self.x[r_idx + 1, f_idx]
                )

        # Frequency 単調性: f < f' => x[r,f] <= x[r,f']
        for r_idx in range(nr):
            for f_idx in range(nf - 1):
                self.constraints.append(
                    self.x[r_idx, f_idx] <= self.x[r_idx, f_idx + 1]
                )

        if kind == "mcc":
            # Recency 凸性: 新しいほど効果が大きい（二階差分 >= 0）
            for r_idx in range(nr - 2):
                for f_idx in range(nf):
                    self.constraints.append(
                        self.x[r_idx, f_idx]
                        - 2 * self.x[r_idx + 1, f_idx]
                        + self.x[r_idx + 2, f_idx]
                        >= 0
                    )

            # Frequency 凹性: 限界効用逓減（二階差分 <= 0）
            for r_idx in range(nr):
                for f_idx in range(nf - 2):
                    self.constraints.append(
                        self.x[r_idx, f_idx]
                        - 2 * self.x[r_idx, f_idx + 1]
                        + self.x[r_idx, f_idx + 2]
                        <= 0
                    )

        self.objectives = []
        for r_idx, r in enumerate(self.R):
            for f_idx, f in enumerate(self.F):
                N = self.RF2N[r, f]
                p = self.RF2Prob[r, f]
                self.objectives.append(N * (self.x[r_idx, f_idx] - p) ** 2)

        self.problem = cp.Problem(
            cp.Minimize(cp.sum(self.objectives)), self.constraints
        )

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
            raise RuntimeError(f"Cannot postprocess: solver status is '{self.status}'")
        self.RF2X = {}
        for r_idx, r in enumerate(self.R):
            for f_idx, f in enumerate(self.F):
                self.RF2X[r, f] = float(self.x.value[r_idx, f_idx])

    def _print_pivot(self, RF2Val, fmt="g"):
        row_w = max(9, max(len(str(r)) for r in self.R) + 1)
        col_w = max(9, max(len(str(f)) for f in self.F) + 1)
        print(f"{'':>{row_w}}" + "".join(f"{f:>{col_w}}" for f in self.F))
        for r in self.R:
            cells = "".join(
                format(float(RF2Val[r, f]), f">{col_w}{fmt}") for f in self.F
            )
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
        obj_str = (
            f"{obj_val:.4f}"
            if (obj_val is not None and math.isfinite(obj_val))
            else "N/A"
        )
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
