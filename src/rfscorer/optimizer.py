import cvxpy as cp

class RFOptimizer:
    def __init__(self):
        self.R = []
        self.F = []
        self.RF2N = {}
        self.RF2P = {}

        self.x = None
        self.constraints = None
        self.problem = None
        self.objectives = None
        self.RF2X = None

    def set_data(self, R, F, RF2N, RF2P):
        self.R = R
        self.F = F
        self.RF2N = RF2N
        self.RF2P = RF2P

    def build_model(self, kind='mono'):
        """Build the optimization model.

        Parameters
        ----------
        kind : {'mono', 'mcc'}, default 'mono'
            'mono' applies monotonicity constraints only.
            'mcc' additionally applies convexity in recency and concavity in
            frequency (diminishing marginal returns).
        """
        if kind not in ('mono', 'mcc'):
            raise ValueError(f"kind must be 'mono' or 'mcc', got '{kind}'")

        nr = len(self.R)
        nf = len(self.F)

        self.x = cp.Variable((nr, nf))

        self.constraints = []

        # Recency 単調性: r < r' => x[r,f] >= x[r',f]
        for r_idx in range(nr - 1):
            for f_idx in range(nf):
                self.constraints.append(self.x[r_idx, f_idx] >= self.x[r_idx + 1, f_idx])

        # Frequency 単調性: f < f' => x[r,f] <= x[r,f']
        for r_idx in range(nr):
            for f_idx in range(nf - 1):
                self.constraints.append(self.x[r_idx, f_idx] <= self.x[r_idx, f_idx + 1])

        if kind == 'mcc':
            # Recency 凸性: 新しいほど効果が大きい（二階差分 >= 0）
            for r_idx in range(nr - 2):
                for f_idx in range(nf):
                    self.constraints.append(
                        self.x[r_idx, f_idx] - 2 * self.x[r_idx + 1, f_idx] + self.x[r_idx + 2, f_idx] >= 0
                    )

            # Frequency 凹性: 限界効用逓減（二階差分 <= 0）
            for r_idx in range(nr):
                for f_idx in range(nf - 2):
                    self.constraints.append(
                        self.x[r_idx, f_idx] - 2 * self.x[r_idx, f_idx + 1] + self.x[r_idx, f_idx + 2] <= 0
                    )

        self.objectives = []
        for r_idx, r in enumerate(self.R):
            for f_idx, f in enumerate(self.F):
                N = self.RF2N[r, f]
                p = self.RF2P[r, f]
                self.objectives.append(N * (self.x[r_idx, f_idx] - p) ** 2)

        self.problem = cp.Problem(cp.Minimize(cp.sum(self.objectives)), self.constraints)

    def solve(self):
        self.problem.solve()

    def postprocess(self):
        self.RF2X = {}
        for r_idx, r in enumerate(self.R):
            for f_idx, f in enumerate(self.F):
                prob = float(self.x.value[r_idx, f_idx])
                self.RF2X[r, f] = prob


if __name__ == '__main__':
    pass
