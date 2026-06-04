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

    def build_model(self):
        nr = len(self.R)
        nf = len(self.F)

        self.x = cp.Variable((nr, nf))

        self.constraints = []

        # Recency 制約: r < r' => x[r,f] >= x[r',f]
        for r_idx in range(nr - 1):
            for f_idx in range(nf):
                self.constraints.append(self.x[r_idx, f_idx] >= self.x[r_idx + 1, f_idx])

        # Frequency 制約: f < f' => x[r,f] <= x[r,f']
        for r_idx in range(nr):
            for f_idx in range(nf - 1):
                self.constraints.append(self.x[r_idx, f_idx] <= self.x[r_idx, f_idx + 1])

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
