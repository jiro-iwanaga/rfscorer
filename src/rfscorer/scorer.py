class RecencyFrequencyScorer:
    """Recency-Frequency based recommendation scorer.

    Estimates re-view probabilities from user-item interaction histories
    using recency and frequency as behavioral signals.
    """

    def fit(self, df, observation_dates, evaluation_dates):
        """Estimate empirical re-view probabilities from interaction history.

        Parameters
        ----------
        df : pd.DataFrame
            Interaction history with columns: user, item, datetime.
        observation_dates : array-like of datetime
            Dates in the observation period.
        evaluation_dates : array-like of datetime
            Dates in the evaluation period.

        Returns
        -------
        self
        """
        raise NotImplementedError

    def optimize(self):
        """Estimate optimized re-view probabilities under RF constraints.

        Solves a convex quadratic programming problem with recency and
        frequency monotonicity constraints.

        Returns
        -------
        self
        """
        raise NotImplementedError
