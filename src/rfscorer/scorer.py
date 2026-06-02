class RecencyFrequencyScorer:
    """Recency-Frequency based recommendation scorer.

    Estimates re-view probabilities from user-item interaction histories
    using recency and frequency as behavioral signals.
    """
    def __init__(self, df, user_col="user", item_col='item', datetime_col='datetime'):
        self.user_col="user"
        self.item_col='item'
        self.datetime_col='datetime'
        self.interaction_log = (
            df[[user_col, item_col, datetime_col]]
            .rename(
                columns={
                    user_col: "user",
                    item_col: "item",
                    datetime_col: "datetime",
                }
            ).copy()
        )


    def fit(self, observation_period, evaluation_period):
        """Estimate empirical re-view probabilities from interaction history.

        Parameters
        ----------
        observation_period : tuple[str | datetime, str | datetime]
            Start and end dates of the observation period.
        evaluation_period : tuple[str | datetime, str | datetime]
            Start and end dates of the evaluation period.
        Returns
        -------
        self
        """
        
        if isinstance(self.interaction_log, pd.DataFrame):
            print('OK')
        else:
            raise ValueError

    def optimize(self):
        """Estimate optimized re-view probabilities under RF constraints.

        Solves a convex quadratic programming problem with recency and
        frequency monotonicity constraints.

        Returns
        -------
        self
        """
        raise NotImplementedError

if __name__ == "__main__":
    print('=== scorer.py ===')
    import pandas as pd
    df = pd.read_csv("../../examples/access_log.csv")
    scorer = RecencyFrequencyScorer(
        df,
        user_col = "user_id",
        item_col = "item_id",
        datetime_col = "date",
        )
    print(scorer.interaction_log.head())

    observation_period = ('2015-07-02', '2015-07-06')
    evaluation_period = ('2015-07-07', '2015-07-08')
    scorer.fit(observation_period, evaluation_period)

