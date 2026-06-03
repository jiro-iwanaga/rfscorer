import pandas as pd
from pandas.api.types import is_datetime64_any_dtype
#import numpy as np
#import matplotlib.pyplot as plt

class RecencyFrequencyScorer:
    """Recency-Frequency based recommendation scorer.

    Estimates revisit probabilities from user-item interaction histories
    using recency and frequency as behavioral signals.
    """

    _USER_COL = "user"
    _ITEM_COL = "item"
    _DATETIME_COL = "datetime"
    _FREQUENCY_LIMIT_RATE = 0.95
    _RECENCY_LIMIT_RATE = 0.95
        
    def __init__(self, df, user_col="user", item_col='item', datetime_col='datetime'):
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        required_columns = [user_col, item_col, datetime_col]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

   
        self.interaction_log = (
            df[[user_col, item_col, datetime_col]]
            .rename(
                columns={
                    user_col: self._USER_COL,
                    item_col: self._ITEM_COL,
                    datetime_col: self._DATETIME_COL,
                }
            ).copy()
        )

        self.interaction_log[self._USER_COL] = self.interaction_log[self._USER_COL].astype(str)
        self.interaction_log[self._ITEM_COL] = self.interaction_log[self._ITEM_COL].astype(str)
        if not is_datetime64_any_dtype(self.interaction_log[self._DATETIME_COL]):
            self.interaction_log[self._DATETIME_COL] = pd.to_datetime(
                self.interaction_log[self._DATETIME_COL]
            )

        self.recency_limit = None
        self.frequency_limit = None
        self.list_recency = []
        self.list_frequency = []

        self.empirical_probability_ = None


    def fit(self, observation_period, evaluation_period, recency_limit=None, frequency_limit=None):
        """Estimate empirical revisit probabilities from interaction history.

        Parameters
        ----------
        observation_period : tuple[str | datetime, str | datetime]
            Start and end dates of the observation period.
        evaluation_period : tuple[str | datetime, str | datetime]
            Start and end dates of the evaluation period.
        recency_limit : int, optional
            Maximum recency rank to include. If None, automatically
            determined from the cumulative cv distribution using
            RECENCY_LIMIT_RATE.
        frequency_limit : int, optional
            Maximum frequency to include. If None, automatically determined
            from the cumulative cv distribution using FREQUENCY_LIMIT_RATE.

        Returns
        -------
        self
        """

        obs_start, obs_end = pd.to_datetime(observation_period)
        eval_start, eval_end = pd.to_datetime(evaluation_period)

        if obs_start > obs_end:
            raise ValueError("observation_period must be ordered as (start, end).")

        if eval_start > eval_end:
            raise ValueError("evaluation_period must be ordered as (start, end).")

        if obs_end >= eval_start:
            raise ValueError(
                "observation_period must end before evaluation_period starts."
            )

        self.observation_start_date_ = obs_start
        self.observation_end_date_ = obs_end
        self.evaluation_start_date_ = eval_start
        self.evaluation_end_date_ = eval_end

        df_obs = self.interaction_log[
            (self.observation_start_date_ <= self.interaction_log.datetime)
            & (self.interaction_log.datetime <= self.observation_end_date_)
            ]
        df_eval = self.interaction_log[
            (self.evaluation_start_date_ <= self.interaction_log.datetime)
            & (self.interaction_log.datetime <= self.evaluation_end_date_)
            ]
        
        #print(df_obs.shape, df_obs.datetime.unique())
        #print(df_eval.shape, df_eval.datetime.unique())

        UI2cv = {}
        for row in df_eval.itertuples():
            UI2cv[row.user, row.item] = 1

        U2I2Recensys = {}
        for row in df_obs.itertuples():
            recency = (self.observation_end_date_ - row.datetime).days + 1
            U2I2Recensys.setdefault(row.user, {})
            U2I2Recensys[row.user].setdefault(row.item, [])
            U2I2Recensys[row.user][row.item].append(recency)
        #print(U2I2Recensys['2497'])
        #print(U2I2Recensys)

        RowsLog = []
        for user, I2Recencys in U2I2Recensys.items():
            for item, Recencys in I2Recencys.items():
                freq = len(Recencys)
                rcen = min(Recencys)
                cv = 1 if (user, item) in UI2cv else 0
                new_row = (user, item, rcen, freq, cv)
                RowsLog.append(new_row)

        df_ui2frc = pd.DataFrame(RowsLog, columns=[self._USER_COL, self._ITEM_COL, 'recency', 'frequency', 'cv'])
        #print(df_ui2frc.shape)
        #print(df_ui2frc.head())
        #print(df_ui2frc.recency.unique())
        #print(df_ui2frc.frequency.unique())

        if recency_limit is not None:
            self.recency_limit = recency_limit
        else:
            df_recency2cv = df_ui2frc.groupby('recency')['cv'].sum().reset_index()
            df_recency2cv.sort_values('recency', inplace=True)
            total_cv = df_recency2cv.cv.sum()
            if total_cv == 0:
                raise ValueError(
                    "No revisits observed in evaluation_period. "
                    "Cannot determine recency_limit automatically."
                )
            cv_sum = 0
            for row in df_recency2cv.itertuples():
                cv_sum += row.cv
                if cv_sum / total_cv >= self._RECENCY_LIMIT_RATE:
                    recency_limit = row.recency
                    break
            self.recency_limit = recency_limit

        if frequency_limit is not None:
            self.frequency_limit = frequency_limit
        else:
            df_frequency2cv = df_ui2frc.groupby('frequency')['cv'].sum().reset_index()
            df_frequency2cv.sort_values('frequency', inplace=True)
            total_cv = df_frequency2cv.cv.sum()
            if total_cv == 0:
                raise ValueError(
                    "No revisits observed in evaluation_period. "
                    "Cannot determine frequency_limit automatically."
                )
            cv_sum = 0
            for row in df_frequency2cv.itertuples():
                cv_sum += row.cv
                if cv_sum / total_cv >= self._FREQUENCY_LIMIT_RATE:
                    frequency_limit = row.frequency
                    break
            self.frequency_limit = frequency_limit

        self.list_recency = list(range(1, self.recency_limit+1))
        self.list_frequency = list(range(1, self.frequency_limit+1))

        #print(self.recency_limit, self.list_recency)
        #print(self.frequency_limit, self.list_frequency)
        df_ui2frc = df_ui2frc[(df_ui2frc.frequency <= self.frequency_limit) & (df_ui2frc.recency <= self.recency_limit)]
        #print('cv:', df_ui2frc.cv.sum())

        RF2N = {}
        RF2CV = {}
        for row in df_ui2frc.itertuples():
            RF2N.setdefault((row.recency, row.frequency), 0)
            RF2CV.setdefault((row.recency, row.frequency), 0)

            RF2N[row.recency, row.frequency] += 1
            if row.cv == 1:
                RF2CV[row.recency, row.frequency] += 1

        for recency in self.list_recency:
            for frequency in self.list_frequency:
                if (recency, frequency) not in RF2CV:
                    RF2CV[recency, frequency] = 0
                    RF2N[recency, frequency] = 0
        
        RowsTable = []
        for (recency, frequency), N in sorted(RF2N.items()):
            cv = RF2CV[recency, frequency]
            prob = cv / N if N!=0 else 0.0
            row_table = (recency, frequency, N, cv, prob)
            RowsTable.append(row_table)

        self.empirical_probability_ = pd.DataFrame(RowsTable, columns = ['recency', 'frequency', 'N', 'cv', 'probability'])
        #print(self.empirical_probability_.shape)
        #print(self.empirical_probability_)
        #self.df_table_empirical = self.empirical_probability_.pivot_table(index='recency', columns='frequency', values='probability')
        #print(self.df_table_empirical)


        #Frequency = df_rf['frequency'].unique().tolist()
        #Recency = df_rf['recency'].unique().tolist()
        #Z = [df_rf[(df_rf['frequency']==freq) & (df_rf['recency']==rcen)]['probability'].iloc[0] for freq in Frequency for rcen in Recency]
        #Z = np.array(Z).reshape((len(Frequency), len(Recency)))
        #X, Y = np.meshgrid(Recency, Frequency)  
        #fig = plt.figure()
        #ax = fig.add_subplot(
        #    111, 
        #    projection='3d',
        #    xlabel='recency',
        #    ylabel='frequency',
        #    zlabel='probability',
        #    )
        #ax.plot_wireframe(X, Y, Z)        


        return self

    def optimize(self):
        """Estimate optimized revisit probabilities under RF constraints.

        Solves a convex quadratic programming problem with recency and
        frequency monotonicity constraints.

        Returns
        -------
        self
        """
        raise NotImplementedError

if __name__ == "__main__":
    print('=== scorer.py ===')

    # サンプルデータの取得
    df = pd.read_csv("../../examples/access_log.csv")

    # スコアリングインスタンスの作成
    scorer = RecencyFrequencyScorer(
        df,
        user_col = "user_id",
        item_col = "item_id",
        datetime_col = "date",
        )
    print(scorer.interaction_log.head())

    # 経験的再閲覧確率の計算
    #observation_period = ('2015-07-01', '2015-07-06')
    #evaluation_period = ('2015-07-07', '2015-07-08')
    observation_period = ('2015-07-01', '2015-07-07')
    evaluation_period = ('2015-07-08', '2015-07-08')
    scorer.fit(observation_period, evaluation_period)
    print('observation:', scorer.observation_start_date_, scorer.observation_end_date_)
    print('evaluation :', scorer.evaluation_start_date_, scorer.evaluation_end_date_)

