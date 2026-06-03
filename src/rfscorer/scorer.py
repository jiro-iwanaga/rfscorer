import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_string_dtype
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
   
        # カラム名を内部処理用に変換
        self.interaction_log = df[[user_col, item_col, datetime_col]].copy() # dfへの影響を与えないため copy を実施
        self.interaction_log.columns = [self._USER_COL, self._ITEM_COL, self._DATETIME_COL]

        # 各カラムのキャストの実施（user:str, item:str, datetime:datetime）
        if not is_string_dtype(self.interaction_log[self._USER_COL]):
            self.interaction_log[self._USER_COL] = self.interaction_log[self._USER_COL].astype(str)
        if not is_string_dtype(self.interaction_log[self._ITEM_COL]):
            self.interaction_log[self._ITEM_COL] = self.interaction_log[self._ITEM_COL].astype(str)
        if not is_datetime64_any_dtype(self.interaction_log[self._DATETIME_COL]):
            self.interaction_log[self._DATETIME_COL] = pd.to_datetime(
                self.interaction_log[self._DATETIME_COL]
            )

        self.observation_start_date_ = None
        self.observation_end_date_ = None
        self.evaluation_start_date_ = None
        self.evaluation_end_date_ = None
        self.recency_limit = None # 最新度の上限値(デフォルトでは _RECENCY_LIMIT_RATE を用いて自動計算)
        self.frequency_limit = None # 頻度の上限値(デフォルトでは _FREQUENCY_LIMIT_RATE を用いて自動計算)
        self.R = [] # 最新度のリスト
        self.F = [] # 頻度のリスト

        self.empirical_probability_ = None # 経験的再閲覧確率データフレーム(縦持ち)
        self.empirical_probability_table = None # 経験的再閲覧確率データフレーム(横持ち)
        self.empirical_probability_dict = None # 経験的再閲覧確率データフレーム(辞書:キーは最新度と頻度のペア)

        # データ解析用
        self.record_num = len(self.interaction_log) # レコード数
        self.record_num_obs = None # 観測期間レコード数
        self.record_num_eval = None # 評価期間レコード数
        self.record_num_target_org = None # 分析対象フィルタリング前レコード数
        self.record_num_target = None # 分析対象レコード数
        self.total_cv_org = None # フィルタリング前 cv 数
        self.total_cv = None # cv 数


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

        # 観測期間の開始日と終了日、評価期間の開始日と終了日を datetime 型に変換
        obs_start, obs_end = pd.to_datetime(observation_period)
        eval_start, eval_end = pd.to_datetime(evaluation_period)

        # 観測期間の開始日と終了日、評価期間の開始日と終了日のバリデーション
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

        # 観測期間のデータフレームの作成
        df_obs = self.interaction_log[
            (self.observation_start_date_ <= self.interaction_log.datetime)
            & (self.interaction_log.datetime <= self.observation_end_date_)
            ]
        # 評価期間のデータフレームの作成
        df_eval = self.interaction_log[
            (self.evaluation_start_date_ <= self.interaction_log.datetime)
            & (self.interaction_log.datetime <= self.evaluation_end_date_)
            ]

        self.record_num_obs = len(df_obs)
        self.record_num_eval = len(df_eval)

        # 評価期間に cv がある user と item のペアを作成
        UIcv = set()
        for row in df_eval.itertuples():
            UIcv.add((row.user, row.item))

        U2I2Recensys = {}
        for row in df_obs.itertuples():
            # 最新度計算
            recency = (self.observation_end_date_ - row.datetime).days + 1
            U2I2Recensys.setdefault(row.user, {})
            U2I2Recensys[row.user].setdefault(row.item, [])
            U2I2Recensys[row.user][row.item].append(recency)
        #print(U2I2Recensys['2497'])
        #print(U2I2Recensys)

        # 閲覧履歴に最新度、頻度、cv の情報を追加
        RowsInteraction = []
        for user, I2Recencys in U2I2Recensys.items():
            for item, Recencys in I2Recencys.items():
                freq = len(Recencys)
                rcen = min(Recencys)
                cv = 1 if (user, item) in UIcv else 0 # 評価期間における cv(0/1) を追加
                new_row = (user, item, rcen, freq, cv)
                RowsInteraction.append(new_row)
        df_ui2frc = pd.DataFrame(RowsInteraction, columns=[self._USER_COL, self._ITEM_COL, 'recency', 'frequency', 'cv'])
        self.record_num_target_org = len(df_ui2frc) # レコード数の計測(フィルタリング前)
        self.total_cv_org = df_ui2frc.cv.sum() # cv数の計測(フィルタリング前)

        #print(df_ui2frc.shape)
        #print(df_ui2frc.head())
        #print(df_ui2frc.recency.unique())
        #print(df_ui2frc.frequency.unique())

        # 最新度の最大値の決定
        # 365日前のデータまで考える必要はない
        # TODO: 適切な最大値は要検討
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

        # 頻度の最大値の決定
        # 100閲覧のデータまで考える必要はない
        # TODO: 適切な最大値は要検討
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

        df_ui2frc = df_ui2frc[
            (df_ui2frc.recency <= self.recency_limit) # 最新度上限値以下のレコードをフィルタリング
              & (df_ui2frc.frequency <= self.frequency_limit) # 頻度上限値以下のレコードをフィルタリング
              ]
        self.record_num_target = len(df_ui2frc) # レコード数の計測
        self.total_cv = df_ui2frc.cv.sum() # cv数の計測

        # 最新度と頻度のリストを作成(上記の処理では抜けがある可能性を否定できない)
        self.R = list(range(1, self.recency_limit+1))
        self.F = list(range(1, self.frequency_limit+1))


        # 最新度と頻度のペアに対して、閲覧数、CV数、経験的再閲覧確率を初期化
        RF2N = {(r,f):0.0 for r in self.R for f in self.F}
        RF2CV = {(r,f):0.0 for r in self.R for f in self.F}

        # 最新度と頻度のペアに対して、閲覧数、CV数の集計
        for row in df_ui2frc.itertuples():
            RF2N[row.recency, row.frequency] += 1
            if row.cv == 1:
                RF2CV[row.recency, row.frequency] += 1
        
        # 経験的再閲覧確率の計算
        RowsRF = []
        for r in self.R:
            for f in self.F:
                if RF2N[r,f]>0:
                    prob = RF2CV[r,f] / RF2N[r,f]
                else:
                    prob = 0.0
                row_rf = (r, f, RF2N[r,f], RF2CV[r,f], prob)
                RowsRF.append(row_rf)

        # 経験的再閲覧確率辞書の作成
        self.empirical_probability_dict = {(r, f): prob for r, f, _, _, prob in RowsRF}

        # 経験的再閲覧確率データフレーム(縦持ち)の作成
        self.empirical_probability_ = pd.DataFrame(
            RowsRF,
            columns = ['recency', 'frequency', 'N', 'cv', 'probability']
            )

        # 経験的再閲覧確率データフレーム(横持ち)の作成
        self.empirical_probability_table = self.empirical_probability_.pivot_table(
            index='recency', 
            columns='frequency', 
            values='probability',
            )        

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
    
    
    def show(self):
        print('=== profiling ===')

        if self.record_num:
            print('record_num:', self.record_num)

        if self.record_num_obs:
            print('record_num_obs:', self.record_num_obs)
        if self.record_num_eval:
            print('record_num_eval:', self.record_num_eval)

        if self.observation_start_date_ and self.observation_end_date_:
            print('observation: {} -> {}'.format(self.observation_start_date_, self.observation_end_date_))
        if self.evaluation_start_date_ and self.evaluation_end_date_:
            print('evaluation: {} -> {}'.format(self.evaluation_start_date_, self.evaluation_end_date_))

        if self.recency_limit:
            print('recency_limit:', self.recency_limit)
        if self.frequency_limit:
            print('frequency_limit:', self.frequency_limit)

        if self.record_num_target_org and self.record_num_target:
            print('target_record_num: {} -> {}'.format(self.record_num_target_org, self.record_num_target))

        if self.total_cv_org and self.total_cv:
            print('total_cv: {} -> {}'.format(self.total_cv_org, self.total_cv))

        if self.empirical_probability_table is not None:
            print('empirical_probability_table:')
            print(self.empirical_probability_table.round(3).to_string())



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
    #print(scorer.interaction_log.head())

    scorer.show()


    # 経験的再閲覧確率の計算
    #observation_period = ('2015-07-01', '2015-07-06')
    #evaluation_period = ('2015-07-07', '2015-07-08')
    observation_period = ('2015-07-01', '2015-07-07')
    evaluation_period = ('2015-07-08', '2015-07-08')
    scorer.fit(observation_period, evaluation_period)

    scorer.show()

