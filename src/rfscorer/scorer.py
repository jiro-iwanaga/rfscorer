import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_string_dtype


class RecencyFrequencyScorer:
    """Recency-Frequency based recommendation scorer.

    Estimates revisit probabilities from user-item interaction histories
    using recency and frequency as behavioral signals.
    """

    _USER_COL = "user"
    _ITEM_COL = "item"
    _DATETIME_COL = "datetime"
    _FREQUENCY_LIMIT_RATE = 0.95 # 最新度上限値自動計算の際に利用する割合
    _RECENCY_LIMIT_RATE = 0.95 # 頻度上限値自動計算の際に利用する割合
        
    def __init__(self, user_col="user", item_col='item', datetime_col='datetime'):
        self.user_col = user_col
        self.item_col = item_col
        self.datetime_col = datetime_col

        self.observation_start_date_ = None
        self.observation_end_date_ = None
        self.evaluation_start_date_ = None
        self.evaluation_end_date_ = None
        self.recency_limit = None # 最新度の上限値(デフォルトでは _RECENCY_LIMIT_RATE を用いて自動計算)
        self.frequency_limit = None # 頻度の上限値(デフォルトでは _FREQUENCY_LIMIT_RATE を用いて自動計算)

        self.R = [] # 最新度のリスト
        self.F = [] # 頻度のリスト
        self.RF2N = {} # 最新度と頻度に対して閲覧数合計を紐づける辞書
        self.RF2CV = {} # 最新度と頻度に対して再閲覧数合計を紐づける辞書
        self.RF2Prob = {} # 最新度と頻度に対して経験的再閲覧確率を紐づける辞書

        # empirical
        self.empirical_probability_ = None # 経験的再閲覧確率データフレーム(縦持ち)
        self.empirical_probability_table_ = None # 経験的再閲覧確率データフレーム(横持ち)
        self.empirical_probability_dict_ = None # 経験的再閲覧確率データフレーム(辞書:キーは最新度と頻度のペア)

        # mono
        self.mono_probability_ = None
        self.mono_probability_table_ = None
        self.mono_probability_dict_ = None

        # mcc
        self.mcc_probability_ = None
        self.mcc_probability_table_ = None
        self.mcc_probability_dict_ = None

        # データ解析用
        self.record_num = None # レコード数（fit() 後に設定）
        self.record_num_obs = None # 観測期間レコード数
        self.record_num_eval = None # 評価期間レコード数
        self.record_num_target_org = None # 分析対象フィルタリング前レコード数
        self.record_num_target = None # 分析対象レコード数
        self.total_cv_org = None # フィルタリング前 cv 数
        self.total_cv = None # cv 数


    def fit(self, df, observation_period, evaluation_period, recency_limit=None, frequency_limit=None):
        """Estimate empirical revisit probabilities from interaction history.

        Parameters
        ----------
        df : pd.DataFrame
            Interaction log containing user, item, and datetime columns.
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

        # df のバリデーションと内部形式への変換
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        required_columns = [self.user_col, self.item_col, self.datetime_col]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        interaction_log = df[[self.user_col, self.item_col, self.datetime_col]].copy()
        interaction_log.columns = [self._USER_COL, self._ITEM_COL, self._DATETIME_COL]

        if not is_string_dtype(interaction_log[self._USER_COL]):
            interaction_log[self._USER_COL] = interaction_log[self._USER_COL].astype(str)
        if not is_string_dtype(interaction_log[self._ITEM_COL]):
            interaction_log[self._ITEM_COL] = interaction_log[self._ITEM_COL].astype(str)
        if not is_datetime64_any_dtype(interaction_log[self._DATETIME_COL]):
            interaction_log[self._DATETIME_COL] = pd.to_datetime(
                interaction_log[self._DATETIME_COL]
            )

        self.record_num = len(interaction_log)

        # 観測期間の開始日と終了日、評価期間の開始日と終了日を datetime 型に変換
        if len(observation_period) != 2:
            raise ValueError("observation_period must be a tuple of (start, end).")
        try:
            obs_start, obs_end = pd.to_datetime(observation_period)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"observation_period could not be parsed as dates: {observation_period}"
            ) from e

        if len(evaluation_period) != 2:
            raise ValueError("evaluation_period must be a tuple of (start, end).")
        try:
            eval_start, eval_end = pd.to_datetime(evaluation_period)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"evaluation_period could not be parsed as dates: {evaluation_period}"
            ) from e

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
        df_obs = interaction_log[
            (self.observation_start_date_ <= interaction_log.datetime)
            & (interaction_log.datetime <= self.observation_end_date_)
            ]
        # 評価期間のデータフレームの作成
        df_eval = interaction_log[
            (self.evaluation_start_date_ <= interaction_log.datetime)
            & (interaction_log.datetime <= self.evaluation_end_date_)
            ]

        self.record_num_obs = len(df_obs)
        self.record_num_eval = len(df_eval)

        # 評価期間に cv がある user と item のペアを作成
        UIcv = {(row.user, row.item) for row in df_eval.itertuples()}

        U2I2Recensys = self._build_u2i2recencies(df_obs, self.observation_end_date_)

        # 閲覧履歴に最新度、頻度、cv の情報を追加
        rows = [
            (user, item, recency, frequency, int((user, item) in UIcv))
            for user, item, recency, frequency in self._iter_u2i_rf(U2I2Recensys)
        ]
        df_ui2frc = pd.DataFrame(
            rows,
            columns=[self._USER_COL, self._ITEM_COL, 'recency', 'frequency', 'cv']
        )
        self.record_num_target_org = len(df_ui2frc) # レコード数の計測(フィルタリング前)
        self.total_cv_org = df_ui2frc.cv.sum() # cv数の計測(フィルタリング前)

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

        # 最新度上限値と頻度上限値でフィルタリング
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
        self.RF2N = {(r,f):0.0 for r in self.R for f in self.F}
        self.RF2CV = {(r,f):0.0 for r in self.R for f in self.F}

        # 最新度と頻度のペアに対して、閲覧数、CV数の集計
        for row in df_ui2frc.itertuples():
            self.RF2N[row.recency, row.frequency] += 1
            if row.cv == 1:
                self.RF2CV[row.recency, row.frequency] += 1
        
        # 経験的再閲覧確率の計算
        RowsRF = []
        for r in self.R:
            for f in self.F:
                if self.RF2N[r,f]>0:
                    prob = self.RF2CV[r,f] / self.RF2N[r,f]
                else:
                    prob = 0.0
                self.RF2Prob[r,f] = prob
                row_rf = (r, f, self.RF2N[r,f], self.RF2CV[r,f], prob)
                RowsRF.append(row_rf)

        # 経験的再閲覧確率辞書の作成
        self.empirical_probability_dict_ = {(r, f): prob for r, f, _, _, prob in RowsRF}

        # 経験的再閲覧確率データフレーム(縦持ち)の作成
        self.empirical_probability_ = pd.DataFrame(
            RowsRF,
            columns = ['recency', 'frequency', 'N', 'cv', 'probability']
            )

        # 経験的再閲覧確率データフレーム(横持ち)の作成
        self.empirical_probability_table_ = self.empirical_probability_.pivot_table(
            index='recency', 
            columns='frequency', 
            values='probability',
            )        

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

        if self.empirical_probability_table_ is not None:
            print('empirical_probability_table_:')
            print(self.empirical_probability_table_.round(3).to_string())



    def plot_probability_surface(self, kind='empirical', path=None):
        """Plot revisit probabilities as a 3D surface and save to file.

        Visualizes the probability table as a 3D wireframe with recency on
        the x-axis, frequency on the y-axis, and probability on the z-axis.

        Parameters
        ----------
        kind : {'empirical', 'optimized'}, default 'empirical'
            Which probability to visualize.
        path : str or None, default None
            Output file path for the PNG image. If None, saves as
            '{kind}_probability_surface.png' in the current directory.

        Returns
        -------
        None
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if kind not in ('empirical', 'mono', 'mcc'):
            raise ValueError(f"kind must be 'empirical', 'mono', or 'mcc', got '{kind}'.")
        if kind == 'empirical' and self.empirical_probability_table_ is None:
            raise RuntimeError("fit() must be called before plot_probability_surface().")
        if kind == 'mono' and self.mono_probability_table_ is None:
            raise RuntimeError("optimize(kind='mono') must be called before plot_probability_surface(kind='mono').")
        if kind == 'mcc' and self.mcc_probability_table_ is None:
            raise RuntimeError("optimize(kind='mcc') must be called before plot_probability_surface(kind='mcc').")

        from pathlib import Path
        default_filename = f'surface_{kind}_probability.png'
        if path is None:
            output_path = Path(default_filename)
        else:
            p = Path(path)
            output_path = p / default_filename if p.is_dir() else p

        if kind == 'empirical':
            table = self.empirical_probability_table_
        elif kind == 'mono':
            table = self.mono_probability_table_
        else:
            table = self.mcc_probability_table_

        recency = table.index.tolist()
        frequency = table.columns.tolist()
        X, Y = np.meshgrid(recency, frequency)
        Z = table.values.T

        fig = plt.figure()
        ax = fig.add_subplot(
            111,
            projection='3d',
            xlabel='recency',
            ylabel='frequency',
            zlabel='probability',
        )
        ax.plot_wireframe(X, Y, Z)
        plt.savefig(output_path)
        plt.close(fig)

    def export_probability_csv(self, kind='empirical', path=None):
        """Export revisit probabilities to a CSV file.

        Parameters
        ----------
        kind : {'empirical', 'optimized', 'all'}, default 'empirical'
            Which probability to export. 'all' merges both empirical and
            optimized into a single file with columns
            empirical_probability and optimized_probability.
        path : str or None, default None
            Output file path for the CSV. If None, saves as
            '{kind}_probability.csv' in the current directory.
            If a directory, saves '{kind}_probability.csv' inside it.

        Returns
        -------
        None
        """
        if kind not in ('empirical', 'mono', 'mcc', 'all'):
            raise ValueError(f"kind must be 'empirical', 'mono', 'mcc', or 'all', got '{kind}'.")
        if kind in ('empirical', 'all') and self.empirical_probability_ is None:
            raise RuntimeError("fit() must be called before export_probability_csv().")
        if kind in ('mono', 'all') and self.mono_probability_ is None:
            raise RuntimeError("optimize(kind='mono') must be called before export_probability_csv(kind='mono').")
        if kind in ('mcc', 'all') and self.mcc_probability_ is None:
            raise RuntimeError("optimize(kind='mcc') must be called before export_probability_csv(kind='mcc').")

        from pathlib import Path
        default_filename = f'{kind}_probability.csv'
        if path is None:
            output_path = Path(default_filename)
        else:
            p = Path(path)
            output_path = p / default_filename if p.is_dir() else p

        if kind == 'all':
            df = (
                self.empirical_probability_
                .rename(columns={'probability': 'empirical_probability'})
                .merge(
                    self.mono_probability_.rename(columns={'probability': 'mono_probability'}),
                    on=['recency', 'frequency'],
                )
                .merge(
                    self.mcc_probability_.rename(columns={'probability': 'mcc_probability'}),
                    on=['recency', 'frequency'],
                )
            )
        elif kind == 'empirical':
            df = self.empirical_probability_
        elif kind == 'mono':
            df = self.mono_probability_
        else:
            df = self.mcc_probability_
        df.to_csv(output_path, index=False)

    def predict(self, r, f, kind='empirical'):
        """Return the revisit probability for a given recency and frequency.

        Parameters
        ----------
        r : int
            Recency rank.
        f : int
            Frequency.
        kind : {'empirical', 'optimized'}, default 'empirical'
            Which probability to use. 'empirical' uses empirical_probability_dict_
            estimated by fit(). 'optimized' uses the result of optimize().

        Returns
        -------
        float
            Revisit probability for the given (r, f).
        """

        # 最新度 r と頻度 f のバリデーション
        if not isinstance(r, int) or r < 1:
            raise TypeError("r must be a positive integer.")
        if not isinstance(f, int) or f < 1:
            raise TypeError("f must be a positive integer.")
        if kind not in ('empirical', 'mono', 'mcc'):
            raise ValueError(f"kind must be 'empirical', 'mono', or 'mcc', got '{kind}'.")
        if kind == 'empirical' and self.empirical_probability_dict_ is None:
            raise RuntimeError("fit() must be called before predict().")
        if kind == 'mono' and self.mono_probability_dict_ is None:
            raise RuntimeError("optimize(kind='mono') must be called before predict(kind='mono').")
        if kind == 'mcc' and self.mcc_probability_dict_ is None:
            raise RuntimeError("optimize(kind='mcc') must be called before predict(kind='mcc').")

        r = min(r, self.recency_limit)
        f = min(f, self.frequency_limit)
        if kind == 'empirical':
            prob = self.empirical_probability_dict_.get((r, f), 0.0)
        elif kind == 'mono':
            prob = self.mono_probability_dict_.get((r, f), 0.0)
        else:
            prob = self.mcc_probability_dict_.get((r, f), 0.0)
        return prob

    def transform(self, df, target_date, kind='empirical', user_col='user', item_col='item', datetime_col='datetime'):
        """Add recency, frequency, and revisit probability columns to a DataFrame.

        Computes recency rank (r) and frequency (f) for each user-item pair
        relative to target_date, then appends the corresponding revisit
        probability from fit() or optimize() results.

        Parameters
        ----------
        df : pd.DataFrame
            User-item interaction history to score.
        target_date : str or datetime
            Reference date for computing recency and frequency.
            Rows after this date are excluded.
        kind : {'empirical', 'optimized'}, default 'empirical'
            Which probability to use.
        user_col : str, optional
            Column name for user. Defaults to the value used in __init__.
        item_col : str, optional
            Column name for item. Defaults to the value used in __init__.
        datetime_col : str, optional
            Column name for datetime. Defaults to the value used in __init__.

        Returns
        -------
        pd.DataFrame
            Copy of df with added columns: recency, frequency, probability.
        """
        if kind not in ('empirical', 'mono', 'mcc'):
            raise ValueError(f"kind must be 'empirical', 'mono', or 'mcc', got '{kind}'.")
        if self.empirical_probability_dict_ is None:
            raise RuntimeError("fit() must be called before transform().")
        if kind == 'mono' and self.mono_probability_dict_ is None:
            raise RuntimeError("optimize(kind='mono') must be called before transform(kind='mono').")
        if kind == 'mcc' and self.mcc_probability_dict_ is None:
            raise RuntimeError("optimize(kind='mcc') must be called before transform(kind='mcc').")

        user_col = user_col or self._USER_COL
        item_col = item_col or self._ITEM_COL
        datetime_col = datetime_col or self._DATETIME_COL

        # 基準日を datetime 型に変換
        try:
            target_date = pd.to_datetime(target_date)
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"target_date could not be parsed as a date: {target_date}"
            ) from e

        df_log = df[[user_col, item_col, datetime_col]].copy()
        df_log.columns = [self._USER_COL, self._ITEM_COL, self._DATETIME_COL]

        if not is_string_dtype(df_log[self._USER_COL]):
            df_log[self._USER_COL] = df_log[self._USER_COL].astype(str)
        if not is_string_dtype(df_log[self._ITEM_COL]):
            df_log[self._ITEM_COL] = df_log[self._ITEM_COL].astype(str)
        if not is_datetime64_any_dtype(df_log[self._DATETIME_COL]):
            df_log[self._DATETIME_COL] = pd.to_datetime(df_log[self._DATETIME_COL])

        # 基準日(target_date)までのレコードに絞る
        df_log = df_log[df_log[self._DATETIME_COL] <= target_date]

        # user と item ごとに最新度を計算
        U2I2Recensys = self._build_u2i2recencies(df_log, target_date)

        # 最新度と頻度を計算し、上限値でクランプしたうえで再閲覧確率を付与
        prob_dict = self._probability_dict(kind)
        rows = []
        for user, item, recency, frequency in self._iter_u2i_rf(U2I2Recensys):
            r_adj = min(recency, self.recency_limit)
            f_adj = min(frequency, self.frequency_limit)
            rows.append((user, item, recency, frequency, prob_dict.get((r_adj, f_adj), 0.0)))
        df_rf = pd.DataFrame(
            rows,
            columns=[self._USER_COL, self._ITEM_COL, 'recency', 'frequency', 'probability']
        )
        df_rf = df_rf.sort_values([self._USER_COL, 'probability'], ascending=[True, False])
        df_rf['order'] = df_rf.groupby(self._USER_COL).cumcount() + 1
        df_rf = df_rf.rename(columns={self._USER_COL: user_col, self._ITEM_COL: item_col})

        # 入力 df の元の dtype に戻す
        df_rf[user_col] = df_rf[user_col].astype(df[user_col].dtype)
        df_rf[item_col] = df_rf[item_col].astype(df[item_col].dtype)
        
        return df_rf
    
    def evaluate(self, df_rec, UIrevisit, order=1, user_col=None, item_col=None):
        """Evaluate recommendation quality at each order cutoff.

        Parameters
        ----------
        df_rec : pd.DataFrame
            Recommendation results from transform(). Must have an 'order' column.
        UIrevisit : set
            Ground truth set of (user, item) pairs that were actually revisited.
        order : int, default 1
            Maximum recommendation rank to evaluate. Results are computed for
            each rank from 1 to order, plus the maximum order in df_rec.
        user_col : str, optional
            Column name for user in df_rec. Defaults to the first column.
        item_col : str, optional
            Column name for item in df_rec. Defaults to the second column.

        Returns
        -------
        pd.DataFrame
            Evaluation metrics for each order cutoff. Columns:
            order, n_recommended, n_hit, precision, recall, f1,
            recall_norm, f1_norm.
        """
        user_col = user_col or df_rec.columns[0]
        item_col = item_col or df_rec.columns[1]

        df_rec = df_rec.copy()
        try:
            df_rec[user_col] = df_rec[user_col].astype(str)
        except Exception as e:
            raise ValueError(
                f"Failed to cast column '{user_col}' to str: {e}"
            ) from e
        try:
            df_rec[item_col] = df_rec[item_col].astype(str)
        except Exception as e:
            raise ValueError(
                f"Failed to cast column '{item_col}' to str: {e}"
            ) from e
        try:
            UIrevisit = {(str(u), str(i)) for u, i in UIrevisit}
        except (TypeError, ValueError) as e:
            raise ValueError(
                "UIrevisit must be a set of (user, item) 2-tuples. "
                f"Failed to convert: {e}"
            ) from e

        order_max = df_rec.order.max()

        target_orders = list(range(1, order+1))
        target_orders += [order_max] if order_max not in set(target_orders) else []
        Rows = []
        for recommend_num in target_orders:
            df_k = df_rec[df_rec['order'] <= recommend_num]
            UIrec = set(zip(df_k[user_col], df_k[item_col]))
            n_hit = len(UIrec & UIrevisit)
            n_recommended = len(UIrec)
            precision = n_hit / n_recommended if n_recommended > 0 else 0.0
            Rows.append((recommend_num, n_recommended, n_hit, precision))
        df_eval = pd.DataFrame(
            Rows,
            columns=['order', 'n_recommended', 'n_hit', 'precision']
        )

        total_hit = df_eval.n_hit.max()
        df_eval['recall'] = df_eval.n_hit / len(UIrevisit)
        denom = df_eval.precision + df_eval.recall
        df_eval['f1'] = (2 * df_eval.precision * df_eval.recall).where(denom > 0, 0.0) / denom.where(denom > 0, 1.0)
        df_eval['recall_norm'] = df_eval.n_hit / total_hit
        denom_norm = df_eval.precision + df_eval.recall_norm
        df_eval['f1_norm'] = (2 * df_eval.precision * df_eval.recall_norm).where(denom_norm > 0, 0.0) / denom_norm.where(denom_norm > 0, 1.0)

        return df_eval


    def _build_u2i2recencies(self, df, ref_date):
        result = {}
        for row in df.itertuples():
            recency = (ref_date - row.datetime).days + 1
            result.setdefault(row.user, {}).setdefault(row.item, []).append(recency)
        return result

    def _iter_u2i_rf(self, U2I2Recensys):
        for user, I2Recencys in U2I2Recensys.items():
            for item, Recencys in I2Recencys.items():
                yield user, item, min(Recencys), len(Recencys)

    def _probability_dict(self, kind):
        if kind == 'mono':
            return self.mono_probability_dict_
        if kind == 'mcc':
            return self.mcc_probability_dict_
        return self.empirical_probability_dict_

    def optimize(self, kind='mono'):
        """Estimate optimized revisit probabilities under RF constraints.

        Solves a convex quadratic programming problem with monotonicity
        constraints (and optionally convexity/concavity constraints).
        Uses weighted least squares as objective.

        Requires fit() to be called first. Depends on cvxpy.

        Parameters
        ----------
        kind : {'mono', 'mcc'}, default 'mono'
            Optimization model to use.
            'mono' applies monotonicity constraints only.
            'mcc' additionally applies convexity in recency and concavity
            in frequency (diminishing marginal returns).

        Returns
        -------
        self
        """
        if kind not in ('mono', 'mcc'):
            raise ValueError(f"kind must be 'mono' or 'mcc', got '{kind}'.")

        try:
            from .optimizer import RFOptimizer
        except ImportError:
            from rfscorer.optimizer import RFOptimizer

        if self.empirical_probability_dict_ is None:
            raise RuntimeError("fit() must be called before optimize().")

        optimizer = RFOptimizer()
        optimizer.set_data(self.R, self.F, self.RF2N, self.RF2Prob)
        optimizer.build_model(kind=kind)
        optimizer.solve()
        optimizer.show_solve_info()
        optimizer.postprocess()

        rows = [(r, f, optimizer.RF2X[(r, f)]) for r in self.R for f in self.F]
        df_opt = pd.DataFrame(rows, columns=['recency', 'frequency', 'probability'])
        table = df_opt.pivot_table(index='recency', columns='frequency', values='probability')

        if kind == 'mono':
            self.mono_probability_dict_ = optimizer.RF2X
            self.mono_probability_ = df_opt
            self.mono_probability_table_ = table
        else:
            self.mcc_probability_dict_ = optimizer.RF2X
            self.mcc_probability_ = df_opt
            self.mcc_probability_table_ = table

        return self



if __name__ == "__main__":
    print('=== scorer.py ===')

    # サンプルデータの取得
    from pathlib import Path
    df = pd.read_csv('../../examples/access_log.csv')
    df_train = df[df.user_id.map(lambda x: hash(x) % 10 < 8)] # hash関数で簡易的に学習データ8割を抽出
    df_test = df[df.user_id.map(lambda x: hash(x) % 10 >= 8)] # hash関数で簡易的にテストデータ2割を抽出

    # スコアリングインスタンスの作成
    scorer = RecencyFrequencyScorer(user_col = "user_id", item_col = "item_id", datetime_col = "date")


    # 経験的再閲覧確率の計算
    observation_period = ('2015-07-01', '2015-07-06')
    evaluation_period = ('2015-07-07', '2015-07-08')
    #observation_period = ('2015-07-01', '2015-07-07')
    #evaluation_period = ('2015-07-08', '2015-07-08')
    scorer.fit(df_train, observation_period, evaluation_period)
    scorer.plot_probability_surface('empirical')
    scorer.show()

    # 最適化(Mono)
    scorer.optimize(kind='mono')
    scorer.plot_probability_surface('mono')

    # 最適化(MCC)
    scorer.optimize(kind='mcc')
    scorer.plot_probability_surface('mcc')

    # 全確率テーブルの出力
    scorer.export_probability_csv('all')

    # テストの実施
    target_date = '2015-07-07'
    df_test_obs = df_test[df_test.date <= target_date] # テストの観測期間データ
    df_test_eval = df_test[df_test.date > target_date] # テストの評価期間データ(正解データ)
    UIrevisit = set([(row.user_id, row.item_id) for row in df_test_eval.itertuples()]) # 正解データ

    print('--- empirical ---')
    df_rec_emp = scorer.transform(
        df_test_obs, target_date, 'empirical',
        user_col='user_id', item_col='item_id', datetime_col='date',
    )
    df_rec_emp.to_csv('df_recommend_emp.csv', index=False)
    print(scorer.evaluate(df_rec_emp, UIrevisit, order=10))

    print('--- mono ---')
    df_rec_mono = scorer.transform(
        df_test_obs, target_date, 'mono',
        user_col='user_id', item_col='item_id', datetime_col='date',
    )
    df_rec_mono.to_csv('df_recommend_mono.csv', index=False)
    print(scorer.evaluate(df_rec_mono, UIrevisit, order=10))

    print('--- mcc ---')
    df_rec_mcc = scorer.transform(
        df_test_obs, target_date, 'mcc',
        user_col='user_id', item_col='item_id', datetime_col='date',
    )
    df_rec_mcc.to_csv('df_recommend_mcc.csv', index=False)
    print(scorer.evaluate(df_rec_mcc, UIrevisit, order=10))

