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
        self.empirical_probability_table_ = None # 経験的再閲覧確率データフレーム(横持ち)
        self.empirical_probability_dict_ = None # 経験的再閲覧確率データフレーム(辞書:キーは最新度と頻度のペア)

        self.optimized_probability_ = None # 最適化再閲覧確率データフレーム(縦持ち)
        self.optimized_probability_table_ = None # 最適化再閲覧確率データフレーム(横持ち)
        self.optimized_probability_dict_ = None # 最適化再閲覧確率データフレーム(辞書:キーは最新度と頻度のペア)

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
        import numpy as np
        import matplotlib.pyplot as plt

        if kind not in ('empirical', 'optimized'):
            raise ValueError(f"kind must be 'empirical' or 'optimized', got '{kind}'.")
        if kind == 'empirical' and self.empirical_probability_table_ is None:
            raise RuntimeError("fit() must be called before plot_probability_surface().")
        if kind == 'optimized' and self.optimized_probability_table_ is None:
            raise RuntimeError("optimize() must be called before plot_probability_surface(kind='optimized').")

        from pathlib import Path
        default_filename = f'{kind}_probability_surface.png'
        if path is None:
            output_path = Path(default_filename)
        else:
            p = Path(path)
            output_path = p / default_filename if p.is_dir() else p

        table = (
            self.empirical_probability_table_
            if kind == 'empirical'
            else self.optimized_probability_table_
        )

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
        if kind not in ('empirical', 'optimized'):
            raise ValueError(f"kind must be 'empirical' or 'optimized', got '{kind}'.")
        if kind == 'empirical' and self.empirical_probability_dict_ is None:
            raise RuntimeError("fit() must be called before predict().")
        if kind == 'optimized' and self.optimized_probability_dict_ is None:
            raise RuntimeError("optimize() must be called before predict(kind='optimized').")

        r = min(r, self.recency_limit)
        f = min(f, self.frequency_limit)
        if kind == 'empirical':
            prob = self.empirical_probability_dict_.get((r, f), 0.0)
        else:
            prob = self.optimized_probability_dict_.get((r, f), 0.0)
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
        if kind not in ('empirical', 'optimized'):
            raise ValueError(f"kind must be 'empirical' or 'optimized', got '{kind}'.")
        if self.empirical_probability_dict_ is None:
            raise RuntimeError("fit() must be called before transform().")
        if kind == 'optimized' and self.optimized_probability_dict_ is None:
            raise RuntimeError("optimize() must be called before transform(kind='optimized').")

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
        U2I2Recensys = {}
        for row in df_log.itertuples():
            recency = (target_date - row.datetime).days + 1
            U2I2Recensys.setdefault(row.user, {})
            U2I2Recensys[row.user].setdefault(row.item, [])
            U2I2Recensys[row.user][row.item].append(recency)

        # 最新度と頻度を計算し、上限値でクランプしたうえで再閲覧確率を付与
        RowsInteraction = []
        for user, I2Recencys in U2I2Recensys.items():
            for item, Recencys in I2Recencys.items():
                freq = len(Recencys)
                rcen = min(Recencys)
                freq_adj = min(freq, self.frequency_limit)
                rcen_adj = min(rcen, self.recency_limit)
                prob_dict = (
                    self.empirical_probability_dict_
                    if kind == 'empirical'
                    else self.optimized_probability_dict_
                )
                prob = prob_dict.get((rcen_adj, freq_adj), 0.0)
                RowsInteraction.append((user, item, rcen, freq, prob))
        df_rf = pd.DataFrame(
            RowsInteraction, 
            columns=[self._USER_COL, self._ITEM_COL, 'recency', 'frequency', 'probability']
            )
        # タイブレイクは後に出てきたitem（行番号が大きい）を優先（後に出てきた商品 item の方が履歴上で最新となる場合が多いと考えられるため）
        df_rf['_order'] = range(len(df_rf))
        df_rf = df_rf.sort_values(
            [self._USER_COL, 'probability', '_order'],
            ascending=[True, False, False],
        )
        df_rf = df_rf.drop(columns='_order')
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
        order_max = df_rec.order.max()

        target_orders = list(range(1, order+1))
        target_orders += [order_max] if order_max not in set(target_orders) else []
        Rows = []
        for ord in target_orders:
            df_k = df_rec[df_rec['order'] <= ord]
            UIrec = set(zip(df_k[user_col], df_k[item_col]))
            n_hit = len(UIrec & UIrevisit)
            n_recommended = len(UIrec)
            precision = n_hit / n_recommended if n_recommended > 0 else 0.0
            Rows.append((ord, n_recommended, n_hit, precision))
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


    def optimize(self):
        """Estimate optimized revisit probabilities under RF constraints.

        Solves a convex quadratic programming problem with recency and
        frequency monotonicity constraints (Recency constraint: lower recency
        rank implies higher probability; Frequency constraint: higher frequency
        implies higher probability). Uses weighted least squares as objective.

        Requires fit() to be called first. Depends on cvxpy.

        Returns
        -------
        self
        """
        import cvxpy as cp

        if self.empirical_probability_dict_ is None:
            raise RuntimeError("fit() must be called before optimize().")

        nr = len(self.R)
        nf = len(self.F)

        # 決定変数: x[r_idx, f_idx] (0-indexed)
        x = cp.Variable((nr, nf))

        constraints = []

        # Recency 制約: r < r' => x[r,f] >= x[r',f]  (小さい r ほど高確率)
        for r_idx in range(nr - 1):
            for f_idx in range(nf):
                constraints.append(x[r_idx, f_idx] >= x[r_idx + 1, f_idx])

        # Frequency 制約: f < f' => x[r,f] <= x[r,f']  (大きい f ほど高確率)
        for r_idx in range(nr):
            for f_idx in range(nf - 1):
                constraints.append(x[r_idx, f_idx] <= x[r_idx, f_idx + 1])

        # 目的関数: Σ N_{r,f} * (x[r,f] - p_{r,f})^2
        objectives = []
        for r_idx, r in enumerate(self.R):
            for f_idx, f in enumerate(self.F):
                N = self.empirical_probability_[
                    (self.empirical_probability_.recency == r)
                    & (self.empirical_probability_.frequency == f)
                ]['N'].values[0]
                p = self.empirical_probability_dict_[(r, f)]
                objectives.append(N * (x[r_idx, f_idx] - p) ** 2)

        problem = cp.Problem(cp.Minimize(cp.sum(objectives)), constraints)
        problem.solve()

        # 結果を属性に格納
        rows = []
        opt_dict = {}
        for r_idx, r in enumerate(self.R):
            for f_idx, f in enumerate(self.F):
                prob = float(x.value[r_idx, f_idx])
                rows.append((r, f, prob))
                opt_dict[(r, f)] = prob

        self.optimized_probability_dict_ = opt_dict
        self.optimized_probability_ = pd.DataFrame(
            rows, columns=['recency', 'frequency', 'probability']
        )
        self.optimized_probability_table_ = self.optimized_probability_.pivot_table(
            index='recency',
            columns='frequency',
            values='probability',
        )

        return self



if __name__ == "__main__":
    print('=== scorer.py ===')

    # サンプルデータの取得
    df = pd.read_csv("../../examples/access_log.csv")
    df_train = df[df.user_id.map(lambda x: x % 10 < 8)] # hash関数で簡易的に学習データ8割を抽出
    df_test = df[df.user_id.map(lambda x: x % 10 >= 8)] # hash関数で簡易的にテストデータ2割を抽出

    # スコアリングインスタンスの作成
    scorer = RecencyFrequencyScorer(
        df_train,
        user_col = "user_id",
        item_col = "item_id",
        datetime_col = "date",
        )
    #print(scorer.interaction_log.head())

    # 経験的再閲覧確率の計算
    #observation_period = ('2015-07-01', '2015-07-06')
    #evaluation_period = ('2015-07-07', '2015-07-08')
    observation_period = ('2015-07-01', '2015-07-07')
    evaluation_period = ('2015-07-08', '2015-07-08')
    scorer.fit(observation_period, evaluation_period)

    scorer.show()
    scorer.plot_probability_surface('empirical')

    scorer.optimize()
    scorer.plot_probability_surface('optimized')

    target_date = '2015-07-07'
    df_test_obs = df_test[df_test.date <= target_date]
    df_test_eval = df_test[df_test.date > target_date]
    UIrevisit = set([(row.user_id, row.item_id) for row in df_test_eval.itertuples()])

    df_rec_emp = scorer.transform(
        df_test_obs, 
        target_date,
        'empirical',
        user_col = 'user_id',
        item_col = 'item_id',
        datetime_col = 'date'
        )
    
    print('--- empirical ---')
    #print(df_rec_emp)
    df_eval_emp = scorer.evaluate(df_rec_emp, UIrevisit, order=10)
    print(df_eval_emp)

    df_rec_opt = scorer.transform(
        df_test_obs, 
        target_date,
        'optimized',
        user_col = 'user_id',
        item_col = 'item_id',
        datetime_col = 'date'
        )

    print('--- optimized ---')
    #print(df_rec_opt)
    df_eval_opt = scorer.evaluate(df_rec_opt, UIrevisit, order=10)
    print(df_eval_opt)

