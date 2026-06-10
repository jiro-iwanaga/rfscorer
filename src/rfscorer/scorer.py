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
    _FREQUENCY_LIMIT_RATE = 0.95  # 最新度上限値自動計算の際に利用する割合
    _RECENCY_LIMIT_RATE = 0.95  # 頻度上限値自動計算の際に利用する割合

    def __init__(self, user_col="user", item_col="item", datetime_col="datetime"):
        """Initialize the scorer with column name mappings.

        Parameters
        ----------
        user_col : str, default "user"
            Column name for user IDs in the interaction log.
        item_col : str, default "item"
            Column name for item IDs in the interaction log.
        datetime_col : str, default "datetime"
            Column name for interaction timestamps in the interaction log.
        """
        self.user_col = user_col
        self.item_col = item_col
        self.datetime_col = datetime_col

        self.observation_start_date_ = None
        self.observation_end_date_ = None
        self.evaluation_start_date_ = None
        self.evaluation_end_date_ = None
        self.recency_limit = (
            None  # 最新度の上限値(デフォルトでは _RECENCY_LIMIT_RATE を用いて自動計算)
        )
        self.frequency_limit = (
            None  # 頻度の上限値(デフォルトでは _FREQUENCY_LIMIT_RATE を用いて自動計算)
        )

        self.R = []  # 最新度のリスト
        self.F = []  # 頻度のリスト
        self.RF2N = {}  # 最新度と頻度に対して閲覧数合計を紐づける辞書
        self.RF2CV = {}  # 最新度と頻度に対して再閲覧数合計を紐づける辞書
        self.RF2Prob = {}  # 最新度と頻度に対して経験的再閲覧確率を紐づける辞書
        self.R2N = {}  # 最新度に対して閲覧数合計を紐づける辞書
        self.R2CV = {}  # 最新度に対して再閲覧数合計を紐づける辞書
        self.R2Prob = {}  # 最新度に対して経験的再閲覧確率を紐づける辞書
        self.F2N = {}  # 頻度に対して閲覧数合計を紐づける辞書
        self.F2CV = {}  # 頻度に対して再閲覧数合計を紐づける辞書
        self.F2Prob = {}  # 頻度に対して経験的再閲覧確率を紐づける辞書

        # empirical
        self.empirical_probability_ = None  # 経験的再閲覧確率データフレーム(縦持ち)
        self.empirical_probability_table_ = None  # 経験的再閲覧確率データフレーム(横持ち)
        self.empirical_probability_dict_ = (
            None  # 経験的再閲覧確率データフレーム(辞書:キーは最新度と頻度のペア)
        )
        self.recency_probability_ = None  # 最新度別経験的再閲覧確率データフレーム
        self.frequency_probability_ = None  # 頻度別経験的再閲覧確率データフレーム

        # er
        self.er_probability_ = None
        self.er_probability_table_ = None
        self.er_probability_dict_ = None

        # ef
        self.ef_probability_ = None
        self.ef_probability_table_ = None
        self.ef_probability_dict_ = None

        # mono
        self.mono_probability_ = None
        self.mono_probability_table_ = None
        self.mono_probability_dict_ = None

        # mr
        self.mr_probability_ = None
        self.mr_probability_table_ = None
        self.mr_probability_dict_ = None

        # mf
        self.mf_probability_ = None
        self.mf_probability_table_ = None
        self.mf_probability_dict_ = None

        # mrc
        self.mrc_probability_ = None
        self.mrc_probability_table_ = None
        self.mrc_probability_dict_ = None

        # mfc
        self.mfc_probability_ = None
        self.mfc_probability_table_ = None
        self.mfc_probability_dict_ = None

        # mcc
        self.mcc_probability_ = None
        self.mcc_probability_table_ = None
        self.mcc_probability_dict_ = None

        # データ解析用
        self.record_num = None  # レコード数（fit() 後に設定）
        self.record_num_obs = None  # 観測期間レコード数
        self.record_num_eval = None  # 評価期間レコード数
        self.record_num_target_org = None  # 分析対象フィルタリング前レコード数
        self.record_num_target = None  # 分析対象レコード数
        self.total_cv_org = None  # フィルタリング前 cv 数
        self.total_cv = None  # cv 数

    def fit(
        self,
        df_obs,
        df_eval,
        ref_date=None,
        recency_limit=None,
        frequency_limit=None,
    ):
        """Estimate empirical revisit probabilities from pre-split interaction data.

        Accepts observation and evaluation DataFrames that have already been
        filtered to the respective periods by the caller.  This is the primary
        fitting method; for convenience wrappers that perform date-based
        splitting automatically, see fit_date() and fit_period().

        Parameters
        ----------
        df_obs : pd.DataFrame
            Observation period interaction log. Must already be filtered to
            the observation period by the caller.
        df_eval : pd.DataFrame
            Evaluation period event log (revisits, purchases, conversions,
            etc.). Must already be filtered to the evaluation period by the caller.
        ref_date : str or datetime, optional
            Reference date for recency computation. Recency of each
            user-item pair is the minimum number of days from any of its
            interactions to ref_date, plus 1 (so the most recent interaction
            on ref_date itself has recency 1). When None, defaults to the
            latest datetime in df_obs.
        recency_limit : int, optional
            Maximum recency rank to include. If None, automatically set to
            the recency rank covering 95% of cumulative revisits.
        frequency_limit : int, optional
            Maximum frequency to include. If None, automatically set to
            the frequency covering 95% of cumulative revisits.

        Returns
        -------
        self
        """
        if not isinstance(df_obs, pd.DataFrame):
            raise TypeError("df_obs must be a pandas DataFrame.")
        if not isinstance(df_eval, pd.DataFrame):
            raise TypeError("df_eval must be a pandas DataFrame.")

        required_columns = [self.user_col, self.item_col, self.datetime_col]
        missing_obs = [c for c in required_columns if c not in df_obs.columns]
        if missing_obs:
            raise ValueError(f"Missing required columns in df_obs: {missing_obs}")
        missing_eval = [c for c in required_columns if c not in df_eval.columns]
        if missing_eval:
            raise ValueError(f"Missing required columns in df_eval: {missing_eval}")

        obs_log = self._to_internal(df_obs)
        eval_log = self._to_internal(df_eval)

        self.record_num = len(obs_log) + len(eval_log)

        if ref_date is None:
            ref_date = obs_log[self._DATETIME_COL].max()
        else:
            try:
                ref_date = pd.to_datetime(ref_date)
            except (ValueError, TypeError) as e:
                raise ValueError(f"ref_date could not be parsed as a date: {ref_date}") from e

        self.observation_start_date_ = (
            obs_log[self._DATETIME_COL].min() if len(obs_log) > 0 else None
        )
        self.observation_end_date_ = ref_date
        self.evaluation_start_date_ = (
            eval_log[self._DATETIME_COL].min() if len(eval_log) > 0 else None
        )
        self.evaluation_end_date_ = (
            eval_log[self._DATETIME_COL].max() if len(eval_log) > 0 else None
        )

        self._fit_impl(obs_log, eval_log, ref_date, recency_limit, frequency_limit)
        return self

    def fit_date(
        self,
        df,
        target_date,
        observation_days=28,
        evaluation_days=7,
        recency_limit=None,
        frequency_limit=None,
    ):
        """Estimate empirical revisit probabilities using target_date as split point.

        Derives observation and evaluation periods automatically from target_date:
        observation spans from at most observation_days before target_date to
        target_date; evaluation spans from the next day to at most evaluation_days
        after target_date.  Pass None for either days argument to use the full
        data range in that direction.

        Parameters
        ----------
        df : pd.DataFrame
            Interaction log containing user, item, and datetime columns.
        target_date : str or datetime
            Reference date used as the observation end / evaluation start boundary.
        observation_days : int or None, default 28
            Maximum number of days to look back from target_date for the
            observation period.  If None, uses all data up to target_date.
        evaluation_days : int or None, default 7
            Maximum number of days to look forward from target_date for the
            evaluation period.  If None, uses all data after target_date.
        recency_limit : int, optional
            Maximum recency rank to include.  If None, determined automatically.
        frequency_limit : int, optional
            Maximum frequency to include.  If None, determined automatically.

        Returns
        -------
        self
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if self.datetime_col not in df.columns:
            raise ValueError(f"Missing required columns: [{self.datetime_col!r}]")

        try:
            target_date = pd.to_datetime(target_date)
        except (ValueError, TypeError) as e:
            raise ValueError(f"target_date could not be parsed as a date: {target_date}") from e

        try:
            dates = pd.to_datetime(df[self.datetime_col])
        except (ValueError, TypeError) as e:
            raise ValueError(f"Column {self.datetime_col!r} could not be parsed as dates.") from e
        df_min = dates.min()
        df_max = dates.max()

        if observation_days is None:
            obs_start = df_min
        else:
            obs_start = max(df_min, target_date - pd.Timedelta(days=observation_days))

        obs_end = target_date
        eval_start = target_date + pd.Timedelta(days=1)

        if evaluation_days is None:
            eval_end = df_max
        else:
            eval_end = min(df_max, target_date + pd.Timedelta(days=evaluation_days))

        interaction_log = self._to_internal(df)
        self.record_num = len(interaction_log)

        obs_log = interaction_log[
            (obs_start <= interaction_log[self._DATETIME_COL])
            & (interaction_log[self._DATETIME_COL] <= obs_end)
        ]
        eval_log = interaction_log[
            (eval_start <= interaction_log[self._DATETIME_COL])
            & (interaction_log[self._DATETIME_COL] <= eval_end)
        ]

        self.observation_start_date_ = obs_start
        self.observation_end_date_ = obs_end
        self.evaluation_start_date_ = eval_start
        self.evaluation_end_date_ = eval_end

        self._fit_impl(obs_log, eval_log, obs_end, recency_limit, frequency_limit)
        return self

    def fit_period(
        self,
        df,
        observation_period,
        evaluation_period,
        recency_limit=None,
        frequency_limit=None,
    ):
        """Estimate empirical revisit probabilities from interaction history.

        Use this when you need explicit control over both periods.
        For the common case of splitting on a single date, prefer fit_date().
        For a sklearn-style interface with pre-split DataFrames, prefer fit().

        Parameters
        ----------
        df : pd.DataFrame
            Interaction log containing user, item, and datetime columns.
        observation_period : tuple[str | datetime, str | datetime]
            Start and end dates of the observation period.
        evaluation_period : tuple[str | datetime, str | datetime]
            Start and end dates of the evaluation period.
            Must start strictly after observation_period ends.
        recency_limit : int, optional
            Maximum recency rank to include. If None, automatically set to
            the recency rank covering 95% of cumulative revisits.
        frequency_limit : int, optional
            Maximum frequency to include. If None, automatically set to
            the frequency covering 95% of cumulative revisits.

        Returns
        -------
        self
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        required_columns = [self.user_col, self.item_col, self.datetime_col]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

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

        if obs_start > obs_end:
            raise ValueError("observation_period must be ordered as (start, end).")
        if eval_start > eval_end:
            raise ValueError("evaluation_period must be ordered as (start, end).")
        if obs_end >= eval_start:
            raise ValueError("observation_period must end before evaluation_period starts.")

        interaction_log = self._to_internal(df)
        self.record_num = len(interaction_log)

        obs_log = interaction_log[
            (obs_start <= interaction_log[self._DATETIME_COL])
            & (interaction_log[self._DATETIME_COL] <= obs_end)
        ]
        eval_log = interaction_log[
            (eval_start <= interaction_log[self._DATETIME_COL])
            & (interaction_log[self._DATETIME_COL] <= eval_end)
        ]

        self.observation_start_date_ = obs_start
        self.observation_end_date_ = obs_end
        self.evaluation_start_date_ = eval_start
        self.evaluation_end_date_ = eval_end

        self._fit_impl(obs_log, eval_log, obs_end, recency_limit, frequency_limit)
        return self

    def _to_internal(self, df):
        """Convert a user-facing DataFrame to internal column names and types."""
        result = df[[self.user_col, self.item_col, self.datetime_col]].copy()
        result.columns = [self._USER_COL, self._ITEM_COL, self._DATETIME_COL]
        if not is_string_dtype(result[self._USER_COL]):
            result[self._USER_COL] = result[self._USER_COL].astype(str)
        if not is_string_dtype(result[self._ITEM_COL]):
            result[self._ITEM_COL] = result[self._ITEM_COL].astype(str)
        if not is_datetime64_any_dtype(result[self._DATETIME_COL]):
            result[self._DATETIME_COL] = pd.to_datetime(result[self._DATETIME_COL])
        return result

    def _fit_impl(self, obs_log, eval_log, ref_date, recency_limit, frequency_limit):
        """Core fitting logic. obs_log and eval_log must use internal column names."""
        self.record_num_obs = len(obs_log)
        self.record_num_eval = len(eval_log)

        UIcv = {(row.user, row.item) for row in eval_log.itertuples()}

        df_ui2frc = self._build_ui_rf_df(obs_log, ref_date)
        df_ui2frc["cv"] = (
            pd.MultiIndex.from_frame(df_ui2frc[[self._USER_COL, self._ITEM_COL]])
            .isin(UIcv)
            .astype(int)
        )
        self.record_num_target_org = len(df_ui2frc)
        self.total_cv_org = df_ui2frc.cv.sum()

        if recency_limit is not None:
            self.recency_limit = recency_limit
        else:
            df_recency2cv = df_ui2frc.groupby("recency")["cv"].sum().reset_index()
            df_recency2cv.sort_values("recency", inplace=True)
            total_cv = df_recency2cv.cv.sum()
            if total_cv == 0:
                raise ValueError(
                    "No revisits observed in evaluation period. "
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
            df_frequency2cv = df_ui2frc.groupby("frequency")["cv"].sum().reset_index()
            df_frequency2cv.sort_values("frequency", inplace=True)
            total_cv = df_frequency2cv.cv.sum()
            if total_cv == 0:
                raise ValueError(
                    "No revisits observed in evaluation period. "
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
            (df_ui2frc.recency <= self.recency_limit)
            & (df_ui2frc.frequency <= self.frequency_limit)
        ]
        self.record_num_target = len(df_ui2frc)
        self.total_cv = df_ui2frc.cv.sum()

        self.R = list(range(1, self.recency_limit + 1))
        self.F = list(range(1, self.frequency_limit + 1))

        self.RF2N = {(r, f): 0.0 for r in self.R for f in self.F}
        self.RF2CV = {(r, f): 0.0 for r in self.R for f in self.F}

        for row in df_ui2frc.itertuples():
            self.RF2N[row.recency, row.frequency] += 1
            if row.cv == 1:
                self.RF2CV[row.recency, row.frequency] += 1

        RowsRF = []
        for r in self.R:
            for f in self.F:
                prob = self.RF2CV[r, f] / self.RF2N[r, f] if self.RF2N[r, f] > 0 else 0.0
                self.RF2Prob[r, f] = prob
                RowsRF.append((r, f, self.RF2N[r, f], self.RF2CV[r, f], prob))

        self.empirical_probability_dict_ = {(r, f): prob for r, f, _, _, prob in RowsRF}
        self.empirical_probability_ = pd.DataFrame(
            RowsRF, columns=["recency", "frequency", "N", "cv", "probability"]
        )
        self.empirical_probability_table_ = self.empirical_probability_.pivot_table(
            index="recency",
            columns="frequency",
            values="probability",
        )

        df_r = self.empirical_probability_.groupby("recency")[["N", "cv"]].sum().reset_index()
        df_r["probability"] = (df_r["cv"] / df_r["N"]).where(df_r["N"] > 0, 0.0)
        self.recency_probability_ = df_r
        self.R2N = dict(zip(df_r["recency"], df_r["N"]))
        self.R2CV = dict(zip(df_r["recency"], df_r["cv"]))
        self.R2Prob = dict(zip(df_r["recency"], df_r["probability"]))

        df_f = self.empirical_probability_.groupby("frequency")[["N", "cv"]].sum().reset_index()
        df_f["probability"] = (df_f["cv"] / df_f["N"]).where(df_f["N"] > 0, 0.0)
        self.frequency_probability_ = df_f
        self.F2N = dict(zip(df_f["frequency"], df_f["N"]))
        self.F2CV = dict(zip(df_f["frequency"], df_f["cv"]))
        self.F2Prob = dict(zip(df_f["frequency"], df_f["probability"]))

        rows_er = [(r, f, self.R2Prob[r]) for r in self.R for f in self.F]
        self.er_probability_dict_ = {(r, f): p for r, f, p in rows_er}
        self.er_probability_ = pd.DataFrame(
            rows_er, columns=["recency", "frequency", "probability"]
        )
        self.er_probability_table_ = self.er_probability_.pivot_table(
            index="recency", columns="frequency", values="probability"
        )

        rows_ef = [(r, f, self.F2Prob[f]) for r in self.R for f in self.F]
        self.ef_probability_dict_ = {(r, f): p for r, f, p in rows_ef}
        self.ef_probability_ = pd.DataFrame(
            rows_ef, columns=["recency", "frequency", "probability"]
        )
        self.ef_probability_table_ = self.ef_probability_.pivot_table(
            index="recency", columns="frequency", values="probability"
        )

    def show(self):
        """Print a summary of fit(), fit_date(), or fit_period() results to stdout."""
        print("=== profiling ===")

        if self.record_num:
            print("record_num:", self.record_num)

        if self.record_num_obs:
            print("record_num_obs:", self.record_num_obs)
        if self.record_num_eval:
            print("record_num_eval:", self.record_num_eval)

        if self.observation_start_date_ and self.observation_end_date_:
            print(
                "observation: {} -> {}".format(
                    self.observation_start_date_, self.observation_end_date_
                )
            )
        if self.evaluation_start_date_ and self.evaluation_end_date_:
            print(
                "evaluation: {} -> {}".format(
                    self.evaluation_start_date_, self.evaluation_end_date_
                )
            )

        if self.recency_limit:
            print("recency_limit:", self.recency_limit)
        if self.frequency_limit:
            print("frequency_limit:", self.frequency_limit)

        if self.record_num_target_org and self.record_num_target:
            print(
                "target_record_num: {} -> {}".format(
                    self.record_num_target_org, self.record_num_target
                )
            )

        if self.total_cv_org and self.total_cv:
            print("total_cv: {} -> {}".format(self.total_cv_org, self.total_cv))

        if self.empirical_probability_table_ is not None:
            print("empirical_probability_table_:")
            print(self.empirical_probability_table_.round(3).to_string())

    def plot_probability_surface(
        self,
        kind="emp",
        title=None,
        figsize=(6, 5),
        fontsize=12,
        recency_label="recency",
        frequency_label="frequency",
        probability_label="probability",
    ):
        """Plot revisit probabilities as a 3D surface.

        Visualizes the probability table as a 3D wireframe with recency on
        the x-axis, frequency on the y-axis, and probability on the z-axis.

        In Jupyter Lab / Colab the returned figure renders inline automatically.
        To save to a file, call ``fig.savefig("output.png")`` on the returned
        figure.

        Parameters
        ----------
        kind : {"emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc"}, default "emp"
            Which probability to visualize. "emp", "er", and "ef" use fit(),
            fit_date(), or fit_period() results; others use optimize() results.
        title : str or None, default None
            Figure title. If None, no title is shown.
        figsize : tuple[float, float], default (6, 5)
            Figure size in inches as (width, height). For publication, set this
            to the final printed size (e.g., (3.5, 3.0) for a single-column
            figure in a two-column journal).
        fontsize : int, default 12
            Font size for axis labels and tick labels. For publication, match
            this to the body text size of the target journal (typically 8–10 pt)
            and set figsize to the final printed size so the font is not scaled.
        recency_label : str, default "recency"
            Label for the x-axis (recency dimension).
        frequency_label : str, default "frequency"
            Label for the y-axis (frequency dimension).
        probability_label : str, default "probability"
            Label for the z-axis (probability dimension).

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import numpy as np

        kind = self._normalize_kind(kind)
        if kind not in ("emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc"):
            raise ValueError(
                f"kind must be 'emp', 'er', 'ef', 'mono', 'mr', 'mf', 'mrc', 'mfc', or 'mcc',"
                f" got {kind!r}."
            )
        if kind in ("emp", "er", "ef") and self.empirical_probability_table_ is None:
            raise RuntimeError(
                "fit(), fit_date(), or fit_period() must be called"
                " before plot_probability_surface()."
            )
        if kind == "mono" and self.mono_probability_table_ is None:
            raise RuntimeError(
                "optimize(kind='mono') must be called before plot_probability_surface(kind='mono')."
            )
        if kind == "mr" and self.mr_probability_table_ is None:
            raise RuntimeError(
                "optimize(kind='mr') must be called before plot_probability_surface(kind='mr')."
            )
        if kind == "mf" and self.mf_probability_table_ is None:
            raise RuntimeError(
                "optimize(kind='mf') must be called before plot_probability_surface(kind='mf')."
            )
        if kind == "mrc" and self.mrc_probability_table_ is None:
            raise RuntimeError(
                "optimize(kind='mrc') must be called before plot_probability_surface(kind='mrc')."
            )
        if kind == "mfc" and self.mfc_probability_table_ is None:
            raise RuntimeError(
                "optimize(kind='mfc') must be called before plot_probability_surface(kind='mfc')."
            )
        if kind == "mcc" and self.mcc_probability_table_ is None:
            raise RuntimeError(
                "optimize(kind='mcc') must be called before plot_probability_surface(kind='mcc')."
            )

        if kind == "emp":
            table = self.empirical_probability_table_
        elif kind == "er":
            table = self.er_probability_table_
        elif kind == "ef":
            table = self.ef_probability_table_
        elif kind == "mono":
            table = self.mono_probability_table_
        elif kind == "mr":
            table = self.mr_probability_table_
        elif kind == "mf":
            table = self.mf_probability_table_
        elif kind == "mrc":
            table = self.mrc_probability_table_
        elif kind == "mfc":
            table = self.mfc_probability_table_
        else:
            table = self.mcc_probability_table_

        recency = table.index.tolist()
        frequency = table.columns.tolist()
        X, Y = np.meshgrid(recency, frequency)
        Z = table.values.T

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_wireframe(X, Y, Z, color="black")
        ax.set_xlabel(recency_label, fontsize=fontsize)
        ax.set_ylabel(frequency_label, fontsize=fontsize)
        ax.set_zlabel(probability_label, fontsize=fontsize, labelpad=10)
        ax.tick_params(labelsize=fontsize)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
        if title is not None:
            ax.set_title(title, fontsize=fontsize)
        return fig

    def plot_marginal_probability(
        self,
        axis="recency",
        kind="emp",
        title=None,
        figsize=(5, 4),
        fontsize=12,
        recency_label="recency",
        frequency_label="frequency",
        probability_label="probability",
    ):
        """Plot revisit probability aggregated along one RF dimension.

        Visualizes R2Prob / mr (when axis='recency') or F2Prob / mf (when
        axis='frequency') as a line chart with markers.

        In Jupyter Lab / Colab the returned figure renders inline automatically.
        To save to a file, call ``fig.savefig("output.png")`` on the returned figure.

        Parameters
        ----------
        axis : {"recency", "frequency"}, default "recency"
            Which dimension to aggregate and plot.
            "recency" plots probability vs recency rank (expected: decreasing).
            "frequency" plots probability vs frequency (expected: increasing).
        kind : {"emp", "mr", "mf", "all"}, default "emp"
            Which probability series to draw.
            "emp" draws the empirical marginal (R2Prob or F2Prob).
            "mr" draws the mr-optimized series (valid only when axis="recency").
            "mf" draws the mf-optimized series (valid only when axis="frequency").
            "all" draws both the empirical and the optimized series together.
        title : str or None, default None
            Figure title. If None, no title is shown.
        figsize : tuple[float, float], default (5, 4)
            Figure size in inches as (width, height). For publication, set this
            to the final printed size (e.g., (3.5, 2.8) for a single-column
            figure in a two-column journal).
        fontsize : int, default 12
            Font size for axis labels and tick labels. For publication, match
            this to the body text size of the target journal (typically 8–10 pt)
            and set figsize to the final printed size so the font is not scaled.
        recency_label : str, default "recency"
            Label for the x-axis when axis="recency".
        frequency_label : str, default "frequency"
            Label for the x-axis when axis="frequency".
        probability_label : str, default "probability"
            Label for the y-axis (probability dimension).

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        kind = self._normalize_kind(kind)
        if axis not in ("recency", "frequency"):
            raise ValueError(f"axis must be 'recency' or 'frequency', got {axis!r}.")
        valid_kinds = ("emp", "mr", "mf", "all")
        if kind not in valid_kinds:
            raise ValueError(f"kind must be one of {valid_kinds}, got {kind!r}.")
        if axis == "recency" and kind == "mf":
            raise ValueError("kind='mf' is not valid when axis='recency'. Use kind='mr'.")
        if axis == "frequency" and kind == "mr":
            raise ValueError("kind='mr' is not valid when axis='frequency'. Use kind='mf'.")
        if self.recency_probability_ is None:
            raise RuntimeError(
                "fit(), fit_date(), or fit_period() must be called"
                " before plot_marginal_probability()."
            )
        opt_kind = "mr" if axis == "recency" else "mf"
        if kind in (opt_kind, "all"):
            opt_attr = f"{opt_kind}_probability_"
            if getattr(self, opt_attr) is None:
                raise RuntimeError(
                    f"optimize(kind='{opt_kind}') must be called before"
                    f" plot_marginal_probability(kind='{kind}')."
                )

        x_col = axis
        x_label = recency_label if axis == "recency" else frequency_label
        if axis == "recency":
            df_emp = self.recency_probability_
        else:
            df_emp = self.frequency_probability_

        if kind in (opt_kind, "all"):
            df_opt_full = getattr(self, f"{opt_kind}_probability_")
            fixed_col = "frequency" if axis == "recency" else "recency"
            fixed_val = df_opt_full[fixed_col].iloc[0]
            df_opt = df_opt_full[df_opt_full[fixed_col] == fixed_val][
                [x_col, "probability"]
            ].reset_index(drop=True)

        fig, ax = plt.subplots(figsize=figsize)
        if kind in ("emp", "all"):
            ax.plot(
                df_emp[x_col],
                df_emp["probability"],
                color="black",
                linestyle="-",
                marker="o",
                linewidth=1.5,
                markersize=6,
                label="emp",
            )
        if kind in (opt_kind, "all"):
            ax.plot(
                df_opt[x_col],
                df_opt["probability"],
                color="black",
                linestyle="--" if kind == "all" else "-",
                marker="s",
                linewidth=1.5,
                markersize=6,
                label=opt_kind,
            )
        if kind == "all":
            ax.legend(fontsize=fontsize)
        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel(probability_label, fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        if title is not None:
            ax.set_title(title, fontsize=fontsize)
        fig.tight_layout()
        return fig

    def export_probability_csv(self, kind="emp", path=None):
        """Export revisit probabilities to a CSV file.

        Parameters
        ----------
        kind : {"emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc", "all"}, default "emp"
            Which probability to export. "emp", "er", and "ef" use fit(),
            fit_date(), or fit_period() results; others use optimize() results;
            "all" merges all nine models into a single file with columns
            empirical_probability, er_probability, ef_probability,
            mono_probability, mr_probability, mf_probability, mrc_probability,
            mfc_probability, mcc_probability (requires all optimize() calls).
        path : str or None, default None
            Output file path for the CSV. If None, saves as
            "{kind}_probability.csv" in the current directory.
            If a directory, saves "{kind}_probability.csv" inside it.

        Returns
        -------
        None
        """
        kind = self._normalize_kind(kind)
        if kind not in ("emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc", "all"):
            raise ValueError(
                f"kind must be 'emp', 'er', 'ef', 'mono', 'mr', 'mf', 'mrc', 'mfc', 'mcc',"
                f" or 'all', got {kind!r}."
            )
        if kind in ("emp", "er", "ef", "all") and self.empirical_probability_ is None:
            raise RuntimeError(
                "fit(), fit_date(), or fit_period() must be called before export_probability_csv()."
            )
        if kind in ("mono", "all") and self.mono_probability_ is None:
            raise RuntimeError(
                "optimize(kind='mono') must be called before export_probability_csv(kind='mono')."
            )
        if kind in ("mr", "all") and self.mr_probability_ is None:
            raise RuntimeError(
                "optimize(kind='mr') must be called before export_probability_csv(kind='mr')."
            )
        if kind in ("mf", "all") and self.mf_probability_ is None:
            raise RuntimeError(
                "optimize(kind='mf') must be called before export_probability_csv(kind='mf')."
            )
        if kind in ("mrc", "all") and self.mrc_probability_ is None:
            raise RuntimeError(
                "optimize(kind='mrc') must be called before export_probability_csv(kind='mrc')."
            )
        if kind in ("mfc", "all") and self.mfc_probability_ is None:
            raise RuntimeError(
                "optimize(kind='mfc') must be called before export_probability_csv(kind='mfc')."
            )
        if kind in ("mcc", "all") and self.mcc_probability_ is None:
            raise RuntimeError(
                "optimize(kind='mcc') must be called before export_probability_csv(kind='mcc')."
            )

        from pathlib import Path

        default_filename = f"{kind}_probability.csv"
        if path is None:
            output_path = Path(default_filename)
        else:
            p = Path(path)
            output_path = p / default_filename if p.is_dir() else p

        if kind == "all":
            df = (
                self.empirical_probability_.rename(columns={"probability": "empirical_probability"})
                .merge(
                    self.er_probability_.rename(columns={"probability": "er_probability"}),
                    on=["recency", "frequency"],
                )
                .merge(
                    self.ef_probability_.rename(columns={"probability": "ef_probability"}),
                    on=["recency", "frequency"],
                )
                .merge(
                    self.mono_probability_.rename(columns={"probability": "mono_probability"}),
                    on=["recency", "frequency"],
                )
                .merge(
                    self.mr_probability_.rename(columns={"probability": "mr_probability"}),
                    on=["recency", "frequency"],
                )
                .merge(
                    self.mf_probability_.rename(columns={"probability": "mf_probability"}),
                    on=["recency", "frequency"],
                )
                .merge(
                    self.mrc_probability_.rename(columns={"probability": "mrc_probability"}),
                    on=["recency", "frequency"],
                )
                .merge(
                    self.mfc_probability_.rename(columns={"probability": "mfc_probability"}),
                    on=["recency", "frequency"],
                )
                .merge(
                    self.mcc_probability_.rename(columns={"probability": "mcc_probability"}),
                    on=["recency", "frequency"],
                )
            )
        elif kind == "emp":
            df = self.empirical_probability_
        elif kind == "er":
            df = self.er_probability_
        elif kind == "ef":
            df = self.ef_probability_
        elif kind == "mono":
            df = self.mono_probability_
        elif kind == "mr":
            df = self.mr_probability_
        elif kind == "mf":
            df = self.mf_probability_
        elif kind == "mrc":
            df = self.mrc_probability_
        elif kind == "mfc":
            df = self.mfc_probability_
        else:
            df = self.mcc_probability_
        df.to_csv(output_path, index=False)

    def predict(self, r, f, kind="emp"):
        """Return the revisit probability for a given recency and frequency.

        Parameters
        ----------
        r : int
            Recency rank (1 = most recently interacted, higher = older).
        f : int
            Frequency (number of interactions in the observation period).
        kind : {"emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc"}, default "emp"
            Which probability to use. "emp", "er", and "ef" use fit(),
            fit_date(), or fit_period() results; others use optimize() results.

        Returns
        -------
        float
            Revisit probability for the given (r, f).
        """

        kind = self._normalize_kind(kind)
        if not isinstance(r, int) or r < 1:
            raise TypeError("r must be a positive integer.")
        if not isinstance(f, int) or f < 1:
            raise TypeError("f must be a positive integer.")
        if kind not in ("emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc"):
            raise ValueError(
                f"kind must be 'emp', 'er', 'ef', 'mono', 'mr', 'mf', 'mrc', 'mfc', or 'mcc',"
                f" got {kind!r}."
            )
        if kind in ("emp", "er", "ef") and self.empirical_probability_dict_ is None:
            raise RuntimeError(
                "fit(), fit_date(), or fit_period() must be called before predict()."
            )
        if kind == "mono" and self.mono_probability_dict_ is None:
            raise RuntimeError("optimize(kind='mono') must be called before predict(kind='mono').")
        if kind == "mr" and self.mr_probability_dict_ is None:
            raise RuntimeError("optimize(kind='mr') must be called before predict(kind='mr').")
        if kind == "mf" and self.mf_probability_dict_ is None:
            raise RuntimeError("optimize(kind='mf') must be called before predict(kind='mf').")
        if kind == "mrc" and self.mrc_probability_dict_ is None:
            raise RuntimeError("optimize(kind='mrc') must be called before predict(kind='mrc').")
        if kind == "mfc" and self.mfc_probability_dict_ is None:
            raise RuntimeError("optimize(kind='mfc') must be called before predict(kind='mfc').")
        if kind == "mcc" and self.mcc_probability_dict_ is None:
            raise RuntimeError("optimize(kind='mcc') must be called before predict(kind='mcc').")

        r = min(r, self.recency_limit)
        f = min(f, self.frequency_limit)
        if kind == "emp":
            prob = self.empirical_probability_dict_.get((r, f), 0.0)
        elif kind == "er":
            prob = self.er_probability_dict_.get((r, f), 0.0)
        elif kind == "ef":
            prob = self.ef_probability_dict_.get((r, f), 0.0)
        elif kind == "mono":
            prob = self.mono_probability_dict_.get((r, f), 0.0)
        elif kind == "mr":
            prob = self.mr_probability_dict_.get((r, f), 0.0)
        elif kind == "mf":
            prob = self.mf_probability_dict_.get((r, f), 0.0)
        elif kind == "mrc":
            prob = self.mrc_probability_dict_.get((r, f), 0.0)
        elif kind == "mfc":
            prob = self.mfc_probability_dict_.get((r, f), 0.0)
        else:
            prob = self.mcc_probability_dict_.get((r, f), 0.0)
        return prob

    def transform(
        self,
        df,
        ref_date=None,
        kind="emp",
        user_col=None,
        item_col=None,
        datetime_col=None,
    ):
        """Add recency, frequency, and revisit probability columns to a DataFrame.

        Computes recency rank and frequency for each user-item pair relative to
        ref_date, then appends the corresponding revisit probability from fit()
        or optimize() results.  The caller is responsible for pre-filtering df
        to the desired observation window before calling this method.

        Parameters
        ----------
        df : pd.DataFrame
            User-item interaction history to score. Should already be filtered
            to the observation period by the caller.
        ref_date : str or datetime, optional
            Reference date for computing recency. When None, defaults to the
            latest datetime in df.
        kind : {"emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc"}, default "emp"
            Which probability to use. "emp", "er", and "ef" use fit(),
            fit_date(), or fit_period() results; others use optimize() results.
        user_col : str, optional
            Column name for user IDs in df. Defaults to the value set in __init__.
        item_col : str, optional
            Column name for item IDs in df. Defaults to the value set in __init__.
        datetime_col : str, optional
            Column name for interaction timestamps in df. Defaults to the value
            set in __init__.

        Returns
        -------
        pd.DataFrame
            One row per user-item pair observed in df.  User and item columns
            retain the names used (from __init__ or the overrides). Additional
            columns: recency, frequency, probability, order. Sorted by user
            ascending and probability descending; order starts at 1.
        """
        kind = self._normalize_kind(kind)
        if kind not in ("emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc"):
            raise ValueError(
                f"kind must be 'emp', 'er', 'ef', 'mono', 'mr', 'mf', 'mrc', 'mfc', or 'mcc',"
                f" got {kind!r}."
            )
        if self.empirical_probability_dict_ is None:
            raise RuntimeError(
                "fit(), fit_date(), or fit_period() must be called before transform()."
            )
        if kind == "mono" and self.mono_probability_dict_ is None:
            raise RuntimeError(
                "optimize(kind='mono') must be called before transform(kind='mono')."
            )
        if kind == "mr" and self.mr_probability_dict_ is None:
            raise RuntimeError("optimize(kind='mr') must be called before transform(kind='mr').")
        if kind == "mf" and self.mf_probability_dict_ is None:
            raise RuntimeError("optimize(kind='mf') must be called before transform(kind='mf').")
        if kind == "mrc" and self.mrc_probability_dict_ is None:
            raise RuntimeError("optimize(kind='mrc') must be called before transform(kind='mrc').")
        if kind == "mfc" and self.mfc_probability_dict_ is None:
            raise RuntimeError("optimize(kind='mfc') must be called before transform(kind='mfc').")
        if kind == "mcc" and self.mcc_probability_dict_ is None:
            raise RuntimeError("optimize(kind='mcc') must be called before transform(kind='mcc').")

        user_col = user_col or self.user_col
        item_col = item_col or self.item_col
        datetime_col = datetime_col or self.datetime_col

        df_log = df[[user_col, item_col, datetime_col]].copy()
        df_log.columns = [self._USER_COL, self._ITEM_COL, self._DATETIME_COL]

        if not is_string_dtype(df_log[self._USER_COL]):
            df_log[self._USER_COL] = df_log[self._USER_COL].astype(str)
        if not is_string_dtype(df_log[self._ITEM_COL]):
            df_log[self._ITEM_COL] = df_log[self._ITEM_COL].astype(str)
        if not is_datetime64_any_dtype(df_log[self._DATETIME_COL]):
            df_log[self._DATETIME_COL] = pd.to_datetime(df_log[self._DATETIME_COL])

        if ref_date is None:
            ref_date = df_log[self._DATETIME_COL].max()
        else:
            try:
                ref_date = pd.to_datetime(ref_date)
            except (ValueError, TypeError) as e:
                raise ValueError(f"ref_date could not be parsed as a date: {ref_date}") from e

        prob_dict = self._probability_dict(kind)
        prob_df = pd.DataFrame(
            [(r, f, p) for (r, f), p in prob_dict.items()],
            columns=["recency_adj", "frequency_adj", "probability"],
        )
        df_rf = self._build_ui_rf_df(df_log, ref_date)
        df_rf["recency_adj"] = df_rf["recency"].clip(upper=self.recency_limit)
        df_rf["frequency_adj"] = df_rf["frequency"].clip(upper=self.frequency_limit)
        df_rf = df_rf.merge(prob_df, on=["recency_adj", "frequency_adj"], how="left")
        df_rf["probability"] = df_rf["probability"].fillna(0.0)
        df_rf = df_rf.drop(columns=["recency_adj", "frequency_adj"])
        df_rf = df_rf.sort_values([self._USER_COL, "probability"], ascending=[True, False])
        df_rf["order"] = df_rf.groupby(self._USER_COL).cumcount() + 1
        df_rf = df_rf.rename(columns={self._USER_COL: user_col, self._ITEM_COL: item_col})

        df_rf[user_col] = df_rf[user_col].astype(df[user_col].dtype)
        df_rf[item_col] = df_rf[item_col].astype(df[item_col].dtype)

        return df_rf

    def transform_date(
        self,
        df,
        target_date,
        kind="emp",
        user_col=None,
        item_col=None,
        datetime_col=None,
    ):
        """Add recency, frequency, and revisit probability columns to a DataFrame.

        Filters df to rows up to target_date, then computes recency rank and
        frequency for each user-item pair relative to target_date, and appends
        the corresponding revisit probability.

        Parameters
        ----------
        df : pd.DataFrame
            User-item interaction history to score.
        target_date : str or datetime
            Reference date for computing recency and frequency.
            Rows after this date are excluded.
        kind : {"emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc"}, default "emp"
            Which probability to use. "emp", "er", and "ef" use fit(),
            fit_date(), or fit_period() results; others use optimize() results.
        user_col : str, optional
            Column name for user IDs in df. Defaults to the value set in __init__.
        item_col : str, optional
            Column name for item IDs in df. Defaults to the value set in __init__.
        datetime_col : str, optional
            Column name for interaction timestamps in df. Defaults to the value
            set in __init__.

        Returns
        -------
        pd.DataFrame
            One row per user-item pair observed in df (up to target_date).
            User and item columns retain the names used (from __init__ or the
            overrides). Additional columns: recency, frequency, probability,
            order. Sorted by user ascending and probability descending; order
            starts at 1.
        """
        datetime_col_name = datetime_col or self.datetime_col

        try:
            target_date = pd.to_datetime(target_date)
        except (ValueError, TypeError) as e:
            raise ValueError(f"target_date could not be parsed as a date: {target_date}") from e

        df_filtered = df[pd.to_datetime(df[datetime_col_name]) <= target_date]
        return self.transform(
            df_filtered,
            ref_date=target_date,
            kind=kind,
            user_col=user_col,
            item_col=item_col,
            datetime_col=datetime_col,
        )

    def evaluate(self, df_rec, df_eval, order=1, user_col=None, item_col=None):
        """Evaluate recommendation quality at each order cutoff.

        Parameters
        ----------
        df_rec : pd.DataFrame
            Recommendation results from transform(). Must have an "order" column.
        df_eval : pd.DataFrame
            Evaluation period event log. Used to derive the ground truth
            set of (user, item) pairs that experienced the target event.
        order : int, default 1
            Maximum recommendation rank to evaluate. Results are computed for
            each rank from 1 to order, plus the maximum order in df_rec.
        user_col : str, optional
            Column name for user in df_rec and df_eval. Defaults to the value
            set in __init__.
        item_col : str, optional
            Column name for item in df_rec and df_eval. Defaults to the value
            set in __init__.

        Returns
        -------
        pd.DataFrame
            Evaluation metrics for each order cutoff. Columns:
            order, n_recommended, n_hit, precision, recall, f1,
            recall_norm (recall normalized by the maximum hits achievable
            within df_rec), f1_norm (f1 using recall_norm instead of recall).
        """
        user_col = user_col or self.user_col
        item_col = item_col or self.item_col

        if not isinstance(df_eval, pd.DataFrame):
            raise TypeError("df_eval must be a pandas DataFrame.")
        missing = [c for c in [user_col, item_col] if c not in df_eval.columns]
        if missing:
            raise ValueError(f"Missing required columns in df_eval: {missing}")

        UIrevisit = set(zip(df_eval[user_col].astype(str), df_eval[item_col].astype(str)))

        df_rec = df_rec.copy()
        try:
            df_rec[user_col] = df_rec[user_col].astype(str)
        except Exception as e:
            raise ValueError(f"Failed to cast column {user_col!r} to str: {e}") from e
        try:
            df_rec[item_col] = df_rec[item_col].astype(str)
        except Exception as e:
            raise ValueError(f"Failed to cast column {item_col!r} to str: {e}") from e

        order_max = df_rec.order.max()

        target_orders = list(range(1, order + 1))
        target_orders += [order_max] if order_max not in set(target_orders) else []
        Rows = []
        for recommend_num in target_orders:
            df_k = df_rec[df_rec["order"] <= recommend_num]
            UIrec = set(zip(df_k[user_col], df_k[item_col]))
            n_hit = len(UIrec & UIrevisit)
            n_recommended = len(UIrec)
            precision = n_hit / n_recommended if n_recommended > 0 else 0.0
            Rows.append((recommend_num, n_recommended, n_hit, precision))
        df_result = pd.DataFrame(Rows, columns=["order", "n_recommended", "n_hit", "precision"])

        total_hit = df_result.n_hit.max()
        df_result["recall"] = df_result.n_hit / len(UIrevisit)
        denom = df_result.precision + df_result.recall
        df_result["f1"] = (2 * df_result.precision * df_result.recall).where(
            denom > 0, 0.0
        ) / denom.where(denom > 0, 1.0)
        df_result["recall_norm"] = df_result.n_hit / total_hit
        denom_norm = df_result.precision + df_result.recall_norm
        df_result["f1_norm"] = (2 * df_result.precision * df_result.recall_norm).where(
            denom_norm > 0, 0.0
        ) / denom_norm.where(denom_norm > 0, 1.0)

        return df_result

    def _build_ui_rf_df(self, df, ref_date):
        """Compute recency and frequency for each (user, item) pair.

        Recency is the minimum days from any interaction to ref_date plus 1
        (1-indexed; most recent interaction on ref_date has recency 1).
        Frequency is the interaction count. Uses pandas groupby.
        """
        tmp = df[[self._USER_COL, self._ITEM_COL]].copy()
        tmp["recency"] = (ref_date - df[self._DATETIME_COL]).dt.days + 1
        return (
            tmp.groupby([self._USER_COL, self._ITEM_COL], sort=False)
            .agg(recency=("recency", "min"), frequency=("recency", "count"))
            .reset_index()
        )

    _KIND_ALIASES = {
        "empirical": "emp",
        "empirical_recency": "er",
        "empirical_frequency": "ef",
        "monotonic": "mono",
        "monotonic_recency": "mr",
        "monotonic_frequency": "mf",
        "monotonic_recency_convex": "mrc",
        "monotonic_frequency_concave": "mfc",
        "monotonic_convex_concave": "mcc",
    }

    def _normalize_kind(self, kind):
        return self._KIND_ALIASES.get(kind, kind)

    def _probability_dict(self, kind):
        if kind == "er":
            return self.er_probability_dict_
        if kind == "ef":
            return self.ef_probability_dict_
        if kind == "mono":
            return self.mono_probability_dict_
        if kind == "mr":
            return self.mr_probability_dict_
        if kind == "mf":
            return self.mf_probability_dict_
        if kind == "mrc":
            return self.mrc_probability_dict_
        if kind == "mfc":
            return self.mfc_probability_dict_
        if kind == "mcc":
            return self.mcc_probability_dict_
        return self.empirical_probability_dict_

    def optimize(self, kind="mono", eps=0.0):
        """Estimate optimized revisit probabilities under RF constraints.

        Solves a convex quadratic programming problem with monotonicity
        constraints (and optionally convexity/concavity constraints).
        Uses weighted least squares as objective.

        Requires fit(), fit_date(), or fit_period() to be called first. Depends on cvxpy.

        Parameters
        ----------
        kind : {"mono", "mr", "mf", "mrc", "mfc", "mcc"}, default "mono"
            Optimization model to use.
            "mono" applies monotonicity constraints only (2D joint model).
            "mr" fits a 1-D recency-only model with monotonicity and convexity.
            "mf" fits a 1-D frequency-only model with monotonicity and concavity.
            "mrc" additionally applies convexity in recency (2D joint model).
            "mfc" additionally applies concavity in frequency (2D joint model).
            "mcc" applies both recency convexity and frequency concavity (2D joint model).
        eps : float, default 0.0
            Minimum gap enforced between adjacent values in monotonicity
            constraints.  When 0.0 (default), broad monotonicity is used.
            When positive, strict monotonicity is enforced, preventing ties
            between adjacent recency or frequency levels.

        Returns
        -------
        self
        """
        kind = self._normalize_kind(kind)
        if kind not in ("mono", "mr", "mf", "mrc", "mfc", "mcc"):
            raise ValueError(
                f"kind must be 'mono', 'mr', 'mf', 'mrc', 'mfc', or 'mcc', got {kind!r}."
            )

        try:
            from .optimizer import RFOptimizer
        except ImportError:
            from rfscorer.optimizer import RFOptimizer

        if self.empirical_probability_dict_ is None:
            raise RuntimeError(
                "fit(), fit_date(), or fit_period() must be called before optimize()."
            )

        optimizer = RFOptimizer()
        optimizer.set_data(self.R, self.F, self.RF2N, self.RF2Prob)
        optimizer.set_marginal_data(self.R2N, self.R2Prob, self.F2N, self.F2Prob)
        optimizer.build_model(kind=kind, eps=eps)
        optimizer.solve()
        optimizer.show_solve_info()
        optimizer.postprocess()

        rows = [(r, f, optimizer.RF2X[(r, f)]) for r in self.R for f in self.F]
        df_opt = pd.DataFrame(rows, columns=["recency", "frequency", "probability"])
        table = df_opt.pivot_table(index="recency", columns="frequency", values="probability")

        if kind == "mono":
            self.mono_probability_dict_ = optimizer.RF2X
            self.mono_probability_ = df_opt
            self.mono_probability_table_ = table
        elif kind == "mr":
            self.mr_probability_dict_ = optimizer.RF2X
            self.mr_probability_ = df_opt
            self.mr_probability_table_ = table
        elif kind == "mf":
            self.mf_probability_dict_ = optimizer.RF2X
            self.mf_probability_ = df_opt
            self.mf_probability_table_ = table
        elif kind == "mrc":
            self.mrc_probability_dict_ = optimizer.RF2X
            self.mrc_probability_ = df_opt
            self.mrc_probability_table_ = table
        elif kind == "mfc":
            self.mfc_probability_dict_ = optimizer.RF2X
            self.mfc_probability_ = df_opt
            self.mfc_probability_table_ = table
        else:
            self.mcc_probability_dict_ = optimizer.RF2X
            self.mcc_probability_ = df_opt
            self.mcc_probability_table_ = table

        return self


if __name__ == "__main__":
    # データの読み込み（オーム社『Pythonではじめる数理最適化』サポートデータより引用）
    url = "https://raw.githubusercontent.com/ohmsha/PyOptBook/main/7.recommendation/access_log.csv"
    df = pd.read_csv(url)
    df_train = df[df.user_id.map(lambda x: hash(x) % 10 < 8)]
    df_test = df[df.user_id.map(lambda x: hash(x) % 10 >= 8)]

    scorer = RecencyFrequencyScorer(user_col="user_id", item_col="item_id", datetime_col="date")

    target_date = "2015-07-07"

    # 観測期間・評価期間に分割してから fit
    df_train_dates = pd.to_datetime(df_train["date"])
    df_train_obs = df_train[df_train_dates <= target_date]
    df_train_eval = df_train[df_train_dates > target_date]
    scorer.fit(df_train_obs, df_train_eval)

    scorer.plot_probability_surface("empirical").savefig("surface_emp_probability.png")
    scorer.plot_marginal_probability("recency").savefig("marginal_recency_probability.png")
    scorer.plot_marginal_probability("frequency").savefig("marginal_frequency_probability.png")
    scorer.plot_probability_surface("er").savefig("surface_er_probability.png")
    scorer.plot_probability_surface("ef").savefig("surface_ef_probability.png")

    scorer.optimize(kind="mr")
    scorer.plot_probability_surface("mr").savefig("surface_mr_probability.png")
    scorer.plot_marginal_probability("recency", kind="mr").savefig(
        "marginal_mono_recency_probability.png"
    )
    scorer.plot_marginal_probability("recency", kind="all").savefig(
        "marginal_all_recency_probability.png"
    )

    scorer.optimize(kind="mf")
    scorer.plot_probability_surface("mf").savefig("surface_mf_probability.png")
    scorer.plot_marginal_probability("frequency", kind="mf").savefig(
        "marginal_mono_frequency_probability.png"
    )
    scorer.plot_marginal_probability("frequency", kind="all").savefig(
        "marginal_all_frequency_probability.png"
    )

    scorer.optimize(kind="mono")
    scorer.plot_probability_surface("mono").savefig("surface_mono_probability.png")
    scorer.optimize(kind="mrc")
    scorer.plot_probability_surface("mrc").savefig("surface_mrc_probability.png")
    scorer.optimize(kind="mfc")
    scorer.plot_probability_surface("mfc").savefig("surface_mfc_probability.png")
    scorer.optimize(kind="mcc")
    scorer.plot_probability_surface("mcc").savefig("surface_mcc_probability.png")
    scorer.export_probability_csv("all")

    df_test_dates = pd.to_datetime(df_test["date"])
    df_test_obs = df_test[df_test_dates <= target_date]
    df_test_eval = df_test[df_test_dates > target_date]

    for kind in ("emp", "er", "ef", "mr", "mf", "mono", "mrc", "mfc", "mcc"):
        print(f"--- {kind} ---")
        df_rec = scorer.transform(df_test_obs, kind=kind)
        df_rec.to_csv(f"df_recommend_{kind}.csv", index=False)
        print(scorer.evaluate(df_rec, df_test_eval, order=10))
