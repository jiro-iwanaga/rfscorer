import datetime

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
)


class RecencyFrequencyScorer:
    """Recency-Frequency based recommendation scorer.

    Estimates product-choice probabilities from user-item interaction histories
    using recency and frequency as behavioral signals.
    """

    _USER_COL = "user"
    _ITEM_COL = "item"
    _SEQUENCE_COL = "datetime"
    _ORDINAL_ORIGIN = pd.Timestamp("0001-01-01")
    _FREQUENCY_LIMIT_RATE = 0.95  # 最新度上限値自動計算の際に利用する割合
    _RECENCY_LIMIT_RATE = 0.95  # 頻度上限値自動計算の際に利用する割合

    def __init__(self, user_col="user", item_col="item", time_col="datetime", unit=1):
        """Initialize the scorer with column name mappings.

        Parameters
        ----------
        user_col : str, default "user"
            Column name for user IDs in the interaction log.
        item_col : str, default "item"
            Column name for item IDs in the interaction log.
        time_col : str, default "datetime"
            Column name for the time axis in the interaction log. Accepts
            datetime (datetime64, str) or integer columns. Datetime values are
            converted to ordinal integers internally; integer values are used
            as-is.
        unit : int, default 1
            Number of days (or integer steps) per recency bin. Recency is
            computed as ``(ref - value) // unit + 1``. Must be a positive
            integer. Use ``unit=7`` for weekly, ``unit=30`` for approximate
            monthly granularity.
        """
        if unit <= 0:
            raise ValueError(f"unit must be a positive integer, got {unit}.")
        self.user_col = user_col
        self.item_col = item_col
        self.time_col = time_col
        self.unit = unit

        self.observation_start_ = None
        self.observation_end_ = None
        self.evaluation_start_ = None
        self.evaluation_end_ = None
        self.recency_limit = (
            None  # 最新度の上限値(デフォルトでは _RECENCY_LIMIT_RATE を用いて自動計算)
        )
        self.frequency_limit = (
            None  # 頻度の上限値(デフォルトでは _FREQUENCY_LIMIT_RATE を用いて自動計算)
        )

        self.R = []  # 最新度のリスト
        self.F = []  # 頻度のリスト
        self.RF2N = {}  # 最新度と頻度に対して閲覧数合計を紐づける辞書
        self.RF2CV = {}  # 最新度と頻度に対して対象イベント発生数合計を紐づける辞書
        self.RF2Prob = {}  # 最新度と頻度に対して経験的商品選択確率を紐づける辞書
        self.R2N = {}  # 最新度に対して閲覧数合計を紐づける辞書
        self.R2CV = {}  # 最新度に対して対象イベント発生数合計を紐づける辞書
        self.R2Prob = {}  # 最新度に対して経験的商品選択確率を紐づける辞書
        self.F2N = {}  # 頻度に対して閲覧数合計を紐づける辞書
        self.F2CV = {}  # 頻度に対して対象イベント発生数合計を紐づける辞書
        self.F2Prob = {}  # 頻度に対して経験的商品選択確率を紐づける辞書

        # empirical
        self.emp_probability_ = None  # 経験的商品選択確率データフレーム(縦持ち)
        self.emp_probability_table_ = None  # 経験的商品選択確率データフレーム(横持ち)
        self.emp_probability_dict_ = (
            None  # 経験的商品選択確率データフレーム(辞書:キーは最新度と頻度のペア)
        )
        self.recency_probability_ = None  # 最新度別経験的商品選択確率データフレーム
        self.frequency_probability_ = None  # 頻度別経験的商品選択確率データフレーム

        # er (1D: recency only)
        self.er_probability_ = None  # DataFrame(recency, probability)
        self.er_probability_dict_ = None  # dict[int, float]

        # ef (1D: frequency only)
        self.ef_probability_ = None  # DataFrame(frequency, probability)
        self.ef_probability_dict_ = None  # dict[int, float]

        # mono
        self.mono_probability_ = None
        self.mono_probability_table_ = None
        self.mono_probability_dict_ = None

        # mr (1D: recency only)
        self.mr_probability_ = None  # DataFrame(recency, probability)
        self.mr_probability_dict_ = None  # dict[int, float]

        # mf (1D: frequency only)
        self.mf_probability_ = None  # DataFrame(frequency, probability)
        self.mf_probability_dict_ = None  # dict[int, float]

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
        ref=None,
        recency_limit=None,
        frequency_limit=None,
    ):
        """Estimate empirical product-choice probabilities from pre-split interaction data.

        Accepts observation and evaluation DataFrames that have already been
        filtered to the respective periods by the caller.  This is the primary
        fitting method; for convenience wrappers that perform automatic
        splitting, see fit_date() and fit_period().

        Parameters
        ----------
        df_obs : pd.DataFrame
            Observation period interaction log. Must already be filtered to
            the observation period by the caller.
        df_eval : pd.DataFrame
            Evaluation period event log (revisits, purchases, conversions,
            etc.). Must already be filtered to the evaluation period by the caller.
        ref : str, datetime, or int, optional
            Reference value for recency computation. Recency of each
            user-item pair is ``(ref - value) // unit + 1``, where the
            minimum across interactions is taken. When None, defaults to the
            maximum value in df_obs time_col.
        recency_limit : int, optional
            Maximum recency rank to include. If None, automatically set to
            the recency rank covering 95% of cumulative events.
        frequency_limit : int, optional
            Maximum frequency to include. If None, automatically set to
            the frequency covering 95% of cumulative events.

        Returns
        -------
        self

        Raises
        ------
        TypeError
            If df_obs or df_eval is not a pandas DataFrame.
        ValueError
            If required columns (user, item, time_col) are missing from
            df_obs or df_eval, if ref cannot be normalized, or if no events
            are observed in the evaluation period (cannot determine
            recency_limit or frequency_limit automatically).

        Notes
        -----
        After a successful call, the following attributes become available
        for use with predict(), transform(), and plot_*() methods:
        ``emp_probability_``, ``emp_probability_table_``,
        ``emp_probability_dict_``, ``recency_probability_``,
        ``frequency_probability_``, ``recency_limit``, ``frequency_limit``.
        """
        if not isinstance(df_obs, pd.DataFrame):
            raise TypeError("df_obs must be a pandas DataFrame.")
        if not isinstance(df_eval, pd.DataFrame):
            raise TypeError("df_eval must be a pandas DataFrame.")

        required_columns = [self.user_col, self.item_col, self.time_col]
        missing_obs = [c for c in required_columns if c not in df_obs.columns]
        if missing_obs:
            raise ValueError(f"Missing required columns in df_obs: {missing_obs}")
        missing_eval = [c for c in required_columns if c not in df_eval.columns]
        if missing_eval:
            raise ValueError(f"Missing required columns in df_eval: {missing_eval}")

        obs_log = self._to_internal(df_obs)
        eval_log = self._to_internal(df_eval)

        self.record_num = len(obs_log) + len(eval_log)

        if ref is None:
            ref_int = int(obs_log[self._SEQUENCE_COL].max())
        else:
            ref_int = self._normalize_ref(ref)

        self.observation_start_ = (
            int(obs_log[self._SEQUENCE_COL].min()) if len(obs_log) > 0 else None
        )
        self.observation_end_ = ref_int
        self.evaluation_start_ = (
            int(eval_log[self._SEQUENCE_COL].min()) if len(eval_log) > 0 else None
        )
        self.evaluation_end_ = (
            int(eval_log[self._SEQUENCE_COL].max()) if len(eval_log) > 0 else None
        )

        self._fit_impl(obs_log, eval_log, ref_int, recency_limit, frequency_limit)
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
        """Estimate empirical product-choice probabilities using target_date as split point.

        Derives observation and evaluation periods automatically from target_date:
        observation spans from at most observation_days before target_date up to
        and including target_date; evaluation spans from the next day to at most
        evaluation_days after target_date.  Pass None for either days argument to
        use the full data range in that direction.

        Parameters
        ----------
        df : pd.DataFrame
            Interaction log containing user, item, and time_col columns.
        target_date : str, datetime, or int
            Reference value used as the observation end / evaluation start
            boundary. Accepts the same types as time_col (datetime or integer).
            target_date is inclusive in the observation period.
        observation_days : int or None, default 28
            Maximum number of units to look back from target_date for the
            observation period.  If None, uses all data up to target_date.
        evaluation_days : int or None, default 7
            Maximum number of units to look forward from target_date for the
            evaluation period.  If None, uses all data after target_date.
        recency_limit : int, optional
            Maximum recency rank to include.  If None, determined automatically.
        frequency_limit : int, optional
            Maximum frequency to include.  If None, determined automatically.

        Returns
        -------
        self

        Raises
        ------
        TypeError
            If df is not a pandas DataFrame.
        ValueError
            If time_col is missing from df, if target_date cannot be
            normalized, or if no events are observed in the evaluation period.

        Notes
        -----
        After a successful call, the same attributes as fit() become available:
        ``emp_probability_``, ``emp_probability_table_``,
        ``emp_probability_dict_``, ``recency_probability_``,
        ``frequency_probability_``, ``recency_limit``, ``frequency_limit``.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if self.time_col not in df.columns:
            raise ValueError(f"Missing required columns: [{self.time_col!r}]")

        interaction_log = self._to_internal(df)
        self.record_num = len(interaction_log)

        target_int = self._normalize_ref(target_date)
        df_min = int(interaction_log[self._SEQUENCE_COL].min())
        df_max = int(interaction_log[self._SEQUENCE_COL].max())

        if observation_days is None:
            obs_start = df_min
        else:
            obs_start = max(df_min, target_int - observation_days)
        obs_end = target_int
        eval_start = target_int + 1
        eval_end = df_max if evaluation_days is None else min(df_max, target_int + evaluation_days)

        obs_log = interaction_log[
            (obs_start <= interaction_log[self._SEQUENCE_COL])
            & (interaction_log[self._SEQUENCE_COL] <= obs_end)
        ]
        eval_log = interaction_log[
            (eval_start <= interaction_log[self._SEQUENCE_COL])
            & (interaction_log[self._SEQUENCE_COL] <= eval_end)
        ]

        self.observation_start_ = obs_start
        self.observation_end_ = obs_end
        self.evaluation_start_ = eval_start
        self.evaluation_end_ = eval_end

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
        """Estimate empirical product-choice probabilities from interaction history.

        Use this when you need explicit control over both periods.
        For the common case of splitting on a single date, prefer fit_date().
        For a sklearn-style interface with pre-split DataFrames, prefer fit().

        Parameters
        ----------
        df : pd.DataFrame
            Interaction log containing user, item, and time_col columns.
        observation_period : tuple[str | datetime | int, str | datetime | int]
            Start and end values of the observation period, both inclusive.
            Accepts the same types as time_col (datetime or integer).
        evaluation_period : tuple[str | datetime | int, str | datetime | int]
            Start and end values of the evaluation period, both inclusive.
            Must start strictly after observation_period ends.
            Accepts the same types as time_col (datetime or integer).
        recency_limit : int, optional
            Maximum recency rank to include. If None, automatically set to
            the recency rank covering 95% of cumulative events.
        frequency_limit : int, optional
            Maximum frequency to include. If None, automatically set to
            the frequency covering 95% of cumulative events.

        Returns
        -------
        self

        Raises
        ------
        TypeError
            If df is not a pandas DataFrame.
        ValueError
            If required columns are missing from df, if observation_period or
            evaluation_period cannot be normalized, if a period is not ordered
            as (start, end), if the periods overlap, or if no events are
            observed in the evaluation period.

        Notes
        -----
        After a successful call, the same attributes as fit() become available:
        ``emp_probability_``, ``emp_probability_table_``,
        ``emp_probability_dict_``, ``recency_probability_``,
        ``frequency_probability_``, ``recency_limit``, ``frequency_limit``.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")

        required_columns = [self.user_col, self.item_col, self.time_col]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if len(observation_period) != 2:
            raise ValueError("observation_period must be a tuple of (start, end).")
        try:
            obs_start, obs_end = (self._normalize_ref(v) for v in observation_period)
        except ValueError as e:
            raise ValueError(
                f"observation_period could not be normalized: {observation_period}"
            ) from e

        if len(evaluation_period) != 2:
            raise ValueError("evaluation_period must be a tuple of (start, end).")
        try:
            eval_start, eval_end = (self._normalize_ref(v) for v in evaluation_period)
        except ValueError as e:
            raise ValueError(
                f"evaluation_period could not be normalized: {evaluation_period}"
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
            (obs_start <= interaction_log[self._SEQUENCE_COL])
            & (interaction_log[self._SEQUENCE_COL] <= obs_end)
        ]
        eval_log = interaction_log[
            (eval_start <= interaction_log[self._SEQUENCE_COL])
            & (interaction_log[self._SEQUENCE_COL] <= eval_end)
        ]

        self.observation_start_ = obs_start
        self.observation_end_ = obs_end
        self.evaluation_start_ = eval_start
        self.evaluation_end_ = eval_end

        self._fit_impl(obs_log, eval_log, obs_end, recency_limit, frequency_limit)
        return self

    def _normalize_ref(self, value) -> int:
        """Normalize a single time reference value (date or integer) to int."""
        if isinstance(value, (pd.Timestamp, datetime.datetime)):
            return value.toordinal()
        elif isinstance(value, str):
            return pd.to_datetime(value).toordinal()
        elif isinstance(value, (int, float, np.integer, np.floating)):
            return int(value)
        else:
            try:
                return int(pd.to_datetime(value).toordinal())
            except Exception:
                raise ValueError(f"time value could not be normalized: {value!r}")

    def _normalize_sequence_col(self, series: pd.Series) -> pd.Series:
        """Normalize a time column (datetime or integer) to an integer Series."""
        if is_datetime64_any_dtype(series):
            return (series - self._ORDINAL_ORIGIN).dt.days + 1
        elif is_string_dtype(series):
            return (pd.to_datetime(series) - self._ORDINAL_ORIGIN).dt.days + 1
        elif is_integer_dtype(series) or is_float_dtype(series):
            return series.astype(int)
        else:
            raise ValueError(f"time_col must be datetime or integer type, got {series.dtype}")

    def _to_internal(self, df):
        """Convert a user-facing DataFrame to internal column names and types."""
        result = df[[self.user_col, self.item_col, self.time_col]].copy()
        result.columns = [self._USER_COL, self._ITEM_COL, self._SEQUENCE_COL]
        if not is_string_dtype(result[self._USER_COL]):
            result[self._USER_COL] = result[self._USER_COL].astype(str)
        if not is_string_dtype(result[self._ITEM_COL]):
            result[self._ITEM_COL] = result[self._ITEM_COL].astype(str)
        result[self._SEQUENCE_COL] = self._normalize_sequence_col(result[self._SEQUENCE_COL])
        return result

    def _fit_impl(self, obs_log, eval_log, ref_int, recency_limit, frequency_limit):
        """Core fitting logic. obs_log and eval_log must use internal column names."""
        self.record_num_obs = len(obs_log)
        self.record_num_eval = len(eval_log)

        UIcv = {(row.user, row.item) for row in eval_log.itertuples()}

        df_ui2frc = self._build_ui_rf_df(obs_log, ref_int)
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
                    "No events observed in evaluation period. "
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
                    "No events observed in evaluation period. "
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

        self.emp_probability_dict_ = {(r, f): prob for r, f, _, _, prob in RowsRF}
        self.emp_probability_ = pd.DataFrame(
            RowsRF, columns=["recency", "frequency", "N", "cv", "probability"]
        )
        self.emp_probability_table_ = self.emp_probability_.pivot_table(
            index="recency",
            columns="frequency",
            values="probability",
        )

        df_r = self.emp_probability_.groupby("recency")[["N", "cv"]].sum().reset_index()
        df_r["probability"] = (df_r["cv"] / df_r["N"]).where(df_r["N"] > 0, 0.0)
        self.recency_probability_ = df_r
        self.R2N = dict(zip(df_r["recency"], df_r["N"]))
        self.R2CV = dict(zip(df_r["recency"], df_r["cv"]))
        self.R2Prob = dict(zip(df_r["recency"], df_r["probability"]))

        df_f = self.emp_probability_.groupby("frequency")[["N", "cv"]].sum().reset_index()
        df_f["probability"] = (df_f["cv"] / df_f["N"]).where(df_f["N"] > 0, 0.0)
        self.frequency_probability_ = df_f
        self.F2N = dict(zip(df_f["frequency"], df_f["N"]))
        self.F2CV = dict(zip(df_f["frequency"], df_f["cv"]))
        self.F2Prob = dict(zip(df_f["frequency"], df_f["probability"]))

        self.er_probability_dict_ = dict(self.R2Prob)
        self.er_probability_ = (
            pd.DataFrame(list(self.R2Prob.items()), columns=["recency", "probability"])
            .sort_values("recency")
            .reset_index(drop=True)
        )

        self.ef_probability_dict_ = dict(self.F2Prob)
        self.ef_probability_ = (
            pd.DataFrame(list(self.F2Prob.items()), columns=["frequency", "probability"])
            .sort_values("frequency")
            .reset_index(drop=True)
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

        if self.observation_start_ and self.observation_end_:
            print("observation: {} -> {}".format(self.observation_start_, self.observation_end_))
        if self.evaluation_start_ and self.evaluation_end_:
            print("evaluation: {} -> {}".format(self.evaluation_start_, self.evaluation_end_))

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

        if self.emp_probability_table_ is not None:
            print("emp_probability_table_:")
            print(self.emp_probability_table_.round(3).to_string())

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
        """Plot product-choice probabilities as a 3D surface.

        Visualizes the probability table as a 3D wireframe with recency on
        the x-axis, frequency on the y-axis, and probability on the z-axis.

        In Jupyter Lab / Colab the returned figure renders inline automatically.
        To save to a file, call ``fig.savefig("output.png")`` on the returned
        figure.

        Parameters
        ----------
        kind : {"emp", "mono", "mrc", "mfc", "mcc"}, default "emp"
            Which probability to visualize. "emp" uses fit(), fit_date(), or
            fit_period() results; others use optimize() results.
            Note: "mr", "mf", "er", and "ef" are 1D marginal models and cannot
            be visualized as a surface; use plot_marginal_probability() instead.
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

        Raises
        ------
        ValueError
            If kind is "mr", "mf", "er", or "ef" (1D marginal models cannot
            be plotted as a surface), or if kind is not one of the accepted
            values.
        RuntimeError
            If fit(), fit_date(), or fit_period() has not been called for
            kind="emp", or if optimize(kind=...) has not been called for
            the requested optimization kind.
        """
        import matplotlib.pyplot as plt

        kind = self._normalize_kind(kind)
        if kind in ("mr", "mf", "er", "ef"):
            raise ValueError(
                f"kind={kind!r} is a 1D marginal model and cannot be plotted as a surface."
                " Use plot_marginal_probability() instead."
            )
        if kind not in ("emp", "mono", "mrc", "mfc", "mcc"):
            raise ValueError(f"kind must be 'emp', 'mono', 'mrc', 'mfc', or 'mcc', got {kind!r}.")
        if kind == "emp" and self.emp_probability_table_ is None:
            raise RuntimeError(
                "fit(), fit_date(), or fit_period() must be called"
                " before plot_probability_surface()."
            )
        if kind == "mono" and self.mono_probability_table_ is None:
            raise RuntimeError(
                "optimize(kind='mono') must be called before plot_probability_surface(kind='mono')."
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
            table = self.emp_probability_table_
        elif kind == "mono":
            table = self.mono_probability_table_
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
        """Plot product-choice probability aggregated along one RF dimension.

        When axis='recency', plots the empirical recency-marginal probability
        (aggregated over all frequency levels) and optionally the mr-optimized
        probability against recency rank.  When axis='frequency', plots the
        empirical frequency-marginal probability and optionally the mf-optimized
        probability against frequency.  Both are shown as line charts with markers.

        In Jupyter Lab / Colab the returned figure renders inline automatically.
        To save to a file, call ``fig.savefig("output.png")`` on the returned figure.

        Parameters
        ----------
        axis : {"recency", "frequency"}, default "recency"
            Which dimension to aggregate and plot.
            "recency" plots probability vs recency rank (expected: decreasing).
            "frequency" plots probability vs frequency (expected: increasing).
        kind : {"emp", "er", "ef", "mr", "mf", "all"}, default "emp"
            Which probability series to draw.
            "emp" draws the empirical marginal (R2Prob or F2Prob).
            "er" draws the empirical recency marginal (valid only when
            axis="recency"). Equivalent to "emp" on the recency axis.
            "ef" draws the empirical frequency marginal (valid only when
            axis="frequency"). Equivalent to "emp" on the frequency axis.
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

        Raises
        ------
        ValueError
            If axis is not "recency" or "frequency", if kind is not one of the
            accepted values, or if the axis/kind combination is invalid (e.g.,
            kind="mf" or "ef" with axis="recency").
        RuntimeError
            If fit(), fit_date(), or fit_period() has not been called, or if
            optimize(kind='mr') / optimize(kind='mf') has not been called when
            kind is "mr", "mf", or "all".
        """
        import matplotlib.pyplot as plt

        kind = self._normalize_kind(kind)
        if axis not in ("recency", "frequency"):
            raise ValueError(f"axis must be 'recency' or 'frequency', got {axis!r}.")
        valid_kinds = ("emp", "er", "ef", "mr", "mf", "all")
        if kind not in valid_kinds:
            raise ValueError(f"kind must be one of {valid_kinds}, got {kind!r}.")
        if axis == "recency" and kind in ("mf", "ef"):
            raise ValueError(f"kind={kind!r} is not valid when axis='recency'. Use 'mr' or 'er'.")
        if axis == "frequency" and kind in ("mr", "er"):
            raise ValueError(f"kind={kind!r} is not valid when axis='frequency'. Use 'mf' or 'ef'.")
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
            df_opt = getattr(self, f"{opt_kind}_probability_")[[x_col, "probability"]].reset_index(
                drop=True
            )

        fig, ax = plt.subplots(figsize=figsize)
        if kind in ("emp", "er", "ef", "all"):
            ax.plot(
                df_emp[x_col],
                df_emp["probability"],
                color="black",
                linestyle="-",
                marker="o",
                linewidth=1.5,
                markersize=6,
                label="emp" if kind == "all" else kind,
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
        """Export product-choice probabilities to a CSV file.

        Parameters
        ----------
        kind : {"emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc", "all"}, default "emp"
            Which probability to export. "emp", "er", and "ef" use fit(),
            fit_date(), or fit_period() results; others use optimize() results;
            "all" merges all nine models into a single file. For 2D models the
            merge key is (recency, frequency); for 1D models mr merges on
            recency and mf merges on frequency, so their probability columns
            are constant along the other axis.
        path : str or None, default None
            Output file path for the CSV. If None, saves as
            "{kind}_probability.csv" in the current directory.
            If a directory, saves "{kind}_probability.csv" inside it.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If kind is not one of the accepted values.
        RuntimeError
            If fit(), fit_date(), or fit_period() has not been called for
            emp/er/ef kinds, or if the required optimize(kind=...) has not been
            called for the requested optimization kind.
        """
        kind = self._normalize_kind(kind)
        if kind not in ("emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc", "all"):
            raise ValueError(
                f"kind must be 'emp', 'er', 'ef', 'mono', 'mr', 'mf', 'mrc', 'mfc', 'mcc',"
                f" or 'all', got {kind!r}."
            )
        if kind in ("emp", "er", "ef", "all") and self.emp_probability_ is None:
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
                self.emp_probability_.rename(columns={"probability": "emp_probability"})
                .merge(
                    self.mono_probability_.rename(columns={"probability": "mono_probability"}),
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
                .merge(
                    self.er_probability_.rename(columns={"probability": "er_probability"}),
                    on="recency",
                )
                .merge(
                    self.mr_probability_.rename(columns={"probability": "mr_probability"}),
                    on="recency",
                )
                .merge(
                    self.ef_probability_.rename(columns={"probability": "ef_probability"}),
                    on="frequency",
                )
                .merge(
                    self.mf_probability_.rename(columns={"probability": "mf_probability"}),
                    on="frequency",
                )
            )
        elif kind == "emp":
            df = self.emp_probability_
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
        """Return the product-choice probability for a given recency and frequency.

        Parameters
        ----------
        r : int
            Recency rank (1 = most recently interacted, higher = older).
        f : int
            Frequency (number of interactions in the observation period).
        kind : {"emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc"}, default "emp"
            Which probability to use. "emp", "er", and "ef" use fit(),
            fit_date(), or fit_period() results; others use optimize() results.
            For 1D marginal models, only the relevant dimension is used:
            "mr" and "er" use r only; "mf" and "ef" use f only.

        Returns
        -------
        float
            Product-choice probability for the given (r, f). If r exceeds
            recency_limit, it is clamped to recency_limit before lookup.
            If f exceeds frequency_limit, it is clamped to frequency_limit.
            For 1D kinds "mr" and "er", f is ignored; for "mf" and "ef",
            r is ignored.

        Raises
        ------
        TypeError
            If r or f is not a positive integer.
        ValueError
            If kind is not one of the accepted values.
        RuntimeError
            If fit(), fit_date(), or fit_period() has not been called for
            emp/er/ef kinds, or if optimize(kind=...) has not been called
            for the requested optimization kind.
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
        if kind in ("emp", "er", "ef") and self.emp_probability_dict_ is None:
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

        if kind in ("mr", "er"):
            r_clipped = min(r, self.recency_limit)
            return self._marginal_dict(kind).get(r_clipped, 0.0)
        if kind in ("mf", "ef"):
            f_clipped = min(f, self.frequency_limit)
            return self._marginal_dict(kind).get(f_clipped, 0.0)

        r = min(r, self.recency_limit)
        f = min(f, self.frequency_limit)
        if kind == "emp":
            prob = self.emp_probability_dict_.get((r, f), 0.0)
        elif kind == "mono":
            prob = self.mono_probability_dict_.get((r, f), 0.0)
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
        ref=None,
        kind="emp",
        user_col=None,
        item_col=None,
        time_col=None,
    ):
        """Add recency, frequency, and product-choice probability columns to a DataFrame.

        Computes recency rank and frequency for each user-item pair relative to
        ref, then appends the corresponding product-choice probability from fit() or
        optimize() results.  Does not filter df; pass a pre-filtered DataFrame
        or use transform_date() to apply automatic filtering.
        Recency values above recency_limit and frequency values above
        frequency_limit are clamped to their respective limits before lookup.

        Parameters
        ----------
        df : pd.DataFrame
            User-item interaction history to score. If this DataFrame is not
            pre-filtered to the desired observation window, use transform_date()
            instead, which filters automatically up to a given value.
        ref : str, datetime, or int, optional
            Reference value for computing recency. When None, defaults to the
            maximum value in df time_col.
        kind : {"emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc"}, default "emp"
            Which probability to use. "emp", "er", and "ef" use fit(),
            fit_date(), or fit_period() results; others use optimize() results.
            For 1D marginal models ("mr", "er", "mf", "ef"), probability is
            looked up by recency or frequency alone.
        user_col : str, optional
            Column name for user IDs in df. Defaults to the value set in __init__.
        item_col : str, optional
            Column name for item IDs in df. Defaults to the value set in __init__.
        time_col : str, optional
            Column name for the time axis in df. Defaults to the value set in
            __init__.

        Returns
        -------
        pd.DataFrame
            One row per user-item pair observed in df.  User and item columns
            retain the names used (from __init__ or the overrides). Additional
            columns: recency, frequency, probability, order. Sorted by user
            ascending and probability descending; order starts at 1.

        Raises
        ------
        ValueError
            If kind is not one of the accepted values, or if ref cannot be
            normalized.
        RuntimeError
            If fit(), fit_date(), or fit_period() has not been called, or if
            optimize(kind=...) has not been called for the requested kind.
        """
        kind = self._normalize_kind(kind)
        if kind not in ("emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc"):
            raise ValueError(
                f"kind must be 'emp', 'er', 'ef', 'mono', 'mr', 'mf', 'mrc', 'mfc', or 'mcc',"
                f" got {kind!r}."
            )
        if self.emp_probability_dict_ is None:
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
        time_col = time_col or self.time_col

        df_log = df[[user_col, item_col, time_col]].copy()
        df_log.columns = [self._USER_COL, self._ITEM_COL, self._SEQUENCE_COL]

        if not is_string_dtype(df_log[self._USER_COL]):
            df_log[self._USER_COL] = df_log[self._USER_COL].astype(str)
        if not is_string_dtype(df_log[self._ITEM_COL]):
            df_log[self._ITEM_COL] = df_log[self._ITEM_COL].astype(str)
        df_log[self._SEQUENCE_COL] = self._normalize_sequence_col(df_log[self._SEQUENCE_COL])

        if ref is None:
            ref_int = int(df_log[self._SEQUENCE_COL].max())
        else:
            ref_int = self._normalize_ref(ref)

        df_rf = self._build_ui_rf_df(df_log, ref_int)
        df_rf["recency_adj"] = df_rf["recency"].clip(upper=self.recency_limit)
        df_rf["frequency_adj"] = df_rf["frequency"].clip(upper=self.frequency_limit)

        if kind in ("mr", "er"):
            prob_df = pd.DataFrame(
                list(self._marginal_dict(kind).items()),
                columns=["recency_adj", "probability"],
            )
            df_rf = df_rf.merge(prob_df, on="recency_adj", how="left")
        elif kind in ("mf", "ef"):
            prob_df = pd.DataFrame(
                list(self._marginal_dict(kind).items()),
                columns=["frequency_adj", "probability"],
            )
            df_rf = df_rf.merge(prob_df, on="frequency_adj", how="left")
        else:
            prob_dict = self._probability_dict(kind)
            prob_df = pd.DataFrame(
                [(r, f, p) for (r, f), p in prob_dict.items()],
                columns=["recency_adj", "frequency_adj", "probability"],
            )
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
        time_col=None,
    ):
        """Score user-item pairs, filtering the interaction log up to target_date.

        Filters df to rows on or before target_date, then delegates to
        transform() using target_date as the recency reference value.
        Use this when df spans multiple values and you want only the
        observation window ending at target_date.

        Parameters
        ----------
        df : pd.DataFrame
            User-item interaction history to score.
        target_date : str, datetime, or int
            Upper bound for the time axis (inclusive). Rows after this value
            are excluded. Also used as the reference value for computing
            recency. Accepts the same types as time_col.
        kind : {"emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc"}, default "emp"
            Which probability to use. "emp", "er", and "ef" use fit(),
            fit_date(), or fit_period() results; others use optimize() results.
        user_col : str, optional
            Column name for user IDs in df. Defaults to the value set in __init__.
        item_col : str, optional
            Column name for item IDs in df. Defaults to the value set in __init__.
        time_col : str, optional
            Column name for the time axis in df. Defaults to the value set in
            __init__.

        Returns
        -------
        pd.DataFrame
            One row per user-item pair observed in df (up to target_date).
            User and item columns retain the names used (from __init__ or the
            overrides). Additional columns: recency, frequency, probability,
            order. Sorted by user ascending and probability descending; order
            starts at 1.

        Raises
        ------
        ValueError
            If target_date cannot be normalized. Also propagates ValueError
            and RuntimeError from transform().
        """
        time_col_name = time_col or self.time_col
        target_int = self._normalize_ref(target_date)
        normalized_col = self._normalize_sequence_col(df[time_col_name])
        df_filtered = df[normalized_col <= target_int]
        return self.transform(
            df_filtered,
            ref=target_int,
            kind=kind,
            user_col=user_col,
            item_col=item_col,
            time_col=time_col,
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

        Raises
        ------
        TypeError
            If df_eval is not a pandas DataFrame.
        ValueError
            If user_col or item_col are missing from df_eval, or if the user
            or item column in df_rec cannot be cast to str.
        """
        user_col = user_col or self.user_col
        item_col = item_col or self.item_col

        if not isinstance(df_eval, pd.DataFrame):
            raise TypeError("df_eval must be a pandas DataFrame.")
        missing = [c for c in [user_col, item_col] if c not in df_eval.columns]
        if missing:
            raise ValueError(f"Missing required columns in df_eval: {missing}")

        UIevent = set(zip(df_eval[user_col].astype(str), df_eval[item_col].astype(str)))

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
            n_hit = len(UIrec & UIevent)
            n_recommended = len(UIrec)
            precision = n_hit / n_recommended if n_recommended > 0 else 0.0
            Rows.append((recommend_num, n_recommended, n_hit, precision))
        df_result = pd.DataFrame(Rows, columns=["order", "n_recommended", "n_hit", "precision"])

        total_hit = df_result.n_hit.max()
        df_result["recall"] = df_result.n_hit / len(UIevent)
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

    def _build_ui_rf_df(self, df, ref_int):
        """Compute recency and frequency for each (user, item) pair.

        Recency is ``(ref_int - value) // unit + 1`` (1-indexed; most recent
        interaction at ref_int has recency 1). Frequency is the interaction
        count. Uses pandas groupby.
        """
        tmp = df[[self._USER_COL, self._ITEM_COL]].copy()
        tmp["recency"] = (ref_int - df[self._SEQUENCE_COL]) // self.unit + 1
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
        if kind == "mono":
            return self.mono_probability_dict_
        if kind == "mrc":
            return self.mrc_probability_dict_
        if kind == "mfc":
            return self.mfc_probability_dict_
        if kind == "mcc":
            return self.mcc_probability_dict_
        return self.emp_probability_dict_

    def _marginal_dict(self, kind):
        if kind == "mr":
            return self.mr_probability_dict_
        if kind == "mf":
            return self.mf_probability_dict_
        if kind == "er":
            return self.er_probability_dict_
        if kind == "ef":
            return self.ef_probability_dict_
        raise ValueError(f"_marginal_dict called with non-marginal kind: {kind!r}")

    def optimize(self, kind="mono", eps=0.0):
        """Estimate optimized product-choice probabilities under RF constraints.

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
                 Result stored as dict[int, float] keyed by recency.
            "mf" fits a 1-D frequency-only model with monotonicity and concavity.
                 Result stored as dict[int, float] keyed by frequency.
            "mrc" additionally applies convexity in recency (2D joint model).
            "mfc" additionally applies concavity in frequency (2D joint model).
            "mcc" applies both recency convexity and frequency concavity (2D joint model).
        eps : float, default 0.0
            Minimum gap enforced between adjacent values in monotonicity
            constraints.  When 0.0 (default), weak monotonicity is used
            (non-strict inequalities allow ties between adjacent levels).
            When positive, strict monotonicity is enforced, preventing ties
            between adjacent recency or frequency levels.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If kind is not one of the accepted values.
        RuntimeError
            If fit(), fit_date(), or fit_period() has not been called first.
        """
        kind = self._normalize_kind(kind)
        if kind not in ("mono", "mr", "mf", "mrc", "mfc", "mcc"):
            raise ValueError(
                f"kind must be 'mono', 'mr', 'mf', 'mrc', 'mfc', or 'mcc', got {kind!r}."
            )

        try:
            from .optimizer import RecencyFrequencyOptimizer
        except ImportError:
            from rfscorer.optimizer import RecencyFrequencyOptimizer

        if self.emp_probability_dict_ is None:
            raise RuntimeError(
                "fit(), fit_date(), or fit_period() must be called before optimize()."
            )

        optimizer = RecencyFrequencyOptimizer()
        optimizer.set_data(self.R, self.F, self.RF2N, self.RF2Prob)
        optimizer.set_marginal_data(self.R2N, self.R2Prob, self.F2N, self.F2Prob)

        if kind == "mr":
            optimizer.build_marginal_model(axis="r", eps=eps)
            optimizer.solve()
            optimizer.show_solve_info()
            optimizer.postprocess()
            self.mr_probability_dict_ = optimizer.R2X
            self.mr_probability_ = (
                pd.DataFrame(list(optimizer.R2X.items()), columns=["recency", "probability"])
                .sort_values("recency")
                .reset_index(drop=True)
            )

        elif kind == "mf":
            optimizer.build_marginal_model(axis="f", eps=eps)
            optimizer.solve()
            optimizer.show_solve_info()
            optimizer.postprocess()
            self.mf_probability_dict_ = optimizer.F2X
            self.mf_probability_ = (
                pd.DataFrame(list(optimizer.F2X.items()), columns=["frequency", "probability"])
                .sort_values("frequency")
                .reset_index(drop=True)
            )

        else:
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
    df.columns = ["user", "item", "datetime"]
    df_train = df[df.user.map(lambda x: hash(x) % 10 < 8)]
    df_test = df[df.user.map(lambda x: hash(x) % 10 >= 8)]

    scorer = RecencyFrequencyScorer()

    target_date = "2015-07-07"

    # 観測期間・評価期間に分割してから fit
    df_train_obs = df_train[df_train.datetime <= target_date]
    df_train_eval = df_train[df_train.datetime > target_date]
    scorer.fit(df_train_obs, df_train_eval)

    scorer.plot_probability_surface("empirical").savefig("surface_emp_probability.png")
    scorer.plot_marginal_probability("recency").savefig("marginal_recency_probability.png")
    scorer.plot_marginal_probability("frequency").savefig("marginal_frequency_probability.png")

    scorer.optimize(kind="mr")
    scorer.plot_marginal_probability("recency", kind="mr").savefig(
        "marginal_mono_recency_probability.png"
    )
    scorer.plot_marginal_probability("recency", kind="all").savefig(
        "marginal_all_recency_probability.png"
    )

    scorer.optimize(kind="mf")
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

    df_test_obs = df_test[df_test.datetime <= target_date]
    df_test_eval = df_test[df_test.datetime > target_date]

    for kind in ("emp", "er", "ef", "mr", "mf", "mono", "mrc", "mfc", "mcc"):
        print(f"--- {kind} ---")
        df_rec = scorer.transform(df_test_obs, kind=kind)
        df_rec.to_csv(f"df_recommend_{kind}.csv", index=False)
        print(scorer.evaluate(df_rec, df_test_eval, order=10))
