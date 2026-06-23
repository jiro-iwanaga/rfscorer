import pandas as pd
from pandas.api.types import is_string_dtype

from ._plotting import PlottingMixin
from ._time_utils import normalize_ref, normalize_sequence_col


class RecencyFrequencyScorer(PlottingMixin):
    """Recency-Frequency based recommendation scorer.

    Estimates product-choice probabilities from user-item behavior histories
    using recency and frequency as behavioral signals.
    """

    _USER_COL = "user"
    _ITEM_COL = "item"
    _SEQUENCE_COL = "datetime"
    _FREQUENCY_LIMIT_RATE = 0.95  # 頻度上限値自動計算の際に利用する割合
    _RECENCY_LIMIT_RATE = 0.95  # 最新度上限値自動計算の際に利用する割合

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

    # Canonical kinds accepted by predict() / transform() (export() also allows "all").
    _INFERENCE_KINDS = ("emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc")

    # Optimization kinds → the dict attribute proving optimize(kind=...) has run.
    _OPT_RESULT_ATTR = {
        "mono": "mono_probability_dict_",
        "mr": "mr_probability_dict_",
        "mf": "mf_probability_dict_",
        "mrc": "mrc_probability_dict_",
        "mfc": "mfc_probability_dict_",
        "mcc": "mcc_probability_dict_",
    }

    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

    def __init__(self, user_col="user", item_col="item", time_col="datetime", unit=1):
        """Initialize the scorer with column name mappings.

        Parameters
        ----------
        user_col : str, default "user"
            Column name for user IDs in the behavior history.
        item_col : str, default "item"
            Column name for item IDs in the behavior history.
        time_col : str, default "datetime"
            Column name for the time axis in the behavior history. Accepts
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
        self.recency_limit = (
            None  # 最新度の上限値(デフォルトでは _RECENCY_LIMIT_RATE を用いて自動計算)
        )
        self.frequency_limit = (
            None  # 頻度の上限値(デフォルトでは _FREQUENCY_LIMIT_RATE を用いて自動計算)
        )

        self._R = []  # 最新度のリスト
        self._F = []  # 頻度のリスト
        self._RF2N = {}  # 最新度と頻度に対して閲覧数合計を紐づける辞書
        self._RF2CV = {}  # 最新度と頻度に対して対象イベント発生数合計を紐づける辞書
        self._RF2Prob = {}  # 最新度と頻度に対して経験的商品選択確率を紐づける辞書
        self._R2N = {}  # 最新度に対して閲覧数合計を紐づける辞書
        self._R2CV = {}  # 最新度に対して対象イベント発生数合計を紐づける辞書
        self._R2Prob = {}  # 最新度に対して経験的商品選択確率を紐づける辞書
        self._F2N = {}  # 頻度に対して閲覧数合計を紐づける辞書
        self._F2CV = {}  # 頻度に対して対象イベント発生数合計を紐づける辞書
        self._F2Prob = {}  # 頻度に対して経験的商品選択確率を紐づける辞書

        # empirical
        self.emp_probability_ = None  # 経験的商品選択確率データフレーム(縦持ち)
        self.emp_probability_table_ = None  # 経験的商品選択確率データフレーム(横持ち)
        self.emp_probability_dict_ = None  # 経験的商品選択確率（辞書：キーは最新度と頻度のペア）
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
        self.recency_corr_ = None  # スピアマン ρ（r 値と P(r) の等重み相関）
        self.recency_corr_pvalue_ = None  # recency_corr_ の p 値
        self.frequency_corr_ = None  # スピアマン ρ（f 値と P(f) の等重み相関）
        self.frequency_corr_pvalue_ = None  # frequency_corr_ の p 値
        self.recency_corr_weighted_ = None  # スピアマン ρ（N_r 重み付き）
        self.frequency_corr_weighted_ = None  # スピアマン ρ（N_f 重み付き）
        self.recency_slice_corr_ = None  # dict[r, float] r 固定スライスの重み付きスピアマン ρ
        self.frequency_slice_corr_ = None  # dict[f, float] f 固定スライスの重み付きスピアマン ρ
        # 実効サンプル（延べ）: fit() では物理と一致、fit_rolling() では全ロール合算
        self.record_num = None  # レコード数（fit() 後に設定）
        self.record_num_obs = None  # 観測期間レコード数
        self.record_num_gt = None  # 正解期間レコード数
        self.record_num_target_org = None  # 分析対象フィルタリング前レコード数
        self.record_num_target = None  # 分析対象レコード数
        self.total_cv_org = None  # フィルタリング前 cv 数
        self.total_cv = None  # cv 数

        # 物理ユニーク（データセット記述用）: 和集合区間を一度だけ数えた実数
        self.n_obs_rows_ = None  # 観測ログの物理ユニーク行数
        self.n_gt_events_ = None  # 正解ログの物理ユニークイベント数
        self.n_users_ = None  # 観測ログのユニークユーザ数
        self.n_items_ = None  # 観測ログのユニーク商品数

        # 学習構成（再現性・表示用）
        self.fit_method_ = None  # "fit" | "fit_rolling"
        self.roll_days_ = None  # fit()→1, fit_rolling()→指定値
        self.observation_days_ = None  # fit()→None, fit_rolling()→観測窓幅
        self.gt_days_ = None  # fit()→None, fit_rolling()→正解窓幅

    # ---------------------------------------------------------------------------
    # Fitting (経験的確率推定)
    # ---------------------------------------------------------------------------

    def fit(
        self,
        df_obs,
        df_gt,
        ref=None,
        recency_limit=None,
        frequency_limit=None,
    ):
        """Estimate empirical product-choice probabilities from pre-split behavior data.

        Accepts observation and ground truth DataFrames that have already been
        filtered to the respective periods by the caller. For convenience,
        use ``rfscorer.split_by_date()`` to obtain a (df_obs, df_gt) pair
        from a single log and target_date.

        Parameters
        ----------
        df_obs : pd.DataFrame
            Observation period behavior history. Must already be filtered to
            the observation period by the caller.
        df_gt : pd.DataFrame
            Ground truth period event log (revisits, purchases, conversions,
            etc.). Must already be filtered to the ground truth period by the caller.
        ref : str, datetime, or int, optional
            Reference value for recency computation. Recency of each
            user-item pair is ``(ref - value) // unit + 1``, where the
            minimum across behavior records is taken. When None, defaults to the
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
            If df_obs or df_gt is not a pandas DataFrame.
        ValueError
            If required columns (user, item, time_col) are missing from
            df_obs, if required columns (user, item) are missing from df_gt,
            if ref cannot be normalized, or if no events are observed in the
            ground truth period (cannot determine recency_limit or
            frequency_limit automatically).

        Notes
        -----
        After a successful call, the following attributes become available
        for use with predict(), transform(), and plot_*() methods:
        ``emp_probability_``, ``emp_probability_table_``,
        ``emp_probability_dict_``, ``er_probability_``, ``ef_probability_``,
        ``recency_limit``, ``frequency_limit``.
        """
        if not isinstance(df_obs, pd.DataFrame):
            raise TypeError("df_obs must be a pandas DataFrame.")
        if not isinstance(df_gt, pd.DataFrame):
            raise TypeError("df_gt must be a pandas DataFrame.")

        missing_obs = [
            c for c in [self.user_col, self.item_col, self.time_col] if c not in df_obs.columns
        ]
        if missing_obs:
            raise ValueError(f"Missing required columns in df_obs: {missing_obs}")
        missing_gt = [c for c in [self.user_col, self.item_col] if c not in df_gt.columns]
        if missing_gt:
            raise ValueError(f"Missing required columns in df_gt: {missing_gt}")

        obs_log = self._to_internal(df_obs)
        gt_log = df_gt[[self.user_col, self.item_col]].copy()
        gt_log.columns = [self._USER_COL, self._ITEM_COL]
        if not is_string_dtype(gt_log[self._USER_COL]):
            gt_log[self._USER_COL] = gt_log[self._USER_COL].astype(str)
        if not is_string_dtype(gt_log[self._ITEM_COL]):
            gt_log[self._ITEM_COL] = gt_log[self._ITEM_COL].astype(str)

        if ref is None:
            ref_int = int(obs_log[self._SEQUENCE_COL].max())
        else:
            ref_int = normalize_ref(ref)

        self.observation_start_ = (
            int(obs_log[self._SEQUENCE_COL].min()) if len(obs_log) > 0 else None
        )
        self.observation_end_ = ref_int

        self._fit_impl(obs_log, gt_log, ref_int, recency_limit, frequency_limit)
        return self

    def _to_internal(self, df, time_col=None):
        """Convert a user-facing DataFrame to internal column names and types."""
        tc = time_col if time_col is not None else self.time_col
        result = df[[self.user_col, self.item_col, tc]].copy()
        result.columns = [self._USER_COL, self._ITEM_COL, self._SEQUENCE_COL]
        if not is_string_dtype(result[self._USER_COL]):
            result[self._USER_COL] = result[self._USER_COL].astype(str)
        if not is_string_dtype(result[self._ITEM_COL]):
            result[self._ITEM_COL] = result[self._ITEM_COL].astype(str)
        result[self._SEQUENCE_COL] = normalize_sequence_col(result[self._SEQUENCE_COL])
        return result

    def _fit_impl(self, obs_log, gt_log, ref_int, recency_limit, frequency_limit):
        """Core fitting logic. obs_log and gt_log must use internal column names."""
        self.record_num_obs = len(obs_log)
        self.record_num_gt = len(gt_log)
        self.record_num = self.record_num_obs + self.record_num_gt
        self.fit_method_ = "fit"
        self.roll_days_ = 1
        self.observation_days_ = None
        self.gt_days_ = None
        self._set_dataset_stats(obs_log, gt_log)
        df_ui2frc = self._build_ui_rf_cv(obs_log, gt_log, ref_int)
        self._aggregate_empirical(df_ui2frc, recency_limit, frequency_limit)

    def _set_dataset_stats(self, obs_df, gt_df):
        """Record physical (de-duplicated) dataset counts for paper reporting.

        obs_df and gt_df must use internal column names and must already be
        filtered to the actual coverage span (a single window for fit(); the
        union of all rolling windows for fit_rolling()), so each physical row
        is counted once. Unlike record_num_* (which pool over rolls), these are
        the true dataset sizes.
        """
        self.n_obs_rows_ = len(obs_df)
        self.n_gt_events_ = len(gt_df)
        self.n_users_ = int(obs_df[self._USER_COL].nunique())
        self.n_items_ = int(obs_df[self._ITEM_COL].nunique())

    def _reset_optimization_results(self):
        """Invalidate previous optimize() results so a re-fit stays consistent.

        fit() / fit_rolling() recompute the empirical attributes but never
        re-run optimize(). Clearing the optimization outputs here (the common
        chokepoint reached once per fit) prevents predict() / transform() /
        export_probability_csv() with an optimization kind from returning
        values tied to a stale fit.
        """
        self.mono_probability_ = self.mono_probability_table_ = self.mono_probability_dict_ = None
        self.mrc_probability_ = self.mrc_probability_table_ = self.mrc_probability_dict_ = None
        self.mfc_probability_ = self.mfc_probability_table_ = self.mfc_probability_dict_ = None
        self.mcc_probability_ = self.mcc_probability_table_ = self.mcc_probability_dict_ = None
        self.mr_probability_ = self.mr_probability_dict_ = None
        self.mf_probability_ = self.mf_probability_dict_ = None

    def _build_ui_rf_cv(self, obs_log, gt_log, ref_int):
        """Build per (user, item) recency/frequency with a cv flag for one window.

        obs_log and gt_log must use internal column names. Returns a DataFrame
        with columns: user, item, recency, frequency, cv. Shared by fit()
        (single window) and fit_rolling() (one call per rolling window).
        """
        UIcv = {(row.user, row.item) for row in gt_log.itertuples()}
        df_ui2frc = self._build_ui_rf_df(obs_log, ref_int)
        df_ui2frc["cv"] = (
            pd.MultiIndex.from_frame(df_ui2frc[[self._USER_COL, self._ITEM_COL]])
            .isin(UIcv)
            .astype(int)
        )
        return df_ui2frc

    def _aggregate_empirical(self, df_ui2frc, recency_limit, frequency_limit):
        """Aggregate empirical probabilities from a (recency, frequency, cv) frame.

        df_ui2frc must contain at least the columns ``recency``, ``frequency``
        and ``cv`` (user/item columns are not required). Shared by fit() and
        fit_rolling(); fit_rolling() passes a frame concatenated across rolls.
        Sets the empirical/er/ef attributes and correlation diagnostics.
        """
        self._reset_optimization_results()
        self.record_num_target_org = len(df_ui2frc)
        self.total_cv_org = int(df_ui2frc.cv.sum())

        if recency_limit is not None:
            self.recency_limit = recency_limit
        else:
            df_recency2cv = df_ui2frc.groupby("recency")["cv"].sum().reset_index()
            df_recency2cv.sort_values("recency", inplace=True)
            total_cv = df_recency2cv.cv.sum()
            if total_cv == 0:
                raise ValueError(
                    "No events observed in ground truth period. "
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
                    "No events observed in ground truth period. "
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
        self.total_cv = int(df_ui2frc.cv.sum())

        self._R = list(range(1, self.recency_limit + 1))
        self._F = list(range(1, self.frequency_limit + 1))

        # Vectorized counts of N (pair count) and CV per (recency, frequency).
        # df_ui2frc is already clamped to the (R, F) grid above; reindex onto the
        # full grid fills unobserved cells with 0. Equivalent to a per-row loop
        # but C-level, which matters for fit_rolling() where rows scale with
        # roll_days.
        full_index = pd.MultiIndex.from_product([self._R, self._F], names=["recency", "frequency"])
        grouped = (
            df_ui2frc.groupby(["recency", "frequency"])
            .agg(N=("cv", "size"), cv=("cv", "sum"))
            .reindex(full_index, fill_value=0)
        )
        self._RF2N = {rf: float(n) for rf, n in grouped["N"].items()}
        self._RF2CV = {rf: float(c) for rf, c in grouped["cv"].items()}

        RowsRF = []
        for r in self._R:
            for f in self._F:
                prob = self._RF2CV[r, f] / self._RF2N[r, f] if self._RF2N[r, f] > 0 else 0.0
                self._RF2Prob[r, f] = prob
                RowsRF.append((r, f, self._RF2N[r, f], self._RF2CV[r, f], prob))

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
        self._R2N = dict(zip(df_r["recency"], df_r["N"]))
        self._R2CV = dict(zip(df_r["recency"], df_r["cv"]))
        self._R2Prob = dict(zip(df_r["recency"], df_r["probability"]))

        df_f = self.emp_probability_.groupby("frequency")[["N", "cv"]].sum().reset_index()
        df_f["probability"] = (df_f["cv"] / df_f["N"]).where(df_f["N"] > 0, 0.0)
        self._F2N = dict(zip(df_f["frequency"], df_f["N"]))
        self._F2CV = dict(zip(df_f["frequency"], df_f["cv"]))
        self._F2Prob = dict(zip(df_f["frequency"], df_f["probability"]))

        self.er_probability_dict_ = dict(self._R2Prob)
        self.er_probability_ = (
            pd.DataFrame(list(self._R2Prob.items()), columns=["recency", "probability"])
            .sort_values("recency")
            .reset_index(drop=True)
        )

        self.ef_probability_dict_ = dict(self._F2Prob)
        self.ef_probability_ = (
            pd.DataFrame(list(self._F2Prob.items()), columns=["frequency", "probability"])
            .sort_values("frequency")
            .reset_index(drop=True)
        )

        r_vals = sorted(r for r in self._R2Prob if self._R2N[r] > 0)
        r_probs = [self._R2Prob[r] for r in r_vals]
        r_weights = [self._R2N[r] for r in r_vals]
        f_vals = sorted(f for f in self._F2Prob if self._F2N[f] > 0)
        f_probs = [self._F2Prob[f] for f in f_vals]
        f_weights = [self._F2N[f] for f in f_vals]
        from scipy.stats import spearmanr

        if len(r_vals) >= 2 and len(set(r_probs)) >= 2:
            _res = spearmanr(r_vals, r_probs)
            self.recency_corr_ = float(_res.statistic)
            self.recency_corr_pvalue_ = float(_res.pvalue)
        else:
            self.recency_corr_ = float("nan")
            self.recency_corr_pvalue_ = float("nan")
        if len(f_vals) >= 2 and len(set(f_probs)) >= 2:
            _res = spearmanr(f_vals, f_probs)
            self.frequency_corr_ = float(_res.statistic)
            self.frequency_corr_pvalue_ = float(_res.pvalue)
        else:
            self.frequency_corr_ = float("nan")
            self.frequency_corr_pvalue_ = float("nan")
        self.recency_corr_weighted_ = self._marginal_spearman(r_vals, r_probs, r_weights)
        self.frequency_corr_weighted_ = self._marginal_spearman(f_vals, f_probs, f_weights)

        rec_slice = {}
        for r in sorted(r for r in self._R if self._R2N.get(r, 0) > 0):
            fs = sorted(f for f in self._F if self._RF2N.get((r, f), 0) > 0)
            if len(fs) < 2:
                rec_slice[r] = float("nan")
            else:
                rec_slice[r] = self._marginal_spearman(
                    fs,
                    [self._RF2Prob[(r, f)] for f in fs],
                    [self._RF2N[(r, f)] for f in fs],
                )
        self.recency_slice_corr_ = rec_slice

        freq_slice = {}
        for f in sorted(f for f in self._F if self._F2N.get(f, 0) > 0):
            rs = sorted(r for r in self._R if self._RF2N.get((r, f), 0) > 0)
            if len(rs) < 2:
                freq_slice[f] = float("nan")
            else:
                freq_slice[f] = self._marginal_spearman(
                    rs,
                    [self._RF2Prob[(r, f)] for r in rs],
                    [self._RF2N[(r, f)] for r in rs],
                )
        self.frequency_slice_corr_ = freq_slice

    def fit_rolling(
        self,
        df_obs,
        df_gt,
        observation_days,
        gt_days,
        roll_days=1,
        end_date=None,
        recency_limit=None,
        frequency_limit=None,
        time_col=None,
    ):
        """Estimate empirical probabilities by rolling the split point over days.

        Aggregates over ``roll_days`` reference dates, rolling the split point
        one time step into the past per roll, and accumulates the counts. This
        increases the effective sample size (stabilizing the empirical
        probabilities) and smooths reference-date-specific bias.

        ``df_obs`` (observation log) and ``df_gt`` (ground truth event log) are
        kept separate, so the target event in ``df_gt`` may be a different event
        type (purchase, conversion, etc.) than the views in ``df_obs``. For each
        roll the observation window is taken from ``df_obs`` and the ground truth
        window from ``df_gt``. To use a single combined log (re-view case), pass
        the same DataFrame as both arguments: ``fit_rolling(df, df, ...)``.

        Unlike ``fit()``, ``df_gt`` must contain ``time_col`` because the ground
        truth window is sliced per roll.

        Parameters
        ----------
        df_obs : pd.DataFrame
            Observation event log (views). Must contain user, item and time_col.
        df_gt : pd.DataFrame
            Ground truth event log (re-views, purchases, conversions, etc.).
            Must contain user, item and time_col.
        observation_days : int
            Length of the observation window (in time_col units), per roll.
        gt_days : int
            Length of the ground truth window (in time_col units), per roll.
        roll_days : int, default 1
            Number of reference dates to aggregate. ``1`` is a single snapshot.
            ``N`` aggregates the anchor down to anchor-(N-1).
        end_date : str, datetime, int, or None, default None
            Last day of usable ground truth data. The most recent ground truth
            window ends exactly at this date. When None, defaults to the maximum
            value in ``df_gt[time_col]``. The anchor (most recent split point) is
            ``end_date - gt_days``.
        recency_limit : int, optional
            Maximum recency rank. When None, determined from the pooled (all
            rolls) cumulative event distribution.
        frequency_limit : int, optional
            Maximum frequency. When None, determined from the pooled distribution.
        time_col : str, optional
            Time column name. When None, uses the value set in ``__init__``.
            Applied to both df_obs and df_gt.

        Returns
        -------
        self

        Raises
        ------
        TypeError
            If df_obs or df_gt is not a pandas DataFrame.
        ValueError
            If required columns are missing, if observation_days / gt_days /
            roll_days are below 1, if end_date exceeds the latest ground truth
            date, or if the oldest roll cannot secure a full observation window
            (the message reports the maximum feasible roll_days).

        Notes
        -----
        On success the same attributes as fit() become available
        (``emp_probability_``, ``emp_probability_table_``,
        ``emp_probability_dict_``, ``er_probability_``, ``ef_probability_``,
        ``recency_limit``, ``frequency_limit`` and the correlation
        diagnostics), so predict(), transform(), optimize(), show() and the
        plot_*() methods behave as after fit(). Any optimize() results from a
        previous fit are cleared, so re-run optimize() after each fit.

        The record counts (``record_num_*``, ``total_cv*``) are summed across
        all rolls (pooled effective sample size; a physical row may be counted
        in several overlapping windows). The de-duplicated dataset sizes are in
        ``n_obs_rows_`` / ``n_gt_events_`` / ``n_users_`` / ``n_items_``.
        ``observation_end_`` is the anchor; ``observation_start_`` is the oldest
        roll's observation window start -- a window boundary, which (unlike
        fit(), where it is the earliest actual record) may precede the first
        record present in the data.
        """
        if not isinstance(df_obs, pd.DataFrame):
            raise TypeError("df_obs must be a pandas DataFrame.")
        if not isinstance(df_gt, pd.DataFrame):
            raise TypeError("df_gt must be a pandas DataFrame.")

        tc = time_col if time_col is not None else self.time_col

        missing_obs = [c for c in [self.user_col, self.item_col, tc] if c not in df_obs.columns]
        if missing_obs:
            raise ValueError(f"Missing required columns in df_obs: {missing_obs}")
        missing_gt = [c for c in [self.user_col, self.item_col, tc] if c not in df_gt.columns]
        if missing_gt:
            raise ValueError(f"Missing required columns in df_gt: {missing_gt}")

        if observation_days < 1:
            raise ValueError(f"observation_days must be >= 1, got {observation_days}.")
        if gt_days < 1:
            raise ValueError(f"gt_days must be >= 1, got {gt_days}.")
        if roll_days < 1:
            raise ValueError(f"roll_days must be >= 1, got {roll_days}.")

        obs_internal = self._to_internal(df_obs, time_col=tc)
        gt_internal = self._to_internal(df_gt, time_col=tc)
        obs_seq = obs_internal[self._SEQUENCE_COL]
        gt_seq = gt_internal[self._SEQUENCE_COL]
        obs_min = int(obs_seq.min())
        gt_max = int(gt_seq.max())

        end_int = gt_max if end_date is None else normalize_ref(end_date)
        anchor = end_int - gt_days

        # Fail-fast: validate before any window slicing or aggregation begins.
        if end_date is not None and end_int > gt_max:
            raise ValueError(
                f"end_date ({end_int}) exceeds the latest ground-truth date "
                f"({gt_max}); the most recent ground-truth window would extend "
                f"beyond available df_gt data."
            )

        oldest_obs_start = anchor - (roll_days - 1) - observation_days + 1
        if oldest_obs_start < obs_min:
            max_roll_days = anchor - obs_min - observation_days + 2
            if max_roll_days < 1:
                raise ValueError(
                    f"Data range is too short for observation_days={observation_days} "
                    f"and gt_days={gt_days}: even roll_days=1 cannot secure a full "
                    f"observation window (anchor={anchor}, obs_min={obs_min})."
                )
            raise ValueError(
                f"roll_days={roll_days} exceeds available observation history. "
                f"Maximum feasible roll_days is {max_roll_days} "
                f"(anchor={anchor}, obs_min={obs_min}, observation_days={observation_days})."
            )

        frames = []
        total_obs_rows = 0
        total_gt_rows = 0
        for k in range(roll_days):
            td = anchor - k
            obs_w = obs_internal[(obs_seq >= td - observation_days + 1) & (obs_seq <= td)]
            gt_w = gt_internal[(gt_seq >= td + 1) & (gt_seq <= td + gt_days)]
            total_obs_rows += len(obs_w)
            total_gt_rows += len(gt_w)
            # cv flag no longer needs user/item; keep 3 columns to bound concat memory.
            frames.append(self._build_ui_rf_cv(obs_w, gt_w, td)[["recency", "frequency", "cv"]])
        combined = pd.concat(frames, ignore_index=True)

        self.record_num_obs = total_obs_rows
        self.record_num_gt = total_gt_rows
        self.record_num = total_obs_rows + total_gt_rows
        self.observation_end_ = anchor
        # The fail-fast check above guarantees oldest_obs_start >= obs_min.
        self.observation_start_ = oldest_obs_start

        self.fit_method_ = "fit_rolling"
        self.roll_days_ = roll_days
        self.observation_days_ = observation_days
        self.gt_days_ = gt_days
        # Physical counts over the union of all rolling windows (counted once,
        # not pooled): observation span [observation_start_, anchor]; ground
        # truth span [anchor - roll_days + 2, end_int].
        obs_union = obs_internal[(obs_seq >= oldest_obs_start) & (obs_seq <= anchor)]
        gt_union = gt_internal[(gt_seq >= anchor - roll_days + 2) & (gt_seq <= end_int)]
        self._set_dataset_stats(obs_union, gt_union)

        self._aggregate_empirical(combined, recency_limit, frequency_limit)
        return self

    # ---------------------------------------------------------------------------
    # Optimization (単調性制約付き再推定)
    # ---------------------------------------------------------------------------

    def optimize(self, kind="mono", eps=0.0, verbose=False):
        """Estimate optimized product-choice probabilities under RF constraints.

        Solves a convex quadratic programming problem with monotonicity
        constraints (and optionally convexity/concavity constraints).
        Uses weighted least squares as objective.

        Requires fit() or fit_rolling() to be called first. Depends on cvxpy.

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
            between adjacent recency or frequency levels. Must be non-negative
            and not exceed the data-dependent maximum feasible gap; otherwise
            ValueError is raised.
        verbose : bool, default False
            If True, print optimization solver status information.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If kind is not one of the accepted values, or if eps is negative
            or exceeds the maximum feasible gap given the data.
        RuntimeError
            If neither fit() nor fit_rolling() has been called first.
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

        self._check_fitted("optimize")

        optimizer = RecencyFrequencyOptimizer()
        optimizer.set_data(self._R, self._F, self._RF2N, self._RF2Prob)
        optimizer.set_marginal_data(self._R2N, self._R2Prob, self._F2N, self._F2Prob)

        if kind == "mr":
            optimizer.build_marginal_model(axis="r", eps=eps)
            optimizer.solve()
            if verbose:
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
            if verbose:
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
            if verbose:
                optimizer.show_solve_info()
            optimizer.postprocess()

            rows = [(r, f, optimizer.RF2X[(r, f)]) for r in self._R for f in self._F]
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

    # ---------------------------------------------------------------------------
    # Inference (推論・スコアリング)
    # ---------------------------------------------------------------------------

    def predict(self, r, f, kind="emp"):
        """Return the product-choice probability for a given recency and frequency.

        Parameters
        ----------
        r : int
            Recency rank (1 = most recently interacted, higher = older).
        f : int
            Frequency (number of behavior records in the observation period).
        kind : {"emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc"}, default "emp"
            Which probability to use. "emp", "er", and "ef" use fit()
            results; others use optimize() results.
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
            If neither fit() nor fit_rolling() has been called for
            emp/er/ef kinds, or if optimize(kind=...) has not been called
            for the requested optimization kind.
        """

        kind = self._normalize_kind(kind)
        if not isinstance(r, int) or r < 1:
            raise TypeError("r must be a positive integer.")
        if not isinstance(f, int) or f < 1:
            raise TypeError("f must be a positive integer.")
        if kind not in self._INFERENCE_KINDS:
            raise ValueError(
                f"kind must be 'emp', 'er', 'ef', 'mono', 'mr', 'mf', 'mrc', 'mfc', or 'mcc',"
                f" got {kind!r}."
            )
        if kind in ("emp", "er", "ef"):
            self._check_fitted("predict")
        else:
            self._check_optimized(kind, "predict")

        if kind in ("mr", "er"):
            r_clipped = min(r, self.recency_limit)
            return self._marginal_dict(kind).get(r_clipped, 0.0)
        if kind in ("mf", "ef"):
            f_clipped = min(f, self.frequency_limit)
            return self._marginal_dict(kind).get(f_clipped, 0.0)

        r = min(r, self.recency_limit)
        f = min(f, self.frequency_limit)
        return self._probability_dict(kind).get((r, f), 0.0)

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
        optimize() results.  Does not filter df; pre-filter the DataFrame
        manually (e.g., ``df[df[time_col] <= target_date]``) before calling
        transform() to score a specific observation window.
        Recency values above recency_limit and frequency values above
        frequency_limit are clamped to their respective limits before lookup.

        Parameters
        ----------
        df : pd.DataFrame
            User-item behavior history to score. Pre-filter to the desired
            observation window before calling.
        ref : str, datetime, or int, optional
            Reference value for computing recency. When None, defaults to the
            maximum value in df time_col.
        kind : {"emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc"}, default "emp"
            Which probability to use. "emp", "er", and "ef" use fit()
            results; others use optimize() results.
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
            If neither fit() nor fit_rolling() has been called, or if
            optimize(kind=...) has not been called for the requested kind.
        """
        kind = self._normalize_kind(kind)
        if kind not in self._INFERENCE_KINDS:
            raise ValueError(
                f"kind must be 'emp', 'er', 'ef', 'mono', 'mr', 'mf', 'mrc', 'mfc', or 'mcc',"
                f" got {kind!r}."
            )
        self._check_fitted("transform")
        self._check_optimized(kind, "transform")

        user_col = user_col or self.user_col
        item_col = item_col or self.item_col
        time_col = time_col or self.time_col

        df_log = df[[user_col, item_col, time_col]].copy()
        df_log.columns = [self._USER_COL, self._ITEM_COL, self._SEQUENCE_COL]

        if not is_string_dtype(df_log[self._USER_COL]):
            df_log[self._USER_COL] = df_log[self._USER_COL].astype(str)
        if not is_string_dtype(df_log[self._ITEM_COL]):
            df_log[self._ITEM_COL] = df_log[self._ITEM_COL].astype(str)
        df_log[self._SEQUENCE_COL] = normalize_sequence_col(df_log[self._SEQUENCE_COL])

        if ref is None:
            ref_int = int(df_log[self._SEQUENCE_COL].max())
        else:
            ref_int = normalize_ref(ref)

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

    # ---------------------------------------------------------------------------
    # Evaluation (推薦精度評価)
    # ---------------------------------------------------------------------------

    def evaluate(self, df_rec, df_gt, order=1, user_col=None, item_col=None):
        """Evaluate recommendation quality at each order cutoff.

        Parameters
        ----------
        df_rec : pd.DataFrame
            Recommendation results from transform(). Must have an "order" column.
        df_gt : pd.DataFrame
            Ground truth period event log. Used to derive the ground truth
            set of (user, item) pairs that experienced the target event.
        order : int, default 1
            Maximum recommendation rank to evaluate. Results are computed for
            each rank from 1 to order, plus the maximum order in df_rec.
        user_col : str, optional
            Column name for user in df_rec and df_gt. Defaults to the value
            set in __init__.
        item_col : str, optional
            Column name for item in df_rec and df_gt. Defaults to the value
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
            If df_gt is not a pandas DataFrame.
        ValueError
            If user_col or item_col are missing from df_gt, or if the user
            or item column in df_rec cannot be cast to str.
        """
        user_col = user_col or self.user_col
        item_col = item_col or self.item_col

        if not isinstance(df_gt, pd.DataFrame):
            raise TypeError("df_gt must be a pandas DataFrame.")
        missing = [c for c in [user_col, item_col] if c not in df_gt.columns]
        if missing:
            raise ValueError(f"Missing required columns in df_gt: {missing}")

        UIevent = set(zip(df_gt[user_col].astype(str), df_gt[item_col].astype(str)))

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

        n_event = len(UIevent)
        total_hit = df_result.n_hit.max()
        df_result["recall"] = df_result.n_hit / n_event if n_event > 0 else 0.0
        denom = df_result.precision + df_result.recall
        df_result["f1"] = (2 * df_result.precision * df_result.recall).where(
            denom > 0, 0.0
        ) / denom.where(denom > 0, 1.0)
        df_result["recall_norm"] = df_result.n_hit / total_hit if total_hit > 0 else 0.0
        denom_norm = df_result.precision + df_result.recall_norm
        df_result["f1_norm"] = (2 * df_result.precision * df_result.recall_norm).where(
            denom_norm > 0, 0.0
        ) / denom_norm.where(denom_norm > 0, 1.0)

        return df_result

    # ---------------------------------------------------------------------------
    # Export (CSV 出力)
    # ---------------------------------------------------------------------------

    def export_probability_csv(self, kind="emp", path=None):
        """Export product-choice probabilities to a CSV file.

        Parameters
        ----------
        kind : {"emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc", "all"}, default "emp"
            Which probability to export. "emp", "er", and "ef" use fit()
            results; others use optimize() results;
            "all" merges all nine models into a single file. For 2D models the
            merge key is (recency, frequency); for 1D models mr merges on
            recency and mf merges on frequency, so their probability columns
            are constant along the other axis.
        path : str or None, default None
            Output file path for the CSV. If None, saves as
            "probability_{kind}.csv" in the current directory.
            If a directory, saves "probability_{kind}.csv" inside it.
            If a file path, saves directly to that path.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If kind is not one of the accepted values.
        RuntimeError
            If neither fit() nor fit_rolling() has been called for
            emp/er/ef kinds, or if the required optimize(kind=...) has not been
            called for the requested optimization kind.
        """
        kind = self._normalize_kind(kind)
        if kind not in (*self._INFERENCE_KINDS, "all"):
            raise ValueError(
                f"kind must be 'emp', 'er', 'ef', 'mono', 'mr', 'mf', 'mrc', 'mfc', 'mcc',"
                f" or 'all', got {kind!r}."
            )
        if kind in ("emp", "er", "ef", "all"):
            self._check_fitted("export_probability_csv")
        if kind == "all":
            for k in self._OPT_RESULT_ATTR:
                self._check_optimized(k, "export_probability_csv")
        else:
            self._check_optimized(kind, "export_probability_csv")

        from pathlib import Path

        default_filename = f"probability_{kind}.csv"
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
            _prob_order = ["er", "ef", "mr", "mf", "emp", "mono", "mrc", "mfc", "mcc"]
            df = df[["recency", "frequency", "N", "cv"] + [f"{k}_probability" for k in _prob_order]]
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

    # ---------------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------------

    def save(self, path=None) -> None:
        """Save the fitted model to a file.

        Parameters
        ----------
        path : str, Path, or None, default None
            Destination for the saved file. If None, saves as "rfscorer.pkl"
            in the current directory. If a directory, saves "rfscorer.pkl"
            inside it. If a file path, saves directly to that path.

        Notes
        -----
        This method uses :mod:`pickle`. Never load files from untrusted sources.
        """
        import pickle
        from importlib.metadata import version
        from pathlib import Path

        payload = {
            "rfscorer_version": version("rfscorer"),
            "scorer": self,
        }
        default_filename = "rfscorer.pkl"
        if path is None:
            output_path = Path(default_filename)
        else:
            p = Path(path)
            output_path = p / default_filename if p.is_dir() else p
        with output_path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path) -> "RecencyFrequencyScorer":
        """Load a fitted model from a file.

        Parameters
        ----------
        path : str or Path
            Path to the saved file.

        Returns
        -------
        RecencyFrequencyScorer

        Warnings
        --------
        This method uses :mod:`pickle`. Never load files from untrusted sources.
        """
        import pickle
        import warnings
        from importlib.metadata import version
        from pathlib import Path

        with Path(path).open("rb") as f:
            payload = pickle.load(f)  # noqa: S301

        saved_ver = payload.get("rfscorer_version", "unknown")
        current_ver = version("rfscorer")

        def _major_minor(v: str) -> tuple[int, int]:
            parts = v.split(".")
            return int(parts[0]), int(parts[1])

        if saved_ver != "unknown" and _major_minor(saved_ver) != _major_minor(current_ver):
            warnings.warn(
                f"Version mismatch: saved={saved_ver}, current={current_ver}. "
                "If you encounter unexpected behavior, re-fit the model with the current version.",
                UserWarning,
                stacklevel=2,
            )

        return payload["scorer"]

    def save_zip(self, path=None) -> None:
        """Save the fitted model as a zip archive with probabilities and plots.

        The archive contains:

        - ``rfscorer.pkl`` — the model for :meth:`load_zip`
        - ``metadata.json`` — version, parameters, and fit statistics
        - ``probabilities/`` — one CSV per computed probability kind
        - ``plots/`` — one PNG per computed probability kind

        Parameters
        ----------
        path : str, Path, or None, default None
            Destination for the zip file. If None, saves as "scorer.zip" in
            the current directory. If a directory, saves "scorer.zip" inside
            it. If a file path, saves directly to that path.
        """
        import io
        import json
        import pickle
        import zipfile
        from importlib.metadata import version
        from pathlib import Path

        import matplotlib.pyplot as plt

        default_filename = "scorer.zip"
        if path is None:
            output_path = Path(default_filename)
        else:
            p = Path(path)
            output_path = p / default_filename if p.is_dir() else p

        current_ver = version("rfscorer")
        optimized_kinds = [
            k
            for k in ("mono", "mr", "mf", "mrc", "mfc", "mcc")
            if getattr(self, f"{k}_probability_", None) is not None
        ]

        def _to_python(v):
            return int(v) if v is not None else None

        metadata = {
            "rfscorer_version": current_ver,
            "user_col": self.user_col,
            "item_col": self.item_col,
            "time_col": self.time_col,
            "unit": _to_python(self.unit),
            "recency_limit": _to_python(self.recency_limit),
            "frequency_limit": _to_python(self.frequency_limit),
            "observation_start": _to_python(self.observation_start_),
            "observation_end": _to_python(self.observation_end_),
            "fit_method": self.fit_method_,
            "roll_days": _to_python(self.roll_days_),
            "observation_days": _to_python(self.observation_days_),
            "gt_days": _to_python(self.gt_days_),
            "n_obs_rows": _to_python(self.n_obs_rows_),
            "n_gt_events": _to_python(self.n_gt_events_),
            "n_users": _to_python(self.n_users_),
            "n_items": _to_python(self.n_items_),
            "record_num": _to_python(self.record_num),
            "total_cv": _to_python(self.total_cv),
            "optimized_kinds": optimized_kinds,
        }

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))

            pkl_buf = io.BytesIO()
            pickle.dump({"rfscorer_version": current_ver, "scorer": self}, pkl_buf)
            zf.writestr("rfscorer.pkl", pkl_buf.getvalue())

            if self.emp_probability_ is not None:
                for kind, df in [
                    ("emp", self.emp_probability_),
                    ("er", self.er_probability_),
                    ("ef", self.ef_probability_),
                ]:
                    buf = io.StringIO()
                    df.to_csv(buf, index=False)
                    zf.writestr(f"probabilities/{kind}_probability.csv", buf.getvalue())

            for kind in optimized_kinds:
                df = getattr(self, f"{kind}_probability_")
                buf = io.StringIO()
                df.to_csv(buf, index=False)
                zf.writestr(f"probabilities/{kind}_probability.csv", buf.getvalue())

            _surface_kinds = ("emp", "mono", "mrc", "mfc", "mcc")
            if self.emp_probability_ is not None:
                for kind in [k for k in ["emp"] + optimized_kinds if k in _surface_kinds]:
                    fig = self.plot_probability_surface(kind=kind)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    zf.writestr(f"plots/{kind}_surface.png", buf.getvalue())
                    plt.close(fig)

                for kind in ("er", "ef"):
                    fig = self.plot_marginal_probability(kind=kind)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    zf.writestr(f"plots/{kind}_marginal.png", buf.getvalue())
                    plt.close(fig)

            for name in ("mr", "mf"):
                if name in optimized_kinds:
                    fig = self.plot_marginal_probability(kind=name)
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight")
                    zf.writestr(f"plots/{name}_marginal.png", buf.getvalue())
                    plt.close(fig)

    @classmethod
    def load_zip(cls, path) -> "RecencyFrequencyScorer":
        """Load a fitted model from a zip archive created by :meth:`save_zip`.

        Parameters
        ----------
        path : str or Path
            Path to the zip file.

        Returns
        -------
        RecencyFrequencyScorer

        Warnings
        --------
        This method uses :mod:`pickle`. Never load files from untrusted sources.
        """
        import io
        import pickle
        import warnings
        import zipfile
        from importlib.metadata import version
        from pathlib import Path

        with zipfile.ZipFile(Path(path), "r") as zf:
            with zf.open("rfscorer.pkl") as f:
                payload = pickle.load(io.BytesIO(f.read()))  # noqa: S301

        saved_ver = payload.get("rfscorer_version", "unknown")
        current_ver = version("rfscorer")

        def _major_minor(v: str) -> tuple[int, int]:
            parts = v.split(".")
            return int(parts[0]), int(parts[1])

        if saved_ver != "unknown" and _major_minor(saved_ver) != _major_minor(current_ver):
            warnings.warn(
                f"Version mismatch: saved={saved_ver}, current={current_ver}. "
                "If you encounter unexpected behavior, re-fit the model with the current version.",
                UserWarning,
                stacklevel=2,
            )

        return payload["scorer"]

    # ---------------------------------------------------------------------------
    # Inspection (デバッグ / 集計情報)
    # ---------------------------------------------------------------------------

    def show(self):
        """Print a structured diagnostic report of the fitted state.

        Available after fit() or fit_rolling(). The report has four sections:

        - **Data**: physical (de-duplicated) dataset sizes (observation rows,
          ground truth events, users, items), the observation span, user-item
          pair counts, and target-event counts (before/after applying limits).
          After fit_rolling() additional lines report the ground truth span,
          the rolling configuration, and the pooled (over rolls) row counts.
        - **Model**: recency_limit and frequency_limit.
        - **Correlation**: Spearman rho (equal-weight and N-weighted) for
          recency and frequency, their p-values, and per-slice correlations.
        - **Empirical Probability Table**: the empirical product-choice
          probabilities in wide form.

        Returns
        -------
        None
        """

        def sec(title=""):
            pad = max(0, 54 - len(title) - 4)
            return f"── {title} " + "─" * pad if title else "─" * 54

        print("=== RecencyFrequencyScorer ===")
        if self.emp_probability_ is None:
            print("  [not fitted]")
            return

        import math

        rolling = self.fit_method_ == "fit_rolling"
        pooled = "  (before → after limits; pooled over rolls)" if rolling else ""

        print()
        print(sec("Data"))
        print(
            f"  dataset          : obs {self.n_obs_rows_} rows,  gt {self.n_gt_events_} events"
            f"  (users: {self.n_users_},  items: {self.n_items_})"
        )
        print(
            f"  observation      : {self._fmt_ordinal(self.observation_start_)}"
            f" → {self._fmt_ordinal(self.observation_end_)}"
        )
        if rolling:
            # Ground-truth coverage is the union of all rolls' gt windows
            # [anchor - roll_days + 2, anchor + gt_days], matching n_gt_events_
            # and consistent with the observation line (also a union span).
            print(
                f"  ground truth     : "
                f"{self._fmt_ordinal(self.observation_end_ - self.roll_days_ + 2)}"
                f" → {self._fmt_ordinal(self.observation_end_ + self.gt_days_)}"
            )
            print(
                f"  rolling          : roll_days={self.roll_days_},"
                f"  obs_window={self.observation_days_},  gt_window={self.gt_days_}"
            )
            print(
                f"  pooled rows      : {self.record_num}"
                f"  (obs: {self.record_num_obs},  gt: {self.record_num_gt})"
            )
        print(
            f"  user×item pairs  : {self.record_num_target_org}"
            f" → {self.record_num_target}"
            f"{pooled if rolling else '  (before → after applying limits)'}"
        )
        print(
            f"  target events    : {self.total_cv_org}"
            f" → {self.total_cv}"
            f"{pooled if rolling else '  (before → after applying limits)'}"
        )

        print()
        print(sec("Model"))
        print(f"  recency_limit    : {self.recency_limit}")
        print(f"  frequency_limit  : {self.frequency_limit}")

        print()
        print(sec("Correlation"))
        print("  [expected: recency ρ < 0  (more recent → higher prob),")
        print("             frequency ρ > 0  (more frequent → higher prob)]")
        n_r = sum(1 for v in self._R2N.values() if v > 0)
        n_f = sum(1 for v in self._F2N.values() if v > 0)

        def _fmt_rho(v):
            return f"{v: .4f}" if not math.isnan(v) else " nan"

        def _fmt_p(v):
            return f"{v:.4f}" if not math.isnan(v) else "nan"

        rho_r = _fmt_rho(self.recency_corr_)
        p_r = _fmt_p(self.recency_corr_pvalue_)
        wrho_r = _fmt_rho(self.recency_corr_weighted_)
        rho_f = _fmt_rho(self.frequency_corr_)
        p_f = _fmt_p(self.frequency_corr_pvalue_)
        wrho_f = _fmt_rho(self.frequency_corr_weighted_)
        print(f"  recency  ρ       : {rho_r}  (p={p_r},  n={n_r},  weighted ρ: {wrho_r})")
        print(f"  frequency ρ      : {rho_f}  (p={p_f},  n={n_f},  weighted ρ: {wrho_f})")
        print()
        print("  Slice ρ by r  [corr(f, P(r,f)),  expected > 0]")
        for line in self._fmt_slice_lines(self.recency_slice_corr_, "r"):
            print(f"    {line}")
        print("  Slice ρ by f  [corr(r, P(r,f)),  expected < 0]")
        for line in self._fmt_slice_lines(self.frequency_slice_corr_, "f"):
            print(f"    {line}")

        print()
        print(sec("Empirical Probability Table"))
        print(self.emp_probability_table_.round(3).to_string())

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _fmt_ordinal(self, v):
        try:
            ts = pd.Timestamp.fromordinal(int(v))
            if 1900 <= ts.year <= 2200:
                return str(ts.date())
        except (ValueError, OverflowError, AttributeError):
            pass
        return str(int(v))

    def _fmt_slice_lines(self, d, prefix):
        import math

        lines = [
            f"{prefix}={k:2d}:  {v: .4f}" if not math.isnan(v) else f"{prefix}={k:2d}:  nan"
            for k, v in sorted(d.items())
        ]
        return lines

    def _marginal_spearman(self, x_vals, y_vals, weights=None):
        """Spearman ρ between x_vals and y_vals, optionally N-weighted.

        When weights is None, equal weights are used (standard Spearman ρ).
        When weights is provided, computes weighted Pearson on ranks (weighted
        Spearman). Returns float("nan") when n < 2 or all ranks are tied.
        """
        import numpy as np
        from scipy.stats import rankdata

        n = len(x_vals)
        if n < 2:
            return float("nan")
        rx = rankdata(x_vals).astype(float)
        ry = rankdata(y_vals).astype(float)
        if weights is None:
            w = np.ones(n) / n
        else:
            w = np.array(weights, dtype=float)
            total = w.sum()
            if total == 0:
                return float("nan")
            w = w / total
        mx = np.dot(w, rx)
        my = np.dot(w, ry)
        cov = np.dot(w, (rx - mx) * (ry - my))
        sx = np.sqrt(np.dot(w, (rx - mx) ** 2))
        sy = np.sqrt(np.dot(w, (ry - my) ** 2))
        if sx == 0.0 or sy == 0.0:
            return float("nan")
        return float(cov / (sx * sy))

    def _build_ui_rf_df(self, df, ref_int):
        """Compute recency and frequency for each (user, item) pair.

        Recency is ``(ref_int - value) // unit + 1`` (1-indexed; most recent
        behavior at ref_int has recency 1). Frequency is the behavior
        count. Uses pandas groupby.
        """
        tmp = df[[self._USER_COL, self._ITEM_COL]].copy()
        tmp["recency"] = (ref_int - df[self._SEQUENCE_COL]) // self.unit + 1
        return (
            tmp.groupby([self._USER_COL, self._ITEM_COL], sort=False)
            .agg(recency=("recency", "min"), frequency=("recency", "count"))
            .reset_index()
        )

    def _normalize_kind(self, kind):
        return self._KIND_ALIASES.get(kind, kind)

    def _check_fitted(self, method):
        """Raise RuntimeError if fit()/fit_rolling() has not produced empirical results."""
        if self.emp_probability_dict_ is None:
            raise RuntimeError(f"fit() or fit_rolling() must be called before {method}().")

    def _check_optimized(self, kind, method):
        """Raise RuntimeError if optimize(kind=...) has not run for an optimization kind.

        No-op for empirical kinds (emp/er/ef), which carry no optimize() state.
        """
        attr = self._OPT_RESULT_ATTR.get(kind)
        if attr is not None and getattr(self, attr) is None:
            raise RuntimeError(
                f"optimize(kind={kind!r}) must be called before {method}(kind={kind!r})."
            )

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


if __name__ == "__main__":
    try:
        from rfscorer import split_by_date
    except ImportError:
        from .utils import split_by_date

    # データの読み込み（オーム社『Pythonではじめる数理最適化』サポートデータより引用）
    url = "https://raw.githubusercontent.com/ohmsha/PyOptBook/main/7.recommendation/access_log.csv"
    df = pd.read_csv(url)
    df.columns = ["user", "item", "datetime"]
    df_train = df[df.user.map(lambda x: hash(x) % 10 < 8)]
    df_test = df[df.user.map(lambda x: hash(x) % 10 >= 8)]

    scorer = RecencyFrequencyScorer()

    target_date = "2015-07-07"

    # split_by_date を使用して観測期間・正解期間に分割
    df_train_obs, df_train_gt = split_by_date(df_train, target_date, observation_days=7, gt_days=1)
    scorer.fit(df_train_obs, df_train_gt)

    scorer.plot_probability_surface("empirical").savefig("surf_emp_prob.png")
    scorer.plot_marginal_probability(kind="er").savefig("marg_emp_recen_prob.png")
    scorer.plot_marginal_probability(kind="ef").savefig("marg_emp_freq_prob.png")

    scorer.optimize(kind="mr")
    scorer.plot_marginal_probability(kind="mr").savefig("marg_mono_recen_prob.png")
    scorer.plot_marginal_probability(kind="rboth").savefig("marg_recens_prob.png")

    scorer.optimize(kind="mf")
    scorer.plot_marginal_probability(kind="mf").savefig("marg_mono_freq_prob.png")
    scorer.plot_marginal_probability(kind="fboth").savefig("marg_freqs_prob.png")

    scorer.optimize(kind="mono")
    scorer.plot_probability_surface("mono").savefig("surf_mono_prob.png")
    scorer.optimize(kind="mrc")
    scorer.plot_probability_surface("mrc").savefig("surf_mrc_prob.png")
    scorer.optimize(kind="mfc")
    scorer.plot_probability_surface("mfc").savefig("surf_mfc_prob.png")
    scorer.optimize(kind="mcc")
    scorer.plot_probability_surface("mcc").savefig("surf_mcc_prob.png")

    scorer.export_probability_csv("all")

    scorer.save("rfscorer.pkl")
    # scorer_loaded = RecencyFrequencyScorer.load("rfscorer.pkl")
    # print("load ok:", scorer_loaded.predict(1, 1, kind="mono"))

    scorer.save_zip("scorer.zip")
    # scorer_loaded_zip = RecencyFrequencyScorer.load_zip("scorer.zip")
    # print("load_zip ok:", scorer_loaded_zip.predict(1, 1, kind="mono"))

    df_test_obs, df_test_gt = split_by_date(df_test, target_date, observation_days=7, gt_days=1)

    for kind in ("emp", "er", "ef", "mr", "mf", "mono", "mrc", "mfc", "mcc"):
        print(f"--- {kind} ---")
        df_rec = scorer.transform(df_test_obs, kind=kind)
        df_rec.to_csv(f"df_rec_{kind}.csv", index=False)

        df_eval = scorer.evaluate(df_rec, df_test_gt, order=10)
        print(df_eval)
        df_eval.to_csv(f"df_eval_{kind}.csv", index=False)
