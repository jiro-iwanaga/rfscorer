import pandas as pd
from pandas.api.types import is_string_dtype

from ._plotting import PlottingMixin
from ._time_utils import normalize_ref, normalize_sequence_col


class RecencyFrequencyScorer(PlottingMixin):
    """Recency-Frequency based recommendation scorer.

    Estimates product-choice probabilities from user-item interaction histories
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

    # ---------------------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------------------

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
        self.gt_start_ = None
        self.gt_end_ = None
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
        self.emp_probability_dict_ = None  # 経験的商品選択確率（辞書：キーは最新度と頻度のペア）
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
        self.record_num_gt = None  # 正解期間レコード数
        self.record_num_target_org = None  # 分析対象フィルタリング前レコード数
        self.record_num_target = None  # 分析対象レコード数
        self.total_cv_org = None  # フィルタリング前 cv 数
        self.total_cv = None  # cv 数

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
        """Estimate empirical product-choice probabilities from pre-split interaction data.

        Accepts observation and ground truth DataFrames that have already been
        filtered to the respective periods by the caller. For convenience,
        use ``rfscorer.split_by_date()`` to obtain a (df_obs, df_gt) pair
        from a single log and target_date.

        Parameters
        ----------
        df_obs : pd.DataFrame
            Observation period interaction log. Must already be filtered to
            the observation period by the caller.
        df_gt : pd.DataFrame
            Ground truth period event log (revisits, purchases, conversions,
            etc.). Must already be filtered to the ground truth period by the caller.
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
            If df_obs or df_gt is not a pandas DataFrame.
        ValueError
            If required columns (user, item, time_col) are missing from
            df_obs or df_gt, if ref cannot be normalized, or if no events
            are observed in the ground truth period (cannot determine
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
        if not isinstance(df_gt, pd.DataFrame):
            raise TypeError("df_gt must be a pandas DataFrame.")

        required_columns = [self.user_col, self.item_col, self.time_col]
        missing_obs = [c for c in required_columns if c not in df_obs.columns]
        if missing_obs:
            raise ValueError(f"Missing required columns in df_obs: {missing_obs}")
        missing_gt = [c for c in required_columns if c not in df_gt.columns]
        if missing_gt:
            raise ValueError(f"Missing required columns in df_gt: {missing_gt}")

        obs_log = self._to_internal(df_obs)
        gt_log = self._to_internal(df_gt)

        self.record_num = len(obs_log) + len(gt_log)

        if ref is None:
            ref_int = int(obs_log[self._SEQUENCE_COL].max())
        else:
            ref_int = normalize_ref(ref)

        self.observation_start_ = (
            int(obs_log[self._SEQUENCE_COL].min()) if len(obs_log) > 0 else None
        )
        self.observation_end_ = ref_int
        self.gt_start_ = int(gt_log[self._SEQUENCE_COL].min()) if len(gt_log) > 0 else None
        self.gt_end_ = int(gt_log[self._SEQUENCE_COL].max()) if len(gt_log) > 0 else None

        self._fit_impl(obs_log, gt_log, ref_int, recency_limit, frequency_limit)
        return self

    def _to_internal(self, df):
        """Convert a user-facing DataFrame to internal column names and types."""
        result = df[[self.user_col, self.item_col, self.time_col]].copy()
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

        UIcv = {(row.user, row.item) for row in gt_log.itertuples()}

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

    # ---------------------------------------------------------------------------
    # Optimization (単調性制約付き再推定)
    # ---------------------------------------------------------------------------

    def optimize(self, kind="mono", eps=0.0):
        """Estimate optimized product-choice probabilities under RF constraints.

        Solves a convex quadratic programming problem with monotonicity
        constraints (and optionally convexity/concavity constraints).
        Uses weighted least squares as objective.

        Requires fit() to be called first. Depends on cvxpy.

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
            If fit() has not been called first.
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
            raise RuntimeError("fit() must be called before optimize().")

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
            Frequency (number of interactions in the observation period).
        kind : {"emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc"}, default "emp"
            Which probability to use. "emp", "er", and "ef" use fit(),
            fit() results; others use optimize() results.
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
            If fit() has not been called for
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
            raise RuntimeError("fit() must be called before predict().")
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
        optimize() results.  Does not filter df; pre-filter the DataFrame
        manually (e.g., ``df[df[time_col] <= target_date]``) before calling
        transform() to score a specific observation window.
        Recency values above recency_limit and frequency values above
        frequency_limit are clamped to their respective limits before lookup.

        Parameters
        ----------
        df : pd.DataFrame
            User-item interaction history to score. Pre-filter to the desired
            observation window before calling.
        ref : str, datetime, or int, optional
            Reference value for computing recency. When None, defaults to the
            maximum value in df time_col.
        kind : {"emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc"}, default "emp"
            Which probability to use. "emp", "er", and "ef" use fit(),
            fit() results; others use optimize() results.
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
            If fit() has not been called, or if
            optimize(kind=...) has not been called for the requested kind.
        """
        kind = self._normalize_kind(kind)
        if kind not in ("emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc"):
            raise ValueError(
                f"kind must be 'emp', 'er', 'ef', 'mono', 'mr', 'mf', 'mrc', 'mfc', or 'mcc',"
                f" got {kind!r}."
            )
        if self.emp_probability_dict_ is None:
            raise RuntimeError("fit() must be called before transform().")
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

    # ---------------------------------------------------------------------------
    # Export (CSV 出力)
    # ---------------------------------------------------------------------------

    def export_probability_csv(self, kind="emp", path=None):
        """Export product-choice probabilities to a CSV file.

        Parameters
        ----------
        kind : {"emp", "er", "ef", "mono", "mr", "mf", "mrc", "mfc", "mcc", "all"}, default "emp"
            Which probability to export. "emp", "er", and "ef" use fit(),
            fit() results; others use optimize() results;
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
            If fit() has not been called for
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
            raise RuntimeError("fit() must be called before export_probability_csv().")
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

    # ---------------------------------------------------------------------------
    # Inspection (デバッグ / 集計情報)
    # ---------------------------------------------------------------------------

    def show(self):
        """Print a summary of fit() results to stdout."""
        print("=== profiling ===")

        if self.record_num:
            print("record_num:", self.record_num)

        if self.record_num_obs:
            print("record_num_obs:", self.record_num_obs)
        if self.record_num_gt:
            print("record_num_gt:", self.record_num_gt)

        if self.observation_start_ and self.observation_end_:
            print("observation: {} -> {}".format(self.observation_start_, self.observation_end_))
        if self.gt_start_ and self.gt_end_:
            print("ground_truth: {} -> {}".format(self.gt_start_, self.gt_end_))

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

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

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


if __name__ == "__main__":
    # データの読み込み（オーム社『Pythonではじめる数理最適化』サポートデータより引用）
    url = "https://raw.githubusercontent.com/ohmsha/PyOptBook/main/7.recommendation/access_log.csv"
    df = pd.read_csv(url)
    df.columns = ["user", "item", "datetime"]
    df_train = df[df.user.map(lambda x: hash(x) % 10 < 8)]
    df_test = df[df.user.map(lambda x: hash(x) % 10 >= 8)]

    scorer = RecencyFrequencyScorer()

    target_date = "2015-07-07"

    # 観測期間・正解期間に分割してから fit
    df_train_obs = df_train[df_train.datetime <= target_date]
    df_train_gt = df_train[df_train.datetime > target_date]
    scorer.fit(df_train_obs, df_train_gt)

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
    df_test_gt = df_test[df_test.datetime > target_date]

    for kind in ("emp", "er", "ef", "mr", "mf", "mono", "mrc", "mfc", "mcc"):
        print(f"--- {kind} ---")
        df_rec = scorer.transform(df_test_obs, kind=kind)
        df_rec.to_csv(f"df_recommend_{kind}.csv", index=False)
        print(scorer.evaluate(df_rec, df_test_gt, order=10))
