"""Public utilities for data preparation outside the main scorer class."""

import pandas as pd

from ._time_utils import normalize_ref, normalize_sequence_col


def split_by_date(
    df,
    target_date,
    observation_days=28,
    evaluation_days=7,
    time_col="datetime",
):
    """Split df into observation and evaluation logs at target_date.

    Parameters
    ----------
    df : pd.DataFrame
        Interaction log containing the time_col column.
    target_date : str, datetime, or int
        Split point. The observation window ends at and includes
        target_date; the evaluation window starts at the next time step.
        Accepts the same types as time_col (datetime or integer).
    observation_days : int or None, default 28
        Number of time units in the observation window ending at target_date
        (inclusive). For example, ``observation_days=7`` covers
        ``[target_date - 6, target_date]`` (7 units). When None, uses df from
        its earliest row. Note: this is in the same time units as time_col
        (days for datetime, integer steps for integer time_col), independent
        of any recency binning unit on the scorer.
    evaluation_days : int or None, default 7
        Number of time units in the evaluation window starting one step after
        target_date. For example, ``evaluation_days=7`` covers
        ``[target_date + 1, target_date + 7]`` (7 units). When None, uses df
        up to its latest row.
    time_col : str, default "datetime"
        Column name of the time axis in df.

    Returns
    -------
    df_obs, df_eval : tuple[pd.DataFrame, pd.DataFrame]
        Observation log and evaluation log. Both preserve the original
        columns and dtypes of df. target_date is inclusive in df_obs and
        exclusive from df_eval.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.
    ValueError
        If time_col is missing from df, or if target_date cannot be
        normalized.

    Examples
    --------
    >>> from rfscorer import RecencyFrequencyScorer, split_by_date
    >>> df_obs, df_eval = split_by_date(df, "2024-01-07")
    >>> scorer = RecencyFrequencyScorer()
    >>> scorer.fit(df_obs, df_eval)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if time_col not in df.columns:
        raise ValueError(f"Missing required column: {time_col!r}")

    target_int = normalize_ref(target_date)
    normalized = normalize_sequence_col(df[time_col])

    df_min = int(normalized.min())
    df_max = int(normalized.max())

    if observation_days is None:
        obs_start = df_min
    else:
        obs_start = max(df_min, target_int - observation_days + 1)
    obs_end = target_int
    eval_start = target_int + 1
    eval_end = df_max if evaluation_days is None else min(df_max, target_int + evaluation_days)

    obs_mask = (obs_start <= normalized) & (normalized <= obs_end)
    eval_mask = (eval_start <= normalized) & (normalized <= eval_end)

    return df[obs_mask], df[eval_mask]
