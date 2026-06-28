"""Internal helpers for normalizing time values and time-column series.

Used by RecencyFrequencyScorer and the split_by_date() utility to support
datetime, string, and integer time inputs through a single ordinal-integer
internal representation.
"""

import datetime

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
)

# Origin for the vectorized ordinal computation in normalize_sequence_col.
# (series - _ORDINAL_ORIGIN).dt.days + _ORDINAL_ORIGIN_OFFSET yields the same
# proleptic Gregorian ordinal as scalar .toordinal() used in normalize_ref.
#
# The origin must stay inside the datetime64[ns] range (about 1677-2262);
# using year 1 (0001-01-01) triggers an OutOfBoundsDatetime overflow when
# pandas aligns resolutions against a nanosecond series. We therefore anchor
# on the Unix epoch and add its ordinal offset back.
_ORDINAL_ORIGIN = pd.Timestamp("1970-01-01")
_ORDINAL_ORIGIN_OFFSET = _ORDINAL_ORIGIN.toordinal()  # 719163


def normalize_ref(value) -> int:
    """Normalize a single time reference value (datetime, date string, or integer) to int."""
    if isinstance(value, (pd.Timestamp, datetime.datetime)):
        return value.toordinal()
    elif isinstance(value, str):
        try:
            return pd.to_datetime(value).toordinal()
        except Exception:
            raise ValueError(f"time value could not be normalized: {value!r}")
    elif isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)
    else:
        try:
            return int(pd.to_datetime(value).toordinal())
        except Exception:
            raise ValueError(f"time value could not be normalized: {value!r}")


def normalize_sequence_col(series: pd.Series) -> pd.Series:
    """Normalize a time column (datetime, string, or integer) to an integer Series."""
    if is_datetime64_any_dtype(series):
        return (series - _ORDINAL_ORIGIN).dt.days + _ORDINAL_ORIGIN_OFFSET
    elif is_string_dtype(series):
        return (pd.to_datetime(series) - _ORDINAL_ORIGIN).dt.days + _ORDINAL_ORIGIN_OFFSET
    elif is_integer_dtype(series) or is_float_dtype(series):
        return series.astype(int)
    else:
        raise ValueError(f"time_col must be datetime or integer type, got {series.dtype}")


def normalize_view_key(series: pd.Series) -> pd.Series:
    """Normalize a time column to an order-preserving integer key (sub-day resolution kept).

    Unlike normalize_sequence_col (which truncates datetime to whole days), this keeps
    full timestamp resolution so view recency can rank events within the same day.
    Used only in view mode. The absolute value is irrelevant; only the per-user ordering
    is used, so mixing nanosecond keys (datetime) and raw integers is fine.
    """
    if is_datetime64_any_dtype(series):
        return (series - _ORDINAL_ORIGIN) // pd.Timedelta("1ns")
    elif is_string_dtype(series):
        return (pd.to_datetime(series) - _ORDINAL_ORIGIN) // pd.Timedelta("1ns")
    elif is_integer_dtype(series) or is_float_dtype(series):
        return series.astype("int64")  # ユーザー指定の粒度をそのまま使う
    else:
        raise ValueError(f"time_col must be datetime or integer type, got {series.dtype}")
