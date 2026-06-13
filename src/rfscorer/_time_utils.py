"""Internal helpers for normalizing time values and time-column series.

Used by RecencyFrequencyScorer and the split_by_date() utility to support
both datetime and integer time inputs through a single ordinal-integer
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
# (series - _ORDINAL_ORIGIN).dt.days + 1 yields the same proleptic Gregorian
# ordinal as scalar .toordinal() used in normalize_ref.
_ORDINAL_ORIGIN = pd.Timestamp("0001-01-01")


def normalize_ref(value) -> int:
    """Normalize a single time reference value (date or integer) to int."""
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
    """Normalize a time column (datetime or integer) to an integer Series."""
    if is_datetime64_any_dtype(series):
        return (series - _ORDINAL_ORIGIN).dt.days + 1
    elif is_string_dtype(series):
        return (pd.to_datetime(series) - _ORDINAL_ORIGIN).dt.days + 1
    elif is_integer_dtype(series) or is_float_dtype(series):
        return series.astype(int)
    else:
        raise ValueError(f"time_col must be datetime or integer type, got {series.dtype}")
