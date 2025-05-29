import pandas as pd 
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np 
import re

sell_csv = Path(__file__).with_name("pge_sell_rates.csv")
def _clean_money(col: pd.Series) -> pd.Series:
    """'$0.08908 ' → 0.08908  (blank → NaN)."""
    return (col.astype(str)
               .str.replace(r"[$,\s]", "", regex=True)
               .replace("", np.nan)
               .astype(float))

# 1) raw read – Python engine tolerates ragged commas
raw = pd.read_csv(sell_csv, engine="python", skipinitialspace=True)

# 2) keep only rows that hold a clock + AM/PM token
clock_raw = raw.iloc[:, 0].astype(str).str.strip()
ampm_raw = raw.iloc[:, 1].astype(str).str.strip()

# Create boolean mask for valid time entries
# Look for pattern like "12:00" in first column and "AM"/"PM" in second column
time_pattern = clock_raw.str.match(r"^\d{1,2}:\d{2}$")
valid_ampm = ampm_raw.isin({"AM", "PM"})
good = time_pattern & valid_ampm

# Filter the dataframe
raw = raw.loc[good].reset_index(drop=True)

# 3) build 0-23 hour index using the filtered data
clock_clean = raw.iloc[:, 0].astype(str).str.strip()
ampm_clean = raw.iloc[:, 1].astype(str).str.strip()

hours = pd.to_datetime(clock_clean + " " + ampm_clean,
                       format="%I:%M %p").dt.hour

raw.insert(0, "hour", hours)                # first column = hour
raw.drop(columns=raw.columns[1:3], inplace=True)  # drop time + AM/PM

# 4) clean money columns (skip the 'hour' index column)
raw.iloc[:, 1:] = raw.iloc[:, 1:].apply(_clean_money)

# 5) move 'hour' to the index *before* assigning the MultiIndex
raw.set_index("hour", inplace=True)         # index = 0-23
price_df = raw.copy()                       # 24 numeric columns remain

# assign a 2-level (month, Produced/Delivered) column index
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
arrays = [np.repeat(months, 2),
          ["Produced", "Delivered"] * 12]
price_df.columns = pd.MultiIndex.from_arrays(arrays)

# 6) collapse Produced + Delivered → final sell_df ($ / kWh)
sell_df = price_df.groupby(level=0, axis=1).sum()   # rows 0-23, cols 12

def _load_weekly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="week")
    # rename h00 … h23 ➜ integers 0 … 23
    df.columns = [int(c[1:]) for c in df.columns]
    return df.fillna(0.0)

irr_upper_df = _load_weekly(Path(__file__).with_name("ghi_weekly_var_min.csv"))
irr_lower_df = _load_weekly(Path(__file__).with_name("ghi_weekly_var_max.csv"))
irr_mean_df = _load_weekly(Path(__file__).with_name("ghi_weekly_mean_kw.csv"))


load_csv = load_csv = Path(__file__).with_name("sac_hourly_profile.csv") 
load_df  = (pd.read_csv(load_csv, index_col="hour")
              .sort_index())  # rows 0-23


#  Time-of-Use definition sent by user
_PEAK_HOURS       = set(range(16, 21))            # 4-8:59 PM
_SEMI_PEAK_HOURS  = {15, 21, 22, 23}              # 3-3:59 PM & 9-11:59 PM
_OFF_PEAK_HOURS   = set(range(0, 24)) - _PEAK_HOURS - _SEMI_PEAK_HOURS

SUMMER_START = (5, 31)   # inclusive  (May 31)
SUMMER_END   = (10, 1)   # exclusive  (Oct 1)

#  rates $/kWh  (order: [summer, winter])
_RATES = {
    "peak":      (0.61418, 0.38266),
    "semi_peak": (0.45230, 0.36057),
    "off_peak":  (0.39562, 0.34671)
}

def _summer_flag(dt: datetime) -> bool:
    """True if dt is in summer season (May 31 ≤ date < Oct 1)."""
    start = datetime(dt.year, *SUMMER_START)
    end   = datetime(dt.year, *SUMMER_END)
    return start <= dt < end

def _tou_bucket(hour: int) -> str:
    if hour in _PEAK_HOURS:
        return "peak"
    if hour in _SEMI_PEAK_HOURS:
        return "semi_peak"
    return "off_peak"

def get_pge_buy_price(stage: int, base_year: int = 2025) -> float:
    """
    Parameters
    ----------
    stage : int
        Hour index since 00:00 on Jan-1  (0 – 8759 for a non-leap year).
    base_year : int
        Calendar year to anchor the simulation (affects summer flag).

    Returns
    -------
    cost : float   buy price $/kWh for that hour
    """
    dt   = datetime(base_year, 1, 1) + timedelta(hours=stage)
    rate = _tou_bucket(dt.hour)
    idx  = 0 if _summer_flag(dt) else 1           # 0 = summer, 1 = winter
    return _RATES[rate][idx]


__all__ = [
    "sell_df",           # Hour × Month table  (sum of Produced+Delivered)
    "irr_upper_df",      # Week × Hour   upper variance factors
    "irr_lower_df",      # Week × Hour   lower variance factors
    "irr_mean_df",
    "load_df"
    "get_pge_buy_price"  # function(stage) -> $/kWh
]


#print("load_df shape:", load_df.shape)
#print("load_df columns:", list(load_df.columns))
