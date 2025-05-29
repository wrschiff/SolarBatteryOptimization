import pandas as pd 
import numpy as np
from pathlib import Path
import glob

CSV_DIR = Path("nsrdb_csv_yearly_sacramento")

frames = []
for csv in glob.glob(str(CSV_DIR / "*.csv")):
    df = pd.read_csv(csv,skiprows=2)
    ts = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']])
    df.index = ts
    frames.append(df[['GHI']])
    
big = pd.concat(frames)

ghi_kw = (big['GHI'] / 1000)
ghi_hourly = ghi_kw.groupby(ghi_kw.index.floor('H')).mean()  # if theres 2 data pts per hour, averages them

iso   = ghi_hourly.index.isocalendar()                  # pandas ≥1.2  :contentReference[oaicite:4]{index=4}
week  = iso.week
hour  = ghi_hourly.index.hour

group = ghi_hourly.groupby([week, hour])

mean_kw = group.mean().unstack(level=1).iloc[:52]       # drop week-53 if you wish
std_kw  = group.std(ddof=0).unstack(level=1).iloc[:52]

cv = std_kw / mean_kw
delta = np.minimum(1.0, cv)          # clip at 1 to avoid negatives

var_min = 1 - delta
var_max = 1 + delta

cols = [f"h{h:02d}" for h in range(24)]
mean_kw.columns = var_min.columns = var_max.columns = cols

mean_kw.to_csv("ghi_weekly_mean_kw.csv", float_format="%.4f")
var_min.to_csv("ghi_weekly_var_min.csv",   float_format="%.4f")
var_max.to_csv("ghi_weekly_var_max.csv",   float_format="%.4f")


print("done")