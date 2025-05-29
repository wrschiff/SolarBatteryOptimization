import requests
import pathlib 
import pandas as pd 
import time
import os 
from dotenv import load_dotenv

API_KEY = os.getenv("MY_API_KEY")
EMAIL = os.getenv("EMAIL")
FULLNAME = os.getenv("FULLNAME")
AFFIL = os.getenv("AFFIL")
REASON = "Research on PV and Battery Optimization"
LAT, LON = 38.58, -121.49 #sac town
YEARS = range(1998,2024)
INTERVAL = 60
ATTRIBUTES = "ghi"

base = ("https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv")
out_dir = pathlib.Path("nsrdb_csv_yearly")
out_dir.mkdir(exist_ok=True)
frames = []

for yr in YEARS:
    params = {
        "api_key":     API_KEY,
        "wkt":         f"POINT({LON} {LAT})",
        "names":       yr,
        "interval":    INTERVAL,
        "attributes":  ATTRIBUTES,
        "leap_day":    "false",
        "utc":         "false",
        "email":       EMAIL,
        "full_name":   FULLNAME,
        "affiliation": AFFIL,
        "reason":      REASON,
        "mailing_list": "false"
    }

    # --- 2) Send GET ------------------------------------------------------------
    r = requests.get(base, params=params, timeout=60, stream=True)
    r.raise_for_status()      # network / auth errors → exception
    csv_path = out_dir / f"{yr}.csv"
    csv_path.write_bytes(r.content)
    df = pd.read_csv(csv_path, skiprows=2)     # 2 metadata rows
    frames.append(df)                          # stash for concat

    time.sleep(2.1)                            # API asks ≥2 s between calls

print("done")

big = pd.concat(frames, ignore_index=True)

