# test_data_sources.py
#
# Quick sanity-checks for:
#   • sell_df              (hour × month, $/kWh)
#   • irr_upper_df / irr_lower_df  (week × hour, dimension-less factors)
#   • get_pge_buy_price()  (TOU logic)

from datetime import datetime
import math

from generate_dynamics_dbs import (
    sell_df,
    irr_upper_df,
    irr_lower_df,
    irr_mean_df,
    load_df,
    get_pge_buy_price,
)


def test_shapes_and_nulls() -> None:
    """DataFrames have the expected shape and no NaNs."""
    assert sell_df.shape == (24, 12), f"sell_df shape {sell_df.shape}"
    assert irr_upper_df.shape == (52, 24), f"irr_upper_df shape {irr_upper_df.shape}"
    assert irr_lower_df.shape == (52, 24), f"irr_lower_df shape {irr_lower_df.shape}"
    assert not sell_df.isnull().values.any(), "sell_df contains NaNs"
    assert not irr_upper_df.isnull().values.any(), "irr_upper_df contains NaNs"
    assert not irr_lower_df.isnull().values.any(), "irr_lower_df contains NaNs"

def test_sample_sell_sum() -> None:
    """Produced + Delivered is summed correctly (Jan, hour 0)."""
    expected = 0.08908 + 0.00082       # $/kWh
    got      = sell_df.loc[0, "January"]
    assert math.isclose(got, expected, rel_tol=1e-6), (
        f"sell_df[0,'January'] = {got:.5f}, expected {expected:.5f}"
    )

def _hours_since_jan1(dt: datetime) -> int:
    return int((dt - datetime(dt.year, 1, 1)).total_seconds() // 3600)

def test_buy_price_summer_peak() -> None:
    """June 15 2025 @ 17:00 → summer PEAK rate = $0.61418."""
    stage  = _hours_since_jan1(datetime(2025, 6, 15, 17))
    price  = get_pge_buy_price(stage)
    assert math.isclose(price, 0.61418, rel_tol=1e-6)

def test_buy_price_winter_offpeak() -> None:
    """February 2 2025 @ 02:00 → winter OFF-peak = $0.34671."""
    stage  = _hours_since_jan1(datetime(2025, 2, 2, 2))
    price  = get_pge_buy_price(stage)
    assert math.isclose(price, 0.34671, rel_tol=1e-6)

def test_buy_price_summer_semipeak() -> None:
    """July 4 2025 @ 21:00 → summer semi-peak = $0.45230."""
    stage  = _hours_since_jan1(datetime(2025, 7, 4, 21))
    price  = get_pge_buy_price(stage)
    assert math.isclose(price, 0.45230, rel_tol=1e-6)

def main() -> None:
    """Run tests without pytest."""
    print(load_df.shape)
    print(irr_mean_df.shape)
    for fn in (
        test_shapes_and_nulls,
        test_sample_sell_sum,
        test_buy_price_summer_peak,
        test_buy_price_winter_offpeak,
        test_buy_price_summer_semipeak,
    ):
        fn()
    print("✅  All data-source tests passed!")

if __name__ == "__main__":
    main()