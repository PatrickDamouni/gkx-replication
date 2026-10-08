"""GKX datashare timing (D023): its DATE is the RETURN month; we store the SIGNAL month, like OSAP.

readme.txt (April 2019): "When DATE=19570329 in our dataset, you can use the monthly RET at
195703 as the response variable." So characteristics on that row are known at end of 195702.
"""

import pandas as pd

from gkx.datashare import to_signal_month


def test_return_month_date_becomes_the_previous_signal_month():
    df = pd.DataFrame({"permno": [1, 2], "DATE": [19570329, 19570131], "mom1m": [0.1, 0.2]})
    out = to_signal_month(df)
    assert out["yyyymm"].tolist() == [195702, 195612]  # January rolls back to December
    assert "DATE" not in out.columns
