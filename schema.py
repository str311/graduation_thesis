import numpy as np


class Col:
    OPEN = 0
    HIGH = 1
    LOW = 2
    CLOSE = 3
    VOLUME = 4
    AMOUNT = 5
    AMPLITUDE = 6
    PCT_CHG = 7
    CHG = 8
    TURNOVER = 9


X_COLS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "amplitude",
    "pct_chg",
    "chg",
    "turnover",
]


DTYPE = np.float64