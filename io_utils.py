from pathlib import Path

import numpy as np
import pandas as pd

from schema import X_COLS, DTYPE


def load_stock_data(path: str | Path):
    df = pd.read_parquet(path)

    dates = pd.to_datetime(df["date"]).to_numpy()
    symbol = str(df["symbol"].iloc[0]).zfill(6)

    x = df[X_COLS].to_numpy(dtype=DTYPE, copy=False)

    return dates, symbol, x


def save_factor_data(
    save_path: str | Path,
    dates: np.ndarray,
    symbol: str,
    factor_values: np.ndarray,
    factor_names: list[str],
):
    out = pd.DataFrame(factor_values, columns=factor_names)
    out.insert(0, "symbol", symbol)
    out.insert(0, "date", dates)

    out.to_parquet(save_path, index=False)