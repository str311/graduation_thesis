from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from factors import FACTOR_SPECS
from labels import LABEL_SPECS
from io_utils import load_stock_data, save_factor_data

index = "zz500"
MODE = "label"   # "factor" or "label"

RAW_DIR = Path(f"/mnt/d/python/graduation_thesis/{index}_parquet")
FACTOR_DIR = Path(f"/mnt/d/python/graduation_thesis/factor_by_stock_{index}")
LABEL_DIR = Path(f"/mnt/d/python/graduation_thesis/label_by_stock_{index}")
FACTOR_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = 10
CUT = 21


def _get_specs_and_dir():
    if MODE == "factor":
        return FACTOR_SPECS, FACTOR_DIR
    if MODE == "label":
        return LABEL_SPECS, LABEL_DIR
    raise ValueError(f"MODE must be 'factor' or 'label', got {MODE}")


def _validate_specs(specs, spec_name: str) -> None:
    if not isinstance(specs, list) or len(specs) == 0:
        raise ValueError(f"{spec_name} must be a non-empty list")
    seen = set()
    for i, spec in enumerate(specs):
        if not isinstance(spec, dict):
            raise TypeError(f"{spec_name}[{i}] must be a dict")
        if "func" not in spec or "names" not in spec:
            raise KeyError(f"{spec_name}[{i}] must contain 'func' and 'names'")
        func = spec["func"]
        names = spec["names"]
        if not callable(func):
            raise TypeError(f"{spec_name}[{i}]['func'] is not callable")
        if not isinstance(names, list) or len(names) == 0:
            raise TypeError(f"{spec_name}[{i}]['names'] must be a non-empty list")
        dup = [n for n in names if n in seen]
        if dup:
            raise ValueError(f"duplicate names found in {spec_name}: {dup[:5]}")
        seen.update(names)


def compute_one_stock(file_path: Path, specs, out_dir: Path):
    dates, symbol, x = load_stock_data(file_path)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError(f"empty or invalid matrix for {file_path}")

    values_list = []
    names_all = []

    for spec in specs:
        func = spec["func"]
        names = spec["names"]
        values = np.asarray(func(x), dtype=np.float64)

        if values.ndim != 2:
            raise ValueError(f"{func.__name__} output ndim={values.ndim}, expected 2")
        if values.shape[0] != x.shape[0]:
            raise ValueError(f"{func.__name__} output T={values.shape[0]}, expected {x.shape[0]}")
        if values.shape[1] != len(names):
            raise ValueError(f"{func.__name__} output K={values.shape[1]}, names={len(names)}")

        values_list.append(values)
        names_all.extend(names)

    out_values = (
        np.concatenate(values_list, axis=1)
        if values_list
        else np.empty((x.shape[0], 0), dtype=np.float64)
    )

    cut = min(CUT, x.shape[0])
    dates = dates[cut:]
    out_values = out_values[cut:, :]

    save_path = out_dir / f"{symbol}.parquet"
    save_factor_data(save_path, dates, symbol, out_values, names_all)
    return symbol, out_values.shape


def main():
    specs, out_dir = _get_specs_and_dir()
    _validate_specs(specs, "FACTOR_SPECS" if MODE == "factor" else "LABEL_SPECS")

    files = sorted(RAW_DIR.glob("*.parquet"))
    # files = [RAW_DIR / "000012.parquet"]
    if not files:
        raise FileNotFoundError(f"no parquet files found in {RAW_DIR}")

    failed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_file = {
            executor.submit(compute_one_stock, fp, specs, out_dir): fp
            for fp in files
        }
        for i, future in enumerate(as_completed(future_to_file), 1):
            fp = future_to_file[future]
            try:
                symbol, shape = future.result()
                print(f"[{i}/{len(files)}] success {MODE} {symbol} shape={shape}")
            except Exception as e:
                failed.append((str(fp), f"{type(e).__name__}: {e}"))
                print(f"[{i}/{len(files)}] failed {fp} | {type(e).__name__}: {e}")

    if failed:
        pd.DataFrame(failed, columns=["file", "error"]).to_csv(
            out_dir / "failed_list.csv", index=False, encoding="utf-8-sig"
        )


if __name__ == "__main__":
    main()