import os
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple

import polars as pl


@dataclass
class NormalizeConfig:
    raw_data_dir: str = "/mnt/d/python/graduation_thesis/data"
    normalized_data_dir: str = "/mnt/d/python/graduation_thesis/normalized_data"
    key_cols: Tuple[str, str] = ("symbol", "date")
    winsor_lower: float = 0.01
    winsor_upper: float = 0.99
    eps: float = 1e-12


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cross_sectional_winsor_zscore(
    df: pl.DataFrame,
    feature_cols: List[str],
    date_col: str = "date",
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    eps: float = 1e-12,
) -> pl.DataFrame:
    out = df

    for c in feature_cols:
        q_low = pl.col(c).quantile(lower_q).over(date_col)
        q_high = pl.col(c).quantile(upper_q).over(date_col)

        clipped = (
            pl.when(pl.col(c).is_null())
            .then(None)
            .when(pl.col(c) < q_low)
            .then(q_low)
            .when(pl.col(c) > q_high)
            .then(q_high)
            .otherwise(pl.col(c))
        )

        mean_cs = clipped.mean().over(date_col)
        std_cs = clipped.std(ddof=0).over(date_col)

        z = (
            pl.when(clipped.is_null())
            .then(None)
            .when((std_cs.is_null()) | (std_cs <= eps))
            .then(0.0)
            .otherwise((clipped - mean_cs) / (std_cs + eps))
        )

        out = out.with_columns(z.alias(c))

    return out


def normalize_one_split(
    in_fp: str,
    out_fp: str,
    key_cols: Tuple[str, str] = ("symbol", "date"),
    winsor_lower: float = 0.01,
    winsor_upper: float = 0.99,
    eps: float = 1e-12,
) -> None:
    symbol_col, date_col = key_cols

    df = pl.read_parquet(in_fp).sort([symbol_col, date_col])
    feature_cols = [c for c in df.columns if c not in [symbol_col, date_col]]

    # inf / -inf -> null
    df = df.with_columns([
        pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
        for c in feature_cols
    ])

    # 按日 winsor + 截面标准化
    df = cross_sectional_winsor_zscore(
        df=df,
        feature_cols=feature_cols,
        date_col=date_col,
        lower_q=winsor_lower,
        upper_q=winsor_upper,
        eps=eps,
    )

    # feature null 填 0
    df = df.with_columns([
        pl.col(c).fill_null(0.0).alias(c)
        for c in feature_cols
    ])

    df.write_parquet(out_fp)
    print(f"saved: {out_fp}, shape={df.shape}")


def main():
    cfg = NormalizeConfig(
        raw_data_dir="/mnt/d/python/graduation_thesis/data",
        normalized_data_dir="/mnt/d/python/graduation_thesis/normalized_data",
        key_cols=("symbol", "date"),
        winsor_lower=0.01,
        winsor_upper=0.99,
        eps=1e-12,
    )

    ensure_dir(cfg.normalized_data_dir)

    split_names = ["train", "valid", "test"]
    for split in split_names:
        in_fp = os.path.join(cfg.raw_data_dir, f"factor_{split}.parquet")
        out_fp = os.path.join(cfg.normalized_data_dir, f"factor_{split}.parquet")

        normalize_one_split(
            in_fp=in_fp,
            out_fp=out_fp,
            key_cols=cfg.key_cols,
            winsor_lower=cfg.winsor_lower,
            winsor_upper=cfg.winsor_upper,
            eps=cfg.eps,
        )

    with open(os.path.join(cfg.normalized_data_dir, "normalize_config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

    print("all normalized factors saved.")


if __name__ == "__main__":
    main()