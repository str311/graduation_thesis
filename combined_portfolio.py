from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================

@dataclass
class SleeveConfig:
    name: str
    pred_path: str
    pred_col: str = "y_pred"
    gross_long: float = 0.1
    gross_short: float = 0.1
    ewm_alpha: Optional[float] = None
    reverse_signal: bool = False


@dataclass
class BacktestConfig:
    ret_path: str
    ret_col: str = "fwd_ret_1"       # 用哪个未来收益来结算组合收益
    symbol_col: str = "symbol"
    date_col: str = "date"
    top_pct: float = 0.05
    bottom_pct: float = 0.05
    fee_bps: float = 10.0
    init_capital: float = 1e8
    end_date: Optional[str] = None


# ============================================================
# Utility
# ============================================================

def _standardize_key_cols(
    df: pd.DataFrame,
    symbol_col: str,
    date_col: str,
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[symbol_col] = df[symbol_col].astype(str).str.zfill(6)
    return df


def _check_sleeves(sleeves: list[SleeveConfig]) -> None:
    if len(sleeves) == 0:
        raise ValueError("sleeves is empty")

    gross_total = sum(s.gross_long + s.gross_short for s in sleeves)
    if not np.isclose(gross_total, 1.0, atol=1e-10):
        raise ValueError(
            f"sum(gross_long + gross_short) must be 1.0, got {gross_total:.12f}"
        )

    for s in sleeves:
        if s.gross_long < 0 or s.gross_short < 0:
            raise ValueError(f"gross must be non-negative: {s.name}")
        if s.ewm_alpha is not None and not (0 < s.ewm_alpha <= 1):
            raise ValueError(f"ewm_alpha must be in (0, 1], got {s.ewm_alpha} for {s.name}")


def align_weights(prev_w: pd.Series, target_w: pd.Series) -> tuple[pd.Series, pd.Series]:
    all_idx = prev_w.index.union(target_w.index)
    prev_w = prev_w.reindex(all_idx, fill_value=0.0)
    target_w = target_w.reindex(all_idx, fill_value=0.0)
    return prev_w, target_w


def calc_turnover(prev_w: pd.Series, new_w: pd.Series) -> float:
    prev_w, new_w = align_weights(prev_w, new_w)
    return 0.5 * np.abs(new_w - prev_w).sum()


# ============================================================
# Data loading
# ============================================================

def load_combo_panel(
    sleeves: list[SleeveConfig],
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """
    输出一个 panel:
    [symbol, date, ret_col, pred_xxx, pred_xxx, ...]
    """
    _check_sleeves(sleeves)

    ret = pd.read_parquet(cfg.ret_path)[[cfg.symbol_col, cfg.date_col, cfg.ret_col]].copy()
    ret = _standardize_key_cols(ret, cfg.symbol_col, cfg.date_col)

    if cfg.end_date is not None:
        end_dt = pd.Timestamp(cfg.end_date)
        ret = ret.loc[ret[cfg.date_col] <= end_dt].copy()

    panel = ret.copy()

    for sleeve in sleeves:
        pred = pd.read_parquet(sleeve.pred_path)[[cfg.symbol_col, cfg.date_col, sleeve.pred_col]].copy()
        pred = _standardize_key_cols(pred, cfg.symbol_col, cfg.date_col)

        if cfg.end_date is not None:
            pred = pred.loc[pred[cfg.date_col] <= end_dt].copy()

        pred = pred.rename(columns={sleeve.pred_col: f"pred_{sleeve.name}"})
        panel = panel.merge(pred, on=[cfg.symbol_col, cfg.date_col], how="inner")

    panel = panel.sort_values([cfg.date_col, cfg.symbol_col]).reset_index(drop=True)
    return panel


# ============================================================
# Signal processing
# ============================================================

def add_time_ewm_signal(
    panel: pd.DataFrame,
    sleeves: list[SleeveConfig],
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """
    对每个 sleeve 单独做按股票时间序列的 EWM。
    """
    panel = panel.sort_values([cfg.symbol_col, cfg.date_col]).copy()

    for sleeve in sleeves:
        pred_col = f"pred_{sleeve.name}"
        signal_col = f"signal_{sleeve.name}"

        if sleeve.ewm_alpha is None:
            panel[signal_col] = panel[pred_col]
        else:
            panel[signal_col] = (
                panel.groupby(cfg.symbol_col, sort=False)[pred_col]
                .transform(lambda s: s.ewm(alpha=sleeve.ewm_alpha, adjust=False, min_periods=1).mean())
            )

        if sleeve.reverse_signal:
            panel[signal_col] = -panel[signal_col]

    panel = panel.sort_values([cfg.date_col, cfg.symbol_col]).reset_index(drop=True)
    return panel


# ============================================================
# Cross-sectional portfolio construction
# ============================================================

def _pick_top_bottom_equal_weight(
    day_df: pd.DataFrame,
    signal_col: str,
    symbol_col: str,
    top_pct: float,
    bottom_pct: float,
    gross_long: float,
    gross_short: float,
) -> pd.Series:
    """
    单个 sleeve 在某一天的目标权重。
    只在该 sleeve 内部做 top/bottom 5% 选股，然后等权。
    """
    sub = day_df[[symbol_col, signal_col]].dropna().copy()
    if sub.empty:
        return pd.Series(dtype=np.float64)

    n = len(sub)
    n_long = max(1, int(np.floor(n * top_pct)))
    n_short = max(1, int(np.floor(n * bottom_pct)))

    sub = sub.sort_values(signal_col, ascending=True).reset_index(drop=True)

    short_names = sub.iloc[:n_short][symbol_col].tolist()
    long_names = sub.iloc[-n_long:][symbol_col].tolist()

    w = pd.Series(0.0, index=pd.Index(sub[symbol_col].unique(), name=symbol_col), dtype=np.float64)

    if n_long > 0 and gross_long > 0:
        w.loc[long_names] += gross_long / n_long
    if n_short > 0 and gross_short > 0:
        w.loc[short_names] -= gross_short / n_short

    w = w[w != 0.0]
    return w


def make_combo_target_weights_for_day(
    day_df: pd.DataFrame,
    sleeves: list[SleeveConfig],
    cfg: BacktestConfig,
) -> pd.Series:
    """
    每个 sleeve 独立建仓，最后直接叠加。
    不做跨-label 的 signal 操作。
    """
    total_w = pd.Series(dtype=np.float64)

    for sleeve in sleeves:
        signal_col = f"signal_{sleeve.name}"

        sleeve_w = _pick_top_bottom_equal_weight(
            day_df=day_df,
            signal_col=signal_col,
            symbol_col=cfg.symbol_col,
            top_pct=cfg.top_pct,
            bottom_pct=cfg.bottom_pct,
            gross_long=sleeve.gross_long,
            gross_short=sleeve.gross_short,
        )

        if total_w.empty:
            total_w = sleeve_w
        else:
            all_idx = total_w.index.union(sleeve_w.index)
            total_w = total_w.reindex(all_idx, fill_value=0.0) + sleeve_w.reindex(all_idx, fill_value=0.0)

    total_w = total_w[total_w != 0.0]
    return total_w


# ============================================================
# Backtest
# ============================================================

def run_combo_backtest(
    panel: pd.DataFrame,
    sleeves: list[SleeveConfig],
    cfg: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    每天直接切到 target weights。
    没有 partial rebalance 参数。
    """
    fee_rate = cfg.fee_bps / 10000.0

    panel = panel.sort_values([cfg.date_col, cfg.symbol_col]).copy()
    panel[cfg.ret_col] = panel[cfg.ret_col].astype(np.float64)

    dates = pd.Index(sorted(panel[cfg.date_col].dropna().unique()))

    prev_w = pd.Series(dtype=np.float64)
    capital = float(cfg.init_capital)

    daily_records = []
    weight_records = []

    for dt in dates:
        use_cols = [cfg.symbol_col, cfg.ret_col] + [f"signal_{s.name}" for s in sleeves]
        day_df = panel.loc[panel[cfg.date_col] == dt, use_cols].copy()
        day_df = day_df.dropna(subset=[cfg.ret_col])

        capital_start = capital

        if day_df.empty:
            daily_records.append(
                {
                    "date": dt,
                    "capital_start": capital_start,
                    "gross_return": 0.0,
                    "turnover": 0.0,
                    "trading_cost": 0.0,
                    "net_return": 0.0,
                    "capital_end": capital_start,
                    "n_universe": 0,
                    "n_pos": 0,
                    "n_neg": 0,
                }
            )
            continue

        target_w = make_combo_target_weights_for_day(day_df, sleeves, cfg)

        if target_w.empty:
            daily_records.append(
                {
                    "date": dt,
                    "capital_start": capital_start,
                    "gross_return": 0.0,
                    "turnover": 0.0,
                    "trading_cost": 0.0,
                    "net_return": 0.0,
                    "capital_end": capital_start,
                    "n_universe": len(day_df),
                    "n_pos": 0,
                    "n_neg": 0,
                }
            )
            continue

        prev_w_aligned, target_w_aligned = align_weights(prev_w, target_w)

        # 不做 partial rebalance，直接切到当天目标仓位
        new_w = target_w_aligned

        turnover = calc_turnover(prev_w_aligned, new_w)

        realized_ret = (
            day_df.set_index(cfg.symbol_col)[cfg.ret_col]
            .reindex(new_w.index, fill_value=0.0)
            .astype(np.float64)
        )

        gross_return = float((new_w * realized_ret).sum())
        trading_cost = float(turnover * fee_rate)
        net_return = gross_return - trading_cost

        capital_end = capital_start * (1.0 + net_return)
        capital = capital_end

        daily_records.append(
            {
                "date": dt,
                "capital_start": capital_start,
                "gross_return": gross_return,
                "turnover": turnover,
                "trading_cost": trading_cost,
                "net_return": net_return,
                "capital_end": capital_end,
                "n_universe": len(day_df),
                "n_pos": int((new_w > 0).sum()),
                "n_neg": int((new_w < 0).sum()),
            }
        )

        w_df = new_w.rename("weight").reset_index()
        w_df.columns = [cfg.symbol_col, "weight"]
        w_df[cfg.date_col] = dt
        w_df["capital_start"] = capital_start
        w_df["position_value"] = w_df["weight"] * capital_start
        weight_records.append(w_df)

        prev_w = new_w

    daily_df = pd.DataFrame(daily_records).sort_values("date").reset_index(drop=True)

    if len(weight_records) > 0:
        weights_df = pd.concat(weight_records, ignore_index=True)
    else:
        weights_df = pd.DataFrame(columns=[cfg.symbol_col, "weight", cfg.date_col, "capital_start", "position_value"])

    if not daily_df.empty:
        daily_df["cum_nav"] = daily_df["capital_end"] / cfg.init_capital
        daily_df["drawdown"] = daily_df["cum_nav"] / daily_df["cum_nav"].cummax() - 1.0

        mean_ret = daily_df["net_return"].mean()
        std_ret = daily_df["net_return"].std(ddof=1)

        ann_return = (1.0 + mean_ret) ** 252 - 1.0 if pd.notna(mean_ret) else np.nan
        ann_vol = std_ret * np.sqrt(252) if (pd.notna(std_ret) and std_ret > 0) else np.nan
        sharpe = mean_ret / std_ret * np.sqrt(252) if (pd.notna(std_ret) and std_ret > 0) else np.nan
        max_dd = daily_df["drawdown"].min()

        gross_total = sum(s.gross_long + s.gross_short for s in sleeves)
        gross_long_total = sum(s.gross_long for s in sleeves)
        gross_short_total = sum(s.gross_short for s in sleeves)

        summary = {
            "init_capital": float(cfg.init_capital),
            "final_capital": float(daily_df["capital_end"].iloc[-1]),
            "cum_return": float(daily_df["capital_end"].iloc[-1] / cfg.init_capital - 1.0),
            "ann_return": float(ann_return) if pd.notna(ann_return) else np.nan,
            "ann_vol": float(ann_vol) if pd.notna(ann_vol) else np.nan,
            "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
            "max_drawdown": float(max_dd) if pd.notna(max_dd) else np.nan,
            "avg_daily_turnover": float(daily_df["turnover"].mean()),
            "avg_n_pos": float(daily_df["n_pos"].mean()),
            "avg_n_neg": float(daily_df["n_neg"].mean()),
            "gross_total": float(gross_total),
            "gross_long_total": float(gross_long_total),
            "gross_short_total": float(gross_short_total),
        }
    else:
        summary = {}

    return daily_df, weights_df, summary


# ============================================================
# Example main
# ============================================================

if __name__ == "__main__":
    cfg = BacktestConfig(
        ret_path="/mnt/d/python/graduation_thesis/data/fwdret.parquet",
        ret_col="fwd_ret_1",
        symbol_col="symbol",
        date_col="date",
        top_pct=0.05,
        bottom_pct=0.05,
        fee_bps=10.0,
        init_capital=1e8,
        end_date=None,
    )

    sleeves = [
        SleeveConfig(
            name="fwd_ret_1",
            pred_path="/mnt/d/python/graduation_thesis/artifacts/fwd_ret_1/test_predictions.parquet",
            pred_col="y_pred",
            gross_long=0.5,
            gross_short=0.3,
            ewm_alpha=0.2,
            reverse_signal=False,
        ),
        SleeveConfig(
            name="fwd_adjvol5",
            pred_path="/mnt/d/python/graduation_thesis/artifacts/fwd_ret_5_voladj/test_predictions.parquet",
            pred_col="y_pred",
            gross_long=0,
            gross_short=0.2,
            ewm_alpha=0.4,
            reverse_signal=False,
        ),
 

    ]

    panel = load_combo_panel(sleeves, cfg)
    panel = add_time_ewm_signal(panel, sleeves, cfg)

    daily_df, weights_df, summary = run_combo_backtest(
        panel=panel,
        sleeves=sleeves,
        cfg=cfg,
    )

    out_dir = Path("/mnt/d/python/graduation_thesis/combo_predict_return/adj5_combo_backtest")
    out_dir.mkdir(parents=True, exist_ok=True)

    panel.to_parquet(out_dir / "combo_panel.parquet", index=False)
    daily_df.to_parquet(out_dir / "daily_perf.parquet", index=False)
    weights_df.to_parquet(out_dir / "daily_weights.parquet", index=False)
    pd.Series(summary).to_json(out_dir / "summary.json", force_ascii=False, indent=2)

    print("===== COMBO SUMMARY =====")
    for k, v in summary.items():
        print(f"{k}: {v}")