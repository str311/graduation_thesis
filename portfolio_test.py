import numpy as np
import pandas as pd
from pathlib import Path


def load_panel(
    pred_path: str,
    ret_path: str,
    pred_col: str = "y_pred",
    ret_col: str = "fwd_ret1",
    symbol_col: str = "symbol",
    date_col: str = "date"
) -> pd.DataFrame:
    """
    pred_path: 预测文件，至少含 [symbol, date, y_pred]
    ret_path : 真实收益文件，至少含 [symbol, date, fwd_ret1]
    """
    end_date = "2024-12-03"
    pred = pd.read_parquet(pred_path)
    ret = pd.read_parquet(ret_path)

    pred = pred[[symbol_col, date_col, pred_col]].copy()
    ret = ret[[symbol_col, date_col, ret_col]].copy()

    pred[date_col] = pd.to_datetime(pred[date_col])
    ret[date_col] = pd.to_datetime(ret[date_col])

    pred[symbol_col] = pred[symbol_col].astype(str).str.zfill(6)
    ret[symbol_col] = ret[symbol_col].astype(str).str.zfill(6)

    if end_date is not None:
        end_date = pd.Timestamp(end_date)
        pred = pred.loc[pred[date_col] <= end_date].copy()
        ret = ret.loc[ret[date_col] <= end_date].copy()

    df = pred.merge(ret, on=[symbol_col, date_col], how="inner")
    df = df.sort_values([symbol_col, date_col]).reset_index(drop=True)
    return df

def add_ewm_signal(
    df: pd.DataFrame,
    pred_col: str = "y_pred",
    symbol_col: str = "symbol",
    alpha: float = 0.2,
    min_periods: int = 1,
    out_col: str = "signal_ewm",
) -> pd.DataFrame:
    """
    对每只股票自己的 y_pred 做时间上的 EWM
    这里只用当下及过去，不泄露未来
    """
    df = df.sort_values([symbol_col, "date"]).copy()
    df[out_col] = (
        df.groupby(symbol_col, sort=False)[pred_col]
        .transform(lambda s: s.ewm(alpha=alpha, adjust=False, min_periods=min_periods).mean())
    )
    return df


def make_target_weights_for_day(
    day_df: pd.DataFrame,
    signal_col: str = "signal_ewm",
    top_pct: float = 0.1,
    bottom_pct: float = 0.1,
    gross_long: float = 0.5,
    gross_short: float = 0.5,
    symbol_col: str = "symbol",
    reverse_signal: bool = False,
) -> pd.Series:
    x = day_df[[symbol_col, signal_col]].dropna().copy()
    if x.empty:
        return pd.Series(dtype=np.float64)

    # 普通收益类：高分在前
    # 波动率类：低分在前
    x = x.sort_values(signal_col, ascending=reverse_signal).reset_index(drop=True)

    n = len(x)
    n_long = max(int(np.floor(n * top_pct)), 1)
    n_short = max(int(np.floor(n * bottom_pct)), 1)

    long_symbols = x.iloc[:n_long][symbol_col].tolist()
    short_symbols = x.iloc[-n_short:][symbol_col].tolist()

    w = pd.Series(0.0, index=x[symbol_col].values, dtype=np.float64)

    if n_long > 0:
        w.loc[long_symbols] = gross_long / n_long
    if n_short > 0:
        w.loc[short_symbols] = -gross_short / n_short

    return w


def normalize_long_short_weights(
    w: pd.Series,
    gross_long: float = 0.5,
    gross_short: float = 0.5,
) -> pd.Series:
    """
    把正负仓分别缩放到固定 gross
    """
    w = w.copy()

    pos_mask = w > 0
    neg_mask = w < 0

    pos_sum = w[pos_mask].sum()
    neg_sum_abs = (-w[neg_mask]).sum()

    if pos_sum > 0:
        w.loc[pos_mask] = w.loc[pos_mask] / pos_sum * gross_long
    if neg_sum_abs > 0:
        w.loc[neg_mask] = w.loc[neg_mask] / neg_sum_abs * gross_short 

    return w


def align_weights(prev_w: pd.Series, target_w: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    把 prev / target 对齐到同一 symbol index
    """
    all_idx = prev_w.index.union(target_w.index)
    prev_w = prev_w.reindex(all_idx, fill_value=0.0)
    target_w = target_w.reindex(all_idx, fill_value=0.0)
    return prev_w, target_w


def run_backtest_ewm_longshort(
    panel: pd.DataFrame,
    signal_col: str = "signal_ewm",
    ret_col: str = "fwd_ret_1",
    symbol_col: str = "symbol",
    date_col: str = "date",
    top_pct: float = 0.1,
    bottom_pct: float = 0.1,
    gross_long: float = 0.5,
    gross_short: float = 0.5,
    rebalance_lambda: float = 0.2,
    init_capital: float = 1e8,
    fee_bps: float = 0.0,
    ret_scale: float = 1.0,
    reverse_signal: bool = True,
):
    """
    EWM + 连续调仓回测
    当天用 signal_col 生成仓位，并用同一行上的 fwd_ret1 结算下一期收益
    假设：
    - panel 里 date=t 的 fwd_ret1，表示从 t 到 t+1 的真实收益
    - fee_bps 是单边 bps；这里按 turnover * fee_rate 扣成本
    """
    panel = panel.sort_values([date_col, symbol_col]).copy()
    panel[ret_col] = panel[ret_col].astype(np.float64) / ret_scale

    fee_rate = fee_bps / 10000.0
    dates = pd.Index(sorted(panel[date_col].dropna().unique()))

    prev_w = pd.Series(dtype=np.float64)
    capital = float(init_capital)

    daily_records = []
    weight_records = []

    for dt in dates:
        day_df = panel.loc[panel[date_col] == dt, [symbol_col, signal_col, ret_col]].copy()
        day_df = day_df.dropna(subset=[signal_col, ret_col])

        if day_df.empty:
            daily_records.append(
                {
                    "date": dt,
                    "capital_start": capital,
                    "gross_return": 0.0,
                    "turnover": 0.0,
                    "trading_cost": 0.0,
                    "net_return": 0.0,
                    "capital_end": capital,
                    "n_universe": 0,
                    "n_long": 0,
                    "n_short": 0,
                }
            )
            continue

        target_w = make_target_weights_for_day(
            day_df=day_df,
            signal_col=signal_col,
            top_pct=top_pct,
            bottom_pct=bottom_pct,
            gross_long=gross_long,
            gross_short=gross_short,
            symbol_col=symbol_col,
            reverse_signal = reverse_signal
        )

        prev_w_aligned, target_w_aligned = align_weights(prev_w, target_w)

        # 温和换手：不是一步调到 target，而是往 target 靠 lambda
        new_w = prev_w_aligned + rebalance_lambda * (target_w_aligned - prev_w_aligned)
        new_w = normalize_long_short_weights(
            new_w,
            gross_long=gross_long,
            gross_short=gross_short,
        )

        # turnover: 0.5 * sum(abs(delta_w))
        # 这里是组合层 turnover，不考虑盘中 drift
        turnover = 0.5 * np.abs(new_w - prev_w_aligned).sum()

        # 当天真实收益
        r = day_df.set_index(symbol_col)[ret_col].reindex(new_w.index, fill_value=0.0)
        gross_return = float((new_w * r).sum())

        trading_cost = turnover * fee_rate
        net_return = gross_return - trading_cost

        capital_start = capital
        capital = capital * (1.0 + net_return)

        pos_n = int((new_w > 0).sum())
        neg_n = int((new_w < 0).sum())

        daily_records.append(
            {
                "date": dt,
                "capital_start": capital_start,
                "gross_return": gross_return,
                "turnover": turnover,
                "trading_cost": trading_cost,
                "net_return": net_return,
                "capital_end": capital,
                "n_universe": len(day_df),
                "n_long": pos_n,
                "n_short": neg_n,
            }
        )

        w_df = new_w.rename("weight").reset_index()
        w_df.columns = [symbol_col, "weight"]
        w_df[date_col] = dt
        w_df["capital_start"] = capital_start
        w_df["position_value"] = w_df["weight"] * capital_start
        weight_records.append(w_df)

        prev_w = new_w

    daily_df = pd.DataFrame(daily_records).sort_values("date").reset_index(drop=True)
    weights_df = pd.concat(weight_records, ignore_index=True) if weight_records else pd.DataFrame()

    # 一些常见统计
    if not daily_df.empty:
        daily_df["cum_nav"] = daily_df["capital_end"] / init_capital
        daily_df["drawdown"] = daily_df["cum_nav"] / daily_df["cum_nav"].cummax() - 1.0

        mean_ret = daily_df["net_return"].mean()
        std_ret = daily_df["net_return"].std(ddof=1)
        ann_ret = (1.0 + mean_ret) ** 252 - 1.0 if pd.notna(mean_ret) else np.nan
        ann_vol = std_ret * np.sqrt(252) if pd.notna(std_ret) else np.nan
        sharpe = mean_ret / std_ret * np.sqrt(252) if (std_ret is not None and std_ret > 0) else np.nan
        max_dd = daily_df["drawdown"].min()

        summary = {
            "init_capital": init_capital,
            "final_capital": float(daily_df["capital_end"].iloc[-1]),
            "cum_return": float(daily_df["capital_end"].iloc[-1] / init_capital - 1.0),
            "ann_return": float(ann_ret) if pd.notna(ann_ret) else np.nan,
            "ann_vol": float(ann_vol) if pd.notna(ann_vol) else np.nan,
            "sharpe": float(sharpe) if pd.notna(sharpe) else np.nan,
            "max_drawdown": float(max_dd) if pd.notna(max_dd) else np.nan,
            "avg_daily_turnover": float(daily_df["turnover"].mean()),
            "avg_n_long": float(daily_df["n_long"].mean()),
            "avg_n_short": float(daily_df["n_short"].mean()),
        }
    else:
        summary = {}

    return daily_df, weights_df, summary


if __name__ == "__main__":
    labels =  ['fwd_ret_1', 'fwd_ret_5', 'fwd_ewm_ret_5', 'fwd_ewm_ret_20',
               'fwd_ret_5_voladj', 'fwd_ret_20_voladj', 'fwd_ret_1_rank',
                'fwd_ret_5_rank', 'fwd_ewm_ret_5_rank', 'fwd_ewm_ret_20_rank']
    # labels = ['fwd_ewm_ret_5']
    for label in labels:
        pred_path = f"/mnt/d/python/graduation_thesis/artifacts/{label}/test_predictions.parquet"
        ret_path = "/mnt/d/python/graduation_thesis/data/fwdret.parquet"
        out_dir = Path(f"/mnt/d/python/graduation_thesis/predict_return/{label}")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # ========= 参数 =========
        reverse_signal = label in {"fwd_vol_5", "fwd_vol_20"}
        pred_col = "y_pred"
        ret_col = "fwd_ret_1"

        ewm_alpha = 0.4              # EWM 强度，越大越看重最近一天，st = 0.2*pt + 0.8*st-1
        top_pct = 0.05
        bottom_pct = 0.05

        gross_long = 0.5             # 多头总权重
        gross_short = 0.5          # 空头总权重（绝对值）
       

        rebalance_lambda = 1       # 每天不进行温和调仓
        init_capital = 1e8
        fee_bps = 10              # 每笔千一的交易成本
        ret_scale = 1.0              # 如果 fwd_ret1 是 1%=1.0，就改成 100.0

        # ========= 跑 =========
        panel = load_panel(
            pred_path=pred_path,
            ret_path=ret_path,
            pred_col=pred_col,
            ret_col=ret_col,
        )

        panel = add_ewm_signal(
            panel,
            pred_col=pred_col,
            alpha=ewm_alpha,
            out_col="signal_ewm",
        )

        daily_df, weights_df, summary = run_backtest_ewm_longshort(
            panel=panel,
            signal_col="signal_ewm",
            ret_col=ret_col,
            top_pct=top_pct,
            bottom_pct=bottom_pct,
            gross_long=gross_long,
            gross_short=gross_short,
            rebalance_lambda=rebalance_lambda,
            init_capital=init_capital,
            fee_bps=fee_bps,
            ret_scale=ret_scale,
            reverse_signal=reverse_signal,
        )

        panel.to_parquet(out_dir / "panel_with_signal.parquet", index=False)
        daily_df.to_parquet(out_dir / "daily_perf.parquet", index=False)
        weights_df.to_parquet(out_dir / "daily_weights.parquet", index=False)

        pd.Series(summary).to_json(out_dir / "summary.json", force_ascii=False, indent=2)

        print(f"====={label} SUMMARY =====")
        for k, v in summary.items():
            print(f"{k}: {v}")
