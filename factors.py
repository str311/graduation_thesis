import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from schema import Col
from ops import delay, delta, pct_change, ts_mean, ts_std, ts_sum, safe_div, stack_cols


def _move_view(x: np.ndarray, window: int):
    x = np.asarray(x, dtype=np.float64)
    if x.size < window:
        return None
    return sliding_window_view(x, window_shape=window)


def _ts_min(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    v = _move_view(x, window)
    if v is None:
        return out
    ok = np.all(np.isfinite(v), axis=1)
    vals = np.min(v, axis=1)
    vals[~ok] = np.nan
    out[window - 1:] = vals
    return out


def _ts_max(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    v = _move_view(x, window)
    if v is None:
        return out
    ok = np.all(np.isfinite(v), axis=1)
    vals = np.max(v, axis=1)
    vals[~ok] = np.nan
    out[window - 1:] = vals
    return out


def _ts_argmax(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    v = _move_view(x, window)
    if v is None:
        return out
    ok = np.all(np.isfinite(v), axis=1)
    vals = (window - 1 - np.argmax(v, axis=1)).astype(np.float64)
    vals[~ok] = np.nan
    out[window - 1:] = vals
    return out


def _ts_argmin(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    v = _move_view(x, window)
    if v is None:
        return out
    ok = np.all(np.isfinite(v), axis=1)
    vals = (window - 1 - np.argmin(v, axis=1)).astype(np.float64)
    vals[~ok] = np.nan
    out[window - 1:] = vals
    return out


def _ts_median(x: np.ndarray, window: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float64)
    v = _move_view(x, window)
    if v is None:
        return out
    ok = np.all(np.isfinite(v), axis=1)
    vals = np.median(v, axis=1)
    vals[~ok] = np.nan
    out[window - 1:] = vals
    return out


def _ts_skew(x: np.ndarray, window: int) -> np.ndarray:
    mu = ts_mean(x, window)
    sd = ts_std(x, window)
    z = safe_div(x - mu, sd)
    return ts_mean(z * z * z, window)


def _ts_kurt(x: np.ndarray, window: int) -> np.ndarray:
    mu = ts_mean(x, window)
    sd = ts_std(x, window)
    z = safe_div(x - mu, sd)
    return ts_mean(z * z * z * z, window)


def _ts_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    ex = ts_mean(x, window)
    ey = ts_mean(y, window)
    exy = ts_mean(x * y, window)
    cov = exy - ex * ey
    return safe_div(cov, ts_std(x, window) * ts_std(y, window))


def _zscore(x: np.ndarray, window: int) -> np.ndarray:
    return safe_div(x - ts_mean(x, window), ts_std(x, window))


def _rolling_pos(x: np.ndarray, window: int) -> np.ndarray:
    lo = _ts_min(x, window)
    hi = _ts_max(x, window)
    return safe_div(x - lo, hi - lo)


def _count_consecutive(cond: np.ndarray) -> np.ndarray:
    cond = np.asarray(cond, dtype=bool)
    out = np.zeros(cond.shape[0], dtype=np.float64)
    cnt = 0.0
    for i, v in enumerate(cond):
        if v:
            cnt += 1.0
            out[i] = cnt
        else:
            cnt = 0.0
            out[i] = 0.0
    out[~np.isfinite(cond.astype(np.float64))] = np.nan
    return out


def _true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = delay(close, 1)
    a = high - low
    b = np.abs(high - prev_close)
    c = np.abs(low - prev_close)
    out = np.maximum(a, np.maximum(b, c))
    bad = ~np.isfinite(a) | ~np.isfinite(b) | ~np.isfinite(c)
    out[bad] = np.nan
    return out


def factor_trend_core_28(x: np.ndarray) -> np.ndarray:
    close = x[:, Col.CLOSE]
    volume = x[:, Col.VOLUME]
    amount = x[:, Col.AMOUNT]
    turnover = x[:, Col.TURNOVER]

    ret_1 = pct_change(close, 1)
    ret_5 = pct_change(close, 5)
    ret_10 = pct_change(close, 10)
    ret_20 = pct_change(close, 20)
    dclose_1 = delta(close, 1)
    absret_1 = np.abs(ret_1)

    ma_5 = ts_mean(close, 5)
    ma_10 = ts_mean(close, 10)
    ma_20 = ts_mean(close, 20)
    vol_ma_5 = ts_mean(volume, 5)
    vol_ma_10 = ts_mean(volume, 10)
    vol_ma_20 = ts_mean(volume, 20)
    amt_ma_20 = ts_mean(amount, 20)
    to_ma_20 = ts_mean(turnover, 20)
    vol_5 = ts_std(ret_1, 5)
    vol_10 = ts_std(ret_1, 10)
    vol_20 = ts_std(ret_1, 20)

    up = (ret_1 > 0).astype(np.float64)
    down = (ret_1 < 0).astype(np.float64)

    cols = [
        ret_1, ret_5, ret_10, ret_20,
        ts_sum(ret_1, 5), ts_sum(ret_1, 10), ts_sum(ret_1, 20),
        vol_5, vol_10, vol_20,
        safe_div(close, ma_5) - 1.0, safe_div(close, ma_10) - 1.0, safe_div(close, ma_20) - 1.0,
        safe_div(ma_5, ma_10) - 1.0, safe_div(ma_5, ma_20) - 1.0, safe_div(ma_10, ma_20) - 1.0,
        safe_div(ma_5 - delay(ma_5, 1), delay(ma_5, 1)), safe_div(ma_10 - delay(ma_10, 1), delay(ma_10, 1)), safe_div(ma_20 - delay(ma_20, 1), delay(ma_20, 1)),
        safe_div(volume, vol_ma_5), safe_div(volume, vol_ma_10), safe_div(volume, vol_ma_20),
        safe_div(amount, amt_ma_20), safe_div(turnover, to_ma_20),
        ts_mean(up, 20), ts_mean(down, 20), safe_div(ts_sum(absret_1, 20), np.abs(ret_20)), _zscore(close, 20),
    ]
    return stack_cols(cols)


factor_trend_core_28_names = [
    "ret_1", "ret_5", "ret_10", "ret_20", "ret_sum_5", "ret_sum_10", "ret_sum_20",
    "vol_5", "vol_10", "vol_20", "bias_5", "bias_10", "bias_20", "ma_gap_5_10", "ma_gap_5_20", "ma_gap_10_20",
    "ma_slope_5", "ma_slope_10", "ma_slope_20", "vol_ratio_5", "vol_ratio_10", "vol_ratio_20", "amt_ratio_20", "turnover_ratio_20",
    "up_ratio_20", "down_ratio_20", "path_to_net_20", "zclose_20",
]


def factor_intraday_shape_24(x: np.ndarray) -> np.ndarray:
    open_ = x[:, Col.OPEN]
    high = x[:, Col.HIGH]
    low = x[:, Col.LOW]
    close = x[:, Col.CLOSE]
    volume = x[:, Col.VOLUME]
    amount = x[:, Col.AMOUNT]
    turnover = x[:, Col.TURNOVER]

    prev_close = delay(close, 1)
    body = close - open_
    body_abs = np.abs(body)
    rng = high - low
    upper = high - np.maximum(open_, close)
    lower = np.minimum(open_, close) - low
    gap = safe_div(open_ - prev_close, prev_close)
    intraday_ret = safe_div(close - open_, open_)
    tr = _true_range(high, low, close)
    typical = (high + low + close) / 3.0

    vol_ma20 = ts_mean(volume, 20)
    amt_ma20 = ts_mean(amount, 20)
    to_ma20 = ts_mean(turnover, 20)
    rng_ma20 = ts_mean(safe_div(rng, close), 20)

    cols = [
        intraday_ret, gap, safe_div(high - low, close), safe_div(body, close), safe_div(body_abs, close),
        safe_div(body, rng), safe_div(upper, rng), safe_div(lower, rng), safe_div(lower - upper, rng),
        safe_div(high - close, rng), safe_div(close - low, rng), safe_div(high - open_, rng), safe_div(open_ - low, rng),
        safe_div(close - typical, rng), safe_div(open_ - typical, rng),
        safe_div(body_abs, high + low + 1e-12),
        safe_div(tr, close), safe_div(safe_div(tr, close), ts_mean(safe_div(tr, close), 20)),
        safe_div(rng, ts_mean(rng, 20)), safe_div(volume, vol_ma20) * intraday_ret,
        safe_div(amount, amt_ma20) * intraday_ret, safe_div(turnover, to_ma20) * intraday_ret,
        safe_div(safe_div(rng, close), rng_ma20), safe_div((close - open_) * volume, amt_ma20),
    ]
    return stack_cols(cols)


factor_intraday_shape_24_names = [
    "intraday_ret", "gap_ret", "hl_spread", "body_pct_close", "body_abs_pct_close", "body_to_range", "upper_to_range", "lower_to_range",
    "shadow_imbalance", "close_to_high_pct", "close_to_low_pct", "open_to_high_pct", "open_to_low_pct", "close_vs_typical", "open_vs_typical",
    "body_vs_hl_sum", "atr_raw_pct", "atr_ratio_20", "range_vs_ma20", "ret_vol_ratio_mix", "ret_amt_ratio_mix", "ret_turnover_ratio_mix",
    "spread_ratio_20", "body_volume_impulse",
]


def factor_path_extreme_24(x: np.ndarray) -> np.ndarray:
    open_ = x[:, Col.OPEN]
    high = x[:, Col.HIGH]
    low = x[:, Col.LOW]
    close = x[:, Col.CLOSE]
    volume = x[:, Col.VOLUME]

    ret_1 = pct_change(close, 1)
    dclose_1 = delta(close, 1)
    abs_dclose_1 = np.abs(dclose_1)
    rng = high - low
    body = close - open_
    close_min20 = _ts_min(close, 20)
    close_max20 = _ts_max(close, 20)
    high_max20 = _ts_max(high, 20)
    low_min20 = _ts_min(low, 20)
    ret_max20 = _ts_max(ret_1, 20)
    ret_min20 = _ts_min(ret_1, 20)
    argmax_close20 = _ts_argmax(close, 20)
    argmin_close20 = _ts_argmin(close, 20)

    up = (ret_1 > 0).astype(np.float64)
    down = (ret_1 < 0).astype(np.float64)
    new_high = (close >= close_max20).astype(np.float64)
    new_low = (close <= close_min20).astype(np.float64)
    long_up_shadow = (safe_div(high - np.maximum(open_, close), rng) > 0.5).astype(np.float64)
    long_low_shadow = (safe_div(np.minimum(open_, close) - low, rng) > 0.5).astype(np.float64)

    up_streak = _count_consecutive(ret_1 > 0)
    down_streak = _count_consecutive(ret_1 < 0)
    sign_flip = np.full_like(close, np.nan)
    sgn = np.sign(np.nan_to_num(ret_1, nan=0.0))
    sign_flip[1:] = (sgn[1:] * sgn[:-1] < 0).astype(np.float64)

    cum_abs20 = ts_sum(np.abs(ret_1), 20)

    cols = [
        _rolling_pos(close, 20), _rolling_pos(open_, 20), _rolling_pos((open_ + close) * 0.5, 20),
        safe_div(close, close_max20) - 1.0, safe_div(close, close_min20) - 1.0,
        safe_div(close - close_min20, close_max20 - close_min20),
        safe_div(close_max20 - close, close_max20 - close_min20),
        safe_div(high_max20 - close, high_max20), safe_div(close - low_min20, low_min20),
        safe_div(np.abs(delta(close, 20)), ts_sum(abs_dclose_1, 20)),
        safe_div(ret_1, ts_std(ret_1, 20)), _ts_skew(ret_1, 20), _ts_kurt(ret_1, 20),
        ret_max20, ret_min20, argmax_close20, argmin_close20,
        ts_mean(new_high, 20), ts_mean(new_low, 20), ts_mean(sign_flip, 20),
        _ts_max(up_streak, 20), _ts_max(down_streak, 20), ts_mean(long_up_shadow, 20), ts_mean(long_low_shadow, 20),
    ]
    return stack_cols(cols)


factor_path_extreme_24_names = [
    "close_pos_20", "open_pos_20", "midbody_pos_20", "dist_high_20", "dist_low_20", "rebound_from_low_20", "drawdown_from_high_20",
    "dist_hh_20", "dist_ll_20", "efficiency_20", "ret_z_20", "skew_20", "kurt_20", "max_ret_20", "min_ret_20", "days_since_high_20",
    "days_since_low_20", "new_high_ratio_20", "new_low_ratio_20", "sign_flip_ratio_20", "max_up_streak_20", "max_down_streak_20", "long_upper_ratio_20", "long_lower_ratio_20",
]


def factor_price_volume_interact_23(x: np.ndarray) -> np.ndarray:
    open_ = x[:, Col.OPEN]
    high = x[:, Col.HIGH]
    low = x[:, Col.LOW]
    close = x[:, Col.CLOSE]
    volume = x[:, Col.VOLUME]
    amount = x[:, Col.AMOUNT]
    turnover = x[:, Col.TURNOVER]

    ret_1 = pct_change(close, 1)
    vol_chg_1 = pct_change(volume, 1)
    amt_chg_1 = pct_change(amount, 1)
    to_chg_1 = pct_change(turnover, 1)
    rng_pct = safe_div(high - low, close)
    tr = _true_range(high, low, close)
    atr20 = ts_mean(tr, 20)
    atr20_pct = safe_div(atr20, close)
    hh20 = _ts_max(high, 20)
    ll20 = _ts_min(low, 20)
    willr20 = -safe_div(hh20 - close, hh20 - ll20)
    typical = (high + low + close) / 3.0
    mf = typical * volume
    pos_mf = np.where(typical > delay(typical, 1), mf, 0.0)
    neg_mf = np.where(typical < delay(typical, 1), mf, 0.0)

    up = (ret_1 > 0).astype(np.float64)
    down = (ret_1 < 0).astype(np.float64)
    vol_ma20 = ts_mean(volume, 20)
    amt_ma20 = ts_mean(amount, 20)
    to_ma20 = ts_mean(turnover, 20)

    signed_vol = np.where(np.isfinite(ret_1), np.sign(np.nan_to_num(ret_1, nan=0.0)) * volume, np.nan)
    signed_amt = np.where(np.isfinite(ret_1), np.sign(np.nan_to_num(ret_1, nan=0.0)) * amount, np.nan)
    obv_step = np.where(np.isfinite(signed_vol), signed_vol, 0.0)
    pvt_step = np.where(np.isfinite(ret_1) & np.isfinite(volume), ret_1 * volume, 0.0)
    obv = np.cumsum(obv_step, dtype=np.float64)
    pvt = np.cumsum(pvt_step, dtype=np.float64)

    large_range = (rng_pct > ts_mean(rng_pct, 20)).astype(np.float64)
    large_vol = (safe_div(volume, vol_ma20) > 1.0).astype(np.float64)
    large_to = (safe_div(turnover, to_ma20) > 1.0).astype(np.float64)

    cols = [
        _ts_corr(ret_1, vol_chg_1, 20), _ts_corr(ret_1, amt_chg_1, 20), _ts_corr(ret_1, to_chg_1, 20),
        _ts_corr(ret_1, rng_pct, 20), _ts_corr(volume, close, 20), _ts_corr(amount, close, 20),
        safe_div(ts_sum(volume * up, 20), ts_sum(volume, 20)), safe_div(ts_sum(volume * down, 20), ts_sum(volume, 20)),
        safe_div(ts_sum(amount * up, 20), ts_sum(amount, 20)), safe_div(ts_sum(amount * down, 20), ts_sum(amount, 20)),
        safe_div(ts_sum(turnover * up, 20), ts_sum(turnover, 20)), safe_div(ts_sum(turnover * down, 20), ts_sum(turnover, 20)),
        ts_mean((up * large_vol), 20), ts_mean((down * large_vol), 20), ts_mean((up * large_to), 20), ts_mean((down * large_to), 20),
        ts_mean((large_range * large_vol), 20), ts_mean((large_range * large_to), 20),
        _zscore(obv, 20), _zscore(pvt, 20), atr20_pct, willr20,
        safe_div(ts_mean(pos_mf, 20), ts_mean(pos_mf, 20) + ts_mean(neg_mf, 20)),
    ]
    return stack_cols(cols)


factor_price_volume_interact_23_names = [
    "corr_ret_volchg_20", "corr_ret_amtchg_20", "corr_ret_tochg_20", "corr_ret_range_20", "corr_vol_close_20", "corr_amt_close_20",
    "up_vol_share_20", "down_vol_share_20", "up_amt_share_20", "down_amt_share_20", "up_to_share_20", "down_to_share_20",
    "up_largevol_ratio_20", "down_largevol_ratio_20", "up_largeto_ratio_20", "down_largeto_ratio_20", "large_range_largevol_ratio_20", "large_range_largeto_ratio_20",
    "obv_z_20", "pvt_z_20", "atr_20_pct", "willr_20", "mfi_20",
]

def factor_classic_extra_20(x: np.ndarray) -> np.ndarray:
    open_ = x[:, Col.OPEN]
    high = x[:, Col.HIGH]
    low = x[:, Col.LOW]
    close = x[:, Col.CLOSE]
    volume = x[:, Col.VOLUME]
    amount = x[:, Col.AMOUNT]
    turnover = x[:, Col.TURNOVER]

    ret_1 = pct_change(close, 1)
    dclose = delta(close, 1)
    gain = np.where(dclose > 0.0, dclose, 0.0)
    loss = np.where(dclose < 0.0, -dclose, 0.0)
    avg_gain20 = ts_mean(gain, 20)
    avg_loss20 = ts_mean(loss, 20)
    rsi20 = safe_div(avg_gain20, avg_gain20 + avg_loss20)

    close_pos20 = _rolling_pos(close, 20)
    stoch_d20 = ts_mean(close_pos20, 3)
    med_ret20 = _ts_median(ret_1, 20)
    med_rng20 = _ts_median(safe_div(high - low, close), 20)
    med_vol20 = _ts_median(volume, 20)
    med_amt20 = _ts_median(amount, 20)

    body = close - open_
    body_abs = np.abs(body)
    rng = high - low
    doji = (safe_div(body_abs, rng) < 0.1).astype(np.float64)
    marubozu = (safe_div(body_abs, rng) > 0.8).astype(np.float64)
    close_up = (close > open_).astype(np.float64)
    close_dn = (close < open_).astype(np.float64)
    amp = x[:, Col.AMPLITUDE]
    amp = amp / 100.0 if np.nanmax(np.abs(amp)) > 50 else amp

    cols = [
        rsi20, close_pos20, stoch_d20,
        med_ret20, med_rng20, med_vol20, med_amt20,
        safe_div(ret_1 - med_ret20, ts_std(ret_1, 20)),
        safe_div(volume - med_vol20, ts_std(volume, 20)),
        safe_div(amount - med_amt20, ts_std(amount, 20)),
        safe_div(turnover - ts_mean(turnover, 20), ts_std(turnover, 20)),
        ts_mean(doji, 20), ts_mean(marubozu, 20), ts_mean(close_up, 20), ts_mean(close_dn, 20),
        safe_div(ts_sum(np.abs(body), 20), ts_sum(rng, 20)),
        safe_div(ts_sum(amp, 20), ts_sum(np.abs(ret_1), 20)),
        _ts_corr(ret_1, turnover, 20), _ts_corr(ret_1, amount, 20), _ts_corr(volume, turnover, 20),
    ]
    return stack_cols(cols)


factor_classic_extra_20_names = [
    "rsi_20", "stoch_k_20", "stoch_d_20", "median_ret_20", "median_range_20", "median_vol_20", "median_amt_20",
    "ret_vs_median_z_20", "vol_vs_median_z_20", "amt_vs_median_z_20", "turnover_z_20", "doji_ratio_20", "marubozu_ratio_20",
    "green_ratio_20", "red_ratio_20", "body_sum_to_range_sum_20", "amplitude_to_absret_sum_20", "corr_ret_turnover_20", "corr_ret_amount_20", "corr_vol_turnover_20",
]


FACTOR_SPECS = [
    {"func": factor_trend_core_28, "names": factor_trend_core_28_names},
    {"func": factor_intraday_shape_24, "names": factor_intraday_shape_24_names},
    {"func": factor_path_extreme_24, "names": factor_path_extreme_24_names},
    {"func": factor_price_volume_interact_23, "names": factor_price_volume_interact_23_names},
    {"func": factor_classic_extra_20, "names": factor_classic_extra_20_names},
]
