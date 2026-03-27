import numpy as np
import bottleneck as bn


def _as_float_array(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _check_window(window: int) -> int:
    if not isinstance(window, int) or window <= 0:
        raise ValueError(f"window must be positive int, got {window}")
    return window


def _check_lag(n: int) -> int:
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"lag n must be non-negative int, got {n}")
    return n


def full_nan_like(x: np.ndarray) -> np.ndarray:
    x = _as_float_array(x)
    return np.full(x.shape, np.nan, dtype=np.float64)


def nan_to_num_posinf(x: np.ndarray, value: float = np.nan) -> np.ndarray:
    x = _as_float_array(x)
    out = x.copy()
    out[~np.isfinite(out)] = value
    return out


def delay(x: np.ndarray, n: int) -> np.ndarray:
    x = _as_float_array(x)
    n = _check_lag(n)
    out = np.empty_like(x)
    if n == 0:
        out[:] = x
        return out
    out[:n] = np.nan
    out[n:] = x[:-n]
    return out


def delta(x: np.ndarray, n: int) -> np.ndarray:
    x = _as_float_array(x)
    n = _check_lag(n)
    out = np.empty_like(x)
    if n == 0:
        out[:] = 0.0
        return out
    out[:n] = np.nan
    out[n:] = x[n:] - x[:-n]
    return out


def pct_change(x: np.ndarray, n: int = 1) -> np.ndarray:
    x = _as_float_array(x)
    n = _check_lag(n)
    out = np.empty_like(x)
    if n == 0:
        out[:] = 0.0
        return out
    out[:n] = np.nan
    prev = x[:-n]
    curr = x[n:]
    valid = np.isfinite(curr) & np.isfinite(prev) & (prev != 0.0)
    out[n:] = np.nan
    out[n:][valid] = curr[valid] / prev[valid] - 1.0
    return out*100


def safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _as_float_array(a)
    b = _as_float_array(b)
    out = np.full(np.broadcast(a, b).shape, np.nan, dtype=np.float64)
    a1, b1 = np.broadcast_arrays(a, b)
    mask = np.isfinite(a1) & np.isfinite(b1) & (b1 != 0.0)
    out[mask] = a1[mask] / b1[mask]
    return out


def clip_inf_to_nan(x: np.ndarray) -> np.ndarray:
    x = _as_float_array(x)
    out = x.copy()
    out[~np.isfinite(out)] = np.nan
    return out


def sign(x: np.ndarray) -> np.ndarray:
    x = _as_float_array(x)
    out = np.sign(x)
    out[~np.isfinite(x)] = np.nan
    return out


def abs_(x: np.ndarray) -> np.ndarray:
    x = _as_float_array(x)
    return np.abs(x)


def log1p_safe(x: np.ndarray) -> np.ndarray:
    x = _as_float_array(x)
    out = np.full_like(x, np.nan)
    mask = np.isfinite(x) & (x > -1.0)
    out[mask] = np.log1p(x[mask])
    return out


def ts_sum(x: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    window = _check_window(window)
    return bn.move_sum(x, window=window, min_count=window)


def ts_mean(x: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    window = _check_window(window)
    return bn.move_mean(x, window=window, min_count=window)


def ts_std(x: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    window = _check_window(window)
    return bn.move_std(x, window=window, min_count=window)


def ts_var(x: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    window = _check_window(window)
    return bn.move_var(x, window=window, min_count=window)


def ts_min(x: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    window = _check_window(window)
    return bn.move_min(x, window=window, min_count=window)


def ts_max(x: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    window = _check_window(window)
    return bn.move_max(x, window=window, min_count=window)


def ts_argmax(x: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    window = _check_window(window)
    idx = bn.move_argmax(x, window=window, min_count=window).astype(np.float64)
    idx[np.isnan(ts_max(x, window))] = np.nan
    return idx


def ts_argmin(x: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    window = _check_window(window)
    idx = bn.move_argmin(x, window=window, min_count=window).astype(np.float64)
    idx[np.isnan(ts_min(x, window))] = np.nan
    return idx


def ts_rank(x: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    window = _check_window(window)
    out = np.full_like(x, np.nan)
    if x.size < window:
        return out
    for i in range(window - 1, x.size):
        w = x[i - window + 1:i + 1]
        if np.any(~np.isfinite(w)):
            continue
        order = np.argsort(np.argsort(w))
        out[i] = order[-1] / (window - 1) if window > 1 else 0.0
    return out


def ts_median(x: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    window = _check_window(window)
    return bn.move_median(x, window=window, min_count=window)


def ts_zscore(x: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    mu = ts_mean(x, window)
    sd = ts_std(x, window)
    return safe_div(x - mu, sd)


def ts_skew(x: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    window = _check_window(window)
    mu = ts_mean(x, window)
    sd = ts_std(x, window)
    z = safe_div(x - mu, sd)
    z3 = z * z * z
    return ts_mean(z3, window)


def ts_kurt(x: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    window = _check_window(window)
    mu = ts_mean(x, window)
    sd = ts_std(x, window)
    z = safe_div(x - mu, sd)
    z4 = z * z * z * z
    return ts_mean(z4, window)


def ts_cov(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    y = _as_float_array(y)
    window = _check_window(window)
    ex = ts_mean(x, window)
    ey = ts_mean(y, window)
    exy = ts_mean(x * y, window)
    return exy - ex * ey


def ts_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    y = _as_float_array(y)
    window = _check_window(window)
    cov = ts_cov(x, y, window)
    sx = ts_std(x, window)
    sy = ts_std(y, window)
    return safe_div(cov, sx * sy)


def ts_product(x: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    window = _check_window(window)
    out = np.full_like(x, np.nan)
    if x.size < window:
        return out
    for i in range(window - 1, x.size):
        w = x[i - window + 1:i + 1]
        if np.any(~np.isfinite(w)):
            continue
        out[i] = np.prod(w)
    return out


def ewm_mean(x: np.ndarray, span: int) -> np.ndarray:
    x = _as_float_array(x)
    span = _check_window(span)
    alpha = 2.0 / (span + 1.0)
    out = np.full_like(x, np.nan)
    if x.size == 0:
        return out
    first_valid = np.flatnonzero(np.isfinite(x))
    if first_valid.size == 0:
        return out
    i0 = first_valid[0]
    out[i0] = x[i0]
    for i in range(i0 + 1, x.size):
        if np.isfinite(x[i]):
            out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
        else:
            out[i] = out[i - 1]
    return out


def linear_decay_mean(x: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    window = _check_window(window)
    out = np.full_like(x, np.nan)
    if x.size < window:
        return out
    w = np.arange(1.0, window + 1.0, dtype=np.float64)
    w /= w.sum()
    for i in range(window - 1, x.size):
        seg = x[i - window + 1:i + 1]
        if np.any(~np.isfinite(seg)):
            continue
        out[i] = np.dot(seg, w)
    return out


def rolling_pos_in_range(x: np.ndarray, window: int) -> np.ndarray:
    x = _as_float_array(x)
    lo = ts_min(x, window)
    hi = ts_max(x, window)
    return safe_div(x - lo, hi - lo)


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    high = _as_float_array(high)
    low = _as_float_array(low)
    close = _as_float_array(close)
    prev_close = delay(close, 1)
    a = high - low
    b = np.abs(high - prev_close)
    c = np.abs(low - prev_close)
    out = np.maximum(a, np.maximum(b, c))
    out[~np.isfinite(a) | ~np.isfinite(b) | ~np.isfinite(c)] = np.nan
    return out


def stack_cols(arr_list: list[np.ndarray]) -> np.ndarray:
    return np.column_stack(arr_list)