import os
import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


@dataclass
class Config:
    normalized_data_dir: str = "/mnt/d/python/graduation_thesis/normalized_data"
    raw_label_dir: str = "/mnt/d/python/graduation_thesis/data"
    artifact_root: str = "/mnt/d/python/graduation_thesis/artifacts"

    label_col: str = "fwd_ret_5"
    key_cols: Tuple[str, str] = ("symbol", "date")

    hidden_dims: Tuple[int, int, int] = (128, 64, 32)
    dropout: float = 0.10

    batch_size: int = 4096
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 5

    num_workers: int = 0
    pin_memory: bool = False

    seed: int = 42
    device: str = "cpu"

    save_by: str = "rank_ic"
    verbose: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def print_shape(name: str, x: np.ndarray, y: np.ndarray) -> None:
    print(f"{name}: X.shape={x.shape}, y.shape={y.shape}")


def np_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    x = x - x.mean()
    y = y - y.mean()

    denom = math.sqrt((x * x).sum()) * math.sqrt((y * y).sum())
    if denom < eps:
        return np.nan
    return float((x * y).sum() / denom)


def simple_rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    rx = simple_rankdata(x)
    ry = simple_rankdata(y)
    return np_corr(rx, ry, eps=eps)


def read_split_pair(
    factor_fp: str,
    label_fp: str,
    label_col: str,
    key_cols: Tuple[str, str] = ("symbol", "date"),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    factor = pl.read_parquet(factor_fp)
    label = pl.read_parquet(label_fp)

    symbol_col, date_col = key_cols

    factor = factor.sort([symbol_col, date_col])
    label = label.sort([symbol_col, date_col])

    factor_key = factor.select([symbol_col, date_col])
    label_key = label.select([symbol_col, date_col])

    if not factor_key.equals(label_key):
        raise ValueError(
            f"factor / label key 不一致:\n"
            f"factor_fp={factor_fp}\nlabel_fp={label_fp}"
        )

    if label_col not in label.columns:
        raise ValueError(f"label_col={label_col} 不在 {label_fp} 里")

    feature_cols = [c for c in factor.columns if c not in [symbol_col, date_col]]

    df = pl.concat(
        [
            factor,
            label.select([label_col]),
        ],
        how="horizontal",
    )

    df = df.filter(pl.col(label_col).is_not_null())

    x = df.select(feature_cols).to_numpy().astype(np.float32, copy=False)
    y = df.select(label_col).to_numpy().reshape(-1).astype(np.float32, copy=False)

    dates = df.select(date_col).to_series().to_numpy()
    symbols = df.select(symbol_col).to_series().to_numpy()

    return x, y, dates, symbols, feature_cols


class PanelDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y).view(-1, 1)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class NN3(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(128, 64, 32), dropout=0.1):
        super().__init__()

        h1, h2, h3 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h2, h3),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(h3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []

    for xb, _ in loader:
        xb = xb.to(device, non_blocking=False)
        pred = model(xb).squeeze(-1).cpu().numpy()
        preds.append(pred)

    return np.concatenate(preds, axis=0)


def daily_ic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: np.ndarray,
    min_n: int = 3,
) -> Dict[str, float]:
    unique_dates = np.unique(dates)

    ic_list = []
    rank_ic_list = []

    for d in unique_dates:
        mask = (dates == d)
        yt = y_true[mask]
        yp = y_pred[mask]

        if yt.shape[0] < min_n:
            continue

        ic = np_corr(yp, yt)
        ric = spearman_corr(yp, yt)

        if not np.isnan(ic):
            ic_list.append(ic)
        if not np.isnan(ric):
            rank_ic_list.append(ric)

    return {
        "ic": float(np.mean(ic_list)) if len(ic_list) > 0 else np.nan,
        "rank_ic": float(np.mean(rank_ic_list)) if len(rank_ic_list) > 0 else np.nan,
        "n_dates": int(len(unique_dates)),
        "n_dates_used_ic": int(len(ic_list)),
        "n_dates_used_rank_ic": int(len(rank_ic_list)),
    }


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    y_true: np.ndarray,
    dates: np.ndarray,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()

    losses = []
    preds = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=False)
            yb = yb.to(device, non_blocking=False)

            pred = model(xb)
            loss = criterion(pred, yb)

            losses.append(loss.item())
            preds.append(pred.squeeze(-1).cpu().numpy())

    y_pred = np.concatenate(preds, axis=0)
    metrics = daily_ic_metrics(y_true=y_true, y_pred=y_pred, dates=dates)
    metrics["loss"] = float(np.mean(losses)) if losses else np.nan
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses = []

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=False)
        yb = yb.to(device, non_blocking=False)

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses)) if losses else np.nan


def save_artifacts(
    save_dir: str,
    config: Config,
    model: nn.Module,
    feature_cols: List[str],
    best_metric: float,
    best_epoch: int,
    history: List[Dict],
) -> None:
    ensure_dir(save_dir)

    torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    with open(os.path.join(save_dir, "feature_cols.json"), "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_metric_name": config.save_by,
                "best_metric_value": best_metric,
                "best_epoch": best_epoch,
                "history": history,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def run_training(config: Config) -> None:
    set_seed(config.seed)
    device = torch.device(config.device)

    factor_train_fp = os.path.join(config.normalized_data_dir, "factor_train.parquet")
    factor_valid_fp = os.path.join(config.normalized_data_dir, "factor_valid.parquet")
    factor_test_fp  = os.path.join(config.normalized_data_dir, "factor_test.parquet")

    label_train_fp = os.path.join(config.raw_label_dir, "label_train.parquet")
    label_valid_fp = os.path.join(config.raw_label_dir, "label_valid.parquet")
    label_test_fp  = os.path.join(config.raw_label_dir, "label_test.parquet")

    X_train, y_train, dates_train, symbols_train, feature_cols = read_split_pair(
        factor_train_fp, label_train_fp, config.label_col, key_cols=config.key_cols
    )
    X_valid, y_valid, dates_valid, symbols_valid, _ = read_split_pair(
        factor_valid_fp, label_valid_fp, config.label_col, key_cols=config.key_cols
    )
    X_test, y_test, dates_test, symbols_test, _ = read_split_pair(
        factor_test_fp, label_test_fp, config.label_col, key_cols=config.key_cols
    )

    print_shape("train", X_train, y_train)
    print_shape("valid", X_valid, y_valid)
    print_shape("test ", X_test, y_test)

    input_dim = X_train.shape[1]
    print("input_dim =", input_dim)

    train_ds = PanelDataset(X_train, y_train)
    valid_ds = PanelDataset(X_valid, y_valid)
    test_ds  = PanelDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
    )

    model = NN3(
        input_dim=input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    criterion = nn.MSELoss()

    best_metric = -np.inf
    best_epoch = -1
    patience_count = 0
    history = []

    save_dir = os.path.join(config.artifact_root, config.label_col)
    ensure_dir(save_dir)
    best_model_fp = os.path.join(save_dir, "best_model.pt")

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        valid_metrics = evaluate_split(
            model=model,
            loader=valid_loader,
            y_true=y_valid,
            dates=dates_valid,
            device=device,
            criterion=criterion,
        )

        current_metric = valid_metrics[config.save_by]

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_metrics["loss"],
            "valid_ic": valid_metrics["ic"],
            "valid_rank_ic": valid_metrics["rank_ic"],
        }
        history.append(row)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} | "
            f"valid_loss={valid_metrics['loss']:.6f} | "
            f"valid_ic={valid_metrics['ic']:.6f} | "
            f"valid_rank_ic={valid_metrics['rank_ic']:.6f}"
        )

        improved = (
            not np.isnan(current_metric)
            and (current_metric > best_metric)
        )

        if improved:
            best_metric = current_metric
            best_epoch = epoch
            patience_count = 0
            torch.save(model.state_dict(), best_model_fp)
        else:
            patience_count += 1

        if patience_count >= config.patience:
            print(f"Early stopping at epoch={epoch}, best_epoch={best_epoch}")
            break

    model.load_state_dict(torch.load(best_model_fp, map_location=device))

    valid_best = evaluate_split(
        model=model,
        loader=valid_loader,
        y_true=y_valid,
        dates=dates_valid,
        device=device,
        criterion=criterion,
    )
    test_best = evaluate_split(
        model=model,
        loader=test_loader,
        y_true=y_test,
        dates=dates_test,
        device=device,
        criterion=criterion,
    )

    print("\n===== BEST VALID =====")
    print(valid_best)
    print("best_epoch =", best_epoch)
    print("best_metric =", best_metric, f"({config.save_by})")

    print("\n===== TEST =====")
    print(test_best)

    save_artifacts(
        save_dir=save_dir,
        config=config,
        model=model,
        feature_cols=feature_cols,
        best_metric=best_metric,
        best_epoch=best_epoch,
        history=history + [
            {"best_valid": valid_best},
            {"test": test_best},
        ],
    )

    test_pred = predict(model, test_loader, device=device)
    pred_df = pl.DataFrame({
        "symbol": symbols_test,
        "date": dates_test,
        "y_true": y_test,
        "y_pred": test_pred.astype(np.float32, copy=False),
    }).sort(["date", "symbol"])

    pred_df.write_parquet(os.path.join(save_dir, "test_predictions.parquet"))
    print(f"\nArtifacts saved to: {save_dir}")


if __name__ == "__main__":
    labels = [ 'fwd_ret_5_rank', 'fwd_ewm_ret_5_rank', 'fwd_ewm_ret_20_rank']
    # labels = ['fwd_ret_1', 'fwd_ret_5', 'fwd_ewm_ret_5', 'fwd_ewm_ret_20', 'fwd_vol_5', 'fwd_vol_20', 'fwd_ret_5_voladj', 'fwd_ret_20_voladj', 'fwd_ret_1_rank',]
    for name in labels:
        cfg = Config(
            normalized_data_dir="/mnt/d/python/graduation_thesis/normalized_data",
            raw_label_dir="/mnt/d/python/graduation_thesis/data",
            artifact_root="/mnt/d/python/graduation_thesis/artifacts",
            label_col=name,
            hidden_dims=(128, 64, 32),
            dropout=0.05,
            batch_size=2048,
            lr=3e-4,
            weight_decay=1e-3,
            max_epochs=25,
            patience=5,
            device="cpu",
            save_by="rank_ic",
        )
        run_training(cfg)