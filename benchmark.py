"""学習率・ミニバッチサイズを変えた条件をまとめて実行するベンチマークスクリプト。

各条件を5回試行し、初回はウォームアップとして時間・MSEの集計から除外する
(ただし終了理由は5回分すべて記録する)。データ分割用のシードとは別に、
各試行のミニバッチシャッフル用シードを固定して記録し、再現性を確保する。
"""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from src.data import DatasetBundle, inverse_transform_y, load_and_prepare
from src.model import MiniBatchGDLinearRegression

DATA_PATH = Path(__file__).parent / "data" / "energy_efficiency.csv"

DEFAULT_LR = 0.01
DEFAULT_BATCH_SIZE: Union[int, str] = 32
DEFAULT_MAX_EPOCHS = 300

N_TRIALS = 5
WARMUP_TRIALS = 1  # 先頭N回をウォームアップとして集計から除外
SHUFFLE_SEED_BASE = 100  # trial_index を加えて各試行のシードにする
HARD_TIME_LIMIT_SEC = 600.0  # 異常時のみ働く安全上限(ベンチマークではタイムアウトなしで計測)

CONDITIONS = [
    {"name": "lr0.01_batch32", "lr": 0.01, "batch_size": 32, "max_epochs": 300},
    {"name": "lr0.01_batch1", "lr": 0.01, "batch_size": 1, "max_epochs": 300},
    {"name": "lr0.01_full", "lr": 0.01, "batch_size": "full", "max_epochs": 300},
    {"name": "lr0.001_batch32", "lr": 0.001, "batch_size": 32, "max_epochs": 300},
    {"name": "lr0.1_batch32", "lr": 0.1, "batch_size": 32, "max_epochs": 300},
    {"name": "lr1.0_batch32", "lr": 1.0, "batch_size": 32, "max_epochs": 300},
    {"name": "batch1_epoch500", "lr": 0.01, "batch_size": 1, "max_epochs": 500},
]


@dataclass
class TrialRecord:
    condition: str
    trial_index: int
    is_warmup: bool
    shuffle_seed: int
    lr: float
    batch_size: Union[int, str]
    max_epochs: int
    epochs_completed: int
    n_updates: int
    elapsed_sec: float
    reason: str
    train_mse_orig: float
    test_mse_orig: float
    best_epoch: int
    divergence_detected: bool


def run_trial(
    bundle: DatasetBundle,
    lr: float,
    batch_size: Union[int, str],
    max_epochs: int,
    shuffle_seed: int,
    condition_name: str,
    trial_index: int,
    is_warmup: bool,
) -> TrialRecord:
    model = MiniBatchGDLinearRegression(n_features=bundle.X_train_std.shape[1])

    last_log = None
    loss_history: List[float] = []
    for log in model.train_generator(
        X_train_std=bundle.X_train_std,
        y_train_std=bundle.y_train_std,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        shuffle_seed=shuffle_seed,
        early_stopping_enabled=False,  # 比較実験のため常に無効
        soft_time_limit_sec=None,  # ベンチマークではタイムアウトなしで実時間を計測
        hard_time_limit_sec=HARD_TIME_LIMIT_SEC,
    ):
        last_log = log
        loss_history.append(log.train_loss_std)

    assert last_log is not None

    pred_train_std = model.predict_std(bundle.X_train_std)
    pred_test_std = model.predict_std(bundle.X_test_std)
    pred_train_orig = inverse_transform_y(pred_train_std, bundle.y_mean, bundle.y_scale)
    pred_test_orig = inverse_transform_y(pred_test_std, bundle.y_mean, bundle.y_scale)
    train_mse_orig = float(np.mean((pred_train_orig - bundle.y_train_orig) ** 2))
    test_mse_orig = float(np.mean((pred_test_orig - bundle.y_test_orig) ** 2))

    finite_history = [v for v in loss_history if np.isfinite(v)]
    best_epoch = (
        int(np.argmin(finite_history)) + 1 if finite_history else 0
    )
    divergence_detected = last_log.reason == "divergence" or len(finite_history) != len(
        loss_history
    )

    return TrialRecord(
        condition=condition_name,
        trial_index=trial_index,
        is_warmup=is_warmup,
        shuffle_seed=shuffle_seed,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        epochs_completed=last_log.epoch,
        n_updates=last_log.n_updates,
        elapsed_sec=last_log.elapsed_sec,
        reason=last_log.reason,
        train_mse_orig=train_mse_orig,
        test_mse_orig=test_mse_orig,
        best_epoch=best_epoch,
        divergence_detected=divergence_detected,
    )


def main() -> None:
    # Windows端末での文字化け防止 (コンソールの既定コードページに依存しないようにする)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

    overall_start = time.perf_counter()

    print("=" * 70)
    print("データ読み込み・前処理")
    print("=" * 70)
    bundle = load_and_prepare(str(DATA_PATH))
    print(f"データ読み込み時間: {bundle.load_time_sec * 1000:.2f} ms")
    print(f"前処理時間 (split+標準化): {bundle.preprocess_time_sec * 1000:.2f} ms")
    print(f"学習データ数: {bundle.n_train}, テストデータ数: {bundle.n_test}")
    print()

    all_records: List[TrialRecord] = []

    for cond in CONDITIONS:
        print("-" * 70)
        print(
            f"条件: {cond['name']}  "
            f"(lr={cond['lr']}, batch_size={cond['batch_size']}, "
            f"max_epochs={cond['max_epochs']})"
        )
        print("-" * 70)
        for trial_index in range(N_TRIALS):
            shuffle_seed = SHUFFLE_SEED_BASE + trial_index
            is_warmup = trial_index < WARMUP_TRIALS
            record = run_trial(
                bundle=bundle,
                lr=cond["lr"],
                batch_size=cond["batch_size"],
                max_epochs=cond["max_epochs"],
                shuffle_seed=shuffle_seed,
                condition_name=cond["name"],
                trial_index=trial_index,
                is_warmup=is_warmup,
            )
            all_records.append(record)
            tag = "warmup" if is_warmup else f"trial{trial_index}"
            print(
                f"  [{tag}] seed={shuffle_seed} "
                f"epoch={record.epochs_completed} updates={record.n_updates} "
                f"time={record.elapsed_sec:.4f}s reason={record.reason} "
                f"test_mse={record.test_mse_orig:.4f}"
            )
        print()

    detail_df = pd.DataFrame([r.__dict__ for r in all_records])

    print("=" * 70)
    print("詳細結果 (全試行)")
    print("=" * 70)
    print(
        detail_df[
            [
                "condition",
                "trial_index",
                "is_warmup",
                "shuffle_seed",
                "epochs_completed",
                "n_updates",
                "elapsed_sec",
                "reason",
                "train_mse_orig",
                "test_mse_orig",
                "best_epoch",
                "divergence_detected",
            ]
        ].to_string(index=False)
    )
    print()

    summary_rows = []
    for cond in CONDITIONS:
        name = cond["name"]
        cond_records = [r for r in all_records if r.condition == name]
        measured = [r for r in cond_records if not r.is_warmup]
        times = [r.elapsed_sec for r in measured]
        test_mses = [r.test_mse_orig for r in measured]
        reasons_all = [r.reason for r in cond_records]
        summary_rows.append(
            {
                "condition": name,
                "lr": cond["lr"],
                "batch_size": cond["batch_size"],
                "max_epochs": cond["max_epochs"],
                "time_median_sec": statistics.median(times),
                "time_min_sec": min(times),
                "time_max_sec": max(times),
                "test_mse_median": statistics.median(test_mses),
                "reasons_all_trials": ",".join(reasons_all),
                "any_divergence": any(r.divergence_detected for r in cond_records),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    print("=" * 70)
    print("条件別サマリー (ウォームアップ1回を除く4回から集計)")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    overall_elapsed = time.perf_counter() - overall_start
    print()
    print(f"ベンチマーク全体の実行時間: {overall_elapsed:.2f} 秒")


if __name__ == "__main__":
    main()
