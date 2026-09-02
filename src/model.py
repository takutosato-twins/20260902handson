"""NumPy自前実装のミニバッチ勾配降下法による線形回帰。

scikit-learnの学習機能は使わず、切片込みの線形回帰をミニバッチ勾配降下法で
自前実装する。学習はエポック単位のジェネレータとして実装し、呼び出し側
(ベンチマークスクリプトや将来のUI)がエポックごとに続行/中断を判断できる
ようにする。特定のUIフレームワークの機構には依存しない。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union

import numpy as np

BatchSize = Union[int, str]  # 数値、または "full"(全件バッチ)

# 終了理由
REASON_NORMAL = "normal"  # 正常終了
REASON_EARLY_STOPPING = "early_stopping"  # 早期終了
REASON_MANUAL_INTERRUPT = "manual_interrupt"  # 手動中断
REASON_TIMEOUT = "timeout"  # 制限時間超過
REASON_DIVERGENCE = "divergence"  # 発散検知
REASON_ERROR = "error"  # エラー

DEFAULT_HARD_TIME_LIMIT_SEC = 600.0  # 異常時のハード上限(常時有効)
DEFAULT_DIVERGENCE_THRESHOLD = 1e8  # 標準化スケールでの損失閾値
DEFAULT_PATIENCE = 20
DEFAULT_MIN_DELTA = 1e-4


@dataclass
class EpochLog:
    """1エポック終了時点のスナップショット。"""

    epoch: int
    n_updates: int
    elapsed_sec: float
    train_loss_std: float  # 標準化スケールでの学習損失(内部監視用)
    shuffle_seed: int
    reason: Optional[str] = None  # Noneでなければこのエポックで学習終了


@dataclass
class TrainResult:
    reason: str
    epochs_completed: int
    n_updates: int
    elapsed_sec: float
    loss_history_std: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_loss_std: float = float("inf")
    error_message: Optional[str] = None


class MiniBatchGDLinearRegression:
    """切片込みの線形回帰をミニバッチ勾配降下法で学習する。"""

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.weights = np.zeros(n_features, dtype=np.float64)
        self.bias = 0.0

    def reset(self) -> None:
        self.weights = np.zeros(self.n_features, dtype=np.float64)
        self.bias = 0.0

    def predict_std(self, X_std: np.ndarray) -> np.ndarray:
        return X_std @ self.weights + self.bias

    def train_generator(
        self,
        X_train_std: np.ndarray,
        y_train_std: np.ndarray,
        lr: float,
        batch_size: BatchSize,
        max_epochs: int,
        shuffle_seed: int,
        early_stopping_enabled: bool = False,
        patience: int = DEFAULT_PATIENCE,
        min_delta: float = DEFAULT_MIN_DELTA,
        soft_time_limit_sec: Optional[float] = None,
        hard_time_limit_sec: float = DEFAULT_HARD_TIME_LIMIT_SEC,
        divergence_loss_threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
        interrupt_check: Optional[Callable[[], bool]] = None,
    ):
        """1エポックごとに EpochLog を yield するジェネレータ。

        - soft_time_limit_sec: Noneなら無効(ベンチマークではNoneにして実時間を計測)。
          Webアプリではこれを30秒などに設定して安全対策として使う。
        - hard_time_limit_sec: 常時有効な異常時の安全上限。
        - early_stopping_enabled: Falseがデフォルト。有効にするとpatienceエポック
          改善がなければ早期終了する。
        - interrupt_check: 呼び出し側が中断要求を伝えるための任意のコールバック。
        """
        n = X_train_std.shape[0]
        eff_batch_size = n if batch_size == "full" else int(batch_size)
        eff_batch_size = max(1, min(eff_batch_size, n))

        rng = np.random.default_rng(shuffle_seed)
        n_updates = 0
        start = time.perf_counter()
        loss_history: List[float] = []
        best_loss = float("inf")
        best_epoch = 0
        epochs_since_improve = 0

        for epoch in range(1, max_epochs + 1):
            perm = rng.permutation(n)
            for start_idx in range(0, n, eff_batch_size):
                batch_idx = perm[start_idx : start_idx + eff_batch_size]
                xb = X_train_std[batch_idx]
                yb = y_train_std[batch_idx]
                pred = xb @ self.weights + self.bias
                err = pred - yb
                m = len(batch_idx)
                grad_w = (2.0 / m) * (xb.T @ err)
                grad_b = (2.0 / m) * err.sum()
                self.weights = self.weights - lr * grad_w
                self.bias = self.bias - lr * grad_b
                n_updates += 1

            pred_full = X_train_std @ self.weights + self.bias
            diff = pred_full - y_train_std
            loss = float(np.mean(diff * diff))
            elapsed = time.perf_counter() - start
            loss_history.append(loss)

            reason: Optional[str] = None
            if not np.isfinite(loss) or loss > divergence_loss_threshold:
                reason = REASON_DIVERGENCE
            elif interrupt_check is not None and interrupt_check():
                reason = REASON_MANUAL_INTERRUPT
            elif soft_time_limit_sec is not None and elapsed > soft_time_limit_sec:
                reason = REASON_TIMEOUT
            elif elapsed > hard_time_limit_sec:
                reason = REASON_TIMEOUT

            if loss < best_loss - min_delta:
                best_loss = loss
                best_epoch = epoch
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            if (
                reason is None
                and early_stopping_enabled
                and epochs_since_improve >= patience
            ):
                reason = REASON_EARLY_STOPPING

            if reason is None and epoch == max_epochs:
                reason = REASON_NORMAL

            yield EpochLog(
                epoch=epoch,
                n_updates=n_updates,
                elapsed_sec=elapsed,
                train_loss_std=loss,
                shuffle_seed=shuffle_seed,
                reason=reason,
            )

            if reason is not None:
                return
