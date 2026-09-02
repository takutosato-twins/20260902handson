"""UCI Energy Efficiencyデータセットの読み込みと前処理。

出典: UCI Machine Learning Repository "Energy Efficiency" データセット
      (Tsanas & Xifara) / ライセンス: CC BY 4.0
取得元: https://archive.ics.uci.edu/static/public/242/data.csv
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
TARGET_COLUMN = "Y1"  # Heating Load
EXPECTED_ROWS = 768
EXPECTED_COLUMNS = FEATURE_COLUMNS + ["Y1", "Y2"]

SPLIT_SEED = 42
TEST_SIZE = 0.2


class DataValidationError(RuntimeError):
    """ダウンロードしたデータが想定と異なる場合に送出する。"""


@dataclass
class DatasetBundle:
    X_train_std: np.ndarray
    X_test_std: np.ndarray
    y_train_std: np.ndarray
    y_test_std: np.ndarray
    y_train_orig: np.ndarray
    y_test_orig: np.ndarray
    feature_names: list
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    n_train: int
    n_test: int
    load_time_sec: float
    preprocess_time_sec: float


def _validate_raw(df: pd.DataFrame) -> None:
    """列名・行数・欠損値・dtypeが想定どおりであることを検証する。

    想定と異なる場合は処理を続行せず例外を送出する。
    """
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"想定していた列が見つかりません: {missing_cols}. "
            f"実際の列: {list(df.columns)}"
        )

    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra_cols:
        raise DataValidationError(
            f"想定外の列が含まれています: {extra_cols}. "
            f"想定していた列: {EXPECTED_COLUMNS}"
        )

    if len(df) != EXPECTED_ROWS:
        raise DataValidationError(
            f"行数が想定と異なります。想定: {EXPECTED_ROWS}行, 実際: {len(df)}行"
        )

    n_missing = int(df[EXPECTED_COLUMNS].isna().sum().sum())
    if n_missing > 0:
        raise DataValidationError(
            f"欠損値が {n_missing} 件見つかりました。想定では欠損値なしです。\n"
            f"{df[EXPECTED_COLUMNS].isna().sum()}"
        )

    non_numeric = [
        c for c in EXPECTED_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        raise DataValidationError(
            f"数値であるべき列が数値型ではありません: {non_numeric}. "
            f"dtypes: {df[EXPECTED_COLUMNS].dtypes.to_dict()}"
        )


def load_and_prepare(
    csv_path: str,
    target_col: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    split_seed: int = SPLIT_SEED,
) -> DatasetBundle:
    """CSVを読み込み、検証・固定split・標準化を行って返す。

    - 入力特徴量・目的変数ともに、標準化の平均/標準偏差は学習データのみから計算する
      (テストデータの情報がリークしないようにする)。
    - 乱数シードを固定してtrain/testを常に同じ分割にする。
    """
    t0 = time.perf_counter()
    df = pd.read_csv(csv_path)
    t1 = time.perf_counter()
    load_time_sec = t1 - t0

    _validate_raw(df)

    t2 = time.perf_counter()

    if target_col not in EXPECTED_COLUMNS:
        raise DataValidationError(f"未知の目的変数列です: {target_col}")

    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df[target_col].to_numpy(dtype=np.float64)
    n = len(df)

    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n)
    n_test = int(round(n * test_size))
    n_train = n - n_test
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train_orig, y_test_orig = y[train_idx], y[test_idx]

    x_mean = X_train.mean(axis=0)
    x_scale = X_train.std(axis=0, ddof=0)
    x_scale = np.where(x_scale == 0, 1.0, x_scale)
    X_train_std = (X_train - x_mean) / x_scale
    X_test_std = (X_test - x_mean) / x_scale

    y_mean = float(y_train_orig.mean())
    y_scale = float(y_train_orig.std(ddof=0))
    if y_scale == 0:
        y_scale = 1.0
    y_train_std = (y_train_orig - y_mean) / y_scale
    y_test_std = (y_test_orig - y_mean) / y_scale

    t3 = time.perf_counter()
    preprocess_time_sec = t3 - t2

    return DatasetBundle(
        X_train_std=X_train_std,
        X_test_std=X_test_std,
        y_train_std=y_train_std,
        y_test_std=y_test_std,
        y_train_orig=y_train_orig,
        y_test_orig=y_test_orig,
        feature_names=list(FEATURE_COLUMNS),
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        n_train=n_train,
        n_test=n_test,
        load_time_sec=load_time_sec,
        preprocess_time_sec=preprocess_time_sec,
    )


def inverse_transform_y(y_std: np.ndarray, y_mean: float, y_scale: float) -> np.ndarray:
    """標準化された目的変数を元のHeating Load単位へ戻す。"""
    return y_std * y_scale + y_mean
