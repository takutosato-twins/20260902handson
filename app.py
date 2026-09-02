"""勾配降下法ハンズオン - 本番Streamlitアプリ。

学習率・ミニバッチサイズ・最大エポック数を変えながら、ミニバッチ勾配降下法の
挙動(速度・安定性・収束)を体験するハンズオン用アプリ。

- 学習は src/model.py の MiniBatchGDLinearRegression.train_generator を使う
  (scikit-learnの学習機能は使わず、NumPy自前実装)。
- 中断可能なUIは st.fragment(run_every=...) で1回の実行につき数エポックだけ
  進める方式で実現する(Phase 2のPoC interrupt_poc.py で検証済みの方式)。
  interrupt_poc.py 自体はここでは使わず、本アプリは独立して実装している。
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.data import inverse_transform_y, load_and_prepare
from src.model import MiniBatchGDLinearRegression

DATA_PATH = Path(__file__).parent / "data" / "energy_efficiency.csv"

# フラグメント1回あたりに進めるエポック数、および自動再実行の間隔。
# streamlit.testing.v1.AppTest による実測(measure_chunk_epochs.py, 開発時のみ使用)
# を踏まえて決定している。環境変数での上書きは計測スクリプト専用で、
# 参加者向けの実行では常にデフォルト値が使われる。
CHUNK_EPOCHS = int(os.environ.get("HANDSON_CHUNK_EPOCHS", "25"))
TICK_INTERVAL_SEC = float(os.environ.get("HANDSON_TICK_INTERVAL_SEC", "0.05"))

SOFT_TIME_LIMIT_SEC = 30.0  # 通常の自動停止時間(Webアプリの安全対策)
HARD_TIME_LIMIT_SEC = 600.0  # 異常時のハード上限
FIXED_SHUFFLE_SEED = 42  # 再現性のため固定

REASON_LABELS = {
    "normal": "正常終了",
    "early_stopping": "早期終了",
    "manual_interrupt": "手動中断",
    "timeout": "制限時間超過",
    "divergence": "発散検知",
    "error": "エラー",
}

LR_OPTIONS = [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
BATCH_OPTIONS = ["1", "2", "4", "8", "16", "32", "64", "128", "256", "全件"]
EPOCH_OPTIONS = [50, 100, 300, 500]

DEFAULT_LR = 0.01
DEFAULT_BATCH_LABEL = "32"
DEFAULT_MAX_EPOCHS = 300

st.set_page_config(page_title="勾配降下法ハンズオン", layout="wide")


@st.cache_resource
def get_bundle():
    return load_and_prepare(str(DATA_PATH))


def init_state() -> None:
    defaults = dict(
        active_run_id=0,
        is_training=False,
        gen=None,
        gen_run_id=None,
        trainer=None,
        stop_requested=False,
        reason=None,
        error_message=None,
        epoch=0,
        n_updates=0,
        loss_history=[],  # 学習データの損失(元単位)を1エポックごとに記録
        run_start_time=None,
        wall_time_total=0.0,
        model_time_total=0.0,
        render_time_total=0.0,
        active_learning_rate=None,
        active_batch_size=None,
        active_max_epochs=None,
        active_seed=None,
        param_lr=DEFAULT_LR,
        param_batch_label=DEFAULT_BATCH_LABEL,
        param_max_epochs=DEFAULT_MAX_EPOCHS,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def start_training() -> None:
    if st.session_state.is_training:
        return  # ボタン連打時の多重起動防止

    bundle = get_bundle()
    lr = st.session_state.param_lr
    batch_label = st.session_state.param_batch_label
    batch_size = "full" if batch_label == "全件" else int(batch_label)
    max_epochs = st.session_state.param_max_epochs

    st.session_state.active_run_id += 1
    run_id = st.session_state.active_run_id

    trainer = MiniBatchGDLinearRegression(n_features=bundle.X_train_std.shape[1])
    gen = trainer.train_generator(
        X_train_std=bundle.X_train_std,
        y_train_std=bundle.y_train_std,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        shuffle_seed=FIXED_SHUFFLE_SEED,
        early_stopping_enabled=False,
        soft_time_limit_sec=SOFT_TIME_LIMIT_SEC,
        hard_time_limit_sec=HARD_TIME_LIMIT_SEC,
        interrupt_check=lambda: st.session_state.stop_requested,
    )

    st.session_state.trainer = trainer
    st.session_state.gen = gen
    st.session_state.gen_run_id = run_id
    st.session_state.is_training = True
    st.session_state.stop_requested = False
    st.session_state.reason = None
    st.session_state.error_message = None
    st.session_state.epoch = 0
    st.session_state.n_updates = 0
    st.session_state.loss_history = []
    st.session_state.run_start_time = time.perf_counter()
    st.session_state.wall_time_total = 0.0
    st.session_state.model_time_total = 0.0
    st.session_state.render_time_total = 0.0
    st.session_state.active_learning_rate = lr
    st.session_state.active_batch_size = batch_label
    st.session_state.active_max_epochs = max_epochs
    st.session_state.active_seed = FIXED_SHUFFLE_SEED


def request_interrupt() -> None:
    if not st.session_state.is_training:
        return
    # 既存のgeneratorはそのまま進行させ、次のエポック境界で
    # interrupt_checkにより自然にmanual_interruptとして終了させる。
    st.session_state.stop_requested = True


def reset_all() -> None:
    # active_run_id を進めることで、リセット直後にfragmentが古いgeneratorの
    # 結果を反映してしまうことを防ぐ。
    st.session_state.active_run_id += 1
    st.session_state.is_training = False
    st.session_state.gen = None
    st.session_state.gen_run_id = None
    st.session_state.trainer = None
    st.session_state.stop_requested = False
    st.session_state.reason = None
    st.session_state.error_message = None
    st.session_state.epoch = 0
    st.session_state.n_updates = 0
    st.session_state.loss_history = []
    st.session_state.run_start_time = None
    st.session_state.wall_time_total = 0.0
    st.session_state.model_time_total = 0.0
    st.session_state.render_time_total = 0.0
    st.session_state.active_learning_rate = None
    st.session_state.active_batch_size = None
    st.session_state.active_max_epochs = None
    st.session_state.active_seed = None
    st.session_state.param_lr = DEFAULT_LR
    st.session_state.param_batch_label = DEFAULT_BATCH_LABEL
    st.session_state.param_max_epochs = DEFAULT_MAX_EPOCHS


init_state()

try:
    _bundle_check = get_bundle()
except Exception as e:  # データ読み込み自体の失敗はアプリ全体を落とさず明示する
    st.title("勾配降下法ハンズオン")
    st.error(f"データの読み込みに失敗しました。管理者に連絡してください。詳細: {e}")
    st.stop()

st.title("勾配降下法ハンズオン")

st.markdown(
    """
**目的**: 学習率とミニバッチサイズを変えながら、勾配降下法の学習速度・安定性・収束の違いを体験します。

**タスク**: 建物の設計情報(コンパクトさ・面積・高さ・窓の面積など)から、暖房負荷(Heating Load)を線形回帰で予測します。

**データについて**: UCI Machine Learning Repository の "Energy Efficiency" データセット
(Tsanas & Xifara、ライセンス: CC BY 4.0)を使用しています。データはこのアプリに同梱済みのため、
参加者がダウンロードする必要はありません。
([データセットのページ](https://archive.ics.uci.edu/dataset/242/energy+efficiency))
"""
)

st.divider()

c1, c2, c3 = st.columns(3)
with c1:
    st.selectbox(
        "学習率", LR_OPTIONS, key="param_lr", disabled=st.session_state.is_training
    )
with c2:
    st.selectbox(
        "ミニバッチサイズ",
        BATCH_OPTIONS,
        key="param_batch_label",
        disabled=st.session_state.is_training,
    )
with c3:
    st.selectbox(
        "最大エポック数",
        EPOCH_OPTIONS,
        key="param_max_epochs",
        disabled=st.session_state.is_training,
    )

b1, b2, b3 = st.columns(3)
with b1:
    st.button(
        "学習開始",
        on_click=start_training,
        disabled=st.session_state.is_training,
        use_container_width=True,
    )
with b2:
    st.button(
        "中断",
        on_click=request_interrupt,
        disabled=not st.session_state.is_training,
        use_container_width=True,
    )
with b3:
    st.button("初期状態に戻す", on_click=reset_all, use_container_width=True)

st.divider()


def render_status() -> None:
    bundle = get_bundle()
    reason = st.session_state.reason
    if reason:
        reason_label = REASON_LABELS.get(reason, reason)
    elif st.session_state.is_training:
        reason_label = "実行中"
    else:
        reason_label = "未実行"

    active_max_epochs = st.session_state.active_max_epochs
    progress_ratio = (
        min(1.0, st.session_state.epoch / active_max_epochs) if active_max_epochs else 0.0
    )
    st.progress(progress_ratio, text=f"進捗状況: {st.session_state.epoch} / {active_max_epochs or '-'} エポック")

    if st.session_state.active_learning_rate is not None:
        st.caption(
            f"今回の実行条件: 学習率={st.session_state.active_learning_rate}, "
            f"ミニバッチサイズ={st.session_state.active_batch_size}, "
            f"最大エポック数={st.session_state.active_max_epochs}, "
            f"乱数シード={st.session_state.active_seed}"
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("完了エポック数", st.session_state.epoch)
    m2.metric("パラメータ更新回数", st.session_state.n_updates)
    m3.metric("経過時間(秒)", f"{st.session_state.wall_time_total:.2f}")
    m4.metric("終了理由", reason_label)

    if reason == "error" and st.session_state.error_message:
        st.error(f"学習中にエラーが発生しました: {st.session_state.error_message}")

    trainer = st.session_state.trainer
    train_mse = st.session_state.loss_history[-1] if st.session_state.loss_history else None
    test_mse = test_rmse = None
    pred_test_orig = None
    if trainer is not None:
        pred_test_orig = inverse_transform_y(
            trainer.predict_std(bundle.X_test_std), bundle.y_mean, bundle.y_scale
        )
        test_mse = float(np.mean((pred_test_orig - bundle.y_test_orig) ** 2))
        test_rmse = float(np.sqrt(test_mse))

    n1, n2, n3 = st.columns(3)
    n1.metric("学習データの損失 (Train MSE)", f"{train_mse:.3f}" if train_mse is not None else "—")
    n2.metric("テストデータのMSE (Test MSE)", f"{test_mse:.3f}" if test_mse is not None else "—")
    n3.metric("テストデータのRMSE (Test RMSE)", f"{test_rmse:.3f}" if test_rmse is not None else "—")
    st.caption(
        "Test MSE: 小さいほどよい。 "
        "Test RMSE: 予測が実際の値からどの程度ずれているかの目安(Heating Loadと同じ単位)。"
    )

    # 学習中か終了後かに関わらず、loss_historyが残っている限り常に描画する。
    # (学習中/終了後で描画有無を切り替えると、フラグメントの出力がその回の
    # 実行内容だけに置き換わるため、描画しなかった回にグラフが消えてしまう)
    if st.session_state.loss_history:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(range(1, len(st.session_state.loss_history) + 1), st.session_state.loss_history)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train MSE")
        ax.set_title("Training Loss")
        st.pyplot(fig)
        plt.close(fig)

        if pred_test_orig is not None:
            fig2, ax2 = plt.subplots(figsize=(4.5, 4.5))
            ax2.scatter(bundle.y_test_orig, pred_test_orig, alpha=0.6, s=15)
            lo = float(min(bundle.y_test_orig.min(), pred_test_orig.min()))
            hi = float(max(bundle.y_test_orig.max(), pred_test_orig.max()))
            ax2.plot([lo, hi], [lo, hi], "r--", linewidth=1)
            ax2.set_xlabel("Actual Heating Load")
            ax2.set_ylabel("Predicted Heating Load")
            ax2.set_title("Actual vs Predicted (Test Set)")
            st.pyplot(fig2)
            plt.close(fig2)

    with st.expander("開発・計測情報"):
        st.write(f"モデル計算時間(合計): {st.session_state.model_time_total:.4f} 秒")
        st.write(f"画面更新(グラフ描画等)時間(合計): {st.session_state.render_time_total:.4f} 秒")
        st.write(f"全体の経過時間: {st.session_state.wall_time_total:.4f} 秒")
        st.write(f"CHUNK_EPOCHS = {CHUNK_EPOCHS}, TICK_INTERVAL_SEC = {TICK_INTERVAL_SEC}")
        st.write(f"Streamlitバージョン: {st.__version__}")


def _advance_training_chunk() -> bool:
    """is_trainingの場合のみ1チャンク分学習を進める。学習を終了させたらTrueを返す。"""
    run_id = st.session_state.active_run_id
    if not (
        st.session_state.is_training
        and st.session_state.gen is not None
        and st.session_state.gen_run_id == run_id
    ):
        return False

    just_finished = False
    try:
        for _ in range(CHUNK_EPOCHS):
            t0 = time.perf_counter()
            log = next(st.session_state.gen)
            st.session_state.model_time_total += time.perf_counter() - t0

            st.session_state.epoch = log.epoch
            st.session_state.n_updates = log.n_updates

            bundle = get_bundle()
            pred_train_orig = inverse_transform_y(
                st.session_state.trainer.predict_std(bundle.X_train_std),
                bundle.y_mean,
                bundle.y_scale,
            )
            train_mse = float(np.mean((pred_train_orig - bundle.y_train_orig) ** 2))
            st.session_state.loss_history.append(train_mse)

            if log.reason is not None:
                st.session_state.reason = log.reason
                just_finished = True
                break
    except StopIteration:
        just_finished = True
    except Exception as e:  # 例外発生時もアプリ全体を落とさない
        st.session_state.reason = "error"
        st.session_state.error_message = str(e)
        just_finished = True

    if just_finished:
        st.session_state.is_training = False
        st.session_state.gen = None
        st.session_state.gen_run_id = None
        st.session_state.stop_requested = False

    return just_finished


# is_trainingの間だけ自動tickを有効にする。フルスクリプト実行のたびに
# training_fragment()はこのデコレータごと再定義されるため、is_trainingが
# Falseになった状態でのフルスクリプト実行(終了時のst.rerun()後、次のボタン
# 操作、ページ再読み込み)では run_every=None で再定義され、以後は自動tickが
# 止まる。学習中はフラグメント単体の自動再実行が繰り返されるだけなので、
# ここで一度決まった run_every=TICK_INTERVAL_SEC がそのまま使われ続ける。
_refresh_interval = TICK_INTERVAL_SEC if st.session_state.is_training else None


@st.fragment(run_every=_refresh_interval)
def training_fragment() -> None:
    # A. 学習中の場合のみ、次のチャンクを進める
    just_finished = _advance_training_chunk()

    # 経過時間は学習中、および終了を検知したその回までのみ更新する。
    # (is_trainingがFalseになった後も毎tick加算し続けると、
    #  終了後も表示上の経過時間が増え続けてしまうため)
    if st.session_state.run_start_time is not None and (
        st.session_state.is_training or just_finished
    ):
        st.session_state.wall_time_total = time.perf_counter() - st.session_state.run_start_time

    if just_finished:
        # 「学習開始」ボタンなど、fragmentの外側にあるウィジェットのdisabled状態を
        # 更新するには、アプリ全体の再実行が必要になる。is_trainingは既にFalseに
        # なっているため、再実行後にこのfragmentへ戻ってきても学習を進める分岐には
        # 入らず、1回限りで無限rerunにはならない。
        st.rerun()

    # B. 学習中か終了後かに関わらず、保存されている結果を常に描画する
    t_render0 = time.perf_counter()
    render_status()
    st.session_state.render_time_total += time.perf_counter() - t_render0


training_fragment()
