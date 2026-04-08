#!/usr/bin/env python3
"""VLM 映像リアルタイムキャプション GUI — Orin Nano 向け

PySide6 でカメラ映像を表示しつつ、
gemma4:e2b (Ollama) で日本語キャプションを生成する。
"""

import base64
import json as _json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import httpx
import numpy as np
from PySide6.QtCore import QThread, Signal, Qt, QSize
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QProgressBar, QPushButton, QSplitter, QVBoxLayout, QWidget,
)

# ===== 設定 =====
CAMERA_SOURCE = 0
DEFAULT_VLM_MODEL = "gemma4:e2b"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_BASE_URL = "http://localhost:11434"
VLM_IMAGE_MAX_SIZE = 640           # VLM 入力画像の最大辺 (小さいほど高速)
VLM_PROMPT_JA = "この画像に映っている状況を日本語で1-2文で簡潔に説明してください。"
VLM_PROMPT_EN = "Describe what is shown in this image in 1-2 concise sentences."

# 日本語プロンプトに対応していないモデル
EN_ONLY_MODELS = {"moondream"}

# VLM 対応モデル候補 (Ollama にインストール済みのもののみ表示)
VLM_CANDIDATES = [
    "moondream",
    "gemma4:e2b",
]

HISTORY_DIR = Path(__file__).parent / "history"
HISTORY_DIR.mkdir(exist_ok=True)


# ===== カメラスレッド =====
class CameraThread(QThread):
    frame_ready = Signal(np.ndarray)

    def __init__(self, source=CAMERA_SOURCE):
        super().__init__()
        self.source = source
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            print(f"ERROR: カメラ {self.source} を開けません")
            return
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {w}x{h}")

        while self.running:
            ret, frame = cap.read()
            if ret:
                self.frame_ready.emit(frame)
            self.msleep(33)  # ~30fps

        cap.release()

    def stop(self):
        self.running = False
        self.wait()


# ===== VLM スレッド =====
class VLMThread(QThread):
    caption_ready = Signal(str, float)  # (caption, elapsed_seconds)
    status_changed = Signal(str)
    loading_progress = Signal(int, str)  # (percent, message)
    snapshot_taken = Signal(np.ndarray)  # VLM に送ったフレーム (リサイズ後)
    warmup_done = Signal()
    model_switched = Signal(str)        # 切替完了通知 (model_name)

    def __init__(self, model: str = DEFAULT_VLM_MODEL):
        super().__init__()
        self.running = True
        self.current_frame = None
        self._request_pending = False
        self._switch_request = None     # 切替先モデル名
        self._ready = False
        self.model = model

    def set_frame(self, frame: np.ndarray):
        self.current_frame = frame

    def request_caption(self):
        if self._ready:
            self._request_pending = True

    def request_switch(self, new_model: str):
        """モデル切替をリクエスト (待機中に処理される)"""
        self._switch_request = new_model

    def _warmup(self, model: str) -> bool:
        self._ready = False
        self.status_changed.emit(f"{model} をロード中...")
        self.loading_progress.emit(0, f"{model} に接続中...")
        t_start = time.time()
        try:
            # 旧モデルをアンロード
            if self.model != model:
                try:
                    httpx.post(
                        OLLAMA_URL,
                        json={"model": self.model, "keep_alive": 0},
                        timeout=10.0,
                    )
                except Exception:
                    pass

            # モデルをロードするだけ (推論はしない)
            self.loading_progress.emit(5, f"{model} をロード中...")
            with httpx.Client(timeout=300.0) as client:
                resp = client.post(
                    f"{OLLAMA_BASE_URL}/api/chat",
                    json={"model": model, "messages": [], "keep_alive": -1},
                )
                if resp.status_code == 200:
                    self.loading_progress.emit(100, "ロード完了")
                else:
                    raise RuntimeError(f"HTTP {resp.status_code}")
            self.model = model
            print(f"VLM warmup done: {model} in {time.time() - t_start:.1f}s")
            return True
        except Exception as e:
            print(f"VLM warmup failed: {e}")
            self.status_changed.emit(f"ロード失敗: {e}")
            self.loading_progress.emit(0, f"エラー: {e}")
            return False

    def run(self):
        if not self._warmup(self.model):
            return

        self.loading_progress.emit(100, "準備完了")
        self._ready = True
        self.warmup_done.emit()
        self.status_changed.emit("準備完了 — ボタンでキャプション開始")

        while self.running:
            # モデル切替リクエストの処理
            if self._switch_request:
                new_model = self._switch_request
                self._switch_request = None
                if new_model != self.model:
                    if self._warmup(new_model):
                        self._ready = True
                        self.warmup_done.emit()
                        self.model_switched.emit(self.model)
                        self.status_changed.emit(f"{self.model} 準備完了 — ボタンでキャプション開始")
                    else:
                        # 失敗時は旧モデルに戻す
                        self._warmup(self.model)
                        self._ready = True
                        self.warmup_done.emit()
                continue

            if not self._request_pending or self.current_frame is None:
                self.msleep(100)
                continue

            self._request_pending = False
            frame = self.current_frame.copy()

            h, w = frame.shape[:2]
            scale = VLM_IMAGE_MAX_SIZE / max(h, w)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(frame, (new_w, new_h))
            else:
                resized = frame

            self.snapshot_taken.emit(resized.copy())
            self.status_changed.emit(f"VLM 処理中... [{self.model}] ({resized.shape[1]}x{resized.shape[0]})")

            t0 = time.time()
            try:
                caption = self._query_vlm(resized)
                t_elapsed = time.time() - t0
                if caption:
                    print(f"📝 [{self.model}] [{t_elapsed:.1f}s] {caption}")
                    self.caption_ready.emit(caption, t_elapsed)
                else:
                    print(f"⚠ [{self.model}] empty response ({t_elapsed:.1f}s)")
                    self.caption_ready.emit("(応答なし — モデルが画像を処理できませんでした)", t_elapsed)
            except Exception as e:
                print(f"VLM error: {e}")
                self.caption_ready.emit(f"エラー: {e}", 0.0)
            finally:
                self.status_changed.emit(f"待機中 [{self.model}] — ボタンでキャプション開始")

    def _query_vlm(self, frame: np.ndarray) -> str:
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        image_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
        prompt = VLM_PROMPT_EN if self.model in EN_ONLY_MODELS else VLM_PROMPT_JA
        resp = httpx.post(
            OLLAMA_URL,
            json={
                "model": self.model,
                "prompt": prompt,
                "images": [image_b64],
                "stream": False,
            },
            timeout=300.0,
        )
        return resp.json().get("response", "").strip()

    def stop(self):
        self.running = False
        self.wait()


# ===== ヘルパー =====
def frame_to_pixmap(frame: np.ndarray, target_size) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(img).scaled(
        target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
    )


def frame_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()


# ===== 履歴エントリ =====
class HistoryEntry:
    def __init__(self, timestamp: str, image_path: str, caption: str, elapsed: float):
        self.timestamp = timestamp
        self.image_path = image_path
        self.caption = caption
        self.elapsed = elapsed


# ===== メインウィンドウ =====
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VLM Real-time Caption")
        self.setMinimumSize(1200, 500)

        self.history: list[HistoryEntry] = []
        self.current_snapshot_frame = None  # 最新の VLM 入力フレーム (numpy)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ===== 左: カメラストリーミング映像 =====
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        left_header = QLabel("Live Camera")
        left_header.setAlignment(Qt.AlignCenter)
        left_header.setStyleSheet(
            "background: #1a1a1a; color: #4CAF50; font-size: 16px; "
            "font-weight: bold; padding: 6px;"
        )
        left_layout.addWidget(left_header)

        self.video_label = QLabel("カメラ起動中...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: black; color: #888;")
        left_layout.addWidget(self.video_label, stretch=1)

        # ===== 中央: スナップショット + キャプション =====
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        # VLM 入力スナップショット
        snap_header = QLabel("VLM Input")
        snap_header.setAlignment(Qt.AlignCenter)
        snap_header.setStyleSheet(
            "background: #1a1a1a; color: #FF9800; font-size: 12px; "
            "font-weight: bold; padding: 4px;"
        )
        center_layout.addWidget(snap_header)

        self.snapshot_label = QLabel("VLM 入力待ち...")
        self.snapshot_label.setAlignment(Qt.AlignCenter)
        self.snapshot_label.setStyleSheet("background: #0a0a0a; color: #888;")
        center_layout.addWidget(self.snapshot_label, stretch=1)

        # キャプション + ボタン + ステータス
        info_widget = QWidget()
        info_layout = QVBoxLayout(info_widget)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(0)

        self.caption_header = QLabel(f"Caption [{DEFAULT_VLM_MODEL}]")
        self.caption_header.setAlignment(Qt.AlignCenter)
        self.caption_header.setStyleSheet(
            "background: #1a1a1a; color: #2196F3; font-size: 16px; "
            "font-weight: bold; padding: 4px; border-top: 1px solid #333;"
        )
        info_layout.addWidget(self.caption_header)

        self.caption_label = QLabel("起動中...")
        self.caption_label.setAlignment(Qt.AlignCenter)
        self.caption_label.setWordWrap(True)
        self.caption_label.setStyleSheet(
            "background: #1a1a1a; color: white; font-size: 20px; padding: 14px;"
        )
        info_layout.addWidget(self.caption_label, stretch=1)

        # モデル選択
        model_row = QHBoxLayout()
        model_row.setContentsMargins(4, 4, 4, 4)
        model_row.setSpacing(4)

        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet(
            "QComboBox { background: #222; color: white; padding: 6px; border: 1px solid #555; font-size: 14px; }"
            "QComboBox:disabled { color: #666; }"
            "QComboBox QAbstractItemView { background: #222; color: white; selection-background-color: #555; selection-color: white; }"
        )
        self.model_combo.setEnabled(False)
        self._populate_models()
        model_row.addWidget(self.model_combo, stretch=1)

        self.switch_btn = QPushButton("切替")
        self.switch_btn.setEnabled(False)
        self.switch_btn.setStyleSheet(
            "QPushButton { background: #FF9800; color: white; font-size: 14px; "
            "font-weight: bold; padding: 8px 14px; border: none; }"
            "QPushButton:hover { background: #F57C00; }"
            "QPushButton:disabled { background: #555; color: #999; }"
        )
        self.switch_btn.clicked.connect(self.on_switch_model)
        model_row.addWidget(self.switch_btn)
        info_layout.addLayout(model_row)

        # キャプションボタン
        self.caption_btn = QPushButton("キャプション生成")
        self.caption_btn.setEnabled(False)
        self.caption_btn.setStyleSheet(
            "QPushButton { background: #4CAF50; color: white; font-size: 18px; "
            "font-weight: bold; padding: 12px; border: none; }"
            "QPushButton:hover { background: #45a049; }"
            "QPushButton:disabled { background: #555; color: #999; }"
            "QPushButton:pressed { background: #388E3C; }"
        )
        self.caption_btn.clicked.connect(self.on_caption_click)
        info_layout.addWidget(self.caption_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet(
            "QProgressBar { background: #222; border: none; height: 22px; color: white; font-size: 13px; }"
            "QProgressBar::chunk { background: #4CAF50; }"
        )
        info_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("接続中...")
        self.status_label.setStyleSheet(
            "background: #111; color: #888; font-size: 14px; padding: 6px 10px;"
        )
        info_layout.addWidget(self.status_label)

        center_layout.addWidget(info_widget, stretch=1)

        # ===== 右: 履歴リスト =====
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        history_header = QLabel("History")
        history_header.setAlignment(Qt.AlignCenter)
        history_header.setStyleSheet(
            "background: #1a1a1a; color: #9C27B0; font-size: 16px; "
            "font-weight: bold; padding: 6px;"
        )
        right_layout.addWidget(history_header)

        self.history_list = QListWidget()
        self.history_list.setStyleSheet(
            "QListWidget { background: #111; color: white; border: none; font-size: 14px; }"
            "QListWidget::item { padding: 6px; border-bottom: 1px solid #333; }"
            "QListWidget::item:selected { background: #333; }"
            "QListWidget::item:hover { background: #222; }"
        )
        self.history_list.setIconSize(QSize(80, 60))
        self.history_list.currentRowChanged.connect(self.on_history_select)
        right_layout.addWidget(self.history_list, stretch=1)

        # ===== スプリッタで3分割 =====
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(center_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 400, 200])
        splitter.setStyleSheet("QSplitter::handle { background: #333; width: 2px; }")
        root_layout.addWidget(splitter)

        # ===== 既存の履歴を読み込み =====
        self._load_history()

        # ===== スレッド起動 =====
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_frame)
        self.camera_thread.start()

        self.vlm_thread = VLMThread()
        self.vlm_thread.caption_ready.connect(self.update_caption)
        self.vlm_thread.status_changed.connect(self.update_status)
        self.vlm_thread.loading_progress.connect(self.update_progress)
        self.vlm_thread.snapshot_taken.connect(self.update_snapshot)
        self.vlm_thread.warmup_done.connect(self.on_warmup_done)
        self.vlm_thread.model_switched.connect(self.on_model_switched)
        self.vlm_thread.start()

    # ----- 履歴の保存/読み込み -----
    def _save_history_entry(self, frame: np.ndarray, caption: str, elapsed: float):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"{ts}.jpg"
        img_path = HISTORY_DIR / img_name
        cv2.imwrite(str(img_path), frame)

        entry = HistoryEntry(
            timestamp=ts,
            image_path=str(img_path),
            caption=caption,
            elapsed=elapsed,
        )
        self.history.append(entry)

        # メタデータを JSON で保存
        meta_path = HISTORY_DIR / f"{ts}.json"
        meta_path.write_text(_json.dumps({
            "timestamp": ts,
            "image": img_name,
            "caption": caption,
            "elapsed": elapsed,
            "model": self.vlm_thread.model if hasattr(self, 'vlm_thread') else DEFAULT_VLM_MODEL,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        self._add_history_item(entry)

    def _load_history(self):
        """起動時に既存の履歴を読み込む"""
        json_files = sorted(HISTORY_DIR.glob("*.json"))
        for jf in json_files:
            try:
                data = _json.loads(jf.read_text(encoding="utf-8"))
                img_path = HISTORY_DIR / data["image"]
                if not img_path.exists():
                    continue
                entry = HistoryEntry(
                    timestamp=data["timestamp"],
                    image_path=str(img_path),
                    caption=data["caption"],
                    elapsed=data.get("elapsed", 0),
                )
                self.history.append(entry)
                self._add_history_item(entry)
            except Exception:
                continue

    def _add_history_item(self, entry: HistoryEntry):
        """履歴リストに1項目追加"""
        # タイムスタンプをフォーマット
        try:
            dt = datetime.strptime(entry.timestamp, "%Y%m%d_%H%M%S")
            time_str = dt.strftime("%H:%M:%S")
        except ValueError:
            time_str = entry.timestamp

        # サムネイル
        pixmap = QPixmap(entry.image_path).scaled(
            80, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        caption_short = entry.caption[:30] + ("..." if len(entry.caption) > 30 else "")
        item = QListWidgetItem(QIcon(pixmap), f"{time_str}\n{caption_short}")
        item.setData(Qt.UserRole, len(self.history) - 1)  # インデックス
        self.history_list.addItem(item)
        self.history_list.scrollToBottom()

    # ----- 履歴クリック -----
    def on_history_select(self, row: int):
        if row < 0:
            return
        item = self.history_list.item(row)
        idx = item.data(Qt.UserRole)
        if idx is None or idx >= len(self.history):
            return
        entry = self.history[idx]

        # スナップショットエリアに画像を表示
        frame = cv2.imread(entry.image_path)
        if frame is not None:
            pixmap = frame_to_pixmap(frame, self.snapshot_label.size())
            self.snapshot_label.setPixmap(pixmap)

        # キャプションエリアに表示
        self.caption_label.setText(entry.caption)
        self.status_label.setText(
            f"履歴表示: {entry.timestamp} ({entry.elapsed:.1f}s)"
        )

    # ----- スロット -----
    def update_frame(self, frame: np.ndarray):
        self.vlm_thread.set_frame(frame)
        pixmap = frame_to_pixmap(frame, self.video_label.size())
        self.video_label.setPixmap(pixmap)

    def update_snapshot(self, frame: np.ndarray):
        self.current_snapshot_frame = frame.copy()
        pixmap = frame_to_pixmap(frame, self.snapshot_label.size())
        self.snapshot_label.setPixmap(pixmap)

    def _populate_models(self):
        """Ollama にインストール済みのビジョンモデルを列挙"""
        models = []
        try:
            resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
            installed = {}
            for m in resp.json().get("models", []):
                name = m["name"]
                # "moondream:latest" → "moondream" に正規化
                short = name.rsplit(":latest", 1)[0] if name.endswith(":latest") else name
                installed[short] = name
            # 候補リストにあるものを順番に追加
            for c in VLM_CANDIDATES:
                if c in installed:
                    models.append(c)
        except Exception:
            pass
        if not models:
            models = [DEFAULT_VLM_MODEL]
        self.model_combo.clear()
        for m in models:
            self.model_combo.addItem(m)
        idx = self.model_combo.findText(DEFAULT_VLM_MODEL)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

    def on_warmup_done(self):
        self.caption_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.switch_btn.setEnabled(True)
        self.caption_header.setText(f"Caption [{self.vlm_thread.model}]")

    def on_switch_model(self):
        new_model = self.model_combo.currentText()
        if new_model == self.vlm_thread.model:
            return
        self.caption_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.switch_btn.setEnabled(False)
        self.progress_bar.show()
        self.vlm_thread.request_switch(new_model)

    def on_model_switched(self, model_name: str):
        idx = self.model_combo.findText(model_name)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
        self.caption_header.setText(f"Caption [{model_name}]")

    def on_caption_click(self):
        self.caption_btn.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.switch_btn.setEnabled(False)
        self.caption_label.setText("推論中...")
        self.history_list.clearSelection()
        self.vlm_thread.request_caption()

    def update_progress(self, pct: int, message: str):
        self.progress_bar.setValue(pct)
        self.progress_bar.setFormat(message)
        self.caption_label.setText(message)
        if pct >= 100:
            self.progress_bar.hide()

    def update_caption(self, caption: str, elapsed: float):
        self.caption_label.setText(caption)
        self.status_label.setText(f"待機中 [{self.vlm_thread.model}] — 最終更新: {time.strftime('%H:%M:%S')} ({elapsed:.1f}s)")
        self.caption_btn.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.switch_btn.setEnabled(True)

        # 履歴に保存
        if self.current_snapshot_frame is not None:
            try:
                self._save_history_entry(self.current_snapshot_frame, caption, elapsed)
            except Exception as e:
                print(f"History save error: {e}")

    def update_status(self, status: str):
        self.status_label.setText(status)

    def closeEvent(self, event):
        self.camera_thread.stop()
        model = self.vlm_thread.model
        self.vlm_thread.stop()
        try:
            httpx.post(OLLAMA_URL, json={"model": model, "keep_alive": 0}, timeout=5.0)
        except Exception:
            pass
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
