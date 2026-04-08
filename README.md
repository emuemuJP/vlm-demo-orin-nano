# VLM Real-time Caption for Jetson Orin Nano

NVIDIA Jetson Orin Nano (8GB) に最適化された映像リアルタイムキャプションデモ。

> **Note**: 本プロジェクトは Jetson Orin Nano (8GB, JetPack 6.x) の統合メモリ環境に特化して設計されています。

## 構成

| コンポーネント | 実装 | 実行先 | 備考 |
|---|---|---|---|
| カメラ | USB カメラ (OpenCV) | CPU | 1280x720, 30fps |
| VLM | gemma4:e2b (Ollama) | GPU | 実効 2.3B パラメータ, Q4_K_M |
| 配信 | FastAPI + MJPEG + WebSocket | CPU | ブラウザで映像+キャプション表示 |

### メモリ配分 (8GB 統合メモリ)

```
┌─────────────────────────────────┐
│ GPU                             │
│  gemma4:e2b Q4     : ~4.0GB    │
├─────────────────────────────────┤
│ CPU/RAM                         │
│  OpenCV + FastAPI  : ~0.3GB    │
│  OS + システム     : ~1.5GB    │
├─────────────────────────────────┤
│ 空き / バッファ    : ~2.2GB    │
└─────────────────────────────────┘
```

## 動作環境

- NVIDIA Jetson Orin Nano (8GB)
- JetPack 6.x (L4T R36)
- Python 3.10+
- USB カメラ

## セットアップ

### 1. Jetson パフォーマンスモード

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

### 2. Ollama + VLM モデル

```bash
# Ollama インストール (未インストールの場合)
curl -fsSL https://ollama.com/install.sh | sh

# gemma4:e2b のダウンロード (7.2GB)
ollama pull gemma4:e2b

# 動作確認
ollama run gemma4:e2b "こんにちは"
```

### 3. Python パッケージ

```bash
pip3 install fastapi uvicorn httpx opencv-python-headless --break-system-packages
```

## 使い方

```bash
cd /mnt/m2ssd/vlm-demo-orin-nano
python3 server.py
```

ブラウザで `http://<orin-nano-ip>:8080` にアクセス。

- 上部: カメラ映像 (MJPEG, ~10fps)
- 下部: VLM による日本語キャプション (5-15秒間隔で更新)
- ステータスバー: VLM 処理中/待機中の表示

## 設定

`server.py` 先頭の定数で調整できます:

```python
CAMERA_SOURCE = 0            # カメラID or 動画ファイルパス
VLM_MODEL = "gemma4:e2b"    # Ollama モデル名
MIN_VLM_INTERVAL = 5.0      # VLM 最小呼出間隔 (秒)
MAX_VLM_INTERVAL = 15.0     # 強制更新間隔 (秒)
CHANGE_THRESHOLD = 0.05     # フレーム間変化の閾値
```

## アーキテクチャ

```
USB カメラ (OpenCV, 30fps)
    │
    ├── 全フレーム ──▶ MJPEG ストリーム ──▶ ブラウザ表示
    │
    └── N秒間隔 + 変化検出 ──▶ gemma4:e2b (Ollama API)
                                    │
                                    ▼
                              WebSocket ──▶ ブラウザにキャプション表示
```

## トラブルシューティング

### カメラが映らない

```bash
ls /dev/video*
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### VLM が応答しない

```bash
systemctl status ollama
ollama list                    # gemma4:e2b がダウンロード済みか確認
curl http://localhost:11434/   # Ollama API の疎通確認
```

### メモリ不足

VLM デモ中は他の Ollama モデル (Qwen3.5:2b 等) を停止してください:

```bash
# 不要なモデルをアンロード
curl -X POST http://localhost:11434/api/generate -d '{"model":"qwen3.5:2b","keep_alive":0}'
```

## 依存ライブラリのライセンス

| パッケージ | ライセンス | 商用利用 |
|---|---|---|
| gemma4:e2b | Apache 2.0 (Gemma) | OK |
| FastAPI | MIT | OK |
| uvicorn | BSD-3-Clause | OK |
| httpx | BSD-3-Clause | OK |
| OpenCV | Apache 2.0 | OK |
| Ollama | MIT | OK |

## ライセンス

MIT License
