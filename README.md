# VLM Real-time Caption for Jetson Orin Nano

NVIDIA Jetson Orin Nano (8GB) に最適化された映像リアルタイムキャプションデモ。

> **Note**: 本プロジェクトは Jetson Orin Nano (8GB, JetPack 6.x) の統合メモリ環境に特化して設計されています。

## 構成

| コンポーネント | 実装 | 実行先 | 備考 |
|---|---|---|---|
| カメラ | USB カメラ (OpenCV) | CPU | 自動検出, 30fps |
| VLM | moondream / gemma4:e2b (Ollama) | GPU | GUI から切替可能 |
| GUI | PySide6 (Qt) | CPU | 3パネルレイアウト |

### 対応モデル

| モデル | サイズ | 推論時間目安 | 日本語 | 備考 |
|---|---|---|---|---|
| **moondream** (デフォルト) | 1.7GB | ~1-23秒 | 英語のみ | 軽量、swap なしで動作 |
| gemma4:e2b | 7.2GB | ~60-160秒 | OK | 高品質だが swap 発生 |

## 動作環境

- NVIDIA Jetson Orin Nano (8GB)
- JetPack 6.x (L4T R36)
- Python 3.10+
- USB カメラ
- PySide6

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

# 軽量モデル (推奨)
ollama pull moondream

# 高品質モデル (オプション、7.2GB)
ollama pull gemma4:e2b
```

### 3. Python パッケージ

```bash
pip3 install PySide6 httpx --break-system-packages
```

OpenCV はシステム版 (`python3-opencv`) を使用します。pip の `opencv-python-headless` は V4L2 カメラが動作しないため非推奨です。

## 使い方

```bash
python3 gui.py
```

### GUI レイアウト

```
┌──────────┬──────────┬────────┐
│          │ VLM Input│        │
│  Live    │ (入力画像) │ History│
│  Camera  ├──────────┤ (履歴)  │
│          │ Caption  │ サムネ  │
│          │ [モデル▼] │ +テキスト│
│          │ [生成]    │        │
└──────────┴──────────┴────────┘
```

- **左**: USB カメラのライブ映像
- **中央上**: VLM に送った入力画像 (640px にリサイズ)
- **中央下**: キャプション + モデル切替 + 生成ボタン
- **右**: 過去の推論履歴 (クリックで詳細表示)

### 操作

1. 起動するとモデルのウォームアップが始まる
2. 「キャプション生成」ボタンでその時点のフレームを VLM に送信
3. モデル切替: ドロップダウンから選択 →「切替」ボタン
4. 履歴: 右パネルのサムネイルをクリックで過去の結果を確認

### 履歴の保存

推論結果は `history/` ディレクトリに自動保存されます:
- `YYYYMMDD_HHMMSS.jpg` — VLM 入力画像
- `YYYYMMDD_HHMMSS.json` — キャプション + メタデータ

## 設定

`gui.py` 先頭の定数で調整できます:

```python
CAMERA_SOURCE = 0              # カメラID
DEFAULT_VLM_MODEL = "moondream" # デフォルトモデル
VLM_IMAGE_MAX_SIZE = 640       # VLM 入力画像の最大辺 (px)
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
ollama list    # モデルがダウンロード済みか確認
```

### メモリ不足 / 推論が遅い

gemma4:e2b (7.2GB) は 8GB 統合メモリでは swap が発生し推論が遅くなります。moondream (1.7GB) を推奨します。

```bash
# 不要なモデルをアンロード
curl -X POST http://localhost:11434/api/generate -d '{"model":"gemma4:e2b","keep_alive":0}'
```

## 依存ライブラリのライセンス

| パッケージ | ライセンス | 商用利用 |
|---|---|---|
| moondream | Apache 2.0 | OK |
| gemma4:e2b | Apache 2.0 (Gemma) | OK |
| PySide6 | LGPL-3.0 | OK |
| httpx | BSD-3-Clause | OK |
| OpenCV | Apache 2.0 | OK |
| Ollama | MIT | OK |

## ライセンス

MIT License
