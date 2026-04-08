#!/usr/bin/env python3
"""VLM 映像リアルタイムキャプション — Orin Nano 向け

カメラ映像を MJPEG でブラウザに配信しつつ、
定期的に gemma4:e2b (Ollama) で日本語キャプションを生成する。
"""

import asyncio
import base64
import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

# ===== 設定 =====
CAMERA_SOURCE = 0                # カメラID (USB カメラ)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
VLM_MODEL = "gemma4:e2b"
OLLAMA_URL = "http://localhost:11434/api/generate"
MIN_VLM_INTERVAL = 5.0           # VLM 最小呼出間隔 (秒)
MAX_VLM_INTERVAL = 15.0          # 変化なくても強制更新する間隔 (秒)
CHANGE_THRESHOLD = 0.05          # フレーム間変化の閾値 (0-1)
JPEG_QUALITY = 70                # MJPEG 配信品質
VLM_JPEG_QUALITY = 80            # VLM に渡す画像品質

VLM_PROMPT = "この画像に映っている状況を日本語で1-2文で簡潔に説明してください。"

# ===== グローバル状態 =====
latest_frame = None              # 最新フレーム (numpy array)
latest_jpeg = None               # 最新 JPEG バイト列
latest_caption = "起動中..."
vlm_processing = False
last_vlm_time = 0.0
prev_vlm_frame = None            # 前回 VLM に渡したフレーム
connected_clients: list[WebSocket] = []


# ===== カメラキャプチャループ =====
async def camera_loop():
    global latest_frame, latest_jpeg

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        print(f"ERROR: カメラ {CAMERA_SOURCE} を開けません")
        return

    print(f"Camera opened: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue

            latest_frame = frame
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            latest_jpeg = buf.tobytes()

            await asyncio.sleep(0.03)  # ~30fps
    finally:
        cap.release()


# ===== フレーム間変化検出 =====
def frame_changed(current: np.ndarray, previous: np.ndarray | None) -> bool:
    if previous is None:
        return True

    # リサイズして高速比較
    small_curr = cv2.resize(current, (160, 120))
    small_prev = cv2.resize(previous, (160, 120))

    # グレースケールで差分
    gray_curr = cv2.cvtColor(small_curr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_prev = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    diff = np.mean(np.abs(gray_curr - gray_prev))
    return diff > CHANGE_THRESHOLD


# ===== VLM 呼び出し =====
async def query_vlm(frame: np.ndarray) -> str:
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, VLM_JPEG_QUALITY])
    image_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(OLLAMA_URL, json={
            "model": VLM_MODEL,
            "prompt": VLM_PROMPT,
            "images": [image_b64],
            "stream": False,
        })
        result = response.json()
        return result.get("response", "").strip()


# ===== VLM キャプションループ =====
async def vlm_loop():
    global latest_caption, vlm_processing, last_vlm_time, prev_vlm_frame

    # カメラが起動するまで待つ
    while latest_frame is None:
        await asyncio.sleep(0.5)

    print("VLM caption loop started")

    while True:
        now = time.time()
        elapsed = now - last_vlm_time
        frame = latest_frame

        if frame is None:
            await asyncio.sleep(1.0)
            continue

        # VLM 呼び出し判定
        should_query = False
        if elapsed >= MAX_VLM_INTERVAL:
            should_query = True  # 強制更新
        elif elapsed >= MIN_VLM_INTERVAL and frame_changed(frame, prev_vlm_frame):
            should_query = True  # 変化検出

        if not should_query:
            await asyncio.sleep(0.5)
            continue

        vlm_processing = True
        t0 = time.time()
        try:
            caption = await query_vlm(frame)
            t_elapsed = time.time() - t0
            if caption:
                latest_caption = caption
                print(f"📝 [{t_elapsed:.1f}s] {caption}")
                await broadcast_caption()
        except Exception as e:
            import traceback
            print(f"VLM error: {e}")
            traceback.print_exc()
        finally:
            vlm_processing = False
            last_vlm_time = time.time()
            prev_vlm_frame = frame.copy()


# ===== WebSocket ブロードキャスト =====
async def broadcast_caption():
    msg = json.dumps({
        "caption": latest_caption,
        "processing": vlm_processing,
        "timestamp": time.time(),
    })
    disconnected = []
    for ws in connected_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        connected_clients.remove(ws)


# ===== MJPEG ストリーム =====
async def mjpeg_generator():
    while True:
        if latest_jpeg is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + latest_jpeg
                + b"\r\n"
            )
        await asyncio.sleep(0.1)  # ~10fps でブラウザに配信


# ===== FastAPI =====
async def warmup_vlm():
    """初回モデルロードを事前に行う"""
    global latest_caption
    latest_caption = "VLM モデルをロード中..."
    print(f"Warming up {VLM_MODEL}...")
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                "http://localhost:11434/api/generate",
                json={"model": VLM_MODEL, "prompt": "hi", "stream": False, "keep_alive": -1},
            )
            print(f"VLM warmup done: {response.status_code}")
            latest_caption = "準備完了 — キャプション待ち..."
    except Exception as e:
        print(f"VLM warmup failed: {e}")
        latest_caption = "VLM ウォームアップ失敗"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cam_task = asyncio.create_task(camera_loop())
    await warmup_vlm()
    vlm_task = asyncio.create_task(vlm_loop())
    yield
    cam_task.cancel()
    vlm_task.cancel()

app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)

    # 接続時に現在のキャプションを送信
    await websocket.send_text(json.dumps({
        "caption": latest_caption,
        "processing": vlm_processing,
        "timestamp": time.time(),
    }))

    try:
        while True:
            await websocket.receive_text()  # クライアントからの ping 待ち
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    # stdout をアンバッファにする
    sys.stdout.reconfigure(line_buffering=True)
    uvicorn.run(app, host="0.0.0.0", port=8080)
