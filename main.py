import os
import time

#AI local using ollama

cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
if os.path.exists(cuda_bin_path):
    os.add_dll_directory(cuda_bin_path)
os.environ["PATH"] = cuda_bin_path + os.pathsep + os.environ.get("PATH", "")

import asyncio
import json
import httpx
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from faster_whisper import WhisperModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat" # api/chat - xu ly hoi thoai #api/generate - tao van ban don thuan
OLLAMA_MODEL    = "qwen2:1.5b"
MAX_HISTORY_TURNS = 3   # nhớ tối đa 3 lượt hỏi-đáp gần nhất

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI. "
    "Chỉ trả lời NGẮN GỌN trong 1 câu, bằng tiếng Việt. "
    "Không giải thích dài dòng."
)


_http_client: httpx.AsyncClient | None = None

async def get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(timeout=60)
    return _http_client


async def ask_ollama(prompt: str, history: list[dict]) -> str:
    print(f"[LLM] Querying: {prompt!r}")
    history.append({"role": "user", "content": prompt})
    remember_segment = history[-(MAX_HISTORY_TURNS * 2):]  # dấu - thể hiện lấy từ cuối danh sách ngược lên
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + remember_segment
    client = await get_http_client()
    resp = await client.post(
        OLLAMA_CHAT_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 30,
            },
        },
    )
    data = resp.json()
    reply = data["message"]["content"].strip()
    first_sentence = reply.split('.')[0] + '.'
    history.append({"role": "assistant", "content": first_sentence})

    if len(history) > MAX_HISTORY_TURNS * 2:
        del history[: len(history) - MAX_HISTORY_TURNS * 2]

    print(f"[LLM] Reply: {first_sentence!r} | history={len(history)//2} turns")
    return first_sentence


model_path = "./models/whisper-medium-ct2"
print("Loading faster-whisper model...")
model = WhisperModel(
    model_path,
    device="cuda",
    compute_type="float16",
    num_workers=1,
    cpu_threads=0,
    download_root=None,
)
print("Model ready!")

_executor = ThreadPoolExecutor(max_workers=1)

# Ngưỡng rolling: 2 giây audio
ROLLING_THRESHOLD_BYTES = 2 * 16_000 * 2


def _transcribe_sync(raw_bytes: bytes) -> str:
    audio_np = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    segments, info = model.transcribe(
        audio_np,
        language="vi",
        beam_size=1,
        best_of=1,
        temperature=0,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
        word_timestamps=False,
    )

    text = " ".join(seg.text for seg in segments).strip()
    print(f"[STT] lang={info.language} prob={info.language_probability:.2f} | {text!r}")
    return text


async def transcribe_audio(raw_bytes: bytes) -> str: #Asyncio: "Bridging Sync and Async" (Cầu nối giữa đồng bộ và bất đồng bộ)
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _transcribe_sync, raw_bytes)


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    await websocket.accept()
    client = websocket.client
    print(f"[WS] Connected: {client}")

    audio_chunks: list[bytes] = []
    pending_tasks: set[asyncio.Task] = set()

    # Mỗi connection có history riêng — tránh lẫn giữa các ESP32
    conversation_history: list[dict] = []

    async def transcribe_and_send(raw: bytes) -> None:
        try:
            t0 = time.perf_counter()
            stt_text = await transcribe_audio(raw)

            if not stt_text:
                await websocket.send_text(
                    json.dumps({"text": "", "status": "empty"})
                )
                return
            await websocket.send_text(
                json.dumps({"stt": stt_text, "text": stt_text.lower(), "status": "ok"})
            )

        except Exception as e:
            print(f"[ERR] {e}")
            await websocket.send_text(
                json.dumps({"text": "", "status": "error", "detail": str(e)})
            )

    try:
        while True:
            message = await websocket.receive()


            if "bytes" in message and message["bytes"]:
                chunk: bytes = message["bytes"]
                audio_chunks.append(chunk)
                total = sum(len(c) for c in audio_chunks)
                print(f"[WS] Chunk {len(chunk):,}B | total {total:,}B")

                if total >= ROLLING_THRESHOLD_BYTES:
                    raw = b"".join(audio_chunks)
                    audio_chunks = []
                    task = asyncio.create_task(transcribe_and_send(raw))
                    pending_tasks.add(task)
                    task.add_done_callback(pending_tasks.discard)


            elif "text" in message:
                text = message["text"].strip()

                if text == "END":
                    if not audio_chunks:
                        if pending_tasks:
                            await asyncio.gather(*pending_tasks, return_exceptions=True)
                        await websocket.send_text(
                            json.dumps({"text": "", "status": "empty"})
                        )
                        continue

                    total_bytes = sum(len(c) for c in audio_chunks)
                    duration = total_bytes / (16_000 * 2)
                    print(f"[WS] END — {total_bytes:,}B ({duration:.1f}s), transcribing...")

                    raw = b"".join(audio_chunks)
                    audio_chunks = []

                    if pending_tasks:
                        await asyncio.gather(*pending_tasks, return_exceptions=True)

                    await transcribe_and_send(raw)

    except WebSocketDisconnect:
        print(f"[WS] Disconnected: {client}")
        for t in pending_tasks:
            t.cancel()
    except Exception as e:
        print(f"[WS] Unexpected error: {e}")
        for t in pending_tasks:
            t.cancel()

@app.post("/transcribe")
async def transcribe_http():
    return {"detail": "Not implemented"}

@app.get("/ping")
async def ping():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        ws_ping_interval=20,
        ws_ping_timeout=30,
    )