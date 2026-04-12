import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
import parameter_secret

app = FastAPI()

client = genai.Client(
    api_key=parameter_secret.API_KEY,
    http_options={"api_version": "v1alpha"},
)

MODEL_ID = "models/gemini-2.5-flash-native-audio-latest"
MIME_PCM = "audio/pcm;rate=16000"

# Giống trợ lý kiểu streaming: kết thúc câu khi không còn chunk mic trong khoảng ngắn.
SILENCE_GAP_SEC = 0.30
# Không nhận được byte audio từ Gemini trong thời gian này → báo ESP32 (TTS local).
NO_FIRST_AUDIO_SEC = 3.0
TURN_COMPLETE_TIMEOUT_SEC = 15.0

SYSTEM_PROMPT = (
    "Bạn là trợ lý AI thông minh tên là Tiểu Trí chạy trên thiết bị ESP32. "
    "Nhiệm vụ: Trò chuyện tự nhiên, trả lời mọi câu hỏi của người dùng. "
    "Quy tắc: Trả lời cực ngắn dưới 20 từ, không dùng Markdown, không lặp từ."
)

LIVE_CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    system_instruction=types.Content(
        parts=[types.Part.from_text(text=SYSTEM_PROMPT)]
    ),
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
        )
    ),
)


def mono_to_stereo(pcm_bytes: bytes) -> bytes:
    if not pcm_bytes:
        return pcm_bytes
    out = bytearray(len(pcm_bytes) * 2)
    mv = memoryview(pcm_bytes)
    j = 0
    for i in range(0, len(pcm_bytes), 2):
        b0, b1 = mv[i], mv[i + 1]
        out[j] = b0
        out[j + 1] = b1
        out[j + 2] = b0
        out[j + 3] = b1
        j += 4
    return bytes(out)


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    await websocket.accept()
    print(f"[WS] Kết nối: {websocket.client}")

    idle = asyncio.Event()
    idle.set()

    awaiting_response = False
    buffer_while_awaiting: list[bytes] = []
    current_utterance_nonempty = False
    lock = asyncio.Lock()
    end_lock = asyncio.Lock()

    silence_task: asyncio.Task | None = None
    no_audio_task: asyncio.Task | None = None

    current_turn_id = 0
    got_first_model_audio = False
    mute_model_audio = False
    turn_complete_already_sent = False

    def cancel_silence_timer() -> None:
        nonlocal silence_task
        if silence_task is not None and not silence_task.done():
            silence_task.cancel()
        silence_task = None

    def cancel_no_audio_watch() -> None:
        nonlocal no_audio_task
        if no_audio_task is not None and not no_audio_task.done():
            no_audio_task.cancel()
        no_audio_task = None

    def schedule_silence_finalize(sess) -> None:
        nonlocal silence_task
        cancel_silence_timer()

        async def _after_silence() -> None:
            try:
                await asyncio.sleep(SILENCE_GAP_SEC)
                async with end_lock:
                    await handle_end(sess)
            except asyncio.CancelledError:
                pass

        silence_task = asyncio.create_task(_after_silence())

    async def forward_pcm(sess, data: bytes) -> None:
        nonlocal current_utterance_nonempty
        async with lock:
            pending = list(buffer_while_awaiting)
            buffer_while_awaiting.clear()
        for blob in pending:
            await sess.send_realtime_input(
                audio=types.Blob(data=blob, mime_type=MIME_PCM)
            )
        await sess.send_realtime_input(
            audio=types.Blob(data=data, mime_type=MIME_PCM)
        )
        async with lock:
            current_utterance_nonempty = True

    async def recv_worker(sess):
        nonlocal awaiting_response, mute_model_audio, turn_complete_already_sent, got_first_model_audio
        try:
            async for response in sess.receive():
                async with lock:
                    ar = awaiting_response
                if not ar:
                    continue
                try:
                    if response.data:
                        async with lock:
                            if mute_model_audio:
                                continue
                            if not got_first_model_audio:
                                got_first_model_audio = True
                        cancel_no_audio_watch()
                        stereo = mono_to_stereo(response.data)
                        await websocket.send_bytes(stereo)
                    sc = response.server_content
                    if sc and sc.turn_complete:
                        async with lock:
                            if not awaiting_response:
                                continue
                            awaiting_response = False
                            already = turn_complete_already_sent
                            turn_complete_already_sent = False
                            mute_model_audio = False
                        try:
                            if not already:
                                await websocket.send_text(
                                    json.dumps({"status": "turn_complete"})
                                )
                        finally:
                            cancel_no_audio_watch()
                            idle.set()
                except (WebSocketDisconnect, RuntimeError):
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[WS] recv_worker: {e}")
            async with lock:
                awaiting_response = False
                mute_model_audio = False
                turn_complete_already_sent = False
                got_first_model_audio = False
            cancel_no_audio_watch()
            idle.set()
            try:
                await websocket.send_text(
                    json.dumps({"status": "turn_complete", "error": str(e)})
                )
            except Exception:
                pass

    async def on_no_first_audio(tid: int) -> None:
        nonlocal mute_model_audio, turn_complete_already_sent
        try:
            await asyncio.sleep(NO_FIRST_AUDIO_SEC)
        except asyncio.CancelledError:
            return
        async with lock:
            if tid != current_turn_id:
                return
            if got_first_model_audio:
                return
            mute_model_audio = True
            turn_complete_already_sent = True
        try:
            await websocket.send_text(
                json.dumps(
                    {
                        "status": "turn_complete",
                        "no_model_audio": True,
                    }
                )
            )
        except Exception:
            pass

    async def handle_end(sess):
        nonlocal current_utterance_nonempty, awaiting_response, current_turn_id, got_first_model_audio
        nonlocal mute_model_audio, turn_complete_already_sent, no_audio_task
        cancel_silence_timer()
        await idle.wait()
        async with lock:
            to_flush = list(buffer_while_awaiting)
            buffer_while_awaiting.clear()
            utterance_ok = current_utterance_nonempty or bool(to_flush)
            if not utterance_ok:
                return
            current_utterance_nonempty = False
            idle.clear()
            awaiting_response = True
            got_first_model_audio = False
            mute_model_audio = False
            turn_complete_already_sent = False
            current_turn_id += 1
            tid = current_turn_id
        for blob in to_flush:
            await sess.send_realtime_input(
                audio=types.Blob(data=blob, mime_type=MIME_PCM)
            )
        await sess.send_realtime_input(audio_stream_end=True)
        cancel_no_audio_watch()

        async def _watch() -> None:
            await on_no_first_audio(tid)

        no_audio_task = asyncio.create_task(_watch())
        try:
            await asyncio.wait_for(idle.wait(), timeout=TURN_COMPLETE_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            async with lock:
                awaiting_response = False
                mute_model_audio = False
                turn_complete_already_sent = False
            cancel_no_audio_watch()
            try:
                await websocket.send_text(
                    json.dumps({"status": "turn_complete", "error": "timeout"})
                )
            except Exception:
                pass
            idle.set()

    try:
        async with client.aio.live.connect(model=MODEL_ID, config=LIVE_CONFIG) as session:
            worker = asyncio.create_task(recv_worker(session))
            try:
                while True:
                    message = await websocket.receive()

                    if "bytes" in message and message["bytes"]:
                        b = message["bytes"]
                        async with lock:
                            if awaiting_response:
                                buffer_while_awaiting.append(b)
                                b = None
                        if b is not None:
                            await forward_pcm(session, b)
                            schedule_silence_finalize(session)

                    elif "text" in message and message["text"].strip() == "END":
                        # Tùy chọn: firmware cũ vẫn dùng được; luồng mặc định là silence.

                        async def run_end():
                            async with end_lock:
                                await handle_end(session)

                        asyncio.create_task(run_end())

            except WebSocketDisconnect:
                print("[WS] Client ngắt")
            finally:
                cancel_silence_timer()
                cancel_no_audio_watch()
                worker.cancel()
                try:
                    await worker
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        import traceback

        print(f"[WS] Live session: {e}")
        traceback.print_exc()
        try:
            await websocket.send_text(
                json.dumps({"status": "turn_complete", "error": str(e)})
            )
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
