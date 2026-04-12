import os
import asyncio
import json
import io
import wave
import unicodedata
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
import mqtt

import parameter_secret

app = FastAPI()


def _normalize_text(text: str) -> str:
    # Normalize unicode + lowercase to avoid misses like "Bật quạt" vs "bật quạt".
    return unicodedata.normalize("NFKC", text).lower()

# ─── Gemini Config ──────────────────────────────────────────────────────────
GEMINI_API_KEY = parameter_secret.API_KEY
client = genai.Client(api_key=GEMINI_API_KEY, http_options={'api_version': 'v1beta'})

MODEL_ID = "gemini-2.5-flash"
SYSTEM_PROMPT = (
    "Bạn là trợ lý AI thông minh tên là tiểu trí, có thể gọi người dùng là đại trí "
    "Nhiệm vụ: Trò chuyện tự nhiên, trả lời mọi câu hỏi của người dùng. "
    "Quy tắc: Trả lời cực ngắn dưới 20 từ, không dùng Markdown, không lặp từ. "
    "Người dùng yêu cầu bật đèn thì hãy trả lời sao cho bao gồm cả cụm từ bật đèn. Tương tự thế đối với các thiết bị điện khác. Không hỏi thêm. "
    "Hãy sử dụng thông tin từ các câu hỏi trước đó trong lịch sử hội thoại để hiểu ngữ cảnh."
)


async def ask_gemini_with_audio(audio_bytes: bytes, history: list) -> str:
    try:
        now = datetime.now().strftime("%H:%M, ngày %d/%m/%Y")

        with io.BytesIO() as wav_io:
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_bytes)
            wav_data = wav_io.getvalue()
        contents = []

        for item in history:
            contents.append(types.Content(role=item["role"], parts=[types.Part.from_text(text=item["content"])]))

        current_audio_part = types.Part.from_bytes(data=wav_data, mime_type="audio/wav")
        context_text = f"Bối cảnh: Bây giờ là {now}. Hãy nghe và trả lời trực tiếp người dùng."

        contents.append(types.Content(role="user", parts=[current_audio_part, types.Part.from_text(text=context_text)]))

        # 4. Gọi API
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=MODEL_ID,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3
            )
        )

        if response.text:
            reply = response.text.strip().replace("\n", " ")

            history.append({"role": "user", "content": "[Người dùng gửi âm thanh]"})
            history.append({"role": "model", "content": reply})

            if len(history) > 6:
                history.pop(0)
                history.pop(0)

            print(f"[Gemini] Reply: {reply}")
            normalized_reply = _normalize_text(reply)
            has_command = False

            if "tắt đèn" in normalized_reply:
                mqtt.publish_control(104, 2, 0)
                has_command = True
            if "bật đèn" in normalized_reply:
                mqtt.publish_control(104, 2, 1)
                has_command = True
            if "bật quạt" in normalized_reply:
                mqtt.publish_control(104, 1, 1)
                has_command = True
            if "tắt quạt" in normalized_reply:
                mqtt.publish_control(104, 1, 0)
                has_command = True

            if not has_command:
                print(f"[MQTT] Không tìm thấy lệnh điều khiển trong reply: {normalized_reply}")
            return reply

        return "Tôi chưa hiểu ý bạn."

    except Exception as e:
        print(f"[ERR Gemini] {e}")
        return "Tôi không phản hồi được, kiểm tra mạng của bạn."


@app.websocket("/ws/transcribe")
async def ws_transcribe(websocket: WebSocket):
    await websocket.accept()
    print(f"[WS] Kết nối mới: {websocket.client}")

    local_history = []
    audio_chunks: list[bytes] = []
    pending_tasks: set[asyncio.Task] = set()

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                audio_chunks.append(message["bytes"])
                # print(f"[WS] Nhận âm thanh: {len(message['bytes'])} bytes, tổng: {sum(len(chunk) for chunk in audio_chunks)} bytes")

            elif "text" in message:
                if message["text"].strip() == "END":
                    print("[WS] Nhận lệnh kết thúc phiên âm.")
                    if not audio_chunks:
                        continue

                    raw_audio = b"".join(audio_chunks)
                    audio_chunks = []

                    async def process_task(data, hist):
                        answer = await ask_gemini_with_audio(data, hist)
                        await websocket.send_text(
                            json.dumps({"stt": "[Gemini 2.5]", "text": answer, "status": "ok"})
                        )

                    task = asyncio.create_task(process_task(raw_audio, local_history))
                    pending_tasks.add(task)
                    task.add_done_callback(pending_tasks.discard)

    except WebSocketDisconnect:
        print("[WS] Ngắt kết nối.")
    except Exception as e:
        print(f"[WS] Lỗi: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)