# from transformers import VitsModel, AutoTokenizer
# # import torch
# # import scipy.io.wavfile as wav
# # import numpy as np
# #
# # # Load model (lần đầu tải ~150MB, sau đó offline hoàn toàn)
# # print("Đang load model...")
# # model     = VitsModel.from_pretrained("facebook/mms-tts-vie")
# # tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")
# #
# # # Dùng GPU nếu có
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # model  = model.to(device)
# # print(f"Dùng: {device.upper()}")
# #
# # def text_to_speech(text, output_file="output.wav"):
# #     inputs = tokenizer(text, return_tensors="pt").to(device)
# #
# #     with torch.no_grad():
# #         output = model(**inputs).waveform
# #
# #     # Chuyển về numpy và lưu file
# #     audio = output.squeeze().cpu().numpy()
# #     audio = (audio * 32767).astype(np.int16)  # chuyển sang int16
# #     wav.write(output_file, rate=model.config.sampling_rate, data=audio)
# #     print(f"Đã lưu: {output_file}")
# #
# # # Thử nghiệm
# # sentences = [
# #     ("Xin chào, tôi là trợ lý giọng nói tiếng Việt.", "output_1.wav"),
# #     ("Hôm nay thời tiết rất đẹp.",                    "output_2.wav"),
# #     ("Học lập trình Python rất thú vị.",              "output_3.wav"),
# # ]
# #
# # for text, filename in sentences:
# #     print(f"Đang xử lý: {text}")
# #     text_to_speech(text, filename)
# #
# # print("\nHoàn tất!")
from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile as wav
import sounddevice as sd
import numpy as np

print("Đang load model...")
model     = VitsModel.from_pretrained("facebook/mms-tts-vie")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-vie")

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = model.to(device)
print(f"Dùng: {device.upper()}\n")

def text_to_speech(text, save_file=None, play=True):
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model(**inputs).waveform

    audio      = output.squeeze().cpu().numpy()
    audio_int  = (audio * 32767).astype(np.int16)
    sample_rate = model.config.sampling_rate

    # Phát audio
    if play:
        print(f"Đang phát: {text}")
        sd.play(audio, samplerate=sample_rate)
        sd.wait()  # chờ phát xong mới tiếp tục

    # Lưu file (tuỳ chọn)
    if save_file:
        wav.write(save_file, rate=sample_rate, data=audio_int)
        print(f"Đã lưu: {save_file}")

# Chỉ phát, không lưu
text_to_speech("Xin chào, tôi là trợ lý giọng nói tiếng Việt.")

# Vừa phát vừa lưu
text_to_speech("Hôm nay thời tiết rất đẹp.")

# Chỉ lưu, không phát
text_to_speech("Học lập trình Python rất thú vị.", play=False)