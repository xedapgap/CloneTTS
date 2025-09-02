import os
import srt
from pydub import AudioSegment, effects
import edge_tts
from IPython.display import Audio, display
import nest_asyncio
import asyncio

nest_asyncio.apply()

# --- SRT sample ---
srt_content = """
1
00:00:03,133 --> 00:00:04,366
Tom lao thẳng lên trời

2
00:00:04,366 --> 00:00:06,300
Ngoảnh đầu nhìn lại thì thấy trạm không gian

3
00:00:06,300 --> 00:00:08,300
Còn có một phi hành gia cũng đeo túi
"""

subs = list(srt.parse(srt_content))
work_dir = "temp_test_srt"
os.makedirs(work_dir, exist_ok=True)
combined = AudioSegment.silent(duration=0)

# --- Async TTS ---
async def synthesize_chunk(text, out_path):
    voice_code = "vi-VN-namminhNeural"
    await edge_tts.Communicate(text, voice=voice_code).save(out_path)
    return out_path

# --- Xử lý từng đoạn SRT với speed giới hạn ---
async def synthesize_srt(subs, min_speed=0.7, max_speed=1.3):
    global combined
    for idx, sub in enumerate(subs):
        start_ms = int(sub.start.total_seconds() * 1000)
        end_ms   = int(sub.end.total_seconds()   * 1000)
        dur_ms   = end_ms - start_ms

        tmp_path = os.path.join(work_dir, f"tmp_{idx}.wav")
        await synthesize_chunk(sub.content, tmp_path)
        tts_audio = AudioSegment.from_file(tmp_path)

        # Tính speed factor và giới hạn trong min/max
        speed_factor = len(tts_audio) / dur_ms
        speed_factor = max(min_speed, min(max_speed, speed_factor))

        tts_audio = effects.speedup(tts_audio, playback_speed=speed_factor)

        # Silence đến start nếu cần
        if start_ms > len(combined):
            combined += AudioSegment.silent(duration=(start_ms - len(combined)))
        combined += tts_audio
        os.remove(tmp_path)

# --- Run ---
await synthesize_srt(subs)

# Xuất file
out_path = os.path.join(work_dir, "test_srt_vi_speed_adj.wav")
combined.export(out_path, format="wav")
display(Audio(out_path, autoplay=True))
