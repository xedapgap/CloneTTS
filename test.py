import os
import asyncio
import tempfile

import pysrt
import torch
import soundfile as sf
from pydub import AudioSegment
import edge_tts
import gradio as gr

from chatterbox.vc import ChatterboxVC

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Lazy load VC model
_vc_model = None
def get_vc():
    global _vc_model
    if _vc_model is None:
        _vc_model = ChatterboxVC.from_pretrained(DEVICE)
    return _vc_model

async def synthesize_segment(text: str, outfile: str, voice: str = "en-US-AriaNeural"):
    """
    Dùng edge-tts để synth một đoạn text ra WAV.
    """
    communicate = edge_tts.Communicate(text, voice)
    # edge-tts tự động xuất MP3 nếu outfile.endswith(".mp3"), WAV nếu .wav
    await communicate.save(outfile)

def synthesize_srt_to_audio(srt_path: str) -> str:
    """
    1) Parse SRT
    2) Với mỗi subtitle: TTS -> seg WAV -> pad/truncate -> overlay lên track silent tổng độ dài
    3) Xuất ra raw_tts.wav
    """
    subs = pysrt.open(srt_path, encoding="utf-8")
    # Tính tổng thời gian (ms) = max end time
    total_ms = max(sub.end.ordinal for sub in subs)
    # Khởi tạo track silent
    out = AudioSegment.silent(duration=total_ms)

    # Từng câu
    for idx, sub in enumerate(subs):
        text = sub.text.replace("\n", " ")
        start_ms = sub.start.ordinal
        dur_ms = sub.end.ordinal - sub.start.ordinal

        # synth bằng edge-tts (async)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        asyncio.get_event_loop().run_until_complete(
            synthesize_segment(text, tmp_path)
        )

        # load bằng pydub
        seg = AudioSegment.from_file(tmp_path, format="wav")
        os.unlink(tmp_path)

        # điều chỉnh độ dài cho đúng SRT
        if len(seg) < dur_ms:
            seg = seg + AudioSegment.silent(duration=(dur_ms - len(seg)))
        else:
            seg = seg[:dur_ms]

        # ghi đè lên vị trí start_ms
        out = out.overlay(seg, position=start_ms)

    # xuất file raw TTS
    os.makedirs("outputs", exist_ok=True)
    raw_path = os.path.join("outputs", "raw_srt_tts.wav")
    out.export(raw_path, format="wav")
    return raw_path

def convert_to_target_voice(raw_tts_path: str, ref_path: str) -> str:
    """
    Clone voice: raw_srt_tts.wav -> target voice
    """
    vc = get_vc()
    wav_np = vc.generate(
        raw_tts_path,
        target_voice_path=ref_path,
        inference_cfg_rate=0.5,
        sigma_min=1e-6
    )
    out_path = os.path.join("outputs", "final_cloned.wav")
    vc.save_wav(wav_np, out_path)
    return out_path

def full_pipeline(srt_file, reference_audio):
    """
    1) TTS theo SRT
    2) Clone voice
    Trả về đường dẫn file WAV kết quả.
    """
    raw = synthesize_srt_to_audio(srt_file.name)
    cloned = convert_to_target_voice(raw, reference_audio)
    return cloned

# Gradio UI
with gr.Blocks(css=".gradio-container {max-width: 600px}") as demo:
    gr.Markdown("## CloneTTS: SRT → Edge‑TTS → Voice‑to‑Voice")

    with gr.Row():
        srt_input = gr.File(label="Upload Subtitle (.srt)")
        ref_audio = gr.Audio(source="upload", type="filepath", label="Reference Voice")

    btn = gr.Button("Convert & Clone")
    out_audio = gr.Audio(label="Final Cloned Audio", type="filepath")

    btn.click(
        fn=full_pipeline,
        inputs=[srt_input, ref_audio],
        outputs=out_audio
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
