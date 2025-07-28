import os
import asyncio
import tempfile

import pysrt
import torch
from pydub import AudioSegment
import edge_tts
import gradio as gr

from chatterbox.vc import ChatterboxVC

# --- 1) Device & lazy-load VC model ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_vc_model = None

def get_vc_model():
    global _vc_model
    if _vc_model is None:
        _vc_model = ChatterboxVC.from_pretrained(DEVICE)
    return _vc_model

# --- 2) Edge‑TTS helper for Vietnamese Male (NamMinh) voice ---
async def synthesize_segment(text: str, out_path: str, voice: str = "vi-VN-NamMinhNeural"):
    """
    Use edge-tts to synthesize `text` → WAV at `out_path`.
    """
    communicator = edge_tts.Communicate(text, voice)
    await communicator.save(out_path)

# --- 3) Build raw TTS track from SRT ---
def synthesize_srt_to_raw_wav(srt_path: str) -> str:
    """
    Parse the .srt, generate each line via Edge‑TTS,
    pad/truncate to the subtitle's duration, and overlay on a silent track.
    Returns the path to the combined raw WAV.
    """
    subs = pysrt.open(srt_path, encoding="utf-8")
    total_ms = max(sub.end.ordinal for sub in subs)
    track = AudioSegment.silent(duration=total_ms)

    for sub in subs:
        text     = sub.text.replace("\n", " ")
        start_ms = sub.start.ordinal
        dur_ms   = sub.end.ordinal - sub.start.ordinal

        # synthesize each segment to a temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        # use asyncio.run to create a fresh event loop
        asyncio.run(synthesize_segment(text, tmp_path))

        seg = AudioSegment.from_file(tmp_path, format="wav")
        os.unlink(tmp_path)

        # pad or truncate to exact duration
        if len(seg) < dur_ms:
            seg = seg + AudioSegment.silent(duration=(dur_ms - len(seg)))
        else:
            seg = seg[:dur_ms]

        # overlay at the correct start time
        track = track.overlay(seg, position=start_ms)

    os.makedirs("outputs", exist_ok=True)
    raw_out = os.path.join("outputs", "raw_srt_tts.wav")
    track.export(raw_out, format="wav")
    return raw_out

# --- 4) Voice‑to‑Voice cloning via ChatterboxVC ---
def clone_voice(raw_wav_path: str, ref_voice_path: str) -> str:
    """
    Run the VC model on the raw TTS WAV, cloning into the reference voice.
    Returns the path to the final cloned WAV.
    """
    vc = get_vc_model()
    wav_np = vc.generate(
        raw_wav_path,
        target_voice_path=ref_voice_path,
        inference_cfg_rate=0.5,
        sigma_min=1e-6
    )
    out_path = os.path.join("outputs", "final_cloned.wav")
    vc.save_wav(wav_np, out_path)
    return out_path

# --- 5) Full pipeline for Gradio ---
def full_pipeline(srt_file, reference_audio):
    """
    1) Build raw TTS from SRT using NamMinh voice
    2) Clone that TTS into the reference voice
    Returns the final audio file path.
    """
    raw_wav = synthesize_srt_to_raw_wav(srt_file.name)
    final_wav = clone_voice(raw_wav, reference_audio)
    return final_wav

# --- 6) Gradio UI Definition ---
with gr.Blocks(css=".gradio-container {max-width:600px;}") as demo:
    gr.Markdown("## CloneTTS: SRT → Edge‑TTS (NamMinh) → Voice‑to‑Voice")
    
    with gr.Row():
        srt_input = gr.File(
            file_types=[".srt"],
            label="Upload Subtitle File (.srt)"
        )
        ref_audio = gr.Audio(
            type="filepath",
            label="Reference Voice Audio (WAV/MP3)"
        )
    
    convert_btn = gr.Button("Convert & Clone")
    out_audio   = gr.Audio(
        type="filepath",
        label="Final Cloned Audio"
    )

    convert_btn.click(
        fn=full_pipeline,
        inputs=[srt_input, ref_audio],
        outputs=out_audio
    )

if __name__ == "__main__":
    demo.launch(share=True)
