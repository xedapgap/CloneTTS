import os
import sys
import asyncio
import importlib
from datetime import datetime
import json

import torch
import gradio as gr
import pydub
import srt
import edge_tts

# --- 1) Thêm src/ vào Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import chatterbox.vc
importlib.reload(chatterbox.vc)
from chatterbox.vc import ChatterboxVC

# --- 2) Edge TTS voices ---
def load_edge_tts_voices(json_path="voices.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        voices = json.load(f)
    display_list, code_map = [], {}
    for lang, genders in voices.items():
        for gender, items in genders.items():
            for v in items:
                disp = f"{lang} - {gender} - {v['display_name']} ({v['voice_code']})"
                display_list.append(disp)
                code_map[disp] = v["voice_code"]
    return display_list, code_map

edge_choices, edge_code_map = load_edge_tts_voices()

async def edge_tts_async(text, voice_code, rate_pct=0, vol_pct=0, out_path="temp.wav"):
    rate_str = f"{rate_pct:+d}%"
    vol_str  = f"{vol_pct:+d}%"
    await edge_tts.Communicate(text, voice=voice_code, rate=rate_str, volume=vol_str).save(out_path)
    return out_path

# --- 3) Synthesize audio from SRT ---
def synthesize_srt_audio(srt_path: str, disp_voice: str, work_dir: str,
                         rate_pct: int=0, vol_pct: int=0) -> str:
    os.makedirs(work_dir, exist_ok=True)
    voice_code = edge_code_map[disp_voice]

    with open(srt_path, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))

    combined = pydub.AudioSegment.empty()
    for idx, sub in enumerate(subs):
        start_ms = int(sub.start.total_seconds() * 1000)
        end_ms   = int(sub.end.total_seconds()   * 1000)
        dur_ms   = end_ms - start_ms

        # Tách sub dài >70 ký tự
        chunks = []
        text = sub.content.strip()
        while len(text) > 70:
            split_pos = text[:70].rfind(" ")
            if split_pos == -1:
                split_pos = 70
            chunks.append(text[:split_pos])
            text = text[split_pos:].strip()
        if text:
            chunks.append(text)

        tmp_audio = pydub.AudioSegment.empty()
        for i, chunk in enumerate(chunks):
            tmp_path = os.path.join(work_dir, f"tmp_{idx}_{i}.wav")
            asyncio.run(edge_tts_async(chunk, voice_code, rate_pct, vol_pct, tmp_path))
            seg = pydub.AudioSegment.from_file(tmp_path)
            tmp_audio += seg
            os.remove(tmp_path)

        # Co giãn thời lượng để match timeline
        if len(tmp_audio) > dur_ms:
            tmp_audio = tmp_audio[:dur_ms]
        else:
            tmp_audio = tmp_audio + pydub.AudioSegment.silent(duration=(dur_ms - len(tmp_audio)))

        # Silence đến start
        if start_ms > len(combined):
            combined += pydub.AudioSegment.silent(duration=(start_ms - len(combined)))
        combined += tmp_audio

    out_path = os.path.join(work_dir, "srt_source.wav")
    combined.export(out_path, format="wav")
    return out_path

# --- 4) VC model ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_vc_model = None
def get_vc_model():
    global _vc_model
    if _vc_model is None:
        _vc_model = ChatterboxVC.from_pretrained(DEVICE)
    return _vc_model

# --- 5) Gradio helper ---
global_log_messages_vc = []
def yield_vc_updates(log_msg=None, audio_data=None, file_list=None, log_append=True):
    global global_log_messages_vc
    if log_msg is not None:
        prefix = datetime.now().strftime("[%H:%M:%S]")
        if log_append:
            global_log_messages_vc.append(f"{prefix} {log_msg}")
        else:
            global_log_messages_vc = [f"{prefix} {log_msg}"]
    log_update = gr.update(value="\n".join(global_log_messages_vc))
    audio_update = gr.update(visible=(audio_data is not None), value=audio_data if audio_data else None)
    files_update = gr.update(visible=(file_list is not None), value=file_list if file_list else [])
    yield log_update, audio_update, files_update

# --- 6) Generate VC ---
def generate_vc(source_audio_path, target_voice_path, cfg_rate, sigma_min):
    model = get_vc_model()
    wav = model.generate(source_audio_path, target_voice_path=target_voice_path,
                         inference_cfg_rate=cfg_rate, sigma_min=sigma_min)
    out_file = f"VC_{datetime.now().strftime('%H%M%S')}.wav"
    model.save_wav(wav, out_file)
    return out_file

# --- 7) Build Gradio UI ---
with gr.Blocks(title="Chuyển Giọng Nói AI") as demo:
    gr.Markdown("## 📣 Chuyển Giọng Nói AI - SRT → VC")
    
    with gr.Row():
        with gr.Column():
            srt_file  = gr.File(file_types=[".srt"], label="Tải lên file SRT")
            srt_voice = gr.Dropdown(choices=edge_choices, label="Giọng Edge TTS")
            rate_pct  = gr.Slider(-100,100,value=0,step=1,label="Tốc độ đọc (%)")
            vol_pct   = gr.Slider(-100,100,value=0,step=1,label="Âm lượng (%)")
            tgt_audio = gr.Audio(sources=["upload","microphone"], type="filepath", label="Giọng mục tiêu")
            cfg_rate  = gr.Slider(0.0,30.0,value=0.5,step=0.1,label="CFG Rate")
            sigma_min = gr.Number(1e-6, label="Sigma Min", minimum=1e-7, maximum=1e-5, step=1e-7)
            run_btn   = gr.Button("🚀 Chuyển giọng")

        with gr.Column():
            log_box  = gr.Textbox(interactive=False, lines=12)
            out_audio = gr.Audio(label="Âm thanh kết quả", type="filepath", visible=False)
            out_files = gr.Files(label="Tải xuống file VC", visible=False)

    def run_pipeline(srt_file, srt_voice, rate_pct, vol_pct, tgt_audio, cfg_rate, sigma_min):
        work_dir = "temp_srt_vc"
        yield from yield_vc_updates("Đang sinh audio từ SRT…")
        src_audio = synthesize_srt_audio(srt_file.name, srt_voice, work_dir, rate_pct, vol_pct)
        yield from yield_vc_updates("Đang chạy VC…")
        out_path = generate_vc(src_audio, tgt_audio, cfg_rate, sigma_min)
        yield from yield_vc_updates("Hoàn thành.", audio_data=out_path, file_list=[out_path])

    run_btn.click(
        fn=run_pipeline,
        inputs=[srt_file, srt_voice, rate_pct, vol_pct, tgt_audio, cfg_rate, sigma_min],
        outputs=[log_box, out_audio, out_files],
        show_progress="minimal"
    )

if __name__ == "__main__":
    demo.launch(share=True)
