import os
import sys
import importlib
import json
import asyncio
from datetime import datetime

import torch
import gradio as gr
import pydub
import edge_tts
import srt

# --- 1) Đảm bảo src/ có trong Python path để import ChatterboxVC ---
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import chatterbox.vc
importlib.reload(chatterbox.vc)
from chatterbox.vc import ChatterboxVC

# --- 2) Khởi tạo model VC ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_vc_model = None
def get_vc_model():
    global _vc_model
    if _vc_model is None:
        print(f"[VC] Đang tải model trên {DEVICE}…")
        _vc_model = ChatterboxVC.from_pretrained(DEVICE)
        print("[VC] Model sẵn sàng.")
    return _vc_model

# --- 3) Helper cập nhật log, audio và file-download ---
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
    audio_update = gr.update(visible=(audio_data is not None),
                             value=audio_data if audio_data is not None else None)
    files_update = gr.update(visible=(file_list is not None),
                             value=file_list if file_list is not None else [])
    yield log_update, audio_update, files_update

# --- 4) Load voices Edge TTS từ voices.json ---
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

# --- 5) TTS Edge với rate & volume ---
async def _edge_tts_async(text, disp, rate_pct, vol_pct):
    code = edge_code_map.get(disp)
    rate_str = f"{rate_pct:+d}%"
    vol_str  = f"{vol_pct:+d}%"
    out = "temp_edge_tts.wav"
    await edge_tts.Communicate(text, voice=code, rate=rate_str, volume=vol_str).save(out)
    return out

def run_edge_tts(text, disp, rate_pct, vol_pct):
    path = asyncio.run(_edge_tts_async(text, disp, rate_pct, vol_pct))
    return path, path

# --- 6) Sinh audio từ SRT theo đoạn, co giãn tốc độ ---
def synthesize_srt_audio(srt_path: str, disp_voice: str, work_dir: str,
                         rate_pct: int, vol_pct: int) -> str:
    with open(srt_path, "r", encoding="utf-8") as f:
        subs = list(srt.parse(f.read()))
    combined = pydub.AudioSegment.empty()
    current_ms = 0

    for sub in subs:
        start_ms = int(sub.start.total_seconds() * 1000)
        end_ms   = int(sub.end.total_seconds()   * 1000)
        dur_ms   = end_ms - start_ms

        # Thêm silence nếu cần
        if start_ms > current_ms:
            combined += pydub.AudioSegment.silent(duration=start_ms - current_ms)

        tmp_wav, _ = run_edge_tts(sub.content, disp_voice, rate_pct, vol_pct)
        tts_audio = pydub.AudioSegment.from_file(tmp_wav)

        # Co giãn tốc độ để match dur_ms
        tts_len = len(tts_audio)
        if tts_len > 0 and tts_len != dur_ms:
            speed_factor = tts_len / dur_ms
            tts_audio = tts_audio._spawn(tts_audio.raw_data, overrides={
                "frame_rate": int(tts_audio.frame_rate * speed_factor)
            }).set_frame_rate(tts_audio.frame_rate)

        combined += tts_audio
        current_ms = end_ms

    out_path = os.path.join(work_dir, "srt_source.wav")
    combined.export(out_path, format="wav")
    return out_path

# --- 7) Voice Conversion chính ---
def generate_vc(
    source_audio_path,
    target_voice_path,
    cfg_rate: float,
    sigma_min: float,
    batch_mode: bool,
    batch_parameter: str,
    batch_values: str
):
    model = get_vc_model()
    yield from yield_vc_updates("Khởi tạo chuyển giọng…", log_append=False)

    date_folder = datetime.now().strftime("%Y%m%d")
    work_dir = os.path.join("outputs/vc", date_folder)
    os.makedirs(work_dir, exist_ok=True)

    def run_once(src, tgt, rate, sigma):
        return model.generate(src, target_voice_path=tgt, inference_cfg_rate=rate, sigma_min=sigma)

    outputs = []
    try:
        if batch_mode:
            vals = [float(v.strip()) for v in batch_values.split(",") if v.strip()]
            yield from yield_vc_updates(f"Chạy batch '{batch_parameter}': {vals}")
            for idx, v in enumerate(vals, 1):
                r, s = cfg_rate, sigma_min
                tag = ""
                if batch_parameter == "Inference CFG Rate":
                    r, tag = v, f"cfg_{v}"
                else:
                    s, tag = v, f"sigma_{v}"
                yield from yield_vc_updates(f" • Mục {idx}/{len(vals)}: {batch_parameter}={v}")
                wav = run_once(source_audio_path, target_voice_path, r, s)
                fn = f"{tag}_{idx}.wav"
                path = os.path.join(work_dir, fn)
                model.save_wav(wav, path)
                outputs.append(path)
                yield from yield_vc_updates(f"Đã lưu: {path}")
        else:
            yield from yield_vc_updates("Đang chuyển giọng…")
            wav = run_once(source_audio_path, target_voice_path, cfg_rate, sigma_min)
            outp = os.path.join(work_dir, f"LyTranTTS_{datetime.now().strftime('%H%M%S')}.wav")
            model.save_wav(wav, outp)
            outputs.append(outp)
            yield from yield_vc_updates("Hoàn thành.")
    except Exception as e:
        yield from yield_vc_updates(f"Lỗi: {e}")
        raise

    first = outputs[0] if outputs else None
    yield from yield_vc_updates(log_msg=None, audio_data=first, file_list=outputs)

# --- 8) Wrapper tổng hợp ---
def run_vc_from_srt_or_file(
    use_srt: bool,
    srt_file, srt_voice, srt_rate, srt_vol,
    edge_text, edge_voice, edge_rate, edge_vol,
    src_audio, tgt_audio,
    cfg_rate, sigma_min,
    batch_mode, batch_parameter, batch_values
):
    yield from yield_vc_updates("Bắt đầu…", log_append=False)

    date_folder = datetime.now().strftime("%Y%m%d")
    work_dir = os.path.join("outputs/vc", date_folder)
    os.makedirs(work_dir, exist_ok=True)

    if use_srt:
        yield from yield_vc_updates("Sinh audio từ SRT…")
        source = synthesize_srt_audio(
            srt_file.name, srt_voice, work_dir,
            rate_pct=srt_rate, vol_pct=srt_vol
        )
    elif edge_text and edge_voice:
        yield from yield_vc_updates("Sinh audio từ Edge TTS…")
        tmp, _ = run_edge_tts(edge_text, edge_voice, edge_rate, edge_vol)
        source = tmp
    else:
        source = src_audio

    yield from generate_vc(
        source, tgt_audio,
        cfg_rate, sigma_min,
        batch_mode, batch_parameter, batch_values
    )

# --- 9) Build Gradio UI ---
with gr.Blocks(title="Chuyển Giọng Nói AI") as demo:
    gr.Markdown("## 📣 Chuyển Giọng Nói AI")
    gr.Markdown("> Tác giả: **Lý Trần**")

    with gr.Row():
        with gr.Column():
            use_srt   = gr.Checkbox(label="Sử dụng file SRT làm nguồn?", value=False)
            srt_file  = gr.File(file_types=[".srt"], label="Tải lên file .srt", visible=False)
            srt_voice = gr.Dropdown
