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

# --- 1) Đảm bảo `src/` có trong Python path để import ChatterboxVC ---
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import chatterbox.vc
importlib.reload(chatterbox.vc)
from chatterbox.vc import ChatterboxVC

# --- 2) Khởi tạo VC model ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_vc_model = None
def get_vc_model():
    global _vc_model
    if _vc_model is None:
        print(f"[VC] Loading model on {DEVICE}…")
        _vc_model = ChatterboxVC.from_pretrained(DEVICE)
        print("[VC] Model ready.")
    return _vc_model

# --- 3) Hàm giúp update log và UI ---
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
    if audio_data is not None:
        audio_update = gr.update(value=audio_data, visible=True)
        files_update = gr.update(visible=False)
    elif file_list:
        audio_update = gr.update(visible=False)
        files_update = gr.update(value=file_list, visible=True)
    else:
        audio_update = gr.update(visible=False)
        files_update = gr.update(visible=False)
    yield log_update, audio_update, files_update

# --- 4) Load Edge TTS voices từ JSON ---
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

# --- 5) TTS với Edge, hỗ trợ tùy chỉnh tốc độ và âm lượng ---
async def _edge_tts_async(text, disp, rate_pct, vol_pct):
    code = edge_code_map.get(disp)
    rate_str = f"{rate_pct:+d}%"
    vol_str  = f"{vol_pct:+d}%"
    out = "temp_edge_tts.wav"
    await edge_tts.Communicate(
        text,
        voice=code,
        rate=rate_str,
        volume=vol_str
    ).save(out)
    return out

def run_edge_tts(text, disp, rate_pct, vol_pct):
    path = asyncio.run(_edge_tts_async(text, disp, rate_pct, vol_pct))
    return path, path

# --- 6) Sinh audio từ SRT ---
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

        if start_ms > current_ms:
            combined += pydub.AudioSegment.silent(duration=start_ms - current_ms)

        tmp_wav, _ = run_edge_tts(sub.content, disp_voice, rate_pct, vol_pct)
        tts_audio = pydub.AudioSegment.from_file(tmp_wav)

        if len(tts_audio) > dur_ms:
            tts_audio = tts_audio[:dur_ms]
        else:
            tts_audio += pydub.AudioSegment.silent(duration=dur_ms - len(tts_audio))

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
    yield from yield_vc_updates(log_msg="Initializing voice conversion…", log_append=False)

    date_folder = datetime.now().strftime("%Y%m%d")
    work_dir = os.path.join("outputs/vc", date_folder)
    os.makedirs(work_dir, exist_ok=True)

    def run_once(src, tgt, rate, sigma):
        wav = model.generate(src, target_voice_path=tgt, inference_cfg_rate=rate, sigma_min=sigma)
        return wav

    try:
        if batch_mode:
            try:
                vals = [float(v.strip()) for v in batch_values.split(",") if v.strip()]
            except:
                raise gr.Error("Batch values must be comma‑separated numbers.")
            yield from yield_vc_updates(f"Batch sweep on '{batch_parameter}': {vals}")
            outputs = []
            for idx, v in enumerate(vals, 1):
                r, s = cfg_rate, sigma_min
                tag = ""
                if batch_parameter == "Inference CFG Rate":
                    r, tag = v, f"cfg_{v}"
                else:
                    s, tag = v, f"sigma_{v}"
                yield from yield_vc_updates(f" • item {idx}/{len(vals)}: {batch_parameter}={v}")
                wav = run_once(source_audio_path, target_voice_path, r, s)
                fn = f"{tag}_{idx}.wav"
                path = os.path.join(work_dir, fn)
                model.save_wav(wav, path)
                outputs.append(path)
                yield from yield_vc_updates(f"Saved: {path}")
            yield from yield_vc_updates("Batch complete.", file_list=outputs)
        else:
            audio = pydub.AudioSegment.from_file(source_audio_path)
            if len(audio) > 40_000:
                yield from yield_vc_updates("Source >40s: splitting into 40s chunks…")
                chunks = [audio[i : i + 40_000] for i in range(0, len(audio), 40_000)]
                paths = []
                for i, chunk in enumerate(chunks):
                    tmp = f"{source_audio_path}_chunk{i}.wav"
                    chunk.export(tmp, format="wav")
                    wav = run_once(tmp, target_voice_path, cfg_rate, sigma_min)
                    outp = os.path.join(work_dir, f"part{i}.wav")
                    model.save_wav(wav, outp)
                    paths.append(outp)
                    os.remove(tmp)
                    yield from yield_vc_updates(f"Processed chunk {i+1}/{len(chunks)}")
                combined = pydub.AudioSegment.empty()
                for p in paths:
                    combined += pydub.AudioSegment.from_file(p)
                final = os.path.join(work_dir, "combined.wav")
                combined.export(final, format="wav")
                yield from yield_vc_updates("Conversion complete.", audio_data=final, file_list=[final])
            else:
                yield from yield_vc_updates("Performing single conversion…")
                wav = run_once(source_audio_path, target_voice_path, cfg_rate, sigma_min)
                outp = os.path.join(work_dir, f"output_{datetime.now().strftime('%H%M%S')}.wav")
                model.save_wav(wav, outp)
                yield from yield_vc_updates("Done.", audio_data=outp, file_list=[outp])
    except Exception as e:
        yield from yield_vc_updates(f"Error: {e}")
        raise

# --- 8) Wrapper tổng hợp ---
def run_vc_from_srt_or_file(
    use_srt: bool,
    srt_file, srt_voice,
    edge_text, edge_voice, edge_rate, edge_vol,
    src_audio, tgt_audio,
    cfg_rate, sigma_min,
    batch_mode, batch_parameter, batch_values
):
    yield from yield_vc_updates(log_msg="Bắt đầu…", log_append=False)

    date_folder = datetime.now().strftime("%Y%m%d")
    work_dir = os.path.join("outputs/vc", date_folder)
    os.makedirs(work_dir, exist_ok=True)

    if use_srt:
        yield from yield_vc_updates("Sinh audio từ SRT…")
        source = synthesize_srt_audio(
            srt_file.name, srt_voice, work_dir,
            rate_pct=edge_rate, vol_pct=edge_vol
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

# --- 9) Xây dựng Gradio UI ---
with gr.Blocks(title="Voice‑to‑Voice Conversion") as demo:
    gr.Markdown("## 📣 Voice‑to‑Voice Conversion")

    with gr.Row():
        with gr.Column():

            # Chọn SRT hay không
            use_srt = gr.Checkbox(label="Use SRT để làm nguồn?", value=False)
            srt_file = gr.File(file_types=[".srt"], label="Upload file .srt", visible=False)
            srt_voice = gr.Dropdown(choices=edge_choices, label="Edge TTS Voice (SRT)", visible=False)

            # Edge TTS source (nếu không dùng SRT)
            use_edge = gr.Checkbox(label="Generate source via Edge TTS?", value=False)
            edge_text   = gr.Textbox(label="Text for Edge TTS", visible=False)
            edge_voice  = gr.Dropdown(choices=edge_choices, label="Edge TTS Voice", visible=False)
            edge_rate   = gr.Slider(-100, 100, value=0, step=1, label="Tốc độ (% chuẩn)", visible=False)
            edge_vol    = gr.Slider(-100, 100, value=0, step=1, label="Âm lượng (% chuẩn)", visible=False)
            gen_edge_btn = gr.Button("Generate Edge TTS", visible=False)
            edge_audio   = gr.Audio(label="Edge‑generated source", type="filepath", visible=False)

            # Manual source nếu không dùng SRT và không dùng Edge
            src_audio = gr.Audio(sources=["upload","microphone"], type="filepath",
                                 label="Upload/Record Source Audio")

            # Target voice
            gr.Markdown("### Reference (Target) Voice")
            tgt_audio = gr.Audio(sources=["upload","microphone"], type="filepath",
                                 label="Upload/Record Target Voice")

            # Tham số VC
            gr.Markdown("### Generation Parameters")
            cfg_slider = gr.Slider(0.0, 30.0, value=0.5, step=0.1, label="Inference CFG Rate")
            sigma_input = gr.Number(1e-6, label="Sigma Min",
                                   minimum=1e-7, maximum=1e-5, step=1e-7)

            with gr.Accordion("Batch Sweep Options", open=False):
                batch_chk    = gr.Checkbox(label="Enable Batch Sweep", value=False)
                batch_param  = gr.Dropdown(choices=["Inference CFG Rate","Sigma Min"],
                                           label="Parameter to Vary")
                batch_vals   = gr.Textbox(placeholder="e.g. 0.5,1.0,2.0",
                                          label="Comma‑separated values")

            # Nút Convert
            run_btn = gr.Button("🚀 Convert Voice")

        with gr.Column():
            gr.Markdown("### Conversion Log")
            log_box = gr.Textbox(interactive=False, lines=12)
            gr.Markdown("### Output")
            out_audio = gr.Audio(label="Result", type="filepath", visible=False)
            out_files = gr.File(label="Download Files", visible=False)

    # Toggle hi/ẩn SRT
    def toggle_srt(v):
        return (
            gr.update(visible=v),  # srt_file
            gr.update(visible=v),  # srt_voice
            gr.update(visible=not v), # use_edge
            gr.update(visible=not v), # edge_text
            gr.update(visible=not v), # edge_voice
            gr.update(visible=not v), # edge_rate
            gr.update(visible=not v), # edge_vol
            gr.update(visible=not v), # gen_edge_btn
            gr.update(visible=not v), # edge_audio
            gr.update(visible=not v)  # src_audio
        )

    use_srt.change(
        fn=toggle_srt,
        inputs=[use_srt],
        outputs=[
            srt_file, srt_voice,
            use_edge, edge_text, edge_voice, edge_rate, edge_vol,
            gen_edge_btn, edge_audio, src_audio
        ]
    )

    # Toggle hi/ẩn Edge TTS
    def toggle_edge(v):
        return (
            gr.update(visible=v),  # edge_text
            gr.update(visible=v),  # edge_voice
            gr.update(visible=v),  # edge_rate
            gr.update(visible=v),  # edge_vol
            gr.update(visible=v),  # gen_edge_btn
            gr.update(visible=v),  # edge_audio
            gr.update(visible=not v) # src_audio
        )

    use_edge.change(
        fn=toggle_edge,
        inputs=[use_edge],
        outputs=[
            edge_text, edge_voice, edge_rate, edge_vol,
            gen_edge_btn, edge_audio, src_audio
        ]
    )

    # Sinh Edge TTS
    gen_edge_btn.click(
        fn=run_edge_tts,
        inputs=[edge_text, edge_voice, edge_rate, edge_vol],
        outputs=[edge_audio, src_audio]
    )

    # Chạy VC
    run_btn.click(
        fn=run_vc_from_srt_or_file,
        inputs=[
            use_srt, srt_file, srt_voice,
            edge_text, edge_voice, edge_rate, edge_vol,
            src_audio, tgt_audio,
            cfg_slider, sigma_input,
            batch_chk, batch_param, batch_vals
        ],
        outputs=[log_box, out_audio, out_files],
        show_progress="minimal"
    )

if __name__ == "__main__":
    demo.launch(share=True)
