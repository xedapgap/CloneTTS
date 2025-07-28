# -*- coding: utf-8 -*-
"""
Ứng dụng: Sao Chép Giọng
Tác giả: Lý Trần
"""

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

# --- 1) Đảm bảo thư mục src/ có trong Python path ---
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import chatterbox.vc
importlib.reload(chatterbox.vc)
from chatterbox.vc import ChatterboxVC

# --- 2) Khởi tạo model chuyển giọng ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_vc_model = None
def lay_model_vc():
    global _vc_model
    if _vc_model is None:
        print(f"[VC] Đang tải model lên {DEVICE}…")
        _vc_model = ChatterboxVC.from_pretrained(DEVICE)
        print("[VC] Model sẵn sàng.")
    return _vc_model

# --- 3) Hàm hỗ trợ cập nhật nhật ký và UI ---
global_log_messages_vc = []
def cap_nhat_log_va_ui(msg_log=None, audio_data=None, file_list=None, log_append=True):
    global global_log_messages_vc
    if msg_log is not None:
        prefix = datetime.now().strftime("[%H:%M:%S]")
        if log_append:
            global_log_messages_vc.append(f"{prefix} {msg_log}")
        else:
            global_log_messages_vc = [f"{prefix} {msg_log}"]
    update_log = gr.update(value="\n".join(global_log_messages_vc))
    if audio_data is not None:
        update_audio = gr.update(value=audio_data, visible=True)
        update_files = gr.update(visible=False)
    elif file_list:
        update_audio = gr.update(visible=False)
        update_files = gr.update(value=file_list, visible=True)
    else:
        update_audio = gr.update(visible=False)
        update_files = gr.update(visible=False)
    yield update_log, update_audio, update_files

# --- 4) Nạp danh sách giọng Edge TTS ---
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

async def _edge_tts_async(text, disp):
    code = edge_code_map.get(disp)
    out = "temp_edge_tts.wav"
    await edge_tts.Communicate(text, code).save(out)
    return out

def sinh_audio_edge(text, disp):
    path = asyncio.run(_edge_tts_async(text, disp))
    return path, path

# --- 5) Chuyển SRT thành audio ---
def synthesize_srt_audio(srt_path: str, disp_voice: str, work_dir: str) -> str:
    # Đọc và parse file .srt
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

        # TTS cho nội dung subtitle
        tmp_wav, _ = sinh_audio_edge(sub.content, disp_voice)
        tts_audio = pydub.AudioSegment.from_file(tmp_wav)

        # Cắt hoặc đệm silence để khớp đúng thời lượng
        if len(tts_audio) > dur_ms:
            tts_audio = tts_audio[:dur_ms]
        elif len(tts_audio) < dur_ms:
            tts_audio += pydub.AudioSegment.silent(duration=dur_ms - len(tts_audio))

        combined += tts_audio
        current_ms = end_ms

    # Xuất file kết quả
    out_path = os.path.join(work_dir, "srt_source.wav")
    combined.export(out_path, format="wav")
    return out_path

# --- 6) Hàm chuyển giọng (VC) ---
def generate_vc(
    source_audio_path,
    target_voice_path,
    cfg_rate: float,
    sigma_min: float,
    batch_mode: bool,
    batch_parameter: str,
    batch_values: str
):
    model = lay_model_vc()
    yield from cap_nhat_log_va_ui(msg_log="Khởi tạo chuyển giọng…", log_append=False)

    # Tạo thư mục output theo ngày
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
                raise gr.Error("Giá trị batch phải là một chuỗi số, cách nhau bởi dấu phẩy.")
            yield from cap_nhat_log_va_ui(f"Quét batch {batch_parameter}: {vals}")
            outputs = []
            for idx, v in enumerate(vals, 1):
                r, s = cfg_rate, sigma_min
                tag = ""
                if batch_parameter == "Inference CFG Rate":
                    r, tag = v, f"cfg_{v}"
                else:
                    s, tag = v, f"sigma_{v}"
                yield from cap_nhat_log_va_ui(f" • mục {idx}/{len(vals)}: {batch_parameter}={v}")
                wav = run_once(source_audio_path, target_voice_path, r, s)
                fn = f"{tag}_{idx}.wav"
                path = os.path.join(work_dir, fn)
                model.save_wav(wav, path)
                outputs.append(path)
                yield from cap_nhat_log_va_ui(f"Đã lưu: {path}")
            yield from cap_nhat_log_va_ui("Hoàn thành batch.", file_list=outputs)
        else:
            audio = pydub.AudioSegment.from_file(source_audio_path)
            if len(audio) > 40_000:
                yield from cap_nhat_log_va_ui("Audio >40s: chia thành các đoạn 40s…")
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
                    yield from cap_nhat_log_va_ui(f"Xử lý đoạn {i+1}/{len(chunks)}")
                combined = pydub.AudioSegment.empty()
                for p in paths:
                    combined += pydub.AudioSegment.from_file(p)
                final = os.path.join(work_dir, "combined.wav")
                combined.export(final, format="wav")
                yield from cap_nhat_log_va_ui("Hoàn thành chuyển giọng.", audio_data=final, file_list=[final])
            else:
                yield from cap_nhat_log_va_ui("Chuyển đơn…")
                wav = run_once(source_audio_path, target_voice_path, cfg_rate, sigma_min)
                outp = os.path.join(work_dir, f"kq_{datetime.now().strftime('%H%M%S')}.wav")
                model.save_wav(wav, outp)
                yield from cap_nhat_log_va_ui("Hoàn thành.", audio_data=outp, file_list=[outp])
    except Exception as e:
        yield from cap_nhat_log_va_ui(f"Lỗi: {e}")
        raise

# --- 7) Wrapper: chọn nguồn SRT hay file ---
def run_vc_tu_srt_hoac_file(
    use_srt: bool,
    srt_file, srt_voice,
    src_audio, tgt_audio,
    cfg_rate, sigma_min,
    batch_mode, batch_parameter, batch_values
):
    yield from cap_nhat_log_va_ui(msg_log="Bắt đầu quy trình…", log_append=False)

    # Thư mục output
    date_folder = datetime.now().strftime("%Y%m%d")
    work_dir = os.path.join("outputs/vc", date_folder)
    os.makedirs(work_dir, exist_ok=True)

    if use_srt:
        yield from cap_nhat_log_va_ui("Đang sinh audio từ SRT…")
        source = synthesize_srt_audio(srt_file.name, srt_voice, work_dir)
    else:
        source = src_audio

    yield from generate_vc(
        source, tgt_audio,
        cfg_rate, sigma_min,
        batch_mode, batch_parameter, batch_values
    )

# --- 8) Xây dựng giao diện Gradio ---
with gr.Blocks(title="Sao Chép Giọng") as demo:
    gr.Markdown("# 🚀 Sao Chép Giọng")
    gr.Markdown("Tác giả: Lý Trần")

    with gr.Row():
        with gr.Column(scale=1):
            # Chọn SRT
            use_srt = gr.Checkbox(label="Sử dụng file SRT làm nguồn", value=False)
            srt_file = gr.File(file_types=[".srt"], label="Tải lên file .srt", visible=False)
            srt_voice = gr.Dropdown(choices=edge_choices, label="Chọn giọng Edge TTS (SRT)", visible=False)

            # Chọn Edge TTS trực tiếp
            use_edge = gr.Checkbox(label="Sinh âm thanh nguồn bằng Edge TTS", value=False)
            edge_text = gr.Textbox(label="Nội dung cho Edge TTS", visible=False)
            edge_voice = gr.Dropdown(choices=edge_choices, label="Chọn giọng Edge TTS", visible=False)
            gen_edge_btn = gr.Button("🌐 Tạo bằng Edge TTS", visible=False)
            edge_audio   = gr.Audio(label="Âm thanh nguồn", type="filepath", visible=False)

            # Hoặc tải lên/ghi âm thủ công
            src_audio = gr.Audio(sources=["upload","microphone"], type="filepath", label="Tải lên/Ghi âm nguồn")

            # Giọng tham chiếu
            gr.Markdown("### Giọng tham chiếu")
            tgt_audio = gr.Audio(sources=["upload","microphone"], type="filepath", label="Tải lên/Ghi âm giọng tham chiếu")

            # Tham số
            gr.Markdown("### Tham số chuyển đổi")
            cfg_slider = gr.Slider(0.0, 30.0, value=0.5, step=0.1, label="Hệ số CFG (Inference CFG Rate)")
            sigma_input = gr.Number(1e-6, label="Giá trị Sigma Min", minimum=1e-7, maximum=1e-5, step=1e-7)

            with gr.Accordion("Tùy chọn quét Batch", open=False):
                batch_chk = gr.Checkbox(label="Kích hoạt quét Batch", value=False)
                batch_param = gr.Dropdown(choices=["Inference CFG Rate","Sigma Min"], label="Tham số thay đổi")
                batch_vals  = gr.Textbox(placeholder="vd. 0.5,1.0,2.0", label="Các giá trị, cách nhau dấu phẩy")

            run_btn = gr.Button("🔄 Chuyển Giọng")

        with gr.Column(scale=1):
            gr.Markdown("### Nhật ký chuyển đổi")
            log_box = gr.Textbox(interactive=False, lines=12)
            gr.Markdown("### Kết quả")
            out_audio = gr.Audio(label="Phát âm thanh kết quả", visible=False)
            out_files = gr.File(label="Tải xuống kết quả", visible=False)

    # Hàm toggle khi chọn SRT
    def toggle_srt(v):
        return (
            gr.update(visible=v),        # srt_file
            gr.update(visible=v),        # srt_voice
            gr.update(visible=not v),    # use_edge
            gr.update(visible=not v),    # edge_text
            gr.update(visible=not v),    # edge_voice
            gr.update(visible=not v),    # gen_edge_btn
            gr.update(visible=not v),    # edge_audio
            gr.update(visible=not v)     # src_audio
        )

    use_srt.change(
        fn=toggle_srt,
        inputs=[use_srt],
        outputs=[srt_file, srt_voice, use_edge, edge_text, edge_voice, gen_edge_btn, edge_audio, src_audio]
    )

    # Hàm toggle khi chọn Edge TTS
    def toggle_edge(v):
        return (
            gr.update(visible=v),
            gr.update(visible=v),
            gr.update(visible=v),
            gr.update(visible=v),
            gr.update(visible=not v)
        )

    use_edge.change(
        fn=toggle_edge,
        inputs=[use_edge],
        outputs=[edge_text, edge_voice, gen_edge_btn, edge_audio, src_audio]
    )

    # Tạo audio bằng Edge TTS
    gen_edge_btn.click(
        fn=sinh_audio_edge,
        inputs=[edge_text, edge_voice],
        outputs=[edge_audio, src_audio]
    )

    # Chạy quy trình chính
    run_btn.click(
        fn=run_vc_tu_srt_hoac_file,
        inputs=[
            use_srt, srt_file, srt_voice,
            src_audio, tgt_audio,
            cfg_slider, sigma_input,
            batch_chk, batch_param, batch_vals
        ],
        outputs=[log_box, out_audio, out_files],
        show_progress="minimal"
    )

if __name__ == "__main__":
    demo.launch(share=True)
