"""Local Gradio UI for Qwen-Image-2.1 Realtime; inference runs on a RunPod Serverless endpoint
(Dockerfile + runpod_worker/).

Nothing is sent to the API until you press Generate. Two modes:
* ⚡ 2-step realtime: 512px, WebP, fastest round trip.
* Normal: chosen resolution (512 / 1K / 1.5K), PNG.

    export RUNPOD_API_KEY=...  RUNPOD_ENDPOINT_ID=...
    python app.py
"""

import base64
import io
import json
import os
import random
import time

import gradio as gr
import requests
from PIL import Image

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "")
RUNPOD_TIMEOUT = int(os.environ.get("RUNPOD_TIMEOUT", "900"))  # covers a cold start
PE_SPACE_ID = os.environ.get("PE_SPACE_ID", "hugging-apps/qwen-image-2-1-prompt-enhancer")
PE_MAX_NEW_TOKENS = int(os.environ.get("PE_MAX_NEW_TOKENS", "1024"))

MAX_SEED = 2**31 - 1
MAX_INPUT_IMAGES = 10
LIVE_RES = 512
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")

ASPECT_RATIOS = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"]
RESOLUTION_CHOICES = [("512 (live)", 512), ("1K (native)", 1024), ("1.5K", 1536)]
DEFAULT_RESOLUTION = 1024
MODE_LIVE = "⚡ 2-step реалтайм (512px)"
MODE_NORMAL = "Обычный (полное разрешение)"

PROMPT_INFO = (
    "Describe what to generate, or how to edit the input images (refer to them as "
    "<image1>…<image10> in upload order). Nothing is generated until you press Generate."
)

# --------------------------------------------------------------------------------------
# RunPod client
# --------------------------------------------------------------------------------------

_session = requests.Session()


def runpod_call(payload):
    """POST /runsync; if the job outlives the sync wait (e.g. cold start), poll /status."""
    if not (RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID):
        raise RuntimeError("set RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID")
    base = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    deadline = time.time() + RUNPOD_TIMEOUT
    resp = _session.post(f"{base}/runsync?wait=90000", json={"input": payload}, headers=headers, timeout=120)
    resp.raise_for_status()
    job = resp.json()
    while job.get("status") in ("IN_QUEUE", "IN_PROGRESS"):
        if time.time() > deadline:
            _session.post(f"{base}/cancel/{job['id']}", headers=headers, timeout=30)
            raise TimeoutError(f"job {job['id']} took longer than {RUNPOD_TIMEOUT}s")
        time.sleep(0.5)
        resp = _session.get(f"{base}/status/{job['id']}", headers=headers, timeout=30)
        resp.raise_for_status()
        job = resp.json()
    if job.get("status") != "COMPLETED":
        raise RuntimeError(f"job {job.get('status')}: {job.get('error') or job}")
    out = job.get("output") or {}
    if "error" in out:
        raise RuntimeError(out["error"])
    out["delay"] = (job.get("delayTime") or 0) / 1000  # queue + cold start, seconds
    return out


def remote_generate(prompt, image_paths, seed, resolution, aspect_ratio, fmt):
    images = [base64.b64encode(open(p, "rb").read()).decode() for p in image_paths]
    started = time.perf_counter()
    out = runpod_call({"prompt": prompt, "images": images, "seed": int(seed), "resolution": int(resolution),
                       "aspect_ratio": aspect_ratio, "format": fmt})
    out["round_trip"] = time.perf_counter() - started
    out["pil"] = Image.open(io.BytesIO(base64.b64decode(out["image"])))
    return out


def status_line(out):
    w, h = out["size"]
    aoti = "AOTI" if out.get("aoti") else "eager"
    return (f"{w}×{h} · GPU {out['elapsed']:.2f} с · всего {out['round_trip']:.2f} с "
            f"(очередь/холодный старт {out['delay']:.2f} с) · {aoti}")


# --------------------------------------------------------------------------------------
# Prompt enhancement (same companion HF Space as the original)
# --------------------------------------------------------------------------------------

_pe_client = None


def enhance_prompt(prompt, image_paths):
    global _pe_client
    from gradio_client import Client, handle_file

    if _pe_client is None:
        _pe_client = Client(PE_SPACE_ID, token=os.environ.get("HF_TOKEN"),
                            httpx_kwargs={"timeout": 900}, verbose=False)
    result = _pe_client.predict(
        prompt=prompt, image_paths=[handle_file(p) for p in image_paths],
        max_new_tokens=PE_MAX_NEW_TOKENS, enable_thinking=False, seed=0, randomize_seed=True,
        api_name="/enhance",
    )
    rewritten, wh_ratio = str(result[0]).strip(), str(result[1]).strip()
    return (rewritten or prompt), wh_ratio


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def normalize_gallery(gallery):
    paths = []
    for item in gallery or []:
        if isinstance(item, (tuple, list)):
            item = item[0]
        if isinstance(item, dict):
            item = item.get("path") or item.get("image") or item.get("name")
            if isinstance(item, dict):
                item = item.get("path")
        if isinstance(item, str) and item:
            paths.append(item)
    return paths


# --------------------------------------------------------------------------------------
# Normal mode
# --------------------------------------------------------------------------------------


def prepare(prompt, input_images, enhance, seed, randomize_seed, aspect_ratio):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt.")
    image_paths = normalize_gallery(input_images)
    if len(image_paths) > MAX_INPUT_IMAGES:
        raise gr.Error(f"Up to {MAX_INPUT_IMAGES} input images are supported.")
    resolved_seed = random.randint(0, MAX_SEED) if randomize_seed else int(seed)
    final_prompt, rewritten = prompt, ""
    if enhance:
        try:
            final_prompt, wh_ratio = enhance_prompt(prompt, image_paths)
            rewritten = final_prompt
            if wh_ratio in ASPECT_RATIOS and aspect_ratio == "Auto" and not image_paths:
                aspect_ratio = wh_ratio
        except Exception as exc:  # noqa: BLE001
            gr.Warning(f"Prompt enhancement unavailable, using the original prompt ({exc}).")
    return final_prompt, rewritten, resolved_seed, aspect_ratio


def generate_image(prompt, input_images, seed, resolution, aspect_ratio, mode):
    prompt = (prompt or "").strip()
    if not prompt:
        raise gr.Error("Please enter a prompt.")
    if mode == MODE_LIVE:
        resolution, fmt = LIVE_RES, "webp"
    else:
        fmt = "png"
    try:
        out = remote_generate(prompt, normalize_gallery(input_images), seed, resolution, aspect_ratio, fmt)
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"Generation failed: {exc}")
    if out["resolution"] != int(resolution):
        gr.Info(f"Input images: rendered at {out['resolution']}px to stay within GPU memory.")
    return out["pil"], ("⚡ " if mode == MODE_LIVE else "") + status_line(out)


# --------------------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------------------


def _load_cases(name):
    with open(os.path.join(EXAMPLE_DIR, name), encoding="utf-8") as fh:
        return json.load(fh)


CASES = _load_cases("turbo_cases.json") + _load_cases("qwen_cases.json")
T2I_RATIO = {"Chemistry study poster (model card)": "2:3", "Travel planner UI": "9:16", "Mathematics exam paper": "3:4"}
T2I_EXAMPLES = [[c["prompt"], T2I_RATIO.get(c["title_en"], "1:1")] for c in CASES if not c["inputs"]]
T2I_LABELS = [c["title_en"] for c in CASES if not c["inputs"]]
EDIT_EXAMPLES = [[c["prompt"], [os.path.join(EXAMPLE_DIR, n) for n in c["inputs"]]] for c in CASES if c["inputs"]]
EDIT_LABELS = [c["title_en"] for c in CASES if c["inputs"]]


# --------------------------------------------------------------------------------------
# UI (same look as the Space)
# --------------------------------------------------------------------------------------

ORANGE = gr.themes.Color(
    c50="#fff7ed", c100="#ffedd5", c200="#fed7aa", c300="#fdba74", c400="#fb923c",
    c500="#f97316", c600="#ea580c", c700="#c2410c", c800="#9a3412", c900="#7c2d12", c950="#431407",
    name="qwenorange",
)
THEME = gr.themes.Base(
    primary_hue=ORANGE, secondary_hue=ORANGE, neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill_dark="#0a0706",
    block_background_fill_dark="#15100b",
    background_fill_secondary_dark="#15100b",
    color_accent="#f97316",
    slider_color="#f97316",
    checkbox_background_color_selected="#f97316",
    checkbox_background_color_selected_dark="#f97316",
)

CSS = """
.main.fillable { max-width: 920px !important; }
#col-container { margin: 0 auto; max-width: 900px; }
.qr-header { padding: 6px 2px 12px; }
.qr-header h1 { font-size: 1.6rem; margin: 0 0 6px; font-weight: 700; }
.qr-header h1 .qr-accent { color: #f97316; }
.qr-header p { color: var(--body-text-color-subdued); margin: 0 0 12px; max-width: 76ch; }
.qr-pills { display: flex; gap: 8px; flex-wrap: wrap; }
.qr-pills a, .qr-pills a:link, .qr-pills a:visited { font-size: .8rem; padding: 4px 12px;
  border-radius: 999px; background: #f97316; color: #fff !important; text-decoration: none; border: none; }
.qr-pills a:hover { background: #ea580c; color: #fff !important; }
#gen-btn, #gen-btn:hover { background: #f97316; border: none; }
#gen-btn:hover { background: #ea580c; }
#gen-btn, #gen-btn * { color: #fff !important; }
#status { font-size: .85rem; color: var(--body-text-color-subdued); min-height: 1.4em; }
"""

HEADER = """
<div class="qr-header">
  <h1>⚡ Qwen Image 2.1 <span class="qr-accent">Realtime</span></h1>
  <p>Real-time text-to-image and multi-image editing with Qwen-Image-2.1, accelerated by a
  2-step distillation LoRA and AOTInductor kernels. UI runs locally, inference runs on a RunPod Serverless GPU
  with the model pre-cached on the host. Pick ⚡ 2-step realtime for a fast 512px result, or Normal for
  full resolution and best quality, then press Generate.</p>
  <div class="qr-pills">
    <a href="https://huggingface.co/linoyts/Qwen-Image-2.1-DMD2-2steps" target="_blank">2-step DMD2 LoRA</a>
    <a href="https://huggingface.co/Qwen/Qwen-Image-2.1" target="_blank">Qwen-Image-2.1</a>
    <a href="https://huggingface.co/hugging-apps/qwen-image-2-1-aoti" target="_blank">AOTI kernels</a>
    <a href="https://huggingface.co/spaces/linoyts/qwen-image-2-1-realtime" target="_blank">Original Space</a>
  </div>
</div>
"""

with gr.Blocks(title="Qwen Image 2.1 Realtime") as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML(HEADER)
        with gr.Row():
            with gr.Column():
                input_images = gr.Gallery(label="Input images (optional, for editing, up to 10)",
                                          type="filepath", interactive=True, columns=5, height=320)
            with gr.Column():
                result = gr.Image(label="Result", type="pil", image_mode="RGBA", format="png", height=320)

        prompt = gr.Textbox(label="Prompt", info=PROMPT_INFO, lines=4, max_lines=24,
                            placeholder="Describe what you want to generate or edit…")
        with gr.Row():
            mode = gr.Radio([MODE_LIVE, MODE_NORMAL], value=MODE_LIVE, label="Режим генерации", scale=2)
            enhance = gr.Checkbox(label="Enhance prompt (slower, rewrites short prompts)", value=False, scale=1)
            generate_button = gr.Button("Generate", variant="primary", size="lg", scale=1, elem_id="gen-btn")
        status = gr.Markdown("", elem_id="status")

        rewritten_prompt_output = gr.Textbox(label="Rewritten prompt", lines=3, max_lines=20, interactive=False,
                                             placeholder="With prompt enhancement on, the rewritten prompt appears here.")

        with gr.Accordion("Advanced settings", open=False):
            with gr.Row():
                resolution = gr.Dropdown(label="Resolution", choices=RESOLUTION_CHOICES, value=DEFAULT_RESOLUTION,
                                         info="For Normal mode. With 2+ input images it is lowered automatically.")
                aspect_ratio = gr.Dropdown(label="Aspect ratio", choices=["Auto"] + ASPECT_RATIOS, value="Auto",
                                           info="Auto follows the input image, or square for text-to-image.")
            with gr.Row():
                seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0)
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
            gr.Markdown("Sampling uses the LoRA's trained 2-step schedule (fixed sigmas [1.0, 0.5]) with CFG 1. "
                        "The RunPod worker stays warm for the endpoint's idle timeout after the last request, "
                        "so repeated generations skip the cold start.")

        final_prompt_box = gr.Textbox(visible=False, label="Prompt")

        gr.Markdown("### Examples")
        gr.Examples(examples=T2I_EXAMPLES, example_labels=T2I_LABELS, inputs=[prompt, aspect_ratio],
                    label="Text to image")
        gr.Examples(examples=EDIT_EXAMPLES, example_labels=EDIT_LABELS, inputs=[prompt, input_images],
                    label="Image editing (loads the reference images too)")

    generate_button.click(
        prepare, [prompt, input_images, enhance, seed, randomize_seed, aspect_ratio],
        [final_prompt_box, rewritten_prompt_output, seed, aspect_ratio],
    ).then(
        generate_image, [final_prompt_box, input_images, seed, resolution, aspect_ratio, mode], [result, status],
        api_name="generate", concurrency_limit=None,
    )

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=None).launch(theme=THEME, css=CSS, allowed_paths=[EXAMPLE_DIR])
