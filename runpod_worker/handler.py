"""RunPod Serverless worker: Qwen-Image-2.1 + 2-step DMD2 LoRA (same setup as the HF Space
linoyts/qwen-image-2-1-realtime): LoRA fused into the weights, fixed sigmas [1.0, 0.5],
CFG 1, AOTInductor kernels for the transformer blocks and VAE decoder (H100/H200 only).

The base model comes from RunPod's cached models (endpoint "Model" field = Qwen/Qwen-Image-2.1),
mounted under /runpod-volume/huggingface-cache/hub. LoRA and AOTI packages are baked into the image.
Everything loads once per worker at import time, before the first job.

Input:  {"prompt": str, "images": [base64, ...], "seed": int, "resolution": 512|1024|1536,
         "aspect_ratio": "Auto"|"1:1"|..., "format": "png"|"webp"}
        {"warmup": true}  -> only reports that the worker is up
Output: {"image": base64, "format": str, "elapsed": s, "size": [w, h], "resolution": int, "aoti": [...]}
"""

import base64
import glob
import io
import os
import time

import runpod
import torch
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
from PIL import Image

MODEL_ID = "Qwen/Qwen-Image-2.1"
CACHED_HUB = "/runpod-volume/huggingface-cache/hub"
LORA_PATH = os.environ.get("LORA_PATH", "/models/lora/pytorch_lora_weights_v1_5.safetensors")
AOTI_DIR = os.environ.get("AOTI_DIR", "/models/aoti")

SIGMAS = [1.0, 0.5]  # the LoRA's trained schedule
NUM_STEPS = len(SIGMAS)
MAX_SEED = 2**31 - 1
MAX_INPUT_IMAGES = 10
BASE_RESOLUTION = 2048
ASPECT_RATIOS = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}
KV_CACHE_BYTES_PER_TOKEN = 32 * 2 * 4096 * 2
KV_CACHE_BUDGET_GB = 10.0


def kv_cache_fits(n_images, resolution):
    return n_images * (int(resolution) // 16) ** 2 * KV_CACHE_BYTES_PER_TOKEN <= KV_CACHE_BUDGET_GB * 1e9


def fit_resolution(n_images, resolution):
    """Largest tier <= request at which the condition-image prefix KV cache stays affordable."""
    resolution = int(resolution)
    if n_images == 0:
        return resolution
    for tier in (1536, 1024, 512):
        if tier <= resolution and kv_cache_fits(n_images, tier):
            return tier
    return min(resolution, 512)


def resolve_size(resolution, aspect_ratio):
    if not aspect_ratio or aspect_ratio not in ASPECT_RATIOS:
        return None, None
    base_w, base_h = ASPECT_RATIOS[aspect_ratio]
    scale = int(resolution) / BASE_RESOLUTION
    return tuple(max(32, int(round(v * scale / 32)) * 32) for v in (base_w, base_h))


def model_path():
    """Local snapshot from RunPod's model cache, else the Hub id (downloads on first start)."""
    snaps = sorted(glob.glob(f"{CACHED_HUB}/models--{MODEL_ID.replace('/', '--')}/snapshots/*/"))
    if snaps:
        return snaps[-1]
    print(f"[load] no cached model under {CACHED_HUB}; downloading {MODEL_ID} from the Hub", flush=True)
    return MODEL_ID


# --------------------------------------------------------------------------------------
# Load once per worker
# --------------------------------------------------------------------------------------

t0 = time.time()
path = model_path()
pipe = QwenImage21Pipeline.from_pretrained(path, torch_dtype=torch.bfloat16)
pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
    pipe.scheduler.config, use_dynamic_shifting=False, shift=1.0, shift_terminal=None
)
# Fuse the LoRA so the AOTI graphs (compiled for the base arch, weights as inputs) apply unchanged.
pipe.load_lora_weights(os.path.dirname(LORA_PATH), weight_name=os.path.basename(LORA_PATH), adapter_name="v1")
pipe.fuse_lora()
pipe.unload_lora_weights()
pipe = pipe.to("cuda")
pipe.vae.enable_tiling(
    tile_sample_min_height=1536,
    tile_sample_min_width=1536,
    tile_sample_stride_height=1152,
    tile_sample_stride_width=1152,
)
print(f"[load] {path} + LoRA on {torch.cuda.get_device_name()} in {time.time() - t0:.0f}s", flush=True)

AOTI = []
capability = torch.cuda.get_device_capability()
if os.environ.get("QWEN21_AOTI", "1") == "0":
    print("[load] AOTI disabled by QWEN21_AOTI=0", flush=True)
elif capability != (9, 0):
    print(f"[load] AOTI kernels are built for sm_90 (H100/H200); this GPU is sm_{capability[0]}{capability[1]}, "
          "running eager", flush=True)
else:
    try:
        from qwen21_aoti import aoti_load_pipeline

        AOTI, cfg = aoti_load_pipeline(pipe, AOTI_DIR)
        print(f"[load] AOTI {AOTI} (built for torch {cfg['torch']})", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[load] AOTI unavailable, running eager: {exc!r}", flush=True)


def render(prompt, images, seed, resolution, aspect_ratio):
    width, height = resolve_size(resolution, aspect_ratio)
    return pipe(
        prompt=prompt,
        image=images or None,
        height=height,
        width=width,
        output_resolution=int(resolution),
        num_inference_steps=NUM_STEPS,
        sigmas=SIGMAS,
        true_cfg_scale=1.0,
        use_kv_cache=kv_cache_fits(len(images), int(resolution)),
        generator=torch.Generator(device="cuda").manual_seed(int(seed) % (MAX_SEED + 1)),
    ).images[0]


# Warm-up: loads the AOTI .so files and initialises cuBLAS/cuDNN before the first real job.
t0 = time.time()
for res in (512, 1024):
    render("warm-up", [], 0, res, None)
torch.cuda.synchronize()
print(f"[load] warm-up done in {time.time() - t0:.0f}s; worker ready", flush=True)


# --------------------------------------------------------------------------------------
# Handler
# --------------------------------------------------------------------------------------


def handler(job):
    inp = job.get("input") or {}
    if inp.get("warmup"):
        return {"ready": True, "aoti": AOTI, "gpu": torch.cuda.get_device_name()}

    prompt = (inp.get("prompt") or "").strip()
    if not prompt:
        return {"error": "empty prompt"}
    images_b64 = inp.get("images") or []
    if len(images_b64) > MAX_INPUT_IMAGES:
        return {"error": f"up to {MAX_INPUT_IMAGES} input images are supported"}
    images = [Image.open(io.BytesIO(base64.b64decode(b))) for b in images_b64]
    resolution = fit_resolution(len(images), inp.get("resolution", 1024))
    fmt = "webp" if inp.get("format") == "webp" else "png"

    started = time.perf_counter()
    out = render(prompt, images, inp.get("seed", 0), resolution, inp.get("aspect_ratio", "Auto"))
    elapsed = time.perf_counter() - started

    buf = io.BytesIO()
    out.save(buf, format="WEBP", quality=90) if fmt == "webp" else out.save(buf, format="PNG")
    print(f"[generate] imgs={len(images)} res={resolution} size={out.size} {elapsed:.2f}s", flush=True)
    return {
        "image": base64.b64encode(buf.getvalue()).decode(),
        "format": fmt,
        "elapsed": elapsed,
        "size": list(out.size),
        "resolution": resolution,
        "aoti": AOTI,
    }


runpod.serverless.start({"handler": handler})
