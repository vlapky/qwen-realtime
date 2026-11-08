# RunPod Serverless worker for Qwen-Image-2.1 + 2-step DMD2 LoRA (see runpod_worker/handler.py).
# The base model is NOT baked in: set the endpoint's "Model" field to Qwen/Qwen-Image-2.1 so RunPod
# pre-caches it on the host (/runpod-volume/huggingface-cache/hub). LoRA + AOTI kernels (~76 MB) are baked.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# AOTI packages were compiled with exactly this torch build (needs a CUDA 13.0 capable driver).
RUN pip install torch==2.13.0 torchvision --index-url https://download.pytorch.org/whl/cu130

# Same diffusers pin as the Space (QwenImage21Pipeline, output_resolution, use_kv_cache).
RUN pip install \
    "git+https://github.com/huggingface/diffusers@da1d3829cf08d4f329b526d89e17cc035c049d8d" \
    "transformers>=5.17.0" accelerate peft pillow safetensors huggingface_hub "runpod==2.1.0"

RUN hf download linoyts/Qwen-Image-2.1-DMD2-2steps pytorch_lora_weights_v1_5.safetensors --local-dir /models/lora \
 && hf download hugging-apps/qwen-image-2-1-aoti --include "config.json" --include "*/package.pt2" --local-dir /models/aoti \
 && rm -rf /models/*/.cache

WORKDIR /app
COPY runpod_worker/handler.py runpod_worker/qwen21_aoti.py ./

CMD ["python", "-u", "handler.py"]
