import json
import os
import threading
import time
from pathlib import Path

import psutil
import torch

PROMPTS = [
    ("01_red_fox_snow_dawn", "a red fox in snow at dawn, photographic"),
    ("02_tavern_sign_open_late", 'a hand-drawn tavern sign that says "OPEN LATE"'),
    ("03_futuristic_city_rain", "a futuristic city street at night in rain"),
    ("04_ramen_closeup", "a bowl of ramen, close-up"),
    ("05_elderly_fisherman", "a portrait of an elderly fisherman"),
]

MODEL = os.environ["MODEL"]
OUT = Path("benchmark-output") / MODEL
OUT.mkdir(parents=True, exist_ok=True)

proc = psutil.Process()
peak_rss = 0
stop = False

def rss_tree():
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total

def sample_memory():
    global peak_rss
    while not stop:
        try:
            peak_rss = max(peak_rss, rss_tree())
        except psutil.Error:
            pass
        time.sleep(0.2)

monitor = threading.Thread(target=sample_memory, daemon=True)
monitor.start()
started = time.perf_counter()
records = []

print(f"model={MODEL}", flush=True)
print(f"torch={torch.__version__} threads={torch.get_num_threads()}", flush=True)
torch.set_num_threads(min(2, os.cpu_count() or 1))

if MODEL == "sd15":
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32, use_safetensors=True
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    steps, guidance = 20, 7.5
elif MODEL == "sdxl-turbo":
    from diffusers import AutoPipelineForText2Image
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float32
    )
    pipe.enable_attention_slicing()
    steps, guidance = 1, 0.0
elif MODEL == "sdxl-lightning":
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    unet = UNet2DConditionModel.from_config(base, subfolder="unet", torch_dtype=torch.float32)
    ckpt = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_1step_unet_x0.safetensors")
    unet.load_state_dict(load_file(ckpt, device="cpu"))
    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=torch.float32)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing", prediction_type="sample"
    )
    pipe.enable_attention_slicing()
    steps, guidance = 1, 0.0
else:
    raise SystemExit(f"unsupported diffusers model: {MODEL}")

pipe = pipe.to("cpu")
load_done = time.perf_counter()
print(f"load_seconds={load_done - started:.3f}", flush=True)

for index, (slug, prompt) in enumerate(PROMPTS):
    path = OUT / f"{slug}.png"
    t0 = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(20261018 + index)
    kwargs = dict(prompt=prompt, width=512, height=512, num_inference_steps=steps,
                  guidance_scale=guidance, generator=generator)
    if MODEL == "sd15":
        kwargs["negative_prompt"] = "blurry, low quality, distorted, watermark, signature"
    result = pipe(**kwargs).images[0]
    result.save(path, format="PNG")
    elapsed = time.perf_counter() - t0
    size = path.stat().st_size
    records.append({"prompt": prompt, "path": str(path), "wall_seconds": round(elapsed, 3), "output_bytes": size})
    print(f"completed={slug} wall_seconds={elapsed:.3f} output_bytes={size}", flush=True)

stop = True
monitor.join(timeout=2)
peak_rss = max(peak_rss, rss_tree())
metrics = {
    "model": MODEL,
    "runner_os": os.environ.get("RUNNER_OS"),
    "python": os.sys.version,
    "torch": torch.__version__,
    "total_wall_seconds": round(time.perf_counter() - started, 3),
    "load_seconds": round(load_done - started, 3),
    "peak_rss_bytes": peak_rss,
    "peak_rss_mib": round(peak_rss / 1024 / 1024, 2),
    "resolution": "512x512",
    "records": records,
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2), flush=True)
