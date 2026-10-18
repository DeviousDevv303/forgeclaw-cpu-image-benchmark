import importlib.metadata
import json
import os
import platform
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
PRECISION = os.environ.get("PRECISION", "float32").lower()
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
if PRECISION not in DTYPES:
    raise SystemExit(f"unsupported precision: {PRECISION}")
DTYPE = DTYPES[PRECISION]
SD15_STEPS = int(os.environ.get("SD15_STEPS", "20"))
if SD15_STEPS not in (20, 30):
    raise SystemExit(f"unsupported SD1.5 step count: {SD15_STEPS}")
OUT = Path("benchmark-output") / MODEL
OUT.mkdir(parents=True, exist_ok=True)

DEPENDENCIES = ["accelerate", "diffusers", "huggingface-hub", "numpy", "pillow", "psutil", "safetensors", "torch", "torchvision", "transformers"]
package_versions = {}
for package in DEPENDENCIES:
    try:
        package_versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        package_versions[package] = "not-installed"

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


print(f"model={MODEL}", flush=True)
print(f"precision={PRECISION} dtype={DTYPE}", flush=True)
print(f"cpu={platform.processor() or platform.machine()} logical_cpus={os.cpu_count()}", flush=True)
print(f"torch_threads_before={torch.get_num_threads()}", flush=True)
torch.set_num_threads(min(2, os.cpu_count() or 1))
print(f"torch_threads_used={torch.get_num_threads()}", flush=True)
print(f"package_versions={json.dumps(package_versions, sort_keys=True)}", flush=True)

monitor = threading.Thread(target=sample_memory, daemon=True)
monitor.start()
started = time.perf_counter()
records = []

if MODEL == "sd15":
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", torch_dtype=DTYPE, use_safetensors=True
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    steps, guidance = SD15_STEPS, 7.5
elif MODEL == "sdxl-turbo":
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=DTYPE
    )
    pipe.enable_attention_slicing()
    steps, guidance = 1, 0.0
elif MODEL == "sdxl-lightning":
    from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    base = "stabilityai/stable-diffusion-xl-base-1.0"
    unet = UNet2DConditionModel.from_config(base, subfolder="unet")
    unet = unet.to(dtype=DTYPE)
    ckpt = hf_hub_download("ByteDance/SDXL-Lightning", "sdxl_lightning_1step_unet_x0.safetensors")
    unet.load_state_dict(load_file(ckpt, device="cpu"))
    pipe = StableDiffusionXLPipeline.from_pretrained(base, unet=unet, torch_dtype=DTYPE)
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, timestep_spacing="trailing", prediction_type="sample"
    )
    pipe.enable_attention_slicing()
    steps, guidance = 1, 0.0
else:
    raise SystemExit(f"unsupported diffusers model: {MODEL}")

pipe = pipe.to("cpu")
unet_dtypes = sorted({str(parameter.dtype) for parameter in pipe.unet.parameters()})
print(f"unet_parameter_dtypes={unet_dtypes}", flush=True)
if unet_dtypes != [str(DTYPE)]:
    raise SystemExit(f"UNet parameter dtype mismatch: expected {DTYPE}, got {unet_dtypes}")
load_done = time.perf_counter()
print(f"load_seconds={load_done - started:.3f}", flush=True)

for index, (slug, prompt) in enumerate(PROMPTS):
    path = OUT / f"{slug}.png"
    t0 = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(20261018 + index)
    kwargs = dict(
        prompt=prompt,
        width=512,
        height=512,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    )
    if MODEL == "sd15":
        kwargs["negative_prompt"] = "blurry, low quality, distorted, watermark, signature"
    with torch.inference_mode():
        result = pipe(**kwargs).images[0]
    result.save(path, format="PNG")
    elapsed = time.perf_counter() - t0
    size = path.stat().st_size
    records.append(
        {"prompt": prompt, "path": str(path), "wall_seconds": round(elapsed, 3), "output_bytes": size}
    )
    print(f"completed={slug} wall_seconds={elapsed:.3f} output_bytes={size}", flush=True)

stop = True
monitor.join(timeout=2)
peak_rss = max(peak_rss, rss_tree())
metrics = {
    "model": MODEL,
    "precision": PRECISION,
    "torch_dtype": str(DTYPE),
    "steps": steps,
    "guidance_scale": guidance,
    "runner_os": os.environ.get("RUNNER_OS"),
    "python": os.sys.version,
    "platform": platform.platform(),
    "logical_cpus": os.cpu_count(),
    "torch_threads": torch.get_num_threads(),
    "package_versions": package_versions,
    "total_wall_seconds": round(time.perf_counter() - started, 3),
    "load_seconds": round(load_done - started, 3),
    "peak_rss_bytes": peak_rss,
    "peak_rss_mib": round(peak_rss / 1024 / 1024, 2),
    "resolution": "512x512",
    "records": records,
}
(OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps(metrics, indent=2), flush=True)
