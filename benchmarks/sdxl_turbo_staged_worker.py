import argparse
import importlib.metadata
import json
import os
import platform
import threading
import time
from pathlib import Path

import psutil
import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline, UNet2DConditionModel

BASE = "stabilityai/sdxl-turbo"
PROMPTS = [
    ("01_red_fox_snow_dawn", "a red fox in snow at dawn, photographic"),
    ("02_tavern_sign_open_late", 'a hand-drawn tavern sign that says "OPEN LATE"'),
    ("03_futuristic_city_rain", "a futuristic city street at night in rain"),
    ("04_ramen_closeup", "a bowl of ramen, close-up"),
    ("05_elderly_fisherman", "a portrait of an elderly fisherman"),
]
SEEDS = [20261018 + i for i in range(len(PROMPTS))]
proc = psutil.Process()
peak_rss = 0
stop_sampling = False


def sample_tree():
    total = proc.memory_info().rss
    for child in proc.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total


def sample_memory():
    global peak_rss
    while not stop_sampling:
        try:
            peak_rss = max(peak_rss, sample_tree())
        except psutil.Error:
            pass
        time.sleep(0.2)


def versions():
    names = ["diffusers", "huggingface-hub", "numpy", "Pillow", "psutil", "safetensors", "torch", "torchvision", "transformers"]
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def write_metrics(path, payload):
    payload["peak_rss_bytes"] = peak_rss
    payload["peak_rss_mib"] = round(peak_rss / 1024 / 1024, 2)
    payload["python"] = os.sys.version
    payload["platform"] = platform.platform()
    payload["package_versions"] = versions()
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def encode(args):
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    started = time.perf_counter()
    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE, unet=None, vae=None, torch_dtype=torch.float32
    )
    model_load_seconds = time.perf_counter() - started
    encode_started = time.perf_counter()
    prompts = [prompt for _, prompt in PROMPTS]
    with torch.inference_mode():
        encoded = pipe.encode_prompt(
            prompt=prompts,
            device="cpu",
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    prompt_embeds, pooled_prompt_embeds = encoded[0], encoded[2]
    torch.save(
        {"prompt_embeds": prompt_embeds.cpu(), "pooled_prompt_embeds": pooled_prompt_embeds.cpu()},
        args.embeddings,
    )
    encoding_seconds = time.perf_counter() - encode_started
    del pipe
    global stop_sampling
    stop_sampling = True
    monitor.join(timeout=2)
    peak = max(peak_rss, sample_tree())
    write_metrics(
        args.metrics,
        {
            "phase": "encode",
            "model": BASE,
            "precision": "float32",
            "prompt_count": len(PROMPTS),
            "model_load_seconds": round(model_load_seconds, 3),
            "prompt_encoding_seconds": round(encoding_seconds, 3),
            "phase_wall_seconds": round(time.perf_counter() - started, 3),
            "peak_rss_bytes": peak,
            "peak_rss_mib": round(peak / 1024 / 1024, 2),
        },
    )


def generate(args):
    torch.set_num_threads(min(2, os.cpu_count() or 1))
    started = time.perf_counter()
    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    embeddings = torch.load(args.embeddings, map_location="cpu", weights_only=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE,
        unet=None,
        vae=None,
        text_encoder=None,
        text_encoder_2=None,
        torch_dtype=torch.float32,
    )
    unet = UNet2DConditionModel.from_pretrained(BASE, subfolder="unet", torch_dtype=torch.float32)
    vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae", torch_dtype=torch.float32)
    pipe.register_modules(unet=unet, vae=vae)
    pipe.enable_attention_slicing()
    pipe = pipe.to("cpu")
    dtypes = sorted({str(parameter.dtype) for parameter in pipe.unet.parameters()})
    if dtypes != [str(torch.float32)]:
        raise SystemExit(f"UNet parameter dtype mismatch: expected float32, got {dtypes}")
    model_load_seconds = time.perf_counter() - started
    print(f"unet_parameter_dtypes={dtypes}", flush=True)
    print(f"load_seconds={model_load_seconds:.3f}", flush=True)
    records = []
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    generation_started = time.perf_counter()
    for index, (slug, prompt) in enumerate(PROMPTS):
        path = output_dir / f"{slug}.png"
        t0 = time.perf_counter()
        generator = torch.Generator(device="cpu").manual_seed(SEEDS[index])
        with torch.inference_mode():
            image = pipe(
                prompt_embeds=embeddings["prompt_embeds"][index:index + 1],
                pooled_prompt_embeds=embeddings["pooled_prompt_embeds"][index:index + 1],
                width=512,
                height=512,
                num_inference_steps=1,
                guidance_scale=0.0,
                generator=generator,
            ).images[0]
        image.save(path, format="PNG")
        record = {
            "prompt": prompt,
            "seed": SEEDS[index],
            "path": str(path),
            "wall_seconds": round(time.perf_counter() - t0, 3),
            "output_bytes": path.stat().st_size,
        }
        records.append(record)
        print(f"completed={slug} wall_seconds={record['wall_seconds']:.3f} output_bytes={record['output_bytes']}", flush=True)
    generation_seconds = time.perf_counter() - generation_started
    global stop_sampling
    stop_sampling = True
    monitor.join(timeout=2)
    peak = max(peak_rss, sample_tree())
    write_metrics(
        args.metrics,
        {
            "phase": "generate",
            "model": BASE,
            "precision": "float32",
            "steps": 1,
            "guidance_scale": 0.0,
            "resolution": "512x512",
            "model_load_seconds": round(model_load_seconds, 3),
            "generation_seconds": round(generation_seconds, 3),
            "total_wall_seconds": round(time.perf_counter() - started, 3),
            "records": records,
            "peak_rss_bytes": peak,
            "peak_rss_mib": round(peak / 1024 / 1024, 2),
        },
    )


parser = argparse.ArgumentParser()
parser.add_argument("phase", choices=["encode", "generate"])
parser.add_argument("--embeddings", required=True)
parser.add_argument("--metrics", required=True)
parser.add_argument("--output", default="benchmark-output/sdxl-turbo-staged")
args = parser.parse_args()
if args.phase == "encode":
    encode(args)
else:
    generate(args)
