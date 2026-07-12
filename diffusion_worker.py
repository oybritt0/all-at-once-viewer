"""
diffusion_worker.py — local hallucination worker for the latent viewer.

FastAPI service the viewer talks to. Receives worlds-mode linework frames
plus a prompt assembled from the latent neighborhood, runs them through the
SDXL + MistoLine + custom-LoRA stack, and serves results back. Strictly
serial (one job at a time) — the A3000's 6 GB leaves no headroom for
concurrency. The viewer polls; latency per frame is ~30–60 s with
sequential CPU offload.

Two job kinds share the same queue:
    single      "hallucinate this view" — result shows up in the viewer strip.
    batch       recorded camera paths — frames land in
                data/hallucinations/<tag>/frame_00000.png etc., then
                assemble_video.py builds the mp4.

Run (from project root, in the same env as your Gradio diffusion app):

    python diffusion_worker.py ^
        --model  path\\to\\juggernautXL_ragnarok.safetensors ^
        --lora   path\\to\\your_lora.safetensors ^
        --port 8787

Requires: fastapi, uvicorn  (pip install fastapi uvicorn)
plus your existing pinned stack: torch 2.5.1+cu121, diffusers 0.30.2,
transformers 4.44.2, peft 0.12.0, huggingface_hub 0.24.6.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import queue
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

DEFAULT_PROMPT_PREFIX = (
    "semi-abstract painterly iconic figures materializing from mist, "
    "faceless cartoon figures, "
)
DEFAULT_NEGATIVE = (
    "photorealistic, photograph, text, watermark, signature, frame, border, "
    "low quality, jpeg artifacts"
)

OUT_ROOT = Path("data") / "hallucinations"


class Job(BaseModel):
    image_b64: str                 # png, black linework on white (viewer capture)
    prompt: str = ""
    negative: str = DEFAULT_NEGATIVE
    seed: int = -1                 # -1 = random
    steps: int = 28
    cfg: float = 4.5
    control_scale: float = 0.9
    control_end: float = 0.6
    lora_scale: float = 0.75
    width: int = 1024
    height: int = 1024
    tag: str = ""                  # batch tag; empty = single job
    frame: int = -1                # frame index within a batch


# ----------------------------------------------------------------------------
# Pipeline (lazy-loaded on first job so the API comes up instantly)
# ----------------------------------------------------------------------------

class Engine:
    def __init__(self, model_path: str, lora_path: str | None, controlnet_id: str):
        self.model_path = model_path
        self.lora_path = lora_path
        self.controlnet_id = controlnet_id
        self.pipe = None
        self.lock = threading.Lock()

    def _load(self):
        import torch
        from diffusers import (
            StableDiffusionXLControlNetPipeline,
            ControlNetModel,
            AutoencoderKL,
            DPMSolverMultistepScheduler,
        )
        print("[engine] loading controlnet:", self.controlnet_id)
        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_id, torch_dtype=torch.float16, variant="fp16"
        )
        print("[engine] loading vae fp16-fix")
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )
        print("[engine] loading base model:", self.model_path)
        pipe = StableDiffusionXLControlNetPipeline.from_single_file(
            self.model_path,
            controlnet=controlnet,
            vae=vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )
        if self.lora_path:
            print("[engine] loading lora:", self.lora_path)
            pipe.load_lora_weights(self.lora_path, adapter_name="custom")
        # 6 GB VRAM: sequential offload (NOT model offload), cpu generator.
        pipe.enable_sequential_cpu_offload()
        try:
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
        except Exception:
            pass
        self.pipe = pipe
        print("[engine] ready")

    @staticmethod
    def prep_control(image_b64: str, w: int, h: int) -> Image.Image:
        """Viewer sends black linework on white; MistoLine wants white on black."""
        raw = base64.b64decode(image_b64.split(",")[-1])
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = ImageOps.invert(img)
        # letterbox onto black to the target size, preserving aspect
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        img.thumbnail((w, h), Image.LANCZOS)
        canvas.paste(img, ((w - img.width) // 2, (h - img.height) // 2))
        return canvas

    def run(self, job: Job) -> Image.Image:
        import torch
        with self.lock:
            if self.pipe is None:
                self._load()
            if self.lora_path:
                self.pipe.set_adapters(["custom"], adapter_weights=[job.lora_scale])
            control = self.prep_control(job.image_b64, job.width, job.height)
            seed = job.seed if job.seed >= 0 else int(time.time_ns() % (2**31))
            gen = torch.Generator(device="cpu").manual_seed(seed)
            prompt = job.prompt.strip() or DEFAULT_PROMPT_PREFIX.rstrip(", ")
            result = self.pipe(
                prompt=prompt,
                negative_prompt=job.negative,
                image=control,
                num_inference_steps=job.steps,
                guidance_scale=job.cfg,
                controlnet_conditioning_scale=job.control_scale,
                control_guidance_end=job.control_end,
                width=job.width,
                height=job.height,
                generator=gen,
            ).images[0]
            return result


# ----------------------------------------------------------------------------
# Queue + API
# ----------------------------------------------------------------------------

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # streamlit component iframe origin varies
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: dict[str, dict] = {}         # id -> {status, job, result_b64, path, error}
Q: "queue.Queue[str]" = queue.Queue()
ENGINE: Engine | None = None


def worker_loop():
    while True:
        jid = Q.get()
        rec = JOBS.get(jid)
        if rec is None:
            continue
        rec["status"] = "running"
        rec["started"] = time.time()
        try:
            img = ENGINE.run(Job(**rec["job"]))
            job = Job(**rec["job"])
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            if job.tag:
                out_dir = OUT_ROOT / job.tag
                out_dir.mkdir(parents=True, exist_ok=True)
                name = f"frame_{job.frame:05d}.png" if job.frame >= 0 else f"{jid}.png"
                out_path = out_dir / name
                out_path.write_bytes(buf.getvalue())
                rec["path"] = str(out_path)
                # sidecar provenance
                (out_dir / (out_path.stem + ".json")).write_text(json.dumps({
                    "prompt": job.prompt, "seed": job.seed, "steps": job.steps,
                    "cfg": job.cfg, "control_scale": job.control_scale,
                    "control_end": job.control_end,
                }, indent=2), encoding="utf-8")
            else:
                out_dir = OUT_ROOT / "singles"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{jid[:8]}.png"
                out_path.write_bytes(buf.getvalue())
                rec["path"] = str(out_path)
            rec["result_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")
            rec["status"] = "done"
        except Exception as ex:
            rec["status"] = "error"
            rec["error"] = str(ex)
            print(f"[worker] job {jid} failed: {ex}")
        finally:
            rec["finished"] = time.time()
            # keep memory bounded: drop b64 payloads of old batch frames
            if len(JOBS) > 40:
                done = [k for k, v in JOBS.items()
                        if v["status"] in ("done", "error") and v.get("job", {}).get("tag")]
                for k in done[:-10]:
                    JOBS[k].pop("result_b64", None)


@app.post("/jobs")
def submit(job: Job):
    jid = uuid.uuid4().hex
    JOBS[jid] = {"status": "queued", "job": job.model_dump(), "submitted": time.time()}
    Q.put(jid)
    return {"id": jid, "queued": Q.qsize()}


@app.get("/jobs/{jid}")
def status(jid: str):
    rec = JOBS.get(jid)
    if rec is None:
        return {"status": "unknown"}
    out = {"status": rec["status"], "queued": Q.qsize()}
    if rec["status"] == "done":
        out["image_b64"] = rec.get("result_b64", "")
        out["path"] = rec.get("path", "")
    if rec["status"] == "error":
        out["error"] = rec.get("error", "")
    return out


@app.get("/health")
def health():
    return {"ok": True, "queued": Q.qsize(),
            "loaded": ENGINE is not None and ENGINE.pipe is not None}


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="juggernautXL_ragnarok .safetensors path")
    ap.add_argument("--lora", default=None, help="custom LoRA .safetensors path")
    ap.add_argument("--controlnet", default="TheMistoAI/MistoLine",
                    help="HF id or local path of the line ControlNet")
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()

    ENGINE = Engine(args.model, args.lora, args.controlnet)
    threading.Thread(target=worker_loop, daemon=True).start()
    print(f"[worker] listening on http://127.0.0.1:{args.port}  "
          f"(viewer connects automatically)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
