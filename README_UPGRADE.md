# Viewer upgrade: figural boundaries + hallucination bridge

Four files. `app.py` replaces the viewer in the viewer repo. The other three
sit next to it in the project root (the directory holding `data/`).

## What changed and why

**Path 1 — figural boundary curves.** The browser pipeline could only trace
isolines of Sobel edge energy, which is why worlds boundaries never resolved
into pillars, arches, or profiles. Extraction now happens offline in
`extract_geometry.py` with learned edges (PiDiNet/HED) and optional SAM
silhouettes, written to `data/embeddings/content_geometry.json`. The viewer
loads that sidecar at startup and uses it in place of the Sobel polygons,
falling back per-image where the sidecar has no entry. Camera responsiveness
is untouched: extraction was already separated from projection, so the
per-frame worlds math is identical.

On top of that sits **curve quoting**, a new `worlds quote` slider under
`worlds unify`. At quote > 0 each union boundary is resampled into segments,
each segment's turning signature retrieves the best-matching curve fragment
from that world's own members (either traversal direction), and the top
fraction of matches replaces the original runs with the fragment geometry,
endpoints pinned so the ring stays closed. The region ends up bounded by
arches and profiles quoted from its own corpus. Fragments join by adjacency
at segment endpoints. Chord, not flow: no interpolated skin anywhere.

Division of labor across media falls out of the extraction itself: line art
(drawings, prints, construction documents) contributes curve fragments;
filled works (paintings, photographs of objects) contribute silhouettes and
medial axes. No media type is privileged.

**Path 2 — hallucination bridge.** Real-time video is out of reach on 6 GB,
so the bridge is asynchronous. `diffusion_worker.py` is a local FastAPI
service wrapping the existing SDXL + MistoLine + LoRA stack (pinned stack,
sequential CPU offload, fp16-fix VAE, cpu generator). The viewer gets a
`hallucinate` panel: the current worlds frame becomes the ControlNet
conditioning image, and the prompt is assembled from the latent neighborhood
around the orbit target, the k nearest members' periods and media,
distance-weighted. Navigating the space navigates the prompt. Results appear
in a strip in the panel (~30–60 s per frame). `auto on settle` fires one
hallucination whenever the camera comes to rest.

For animation: `record path` samples the camera while you navigate, `queue
path` interpolates it to N frames and submits the batch. The worker writes
frames plus provenance sidecars to `data/hallucinations/<tag>/`, and
`assemble_video.py` builds the mp4. Per-frame flicker is deliberately not
smoothed: adjacent frames are independent registrations of the same linework
and their disagreement is content. Temporal smoothing would trade that for
parametricist flow.

## Install

In the viewer environment:

    pip install controlnet-aux

Optional but the biggest fidelity win, in the same environment:

    pip install segment-anything
    # download sam_vit_b_01ec64.pth (~375 MB) from the segment-anything repo

In the diffusion environment (same one as the Gradio app):

    pip install fastapi uvicorn

`ffmpeg` on PATH for video assembly.

## Run order

1. Extract, from the project root (same place you run streamlit):

       python extract_geometry.py
       python extract_geometry.py --sam sam_vit_b_01ec64.pth   # better

   `--limit 20` for a quick smoke test, `--edges xdog` if controlnet-aux
   gives trouble, `--root PATH` if running from elsewhere. Re-run after any
   corpus change; the viewer picks up the new sidecar on reload.

2. Viewer as usual. Worlds mode now draws sidecar geometry. Set
   `worlds unify` > 0, then raise `worlds quote`.

3. Worker, whenever hallucination is wanted:

       python diffusion_worker.py --model path\to\juggernautXL_ragnarok.safetensors --lora path\to\your_lora.safetensors

   The panel header flips to `(worker online)` within a few seconds.

4. After a queued path finishes:

       python assemble_video.py --tag path_2026-07-07T21-30-00 --fps 12

## Tuning notes

- Quote segment scale is `minDim * 0.06` minimum, ~28 segments per boundary.
  If quotes read too small at exhibition scale, raise the `0.06` and `/ 28`
  in `quoteContour`.
- Match acceptance threshold is `0.45` mean turning distance plus a
  chord-ratio penalty. Lower = stricter quoting, fewer replacements.
- Fragment pool per world caps at 360 to keep the redraw cheap.
- The hallucination prompt template lives in `halluNeighborhoodPrompt` in
  app.py and `DEFAULT_PROMPT_PREFIX` in the worker. The panel's `prompt +`
  field appends without editing code.
