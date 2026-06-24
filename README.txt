All, At Once - latent space viewer
==================================

WHAT YOU NEED
  - Python 3.10 or 3.11 installed (python.org).
  - An internet connection the first time you launch (it installs a few
    packages) and while using the 3D worlds view (it loads the 3D library
    from a CDN). Everything else runs locally on your machine.

TO RUN
  Windows:        double-click  run-windows.bat
  macOS / Linux:  in a terminal, run:  bash run-mac-linux.command

  The first launch takes a minute while it sets things up. After that it
  opens in your web browser. Leave the little terminal window open while
  you use it; closing it stops the viewer.

NOTES
  - Keep this folder intact. The app finds its data by the folder layout,
    so don't move app.py or the data/ folder out of here.
  - Your exports are written into data/exports/ inside this folder.
# All At Once — Latent / Constellation Viewer

Streamlit + Three.js viewers for the *All, At Once* / *Histories* latent space. The app reads CLIP embeddings and a 3D UMAP projection over the heterogeneous dataset (ten art-historical periods) and renders an interactive explorer.

## Important: the dataset is not in this repo

The repo contains **viewer code and small metadata only**. The image dataset (the `NN_period` folders) and the large embedding arrays are maintained separately by the team and are excluded by `.gitignore`. A fresh clone will start but will report missing data files until you add them.

You must obtain the following from the team's data source and place them as shown:

```
all-at-once-viewer/
├── app.py                     # canonical entry point
├── data/
│   ├── catalog/
│   │   └── manifest.json      # tracked (small)
│   └── embeddings/
│       ├── index.json         # tracked (small)
│       ├── latent_2d.json     # tracked (small)
│       ├── clip_features.npy  # NOT tracked — add this
│       └── clip_umap_3d.npy   # NOT tracked — add this (optional, see below)
└── 01_prehistoric/ … 10_parametric/   # NOT tracked — add the period image tree
```

`clip_features.npy` and the period image folders are required. `clip_umap_3d.npy` is optional: if absent, the app computes the 3D UMAP on first run and writes it to that path (slow once, cached after).

## Setup

Requires Python 3.10. Create a fresh virtual environment — the repo does not ship one.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The UI opens at http://localhost:8501.

On Windows you can instead double-click `run-windows.bat`; on macOS/Linux run `run-mac-linux.command`. Both wrap the same `streamlit run app.py`.

## Entry points

`app.py` is the canonical viewer. The other files are kept for history and reference:

- `app.py` — **canonical** latent explorer.
- `app_constellation.py` — star-field constellation viewer.
- `app_detailed_trace.py` — variant with enhanced image-tracing fidelity for worlds mode.
- `og_app.py`, `original_app.py`, `original_app_constellation.py` — earlier snapshots. Do not edit; superseded by the files above.
- `hull_from_worlds.py` — shape-from-silhouette tool that turns worlds-mode linework into a 3D hull (`example_worlds_hull.obj/.mtl` is sample output).

## How data is located

`app.py` walks up from its own location looking for `data/catalog/manifest.json` to find the project root, then loads everything under `data/`. Keep `app.py` and the `data/` folder together at the repo root and the resolution works without configuration.

## Notes

- Streamlit caches the embedded component aggressively. After changing rendering code, do a full server restart and a hard browser refresh.
- The dataset is heterogeneous (photographs, drawings, paintings, sculpture, point clouds, construction documents). It is not buildings-only; treat all members as co-equal representations.
