"""
Constellation Explorer — a second viewer for the latent space.

Runs alongside app.py and reads the same artifacts (manifest, CLIP features,
2D/3D UMAP) that latent_embedding.ipynb writes into data/embeddings/. The
shape-generation logic is unchanged: positions are the 3D UMAP, and the same
Gaussian density field that drives the contours and enclosing volume in the
original viewer is reused here. The only thing that changes is the
representation.

Each representation becomes a star at its UMAP position. The density field is
re-read as light: regions where representations agree bloom into a bright,
milky core; conflict zones stay sparse and singular. Clusters become forms you
can ignite — the world-view act of drawing a boundary, translated into making
those stars brighter. The scene is WebXR-capable so it can be walked through in
a headset for the exhibition.

    streamlit run app_constellation.py

The "Download standalone VR page" button writes a self-contained .html with the
data baked in. Serve that over https (or localhost) for the headset; WebXR is
usually blocked inside Streamlit's component iframe.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import matplotlib
import matplotlib.colors as mcolors


# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Constellation Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="✦",
)
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 100%; }
  [data-testid="stSidebar"] { min-width: 340px; }
  h1 { margin-top: 0; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Paths  (identical resolution to app.py so the second copy finds the same data)
# =============================================================================
def _find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents][:5]:
        if (candidate / "data" / "catalog" / "manifest.json").exists():
            return candidate
    if start.name == "notebooks":
        return start.parent
    return start

_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _find_project_root(_HERE)
MANIFEST_PATH = PROJECT_ROOT / "data" / "catalog" / "manifest.json"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EXPORTS_DIR = PROJECT_ROOT / "data" / "exports" / "constellation"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

PALETTE_NAMES = [
    "tab10", "tab20", "Set1", "Set2", "Set3", "Dark2", "Paired", "Accent",
    "viridis", "plasma", "magma", "inferno", "cividis", "twilight", "hsv",
]
COLOR_FIELDS = ["cluster", "category", "subtype", "era", "none"]


# =============================================================================
# Palettes  (same categorical assignment as app.py)
# =============================================================================
def categorical_palette(values, palette_name="tab10"):
    uniq = sorted({str(v) for v in values}, key=lambda s: (s == "-1", s))
    try:
        cmap = matplotlib.colormaps[palette_name]
    except KeyError:
        cmap = matplotlib.colormaps["tab10"]
    N = getattr(cmap, "N", 256)
    is_listed = N <= 32
    n_non_noise = sum(1 for v in uniq if v != "-1")
    out = {}
    color_index = 0
    for v in uniq:
        if v == "-1":
            out[v] = "#888888"
            continue
        if is_listed:
            color = cmap(color_index % N)
        else:
            denom = max(n_non_noise - 1, 1)
            color = cmap(color_index / denom)
        out[v] = mcolors.to_hex(color)
        color_index += 1
    return out


def _hex_to_rgb01(h):
    c = mcolors.to_rgb(h)
    return [round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4)]


# =============================================================================
# Data loading  (same artifacts and conventions as app.py)
# =============================================================================
@st.cache_data(show_spinner="Loading manifest and features...")
def load_data():
    missing = [p for p in [
        MANIFEST_PATH,
        EMBEDDINGS_DIR / "clip_features.npy",
        EMBEDDINGS_DIR / "index.json",
        EMBEDDINGS_DIR / "latent_2d.json",
    ] if not p.exists()]
    if missing:
        return None, None, None, [str(p) for p in missing]

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    manifest_df = pd.DataFrame(manifest)

    features = np.load(EMBEDDINGS_DIR / "clip_features.npy")
    with open(EMBEDDINGS_DIR / "index.json") as f:
        feature_index = json.load(f)
    feature_ids = feature_index["ids"]
    success_mask = np.array(feature_index["success"], dtype=bool)

    with open(EMBEDDINGS_DIR / "latent_2d.json") as f:
        latent_2d = json.load(f)
    latent_df = pd.DataFrame(latent_2d["entries"])

    df = pd.DataFrame({"id": feature_ids})
    df = df.merge(manifest_df, on="id", how="left")
    df = df.merge(latent_df, on="id", how="left")
    df_valid = df[success_mask].reset_index(drop=True)
    valid_features = features[success_mask]

    # 3D UMAP — reuse the cached projection app.py writes; compute if absent.
    umap_3d_path = EMBEDDINGS_DIR / "clip_umap_3d.npy"
    if umap_3d_path.exists():
        embeddings_3d = np.load(umap_3d_path)
    else:
        from sklearn.decomposition import PCA
        import umap
        n_pca = min(50, valid_features.shape[0] - 1)
        pca = PCA(n_components=n_pca, random_state=RANDOM_STATE)
        features_pca = pca.fit_transform(valid_features)
        reducer = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1,
                            metric="cosine", random_state=RANDOM_STATE)
        embeddings_3d = reducer.fit_transform(features_pca)
        np.save(umap_3d_path, embeddings_3d)

    df_valid["umap_x_3d"] = embeddings_3d[:, 0]
    df_valid["umap_y_3d"] = embeddings_3d[:, 1]
    df_valid["umap_z_3d"] = embeddings_3d[:, 2]

    for col in ["category", "subtype", "era", "cluster"]:
        if col in df_valid.columns:
            df_valid[col] = df_valid[col].astype("object").where(df_valid[col].notna(), "unknown")
            df_valid[col] = df_valid[col].astype(str)

    return df_valid, valid_features, embeddings_3d, []


@st.cache_data(show_spinner="Measuring agreement density...")
def compute_density(features_bytes: bytes, n: int, dim: int, k: int):
    """Per-point agreement density in CLIP feature space.

    Mean cosine distance to the k nearest neighbours, inverted and rank-
    normalised to [0, 1]. This is the k-NN agreement measure used elsewhere in
    the project: a star sitting among many close neighbours is a dense point of
    representational agreement and reads bright; an outlier sits in a sparse
    region of conflict and reads dim. It does not collapse heterogeneous media
    into one type — it only measures neighbourhood proximity in the shared
    embedding.
    """
    from sklearn.neighbors import NearestNeighbors
    features = np.frombuffer(features_bytes, dtype=np.float32).reshape(n, dim)
    kk = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=kk, metric="cosine").fit(features)
    dists, _ = nn.kneighbors(features)
    mean_d = dists[:, 1:].mean(axis=1) if kk > 1 else dists[:, 0]
    dens = 1.0 / (mean_d + 1e-6)
    order = dens.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.linspace(0.0, 1.0, len(order))
    return ranks


def period_order(df_valid):
    """Ordered list of distinct periods (the `era` field).

    Values are numeric-prefixed (e.g. 07_industrial_modernism), so a string
    sort puts them in chronological order. Whatever periods are present in the
    data define the toggle menu.
    """
    if "era" in df_valid.columns:
        return sorted({str(v) for v in df_valid["era"]})
    return ["unknown"]


def cluster_ints(df_valid):
    """Map cluster labels to stable ints aligned with row order.

    Noise / unknown clusters map to -1 and never form a world. This keeps
    conflict zones singular: a representation that HDBSCAN could not place in a
    world stays an isolated star that unify never brightens.
    """
    if "cluster" in df_valid.columns:
        clu = df_valid["cluster"].astype(str)
    else:
        clu = pd.Series(["unknown"] * len(df_valid))
    mapping, nxt, ci = {}, 0, []
    for c in clu:
        if c not in mapping:
            if c in ("-1", "unknown", "nan", "None", ""):
                mapping[c] = -1
            else:
                mapping[c] = nxt
                nxt += 1
        ci.append(mapping[c])
    return ci, mapping


@st.cache_data(show_spinner="Linking neighbours...")
def compute_knn_edges(features_bytes: bytes, n: int, dim: int, k: int):
    from sklearn.neighbors import NearestNeighbors
    features = np.frombuffer(features_bytes, dtype=np.float32).reshape(n, dim)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine").fit(features)
    _, indices = nn.kneighbors(features)
    seen = set()
    edges = []
    for i in range(n):
        for j in indices[i, 1:]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append([a, b])
    return edges


# =============================================================================
# Constellation template  (three.js r160 + WebXR; same CDN as app.py)
# =============================================================================
CONSTELLATION_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin: 0; padding: 0; overflow: hidden; background: __BG__; }
  canvas { display: block; }
  #overlay {
    position: absolute; top: 10px; left: 10px; z-index: 10;
    color: #cdd2dc; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11px; background: rgba(8,9,14,0.6); padding: 9px 12px;
    border-radius: 5px; backdrop-filter: blur(5px); user-select: none;
    max-width: 230px;
  }
  #overlay b { color: #eef1f6; }
  #overlay .row { margin: 5px 0; }
  #overlay input[type=range] { width: 120px; vertical-align: middle; }
  #overlay .v { color: #8b93a3; }
  #overlay hr { border: 0; border-top: 1px solid #23262e; margin: 8px 0; }
  #overlay button {
    background: #1b1e26; color: #cdd2dc; border: 1px solid #3a3f4b;
    padding: 4px 9px; cursor: pointer; font: inherit; border-radius: 4px;
    margin-top: 4px;
  }
  #overlay button:hover { background: #262a34; }
  #tooltip {
    position: absolute; pointer-events: none; z-index: 11; color: #fff;
    font-family: ui-monospace, monospace; font-size: 11px;
    background: rgba(0,0,0,0.82); padding: 5px 9px; border-radius: 4px;
    display: none; max-width: 260px; line-height: 1.4;
  }
  #vrhint {
    position: absolute; bottom: 14px; left: 50%; transform: translateX(-50%);
    z-index: 10; color: #6a7180; font-family: ui-monospace, monospace;
    font-size: 10px; text-align: center;
  }
</style>
</head>
<body>
<div id="overlay">
  <div class="row"><b>Constellation</b> &nbsp;<span class="v" id="count"></span></div>
  <hr>
  <div class="row">star size <span class="v" id="vSize">0.9</span><br>
    <input id="size" type="range" min="0.3" max="2.5" step="0.05" value="0.9"></div>
  <div class="row">glow <span class="v" id="vGlow">0.6</span><br>
    <input id="glow" type="range" min="0.15" max="2" step="0.05" value="0.6"></div>
  <div class="row">milkiness <span class="v" id="vNeb">0.5</span><br>
    <input id="neb" type="range" min="0" max="2" step="0.05" value="0.5"></div>
  <div class="row">density &rarr; brightness <span class="v" id="vDens">0.5</span><br>
    <input id="dens" type="range" min="0" max="1" step="0.02" value="0.5"></div>
  <div class="row"><b>worlds unify</b> <span class="v" id="vUnify">0.00</span><br>
    <input id="unify" type="range" min="0" max="1" step="0.02" value="0"></div>
  <div class="row">unify shows
    <select id="unifyMode" style="background:#11131a;color:#cdd2dc;border:1px solid #3a3f4b;font:inherit;border-radius:3px;">
      <option value="lines">lines</option>
      <option value="forms">forms</option>
      <option value="both" selected>both</option>
    </select></div>
  <div class="row">form fill <span class="v" id="vFill">0.7</span><br>
    <input id="fill" type="range" min="0" max="1.5" step="0.05" value="0.7"></div>
  <div class="row">twinkle <span class="v" id="vTwk">0.22</span><br>
    <input id="twk" type="range" min="0" max="1" step="0.02" value="0.22"></div>
  <div class="row">lines <span class="v" id="vLine">0.10</span><br>
    <input id="line" type="range" min="0" max="0.8" step="0.01" value="0.10"></div>
  <hr>
  <div class="row"><b>polygon worlds</b> <button id="polyToggle">off</button>
    <span style="color:#5c6270;font-size:10px;">original forms, sanity check</span></div>
  <div class="row">opacity <span class="v" id="vPolyOp">0.45</span><br>
    <input id="polyOp" type="range" min="0" max="1" step="0.02" value="0.45"></div>
  <div class="row">blend
    <select id="polyBlend" style="background:#11131a;color:#cdd2dc;border:1px solid #3a3f4b;font:inherit;border-radius:3px;">
      <option value="source-over">normal</option>
      <option value="screen" selected>screen</option>
      <option value="lighten">lighten</option>
      <option value="multiply">multiply</option>
      <option value="overlay">overlay</option>
      <option value="color-dodge">dodge</option>
      <option value="difference">difference</option>
    </select>
    <label style="font-size:10px;color:#9aa3b5;margin-left:4px;"><input type="checkbox" id="polyStroke" checked style="vertical-align:middle;">outline</label></div>
  <div class="row">lens <span class="v" id="vLens">55°</span> <span style="color:#5c6270;font-size:10px;">wide = strong</span><br>
    <input id="lens" type="range" min="10" max="150" step="1" value="55"></div>
  <div class="row">drift <span class="v" id="vDrift">0.05</span><br>
    <input id="drift" type="range" min="0" max="0.4" step="0.01" value="0.05"></div>
  <hr>
  <div class="row"><b>periods</b>
    <button id="periodsAll" style="margin-left:6px;">all</button>
    <button id="periodsNone">none</button></div>
  <div id="periodList" style="max-height:150px;overflow:auto;margin-top:2px;padding-right:4px;"></div>
  <hr>
  <div class="row v" id="ignState">click a star to ignite its cluster</div>
  <button id="release">release form</button>
  <button id="reset">reset view</button>
  <hr>
  <div class="row"><b>export</b>
    <button id="exportSVG">SVG</button>
    <button id="exportPNG">PNG</button></div>
</div>
<div id="tooltip"></div>
<div id="vrhint"></div>

<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRButton } from 'three/addons/webxr/VRButton.js';

const POINTS = __POINTS_JSON__;
const EDGES  = __EDGES_JSON__;
const PERIODS = __PERIODS_JSON__;   // ordered period (era) names; point.pi indexes this
const CFG    = __SETTINGS_JSON__;
const N = POINTS.length;

// ---- framing -------------------------------------------------------------
let cx=0, cy=0, cz=0;
for (const p of POINTS) { cx+=p.x; cy+=p.y; cz+=p.z; }
cx/=N; cy/=N; cz/=N;
let radius = 1e-6;
for (const p of POINTS) {
  const dx=p.x-cx, dy=p.y-cy, dz=p.z-cz;
  radius = Math.max(radius, Math.hypot(dx,dy,dz));
}
const W = () => window.innerWidth, H = () => window.innerHeight;

const scene = new THREE.Scene();
scene.background = new THREE.Color('__BG__');
scene.fog = new THREE.FogExp2(new THREE.Color('__BG__').getHex(), 0.6 / (radius * 6));

const camera = new THREE.PerspectiveCamera(55, W()/H(), radius*0.002, radius*200);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, preserveDrawingBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(W(), H());
renderer.xr.enabled = true;
document.body.appendChild(renderer.domElement);

// 2D overlay canvas for the polygon-worlds layer (screen space, desktop only)
const polyCanvas = document.createElement('canvas');
polyCanvas.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;z-index:5;';
document.body.appendChild(polyCanvas);
const polyCtx = polyCanvas.getContext('2d');

// world group holds everything so it can be rescaled for VR without disturbing
// the authored coordinates; dolly carries the camera for headset locomotion.
const world = new THREE.Group();
scene.add(world);
const dolly = new THREE.Group();
dolly.add(camera);
scene.add(dolly);

const DESK_CAM = new THREE.Vector3(cx, cy + radius*0.35, cz + radius*2.4);
const DEFAULT_FOV = camera.fov;
camera.position.copy(DESK_CAM);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(cx, cy, cz);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.update();

// ---- sprite textures -----------------------------------------------------
// star: a small bright pinpoint with a thin, faint halo — a night-sky point
// rather than a glowing orb. fill/nebula: a wide, very soft falloff that only
// reads when many overlap (milkiness).
function spriteTexture(stops) {
  const size = 128;
  const c = document.createElement('canvas'); c.width = c.height = size;
  const g = c.getContext('2d');
  const grd = g.createRadialGradient(size/2,size/2,0, size/2,size/2,size/2);
  for (const s of stops) grd.addColorStop(s[0], 'rgba(255,255,255,'+s[1]+')');
  g.fillStyle = grd; g.fillRect(0,0,size,size);
  const t = new THREE.CanvasTexture(c); t.needsUpdate = true; return t;
}
const starTex = spriteTexture([[0.0,1.0],[0.10,0.85],[0.22,0.22],[0.5,0.04],[1.0,0.0]]);
const softTex = spriteTexture([[0.0,1.0],[0.5,0.18],[1.0,0.0]]);
const nebTex  = softTex;

// ---- attribute buffers ---------------------------------------------------
const pos = new Float32Array(N*3);
const col = new Float32Array(N*3);
const aBright = new Float32Array(N);   // base brightness (density-driven)
const aSize   = new Float32Array(N);
const aPhase  = new Float32Array(N);
const aClu    = new Float32Array(N);
const aUnion  = new Float32Array(N);   // fraction of own world fused at current unify
const aVis    = new Float32Array(N);   // 1 = period shown, 0 = hidden
for (let i=0;i<N;i++){
  const p = POINTS[i];
  pos[i*3]=p.x; pos[i*3+1]=p.y; pos[i*3+2]=p.z;
  col[i*3]=p.r; col[i*3+1]=p.g; col[i*3+2]=p.b;
  aBright[i]=p.d;                       // 0..1 agreement density
  aSize[i]=0.7 + p.d*0.6;               // subtle size variation; brightness carries magnitude
  aPhase[i]=Math.random()*6.2832;
  aClu[i]=p.ci;
  aUnion[i]=0.0;
  aVis[i]=1.0;
}
// one geometry, shared by the star pass and the nebula pass
const geom = new THREE.BufferGeometry();
geom.setAttribute('position', new THREE.BufferAttribute(pos,3));
geom.setAttribute('aColor',   new THREE.BufferAttribute(col,3));
geom.setAttribute('aBright',  new THREE.BufferAttribute(aBright,1));
geom.setAttribute('aSize',    new THREE.BufferAttribute(aSize,1));
geom.setAttribute('aPhase',   new THREE.BufferAttribute(aPhase,1));
geom.setAttribute('aClu',     new THREE.BufferAttribute(aClu,1));
const aUnionAttr = new THREE.BufferAttribute(aUnion,1);
geom.setAttribute('aUnion', aUnionAttr);
const aVisAttr = new THREE.BufferAttribute(aVis,1);
geom.setAttribute('aVis', aVisAttr);

// ---- shared uniforms ------------------------------------------------------
const uniforms = {
  uSize:    { value: 0.9 },
  uGlow:    { value: 0.6 },
  uTwk:     { value: 0.22 },
  uTime:    { value: 0 },
  uDens:    { value: 0.5 },
  uIgnited: { value: -2.0 },     // cluster id currently ignited (-2 = none)
  uUnify:   { value: 0.0 },      // worlds unify amount (0..1)
  uFill:    { value: 0.7 },      // form-fill (substar) intensity
  uH:       { value: H() },
  uScale:   { value: 1.0 },      // world scale (changes in VR)
  uFovScale:{ value: 1.0 / Math.tan(DEFAULT_FOV * Math.PI / 360) },
  uNeb:     { value: 0.5 },
  uTex:     { value: starTex },
};

const STAR_VS = `
attribute vec3 aColor; attribute float aBright; attribute float aSize;
attribute float aPhase; attribute float aClu; attribute float aUnion; attribute float aVis;
uniform float uSize, uTwk, uTime, uDens, uIgnited, uH, uScale, uUnify, uFovScale;
varying vec3 vCol; varying float vB;
void main(){
  if (aVis < 0.5){ gl_Position = vec4(0.0,0.0,2.0,1.0); gl_PointSize = 0.0; return; }
  float ignited = (abs(aClu - uIgnited) < 0.5) ? 1.0 : 0.0;
  // brightness: a low floor, plus a gentle density influence
  float b = mix(0.22, 0.22 + aBright*0.7, uDens);
  // worlds unify: a restrained lift as a star's world fuses; the form is read
  // mainly through the fill, not by blowing out the member stars
  b *= 1.0 + 0.55 * uUnify * aUnion;
  if (uIgnited > -1.5) { b = ignited > 0.5 ? (b + 0.45) : b*0.28; }
  b *= 1.0 + uTwk*0.45*sin(uTime*2.2 + aPhase);
  vB = b; vCol = aColor;
  vec4 mv = modelViewMatrix * vec4(position,1.0);
  float ps = uSize * aSize * (0.8 + b*0.45) * uScale * (156.0 * uFovScale / -mv.z);
  gl_PointSize = clamp(ps, 1.0, 0.18*uH);
  gl_Position = projectionMatrix * mv;
}`;
const STAR_FS = `
precision mediump float;
uniform sampler2D uTex; uniform float uGlow;
varying vec3 vCol; varying float vB;
void main(){
  vec4 t = texture2D(uTex, gl_PointCoord);
  float a = t.a;
  if (a < 0.01) discard;
  vec3 c = vCol * vB * uGlow;
  // a faint warm-white core only at the very centre of bright stars
  c = mix(c, vec3(1.0), smoothstep(0.55,1.0,a) * clamp(vB,0.0,1.0) * 0.3);
  gl_FragColor = vec4(c * a, a);
}`;

const starMat = new THREE.ShaderMaterial({
  uniforms, vertexShader: STAR_VS, fragmentShader: STAR_FS,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
});
const stars = new THREE.Points(geom, starMat);
world.add(stars);

// ---- nebula / milkiness ---------------------------------------------------
// Same points, large and faint, additively blended. Where stars pile up the
// halos sum into milky cloud; sparse conflict zones stay dark. This is the
// density field re-read as light rather than as contour lines.
const NEB_VS = `
attribute vec3 aColor; attribute float aBright; attribute float aClu;
attribute float aUnion; attribute float aVis;
uniform float uSize, uTime, uIgnited, uH, uScale, uNeb, uUnify, uFovScale;
varying vec3 vCol; varying float vB;
void main(){
  if (aVis < 0.5){ gl_Position = vec4(0.0,0.0,2.0,1.0); gl_PointSize = 0.0; return; }
  float ignited = (abs(aClu - uIgnited) < 0.5) ? 1.0 : 0.0;
  float b = aBright;                       // milkiness follows density
  b += uUnify * aUnion * 1.3;              // and thickens as the world fuses
  if (uIgnited > -1.5) b = ignited > 0.5 ? b + 0.5 : b*0.2;
  vB = b * uNeb;
  vCol = aColor;
  vec4 mv = modelViewMatrix * vec4(position,1.0);
  float ps = uSize * (10.0 + aBright*26.0) * uScale * (156.0 * uFovScale / -mv.z);
  gl_PointSize = clamp(ps, 2.0, 0.9*uH);
  gl_Position = projectionMatrix * mv;
}`;
const NEB_FS = `
precision mediump float;
uniform sampler2D uTex;
varying vec3 vCol; varying float vB;
void main(){
  float a = texture2D(uTex, gl_PointCoord).a;
  if (a < 0.004) discard;
  // tint milkiness toward the cluster colour but keep it pale
  vec3 c = mix(vec3(0.7,0.75,0.85), vCol, 0.5) * vB;
  gl_FragColor = vec4(c * a * 0.035, a);
}`;
const nebMat = new THREE.ShaderMaterial({
  uniforms: Object.assign({}, uniforms, { uTex: { value: nebTex } }),
  vertexShader: NEB_VS, fragmentShader: NEB_FS,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
});
// nebula shares the live uniforms except its own texture; relink the shared ones
nebMat.uniforms.uSize = uniforms.uSize;
nebMat.uniforms.uTime = uniforms.uTime;
nebMat.uniforms.uIgnited = uniforms.uIgnited;
nebMat.uniforms.uH = uniforms.uH;
nebMat.uniforms.uScale = uniforms.uScale;
nebMat.uniforms.uNeb = uniforms.uNeb;
nebMat.uniforms.uUnify = uniforms.uUnify;
nebMat.uniforms.uFovScale = uniforms.uFovScale;
const nebula = new THREE.Points(geom, nebMat);
world.add(nebula);

// ---- constellation lines (k-NN) ------------------------------------------
// Rebuilt when the period filter changes so an edge touching a hidden star is
// dropped rather than left dangling.
let lineSeg = null, linePos = null, lineCol = null, linePosAttr = null, lineColAttr = null;
if (EDGES.length) {
  linePos = new Float32Array(EDGES.length*6);
  lineCol = new Float32Array(EDGES.length*6);
  linePosAttr = new THREE.BufferAttribute(linePos,3);
  lineColAttr = new THREE.BufferAttribute(lineCol,3);
  const lg = new THREE.BufferGeometry();
  lg.setAttribute('position', linePosAttr);
  lg.setAttribute('color', lineColAttr);
  lg.setDrawRange(0,0);
  const lm = new THREE.LineBasicMaterial({
    vertexColors:true, transparent:true, opacity:0.10,
    depthWrite:false, blending:THREE.AdditiveBlending });
  lineSeg = new THREE.LineSegments(lg, lm);
  world.add(lineSeg);
}
function rebuildLines(){
  if (!lineSeg) return;
  let n = 0;
  for (let e=0;e<EDGES.length;e++){
    const a=EDGES[e][0], b=EDGES[e][1];
    if (aVis[a] < 0.5 || aVis[b] < 0.5) continue;
    const o=n*6, A=POINTS[a], B=POINTS[b];
    linePos[o]=A.x; linePos[o+1]=A.y; linePos[o+2]=A.z;
    linePos[o+3]=B.x; linePos[o+4]=B.y; linePos[o+5]=B.z;
    lineCol[o]=A.r; lineCol[o+1]=A.g; lineCol[o+2]=A.b;
    lineCol[o+3]=B.r; lineCol[o+4]=B.g; lineCol[o+5]=B.b;
    n++;
  }
  linePosAttr.needsUpdate=true; lineColAttr.needsUpdate=true;
  lineSeg.geometry.setDrawRange(0, n*2);
}

// ---- form fill: soft substars seeded inside fused components --------------
// The other way to read unify. As a component fuses, its interior fills with
// small soft substars so the figure reads as a body — sparse substars at low
// unify thickening into milkiness as it rises — the way the original traces and
// fills its boolean-union polygons. Substars are convex combinations of member
// positions, so they always land inside the figure.
const SUB_MAX = N * 4;
const subPos = new Float32Array(SUB_MAX*3);
const subCol = new Float32Array(SUB_MAX*3);
const subClu = new Float32Array(SUB_MAX);
const subPosAttr = new THREE.BufferAttribute(subPos,3);
const subColAttr = new THREE.BufferAttribute(subCol,3);
const subCluAttr = new THREE.BufferAttribute(subClu,1);
const subGeom = new THREE.BufferGeometry();
subGeom.setAttribute('position', subPosAttr);
subGeom.setAttribute('aColor', subColAttr);
subGeom.setAttribute('aClu', subCluAttr);
subGeom.setDrawRange(0,0);
const SUB_VS = `
attribute vec3 aColor; attribute float aClu;
uniform float uSize, uScale, uH, uIgnited, uFill, uUnify, uFovScale;
varying vec3 vCol; varying float vB;
void main(){
  float ignited = (abs(aClu - uIgnited) < 0.5) ? 1.0 : 0.0;
  float b = uFill * (0.3 + 0.7*uUnify);
  if (uIgnited > -1.5) b = ignited > 0.5 ? b : b*0.18;
  vB = b; vCol = aColor;
  vec4 mv = modelViewMatrix * vec4(position,1.0);
  float ps = uSize * (1.6 + 2.2*uUnify) * uScale * (156.0 * uFovScale / -mv.z);
  gl_PointSize = clamp(ps, 1.0, 0.12*uH);
  gl_Position = projectionMatrix * mv;
}`;
const SUB_FS = `
precision mediump float;
uniform sampler2D uTex;
varying vec3 vCol; varying float vB;
void main(){
  float a = texture2D(uTex, gl_PointCoord).a;
  if (a < 0.01) discard;
  vec3 c = mix(vec3(0.72,0.78,0.9), vCol, 0.5) * vB;
  gl_FragColor = vec4(c * a * 0.06, a);
}`;
const subMat = new THREE.ShaderMaterial({
  uniforms: { uSize:uniforms.uSize, uScale:uniforms.uScale, uH:uniforms.uH,
              uIgnited:uniforms.uIgnited, uFill:uniforms.uFill,
              uUnify:uniforms.uUnify, uFovScale:uniforms.uFovScale, uTex:{value:softTex} },
  vertexShader: SUB_VS, fragmentShader: SUB_FS,
  transparent:true, depthWrite:false, blending:THREE.AdditiveBlending });
const substars = new THREE.Points(subGeom, subMat);
substars.visible = false;
world.add(substars);

// deterministic PRNG so a stable component yields a stable fill (no boiling);
// it only re-rolls when the component's membership changes (i.e. when you move)
function mulberry32(a){ return function(){ a|=0; a=a+0x6D2B79F5|0; let t=Math.imul(a^a>>>15,1|a); t=t+Math.imul(t^t>>>7,61|t)^t; return ((t^t>>>14)>>>0)/4294967296; }; }

// how unify is expressed: 'lines' (welds), 'forms' (fill), or 'both'
let unifyMode = 'both';

// ---- worlds: a viewpoint-relative, unstable constellation of forms --------
// This is the original viewer's logic kept intact: members of a world are
// linked by a Kruskal spanning forest over candidate edges within `reach`,
// fragments fusing into figures as unify rises. The one faithful change for an
// immersive scene is the metric. The original measures distance in the
// projected screen, so the figures depend on where the camera stands; here the
// metric is angular separation from the eye, which is the same projection made
// continuous. Two stars aligned along your sightline read as one and fuse; move,
// and the alignment breaks and the figure dissolves. The forest is recomputed
// as you move, so the forms are never fixed objects — they assemble and come
// apart with your position in the space.
// angular reach scales with the lens: a wide lens fuses across larger angles,
// a long lens only fuses near-aligned stars, so the lens reshapes the forms the
// way it does in the original. In VR the headset optics set the field instead.
function reachBaseRad(){
  const fovDeg = renderer.xr.isPresenting ? 60.0 : camera.fov;
  return THREE.MathUtils.degToRad(fovDeg) * 0.72;
}

// group members by world (ci >= 0), capped like the original's lim
const worldGroups = new Map();
for (let i=0;i<N;i++){
  const c=POINTS[i].ci; if(c<0) continue;
  if(!worldGroups.has(c)) worldGroups.set(c,[]);
  const arr=worldGroups.get(c); if(arr.length<400) arr.push(i);
}
const cluSize = new Map();
for (const [c,arr] of worldGroups) cluSize.set(c, arr.length);

// dynamic weld geometry (a forest has < N edges, so N segments is plenty)
const weldPos = new Float32Array(N*6);
const weldCol = new Float32Array(N*6);
const weldPosAttr = new THREE.BufferAttribute(weldPos,3);
const weldColAttr = new THREE.BufferAttribute(weldCol,3);
const weldGeom = new THREE.BufferGeometry();
weldGeom.setAttribute('position', weldPosAttr);
weldGeom.setAttribute('color', weldColAttr);
weldGeom.setDrawRange(0,0);
const weldSeg = new THREE.LineSegments(weldGeom, new THREE.LineBasicMaterial({
  vertexColors:true, transparent:true, opacity:0.0,
  depthWrite:false, blending:THREE.AdditiveBlending }));
weldSeg.visible = false;
world.add(weldSeg);

// union-find scratch
const ufParent = new Int32Array(N);
function ufFind(x){ while(ufParent[x]!==x){ ufParent[x]=ufParent[ufParent[x]]; x=ufParent[x]; } return x; }

let currentUnify = 0.0;
const _eye = new THREE.Vector3(), _fwd = new THREE.Vector3(), _v = new THREE.Vector3();
function getEye(){
  if (renderer.xr.isPresenting){
    const c = renderer.xr.getCamera();
    c.getWorldPosition(_eye); c.getWorldDirection(_fwd);
  } else {
    _eye.copy(camera.position); camera.getWorldDirection(_fwd);
  }
}

function recomputeWorlds(){
  uniforms.uUnify.value = currentUnify;
  if (currentUnify <= 0.001){
    aUnion.fill(0); aUnionAttr.needsUpdate = true;
    weldGeom.setDrawRange(0,0); weldSeg.visible = false;
    subGeom.setDrawRange(0,0); substars.visible = false;
    return;
  }
  getEye();
  const reach = Math.pow(currentUnify, 1.5) * reachBaseRad();
  const reachCos = Math.cos(reach);           // angle <= reach  <=>  dot >= reachCos
  for (let i=0;i<N;i++) ufParent[i]=i;
  let weldCount = 0;
  // visible members per world (denominator for the fused fraction)
  const visClu = new Map();
  for (let i=0;i<N;i++){ const c=POINTS[i].ci; if(c>=0 && aVis[i]>=0.5) visClu.set(c,(visClu.get(c)||0)+1); }

  for (const [c, members] of worldGroups){
    const M = members.length; if (M < 2) continue;
    // unit direction from eye to each member (world space); cull behind the head
    // and cull members whose period is hidden
    const dirs = new Array(M);
    for (let t=0;t<M;t++){
      const idx = members[t];
      if (aVis[idx] < 0.5){ dirs[t]=null; continue; }
      const p = POINTS[idx];
      _v.set(p.x,p.y,p.z).applyMatrix4(world.matrixWorld).sub(_eye);
      const L = _v.length() || 1; _v.multiplyScalar(1/L);
      dirs[t] = (_v.dot(_fwd) > 0.0) ? [_v.x,_v.y,_v.z] : null;
    }
    // candidate edges within angular reach. dot is monotonic in angle, so we
    // threshold and sort on it directly (largest dot = smallest angle first),
    // which is the same Kruskal order without a per-pair acos.
    const cand = [];
    for (let a=0;a<M;a++){ const da=dirs[a]; if(!da) continue;
      for (let b=a+1;b<M;b++){ const db=dirs[b]; if(!db) continue;
        const dot = da[0]*db[0]+da[1]*db[1]+da[2]*db[2];
        if (dot >= reachCos) cand.push([dot, members[a], members[b]]);
      }
    }
    cand.sort((e1,e2)=>e2[0]-e1[0]);   // shortest angle first
    for (const e of cand){
      const ra=ufFind(e[1]), rb=ufFind(e[2]);
      if (ra===rb) continue;            // forest: skip edges that close a loop
      ufParent[ra]=rb;
      const A=POINTS[e[1]], B=POINTS[e[2]], o=weldCount*6;
      weldPos[o]=A.x;weldPos[o+1]=A.y;weldPos[o+2]=A.z;
      weldPos[o+3]=B.x;weldPos[o+4]=B.y;weldPos[o+5]=B.z;
      weldCol[o]=A.r;weldCol[o+1]=A.g;weldCol[o+2]=A.b;
      weldCol[o+3]=B.r;weldCol[o+4]=B.g;weldCol[o+5]=B.b;
      weldCount++;
    }
  }
  // group visible clustered points into their fused components
  const comps = new Map();
  for (let i=0;i<N;i++){
    if (POINTS[i].ci<0 || aVis[i]<0.5) continue;
    const r=ufFind(i);
    let g=comps.get(r); if(!g){ g=[]; comps.set(r,g); } g.push(i);
  }
  // aUnion = fraction of its (visible) world that a star's component spans
  for (let i=0;i<N;i++){
    const c=POINTS[i].ci;
    if (c<0 || aVis[i]<0.5){ aUnion[i]=0; continue; }
    const g=comps.get(ufFind(i));
    aUnion[i] = (g?g.length:1) / (visClu.get(c)||1);
  }
  aUnionAttr.needsUpdate = true;

  // weld lines (shown unless mode is forms-only)
  weldPosAttr.needsUpdate = true; weldColAttr.needsUpdate = true;
  weldGeom.setDrawRange(0, weldCount*2);
  weldSeg.material.opacity = 0.10 + 0.42*currentUnify;
  weldSeg.visible = (unifyMode !== 'forms') && weldCount>0;

  // form fill (shown unless mode is lines-only)
  let subCount = 0;
  if (unifyMode !== 'lines'){
    for (const g of comps.values()){
      const M=g.length; if (M<3) continue;            // figures need >= 3 members
      let seed=1<<30; for (const idx of g) if (idx<seed) seed=idx;  // stable canonical id
      const rng=mulberry32((seed*2654435761)>>>0);
      const per=Math.max(1, Math.round(M * 0.7 * currentUnify)); // density grows with unify
      let total=Math.min(M*per, SUB_MAX-subCount);
      const cc=POINTS[g[0]];
      for (let s=0;s<total;s++){
        const A=POINTS[g[(rng()*M)|0]], B=POINTS[g[(rng()*M)|0]], C=POINTS[g[(rng()*M)|0]];
        let w0=rng(), w1=rng(), w2=rng(); const ws=w0+w1+w2||1; w0/=ws; w1/=ws; w2/=ws;
        const o=subCount*3;
        subPos[o]  =A.x*w0+B.x*w1+C.x*w2;
        subPos[o+1]=A.y*w0+B.y*w1+C.y*w2;
        subPos[o+2]=A.z*w0+B.z*w1+C.z*w2;
        subCol[o]=cc.r; subCol[o+1]=cc.g; subCol[o+2]=cc.b;
        subClu[subCount]=cc.ci;
        subCount++;
      }
      if (subCount>=SUB_MAX) break;
    }
  }
  subPosAttr.needsUpdate=true; subColAttr.needsUpdate=true; subCluAttr.needsUpdate=true;
  subGeom.setDrawRange(0, subCount);
  substars.visible = (unifyMode !== 'lines') && subCount>0;
}

// recompute when the view moves enough, throttled; immediate on slider change
let unifyDirty = true;
const _lastEye = new THREE.Vector3(Infinity,0,0), _lastFwd = new THREE.Vector3();
let unifyAccum = 0;
function maybeRecomputeWorlds(dt){
  if (currentUnify <= 0.001){ if(unifyDirty){ recomputeWorlds(); unifyDirty=false; } return; }
  unifyAccum += dt;
  getEye();
  const moved = _eye.distanceToSquared(_lastEye) > (1e-4)*( _eye.lengthSq()+1 )
             || _fwd.distanceToSquared(_lastFwd) > 1e-5;
  const animating = driftSpeed > 0.0001 || renderer.xr.isPresenting;
  if (unifyDirty || (unifyAccum > 0.07 && (moved || animating))){
    recomputeWorlds();
    _lastEye.copy(_eye); _lastFwd.copy(_fwd);
    unifyAccum = 0; unifyDirty = false;
  }
}

// ===========================================================================
// Polygon worlds overlay — the original viewer's linework worlds-unify forms,
// ported verbatim (raster + spanning-forest necks + morphological close +
// crack-followed contour trace) and drawn on the 2D overlay in screen space.
// Driven by the SAME clusters and the SAME unify value as the stars, so the
// traced figures are a ground-truth check on whether the constellation's fused
// components match. Members here are disks (the constellation has no per-image
// silhouette), but the fusion logic that produces the forms is identical.
// ===========================================================================
let polyOn=false, polyDirty=true, polyOpacity=0.45, polyBlend='screen', polyStroke=true, polyFigures=[];
const cssW=()=>renderer.domElement.clientWidth||window.innerWidth;
const cssH=()=>renderer.domElement.clientHeight||window.innerHeight;

function polyAreaSigned(p){ let a=0; for(let i=0;i<p.length;i++){ const j=(i+1)%p.length; a+=p[i][0]*p[j][1]-p[j][0]*p[i][1]; } return a/2; }
function ringDropCollinear(poly){ const n=poly.length; if(n<3) return poly; const out=[];
  for(let i=0;i<n;i++){ const a=poly[(i-1+n)%n],b=poly[i],c=poly[(i+1)%n];
    const cr=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0]); if(Math.abs(cr)>1e-9) out.push(b); }
  return out.length>=3?out:poly; }
function ringSelfIntersects(poly){ const n=poly.length;
  const ci=(a,b,d)=>(d[1]-a[1])*(b[0]-a[0])-(b[1]-a[1])*(d[0]-a[0]);
  const si=(p1,p2,p3,p4)=>ci(p1,p3,p4)*ci(p2,p3,p4)<0 && ci(p1,p2,p3)*ci(p1,p2,p4)<0;
  for(let i=0;i<n;i++)for(let j=i+2;j<n;j++){ if(i===0&&j===n-1) continue; if(si(poly[i],poly[(i+1)%n],poly[j],poly[(j+1)%n])) return true; } return false; }
function ringSimplifySafe(ring,eps){ if(ring.length<4) return ring;
  const perp=(p,a,b)=>{ const dx=b[0]-a[0],dy=b[1]-a[1],L=Math.hypot(dx,dy)||1; return Math.abs((p[0]-a[0])*dy-(p[1]-a[1])*dx)/L; };
  function rdp(pts,e){ const keep=new Uint8Array(pts.length); keep[0]=1; keep[pts.length-1]=1; const st=[[0,pts.length-1]];
    while(st.length){ const seg=st.pop(),s=seg[0],en=seg[1]; let mx=0,mi=-1;
      for(let i=s+1;i<en;i++){ const d=perp(pts[i],pts[s],pts[en]); if(d>mx){mx=d;mi=i;} }
      if(mx>e&&mi>0){ keep[mi]=1; st.push([s,mi],[mi,en]); } }
    const o=[]; for(let i=0;i<pts.length;i++) if(keep[i]) o.push(pts[i]); return o; }
  for(let f=1;f<=4;f++){ const open=ring.slice(); open.push(ring[0]); let s=rdp(open,eps/f); s=s.slice(0,s.length-1);
    if(s.length>=3 && !ringSelfIntersects(s)) return s; }
  return ringDropCollinear(ring); }
function fillPolyMask(mask,gw,gh,pts){ if(!pts||pts.length<3) return; let ymin=Infinity,ymax=-Infinity;
  for(const p of pts){ if(p[1]<ymin)ymin=p[1]; if(p[1]>ymax)ymax=p[1]; }
  ymin=Math.max(0,Math.floor(ymin)); ymax=Math.min(gh-1,Math.ceil(ymax));
  for(let y=ymin;y<=ymax;y++){ const yc=y+0.5; const xs=[];
    for(let i=0;i<pts.length;i++){ const a=pts[i],b=pts[(i+1)%pts.length];
      if((a[1]<=yc&&b[1]>yc)||(b[1]<=yc&&a[1]>yc)) xs.push(a[0]+(yc-a[1])/(b[1]-a[1])*(b[0]-a[0])); }
    xs.sort((p,q)=>p-q);
    for(let k=0;k+1<xs.length;k+=2){ const x0=Math.max(0,Math.ceil(xs[k]-0.5)),x1=Math.min(gw-1,Math.floor(xs[k+1]-0.5));
      for(let x=x0;x<=x1;x++) mask[y*gw+x]=1; } } }
function maxMinFilter(mask,gw,gh,r,dilate){ const out=new Uint8Array(gw*gh);
  for(let y=0;y<gh;y++)for(let x=0;x<gw;x++){ let v=dilate?0:1;
    for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++){ const xx=x+dx,yy=y+dy;
      const s=(xx>=0&&yy>=0&&xx<gw&&yy<gh)?mask[yy*gw+xx]:0; v=dilate?Math.max(v,s):Math.min(v,s); if(dilate&&v) break; }
    out[y*gw+x]=v; } return out; }
function closeMaskBin(mask,gw,gh,r){ if(r<=0) return mask; return maxMinFilter(maxMinFilter(mask,gw,gh,r,true),gw,gh,r,false); }
function labelMask(mask,gw,gh){ const lab=new Int32Array(gw*gh).fill(-1); let n=0; const stack=[];
  for(let i=0;i<gw*gh;i++){ if(!mask[i]||lab[i]>=0) continue; lab[i]=n; stack.length=0; stack.push(i);
    while(stack.length){ const p=stack.pop(),x=p%gw,y=(p/gw)|0;
      if(x>0&&mask[p-1]&&lab[p-1]<0){lab[p-1]=n;stack.push(p-1);}
      if(x<gw-1&&mask[p+1]&&lab[p+1]<0){lab[p+1]=n;stack.push(p+1);}
      if(y>0&&mask[p-gw]&&lab[p-gw]<0){lab[p-gw]=n;stack.push(p-gw);}
      if(y<gh-1&&mask[p+gw]&&lab[p+gw]<0){lab[p+gw]=n;stack.push(p+gw);} }
    n++; } return {lab,n}; }
function traceMaskContours(mask,gw,gh){ const at=(x,y)=>(x>=0&&y>=0&&x<gw&&y<gh)?mask[y*gw+x]:0;
  const cw=gw+1, cid=(x,y)=>y*cw+x; const ea=[],eb=[],eax=[],eay=[],ebx=[],eby=[];
  const push=(ax,ay,bx,by)=>{ ea.push(cid(ax,ay));eb.push(cid(bx,by));eax.push(ax);eay.push(ay);ebx.push(bx);eby.push(by); };
  for(let y=0;y<gh;y++)for(let x=0;x<gw;x++){ if(!at(x,y)) continue;
    if(!at(x,y-1)) push(x,y,x+1,y); if(!at(x+1,y)) push(x+1,y,x+1,y+1);
    if(!at(x,y+1)) push(x+1,y+1,x,y+1); if(!at(x-1,y)) push(x,y+1,x,y); }
  const m=ea.length; const outMap=new Map();
  for(let i=0;i<m;i++){ const k=ea[i]; if(!outMap.has(k)) outMap.set(k,[]); outMap.get(k).push(i); }
  const used=new Uint8Array(m); const loops=[];
  for(let s=0;s<m;s++){ if(used[s]) continue; const loop=[]; let ei=s,guard=0;
    while(ei>=0&&!used[ei]&&guard++<m+5){ used[ei]=1; loop.push([eax[ei],eay[ei]]);
      const inx=Math.sign(ebx[ei]-eax[ei]),iny=Math.sign(eby[ei]-eay[ei]); const cands=outMap.get(eb[ei])||[];
      let best=-1,bs=Infinity;
      for(const cidx of cands){ if(used[cidx]) continue; const dx=Math.sign(ebx[cidx]-eax[cidx]),dy=Math.sign(eby[cidx]-eay[cidx]);
        const cross=inx*dy-iny*dx, dot=inx*dx+iny*dy; let score=cross; if(dot<0&&Math.abs(cross)<1e-9) score=1e6;
        if(score<bs){ bs=score; best=cidx; } } ei=best; }
    if(loop.length>=4) loops.push(loop); } return loops; }
function unionFindMake(n){ const p=new Int32Array(n); for(let i=0;i<n;i++)p[i]=i;
  const find=(i)=>{ while(p[i]!==i){ p[i]=p[p[i]]; i=p[i]; } return i; };
  return { find, union:(a,b)=>{ a=find(a); b=find(b); if(a!==b){ p[a]=b; return true; } return false; } }; }

const UNIT_DISK=(()=>{ const a=[],K=12; for(let i=0;i<K;i++){ const t=i/K*6.2831853; a.push([0.5+0.5*Math.cos(t),0.5+0.5*Math.sin(t)]); } return a; })();

// faithful boolean-union footprint (no articulation/gesture machinery)
function rasterUnion(mem,unify,minDim){
  let xmn=Infinity,xmx=-Infinity,ymn=Infinity,ymx=-Infinity,avg=0;
  for(const m of mem){ const h=m.size*0.6; avg+=m.size;
    if(m.cx-h<xmn)xmn=m.cx-h; if(m.cx+h>xmx)xmx=m.cx+h; if(m.cy-h<ymn)ymn=m.cy-h; if(m.cy+h>ymx)ymx=m.cy+h; }
  avg/=Math.max(1,mem.length); const pad=avg*0.8; xmn-=pad;xmx+=pad;ymn-=pad;ymx+=pad;
  const bw=xmx-xmn,bh=ymx-ymn; if(!(bw>0)||!(bh>0)) return [];
  const GMAX=240, gscale=Math.min(GMAX/bw,GMAX/bh,2.5);
  const gw=Math.max(8,Math.ceil(bw*gscale)), gh=Math.max(8,Math.ceil(bh*gscale));
  const mask=new Uint8Array(gw*gh);
  const toG=(m,p)=>[ (m.cx+(p[0]-0.5)*m.size-xmn)*gscale, (m.cy+(p[1]-0.5)*m.size-ymn)*gscale ];
  for(const m of mem){ fillPolyMask(mask,gw,gh,UNIT_DISK.map(p=>toG(m,p))); }
  const reach=Math.pow(unify,1.5)*minDim*0.7;
  if(reach>1&&mem.length>1){ const edges=[],lim=Math.min(mem.length,400);
    for(let i=0;i<lim;i++)for(let j=i+1;j<lim;j++){ const d=Math.hypot(mem[i].cx-mem[j].cx,mem[i].cy-mem[j].cy); if(d<=reach) edges.push([d,i,j]); }
    edges.sort((a,b)=>a[0]-b[0]); const uf=unionFindMake(mem.length);
    for(const e of edges){ if(!uf.union(e[1],e[2])) continue; const a=mem[e[1]],b=mem[e[2]];
      const ax=(a.cx-xmn)*gscale,ay=(a.cy-ymn)*gscale,bx=(b.cx-xmn)*gscale,by=(b.cy-ymn)*gscale;
      const vx=bx-ax,vy=by-ay,L=Math.hypot(vx,vy)||1,px=-vy/L,py=vx/L;
      const wScreen=Math.max(minDim*0.004*unify, Math.min(a.size,b.size)*(0.12+unify*1.1)), w=wScreen*gscale;
      fillPolyMask(mask,gw,gh,[[ax+px*w/2,ay+py*w/2],[bx+px*w/2,by+py*w/2],[bx-px*w/2,by-py*w/2],[ax-px*w/2,ay-py*w/2]]); } }
  const closeR=Math.max(0,Math.round(gscale*(unify*unify*3.2)));
  const closed=closeMaskBin(mask,gw,gh,closeR); const lm=labelMask(closed,gw,gh);
  const depthSum=new Float64Array(lm.n),depthCnt=new Int32Array(lm.n);
  for(const m of mem){ const gx=Math.round((m.cx-xmn)*gscale),gy=Math.round((m.cy-ymn)*gscale);
    if(gx<0||gy<0||gx>=gw||gy>=gh) continue; const c=lm.lab[gy*gw+gx]; if(c<0) continue; depthSum[c]+=m.depth; depthCnt[c]++; }
  const cellCnt=new Int32Array(lm.n); for(let i=0;i<gw*gh;i++) if(lm.lab[i]>=0) cellCnt[lm.lab[i]]++;
  const minCells=Math.max(12,gscale*gscale*6); const figures=[];
  for(let c=0;c<lm.n;c++){ if(cellCnt[c]<minCells) continue;
    const sub=new Uint8Array(gw*gh); for(let i=0;i<gw*gh;i++) if(lm.lab[i]===c) sub[i]=1;
    const loops=traceMaskContours(sub,gw,gh); const contours=[];
    for(const lpRaw of loops){ let lp=ringDropCollinear(lpRaw); if(lp.length<3) continue; lp=ringSimplifySafe(lp,1.2); if(lp.length<3) continue;
      contours.push(lp.map(q=>[xmn+q[0]/gscale,ymn+q[1]/gscale])); }
    if(!contours.length) continue;
    const ord=contours.map((c,i)=>i).sort((a,b)=>Math.abs(polyAreaSigned(contours[b]))-Math.abs(polyAreaSigned(contours[a])));
    figures.push({ contours: ord.map(i=>contours[i]), depth: depthCnt[c]>0?depthSum[c]/depthCnt[c]:1 }); }
  return figures;
}

// per-cluster representative colour (matches the stars)
const cluColorMap=new Map();
for(let i=0;i<N;i++){ const c=POINTS[i].ci; if(c>=0&&!cluColorMap.has(c)) cluColorMap.set(c,[POINTS[i].r,POINTS[i].g,POINTS[i].b]); }
function rgbOf(i){ const p=POINTS[i]; return 'rgb('+((p.r*255)|0)+','+((p.g*255)|0)+','+((p.b*255)|0)+')'; }
function cluColor(c){ const v=cluColorMap.get(c)||[0.8,0.8,0.9]; return 'rgb('+((v[0]*255)|0)+','+((v[1]*255)|0)+','+((v[2]*255)|0)+')'; }

function projectScreen(W,H){
  camera.updateMatrixWorld(true);
  const camPos=camera.position.clone(), camFwd=new THREE.Vector3(); camera.getWorldDirection(camFwd);
  const tmp=new THREE.Vector3(); const proj=new Array(N).fill(null); const depths=[];
  for(let i=0;i<N;i++){ if(aVis[i]<0.5) continue;
    tmp.set(POINTS[i].x,POINTS[i].y,POINTS[i].z).applyMatrix4(world.matrixWorld);
    const along=tmp.clone().sub(camPos).dot(camFwd); if(along<=camera.near) continue;
    const wd=tmp.distanceTo(camPos); tmp.project(camera);
    proj[i]={x:(tmp.x*0.5+0.5)*W, y:(1-(tmp.y*0.5+0.5))*H, depth:wd}; depths.push(wd); }
  depths.sort((a,b)=>a-b); const refDist=depths.length?depths[Math.floor(depths.length/2)]:1;
  return {proj,refDist};
}
function computePolyFigures(){
  const W=cssW(),H=cssH(),minDim=Math.min(W,H); const {proj,refDist}=projectScreen(W,H);
  const base=minDim*0.05,minS=minDim*0.018,maxS=minDim*0.5; const byClu=new Map();
  for(let i=0;i<N;i++){ const pr=proj[i]; if(!pr) continue; const c=POINTS[i].ci; if(c<0) continue;
    let size=base*(refDist/pr.depth); size=Math.max(minS,Math.min(maxS,size));
    if(!byClu.has(c)) byClu.set(c,[]); byClu.get(c).push({cx:pr.x,cy:pr.y,size,depth:pr.depth}); }
  const figs=[];
  for(const [c,mem] of byClu){ if(mem.length<3) continue; const col=cluColor(c);
    for(const f of rasterUnion(mem,currentUnify,minDim)) if(f.contours&&f.contours.length) figs.push({contours:f.contours,depth:f.depth,col}); }
  figs.sort((a,b)=>b.depth-a.depth); polyFigures=figs; polyDirty=false;
}
function drawPolyFigures(){
  const W=cssW(),H=cssH(); polyCtx.clearRect(0,0,W,H); if(!polyOn) return;
  polyCtx.save(); polyCtx.globalCompositeOperation=polyBlend; polyCtx.globalAlpha=polyOpacity; polyCtx.lineJoin='round';
  for(const f of polyFigures){ polyCtx.beginPath();
    for(const ring of f.contours){ polyCtx.moveTo(ring[0][0],ring[0][1]); for(let i=1;i<ring.length;i++) polyCtx.lineTo(ring[i][0],ring[i][1]); polyCtx.closePath(); }
    polyCtx.fillStyle=f.col; polyCtx.fill('evenodd');
    if(polyStroke){ polyCtx.globalAlpha=Math.min(1,polyOpacity*1.7); polyCtx.lineWidth=1.0; polyCtx.strokeStyle=f.col; polyCtx.stroke(); polyCtx.globalAlpha=polyOpacity; } }
  polyCtx.restore();
}
let polyAccum=0; const _pEye=new THREE.Vector3(Infinity,0,0), _pFwd=new THREE.Vector3();
function maybePoly(dt){
  if(!polyOn) return; polyAccum+=dt; getEye();
  const moved=_eye.distanceToSquared(_pEye)>1e-4*(_eye.lengthSq()+1) || _fwd.distanceToSquared(_pFwd)>1e-5;
  const animating=driftSpeed>0.0001;
  if(polyDirty || (polyAccum>0.1 && (moved||animating))){ computePolyFigures(); drawPolyFigures(); _pEye.copy(_eye); _pFwd.copy(_fwd); polyAccum=0; }
}

function downloadBlob(blob,name){ const u=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=u; a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(u),2000); }

// ---- faint background starfield for depth --------------------------------
(function backdrop(){
  const M = 1400; const bp = new Float32Array(M*3);
  for (let i=0;i<M;i++){
    const r = radius*(12+Math.random()*30);
    const th = Math.random()*6.2832, ph = Math.acos(2*Math.random()-1);
    bp[i*3]=cx+r*Math.sin(ph)*Math.cos(th);
    bp[i*3+1]=cy+r*Math.sin(ph)*Math.sin(th);
    bp[i*3+2]=cz+r*Math.cos(ph);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(bp,3));
  const m = new THREE.PointsMaterial({ size: radius*0.006, color:0x8a93a8,
    transparent:true, opacity:0.32, depthWrite:false, blending:THREE.AdditiveBlending });
  scene.add(new THREE.Points(g, m));
})();

// =========================================================================
// Picking: nearest point in screen space (robust for additive point clouds)
// =========================================================================
const proj = new THREE.Vector3();
function nearestPoint(clientX, clientY){
  const rect = renderer.domElement.getBoundingClientRect();
  const mx = clientX-rect.left, my = clientY-rect.top;
  let best=-1, bestD=18*18;          // 18px pick radius
  for (let i=0;i<N;i++){
    if (aVis[i] < 0.5) continue;
    proj.set(POINTS[i].x, POINTS[i].y, POINTS[i].z);
    world.localToWorld(proj); proj.project(camera);
    if (proj.z>1) continue;
    const sx=(proj.x*0.5+0.5)*rect.width, sy=(-proj.y*0.5+0.5)*rect.height;
    const d=(sx-mx)*(sx-mx)+(sy-my)*(sy-my);
    if (d<bestD){ bestD=d; best=i; }
  }
  return best;
}
const tip = document.getElementById('tooltip');
renderer.domElement.addEventListener('pointermove', (e)=>{
  if (renderer.xr.isPresenting) return;
  const i = nearestPoint(e.clientX, e.clientY);
  if (i<0){ tip.style.display='none'; return; }
  const p = POINTS[i];
  tip.innerHTML = `<b>${p.id}</b><br>${p.cat}${p.sub?' · '+p.sub:''}`
    + `<br>${p.era} · cluster ${p.clu}`;
  tip.style.display='block';
  tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY+14)+'px';
});
let ignited = -2;
function setIgnited(ci, label){
  ignited = ci; uniforms.uIgnited.value = ci;
  document.getElementById('ignState').textContent =
    ci > -1.5 ? ('ignited: cluster '+label) : 'click a star to ignite its cluster';
}
renderer.domElement.addEventListener('click', (e)=>{
  const i = nearestPoint(e.clientX, e.clientY);
  if (i<0){ setIgnited(-2,''); return; }
  setIgnited(POINTS[i].ci, POINTS[i].clu);
});

// =========================================================================
// Overlay controls (live, no Streamlit rerun)
// =========================================================================
function bind(id, vid, fn, fmt){
  const el=document.getElementById(id), v=document.getElementById(vid);
  const upd=()=>{ const x=parseFloat(el.value); fn(x); if(v) v.textContent=(fmt?fmt(x):x.toFixed(2)); };
  el.addEventListener('input', upd); upd();
}
bind('size','vSize', x=>uniforms.uSize.value=x);
bind('glow','vGlow', x=>uniforms.uGlow.value=x);
bind('neb','vNeb',   x=>{ uniforms.uNeb.value=x; nebula.visible = x>0.001; });
bind('dens','vDens', x=>uniforms.uDens.value=x);
bind('unify','vUnify', x=>{ currentUnify = x; unifyDirty = true; });
bind('fill','vFill', x=>{ uniforms.uFill.value = x; unifyDirty = true; });
document.getElementById('unifyMode').addEventListener('change', e=>{
  unifyMode = e.target.value; unifyDirty = true;
});
bind('twk','vTwk',   x=>uniforms.uTwk.value=x);
bind('line','vLine', x=>{ if(lineSeg){ lineSeg.material.opacity=x; lineSeg.visible=x>0.001; }});
let driftSpeed=0.05;
bind('drift','vDrift', x=>driftSpeed=x);

// lens: dolly-zoom. Change FOV and slide the camera along its view direction to
// hold the framing, so only perspective strength changes. uFovScale keeps the
// stars' apparent size constant through the zoom; the depth relationships are
// what compress (long lens) or exaggerate (wide lens). The form reach follows
// the lens too, so a long lens fragments the worlds and a wide lens fuses them.
const lensEl = document.getElementById('lens'), lensVal = document.getElementById('vLens');
function applyLens(newFov){
  const oldFov = camera.fov;
  const dir = new THREE.Vector3().subVectors(camera.position, controls.target);
  const dist = dir.length() || 1;
  const newDist = dist * Math.tan(oldFov*Math.PI/360) / Math.tan(newFov*Math.PI/360);
  dir.setLength(newDist);
  camera.position.copy(controls.target).add(dir);
  camera.fov = newFov;
  camera.updateProjectionMatrix();
  controls.update();
  uniforms.uFovScale.value = 1.0 / Math.tan(newFov*Math.PI/360);
  lensVal.textContent = newFov.toFixed(0)+'\u00B0';
  unifyDirty = true;   // angular reach changed -> reform the worlds
}
lensEl.addEventListener('input', e=>applyLens(parseFloat(e.target.value)));

document.getElementById('release').onclick=()=>setIgnited(-2,'');
document.getElementById('reset').onclick=()=>{
  camera.fov = DEFAULT_FOV; camera.updateProjectionMatrix();
  uniforms.uFovScale.value = 1.0 / Math.tan(DEFAULT_FOV*Math.PI/360);
  lensEl.value = DEFAULT_FOV; lensVal.textContent = DEFAULT_FOV.toFixed(0)+'\u00B0';
  camera.position.copy(DESK_CAM); controls.target.set(cx,cy,cz); controls.update();
  unifyDirty = true;
};
document.getElementById('count').textContent = N + ' stars';

// ---- period filter -------------------------------------------------------
const periodVisible = PERIODS.map(()=>true);
function prettyPeriod(s){ return s.replace(/^[0-9]+_/,'').replace(/_/g,' '); }
function applyVisibility(){
  for (let i=0;i<N;i++) aVis[i] = periodVisible[POINTS[i].pi] ? 1.0 : 0.0;
  aVisAttr.needsUpdate = true;
  rebuildLines();
  unifyDirty = true;          // re-form worlds without the hidden periods
}
function setAllPeriods(v){
  for (let i=0;i<periodVisible.length;i++){
    periodVisible[i]=v;
    const el=document.getElementById('per_'+i); if(el) el.checked=v;
  }
  applyVisibility();
}
(function buildPeriodMenu(){
  const list = document.getElementById('periodList');
  PERIODS.forEach((name,idx)=>{
    const row = document.createElement('label');
    row.style.cssText='display:block;font-size:11px;margin:2px 0;cursor:pointer;color:#cdd2dc;white-space:nowrap;';
    row.innerHTML = '<input type="checkbox" id="per_'+idx+'" checked style="vertical-align:middle;margin-right:5px;">'+prettyPeriod(name);
    list.appendChild(row);
    row.querySelector('input').addEventListener('change', e=>{
      periodVisible[idx] = e.target.checked; applyVisibility();
    });
  });
  document.getElementById('periodsAll').onclick = ()=>setAllPeriods(true);
  document.getElementById('periodsNone').onclick = ()=>setAllPeriods(false);
})();
applyVisibility();   // initialise aVis + lines

// ---- polygon worlds + export wiring --------------------------------------
const polyToggleBtn = document.getElementById('polyToggle');
polyToggleBtn.onclick = ()=>{
  polyOn = !polyOn; polyToggleBtn.textContent = polyOn ? 'on' : 'off'; polyDirty = true;
  if (polyOn){ computePolyFigures(); drawPolyFigures(); } else { drawPolyFigures(); }
};
bind('polyOp','vPolyOp', x=>{ polyOpacity=x; drawPolyFigures(); });
document.getElementById('polyBlend').addEventListener('change', e=>{ polyBlend=e.target.value; drawPolyFigures(); });
document.getElementById('polyStroke').addEventListener('change', e=>{ polyStroke=e.target.checked; drawPolyFigures(); });

document.getElementById('exportPNG').onclick = ()=>{
  renderer.render(scene, camera);
  const dpr=Math.min(window.devicePixelRatio,2), W=cssW(), H=cssH();
  const c=document.createElement('canvas'); c.width=W*dpr; c.height=H*dpr;
  const g=c.getContext('2d');
  g.fillStyle='__BG__'; g.fillRect(0,0,c.width,c.height);
  g.drawImage(renderer.domElement,0,0,c.width,c.height);
  if (polyOn) g.drawImage(polyCanvas,0,0,c.width,c.height);
  c.toBlob(b=>downloadBlob(b,'constellation.png'));
};
document.getElementById('exportSVG').onclick = ()=>{
  const W=cssW(), H=cssH(); const {proj,refDist}=projectScreen(W,H);
  let s='<svg xmlns="http://www.w3.org/2000/svg" width="'+W+'" height="'+H+'" viewBox="0 0 '+W+' '+H+'">';
  s+='<rect width="100%" height="100%" fill="__BG__"/>';
  if (polyOn){ if(polyDirty) computePolyFigures();
    const mb = polyBlend==='source-over' ? 'normal' : polyBlend;
    s+='<g style="mix-blend-mode:'+mb+'" fill-opacity="'+polyOpacity.toFixed(2)+'">';
    for(const f of polyFigures){ let d='';
      for(const ring of f.contours){ d+='M'+ring.map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join('L')+'Z'; }
      s+='<path d="'+d+'" fill="'+f.col+'" fill-rule="evenodd"'+(polyStroke?(' stroke="'+f.col+'" stroke-width="1"'):'')+'/>'; }
    s+='</g>'; }
  if (lineSeg && lineSeg.visible){ s+='<g stroke-opacity="'+lineSeg.material.opacity.toFixed(2)+'" stroke-width="0.6">';
    for(let e=0;e<EDGES.length;e++){ const a=EDGES[e][0],b=EDGES[e][1]; if(aVis[a]<0.5||aVis[b]<0.5) continue;
      const pa=proj[a],pb=proj[b]; if(!pa||!pb) continue;
      s+='<line x1="'+pa.x.toFixed(1)+'" y1="'+pa.y.toFixed(1)+'" x2="'+pb.x.toFixed(1)+'" y2="'+pb.y.toFixed(1)+'" stroke="'+rgbOf(a)+'"/>'; }
    s+='</g>'; }
  s+='<g>';
  for(let i=0;i<N;i++){ const pr=proj[i]; if(!pr) continue; const p=POINTS[i]; const b=Math.max(0.2,p.d);
    let r=(0.9+b*0.8)*(refDist/pr.depth)*1.2; r=Math.max(0.4,Math.min(6,r));
    s+='<circle cx="'+pr.x.toFixed(1)+'" cy="'+pr.y.toFixed(1)+'" r="'+r.toFixed(2)+'" fill="'+rgbOf(i)+'" fill-opacity="'+Math.min(1,0.4+b*0.6).toFixed(2)+'"/>'; }
  s+='</g></svg>';
  downloadBlob(new Blob([s],{type:'image/svg+xml'}),'constellation.svg');
};

// =========================================================================
// WebXR: place the viewer inside the cloud at a comfortable scale
// =========================================================================
document.body.appendChild(VRButton.createButton(renderer));
const VR_RADIUS = 5.0;        // metres from centre to the outermost star
let driftEnabledDesktop = true;

renderer.xr.addEventListener('sessionstart', ()=>{
  const s = VR_RADIUS / radius;
  world.scale.setScalar(s);
  world.position.set(-cx*s, -cy*s + 1.4, -cz*s);  // eye height ~1.4m
  uniforms.uScale.value = s;
  driftEnabledDesktop = false;
  dolly.position.set(0,0,0);
  polyCtx.clearRect(0,0,cssW(),cssH());
  document.getElementById('vrhint').textContent =
    'left stick move · right stick turn · trigger ignites nearest cluster';
});
renderer.xr.addEventListener('sessionend', ()=>{
  world.scale.setScalar(1); world.position.set(0,0,0);
  uniforms.uScale.value = 1; driftEnabledDesktop = true;
  dolly.position.set(0,0,0); dolly.rotation.set(0,0,0);
  camera.position.copy(DESK_CAM); controls.target.set(cx,cy,cz); controls.update();
});

// controllers
const c0 = renderer.xr.getController(0);
const c1 = renderer.xr.getController(1);
dolly.add(c0); dolly.add(c1);
function igniteFromController(ctrl){
  // ray from controller; ignite cluster of nearest star within an angle
  const o = new THREE.Vector3(), d = new THREE.Vector3(0,0,-1);
  ctrl.getWorldPosition(o); d.applyQuaternion(ctrl.getWorldQuaternion(new THREE.Quaternion()));
  let best=-1, bestDot=0.985, v=new THREE.Vector3();
  for (let i=0;i<N;i++){
    v.set(POINTS[i].x,POINTS[i].y,POINTS[i].z); world.localToWorld(v); v.sub(o).normalize();
    const dot = v.dot(d); if (dot>bestDot){ bestDot=dot; best=i; }
  }
  if (best>=0) setIgnited(POINTS[best].ci, POINTS[best].clu); else setIgnited(-2,'');
}
c0.addEventListener('selectstart', ()=>igniteFromController(c0));
c1.addEventListener('selectstart', ()=>igniteFromController(c1));

const fwd=new THREE.Vector3(), right=new THREE.Vector3(), up=new THREE.Vector3(0,1,0);
function vrLocomotion(dt){
  const session = renderer.xr.getSession(); if(!session) return;
  const head = renderer.xr.getCamera();
  head.getWorldDirection(fwd); fwd.y=0; fwd.normalize();
  right.crossVectors(fwd, up).normalize();
  const speed = VR_RADIUS*0.6;
  for (const src of session.inputSources){
    if (!src.gamepad) continue;
    const ax = src.gamepad.axes;
    const gx = ax.length>=4 ? ax[2] : (ax[0]||0);
    const gy = ax.length>=4 ? ax[3] : (ax[1]||0);
    const hand = src.handedness;
    if (hand==='left' || hand==='none'){
      dolly.position.addScaledVector(fwd, -gy*speed*dt);
      dolly.position.addScaledVector(right, gx*speed*dt);
    } else if (hand==='right'){
      if (Math.abs(gx)>0.6) dolly.rotateY(-Math.sign(gx)*1.8*dt);
    }
  }
}

// =========================================================================
// Loop
// =========================================================================
const clock = new THREE.Clock();
function onResize(){
  camera.aspect=W()/H(); camera.updateProjectionMatrix();
  renderer.setSize(W(),H()); uniforms.uH.value=H();
  const dpr=Math.min(window.devicePixelRatio,2);
  polyCanvas.width=W()*dpr; polyCanvas.height=H()*dpr;
  polyCanvas.style.width=W()+'px'; polyCanvas.style.height=H()+'px';
  polyCtx.setTransform(dpr,0,0,dpr,0,0);
  polyDirty=true;
}
window.addEventListener('resize', onResize);
onResize();   // size the WebGL + polygon canvases up front

renderer.setAnimationLoop(()=>{
  const dt = Math.min(clock.getDelta(), 0.05);
  uniforms.uTime.value += dt;
  maybeRecomputeWorlds(dt);
  if (renderer.xr.isPresenting){
    vrLocomotion(dt);
  } else {
    if (driftEnabledDesktop && driftSpeed>0.0001) world.rotation.y += driftSpeed*dt;
    controls.update();
    maybePoly(dt);
  }
  renderer.render(scene, camera);
});
</script>
</body>
</html>"""


# =============================================================================
# Build the embeddable / standalone HTML
# =============================================================================
def build_points(df_valid, embeddings_3d, density, ci, periods, color_field, palette_name):
    if color_field == "none" or color_field not in df_valid.columns:
        rgb_lookup = None
    else:
        vals = df_valid[color_field].astype(str)
        pal = categorical_palette(vals, palette_name)
        rgb_lookup = {v: _hex_to_rgb01(c) for v, c in pal.items()}

    pidx = {name: i for i, name in enumerate(periods)}
    pts = []
    for i in range(len(df_valid)):
        row = df_valid.iloc[i]
        x, y, z = embeddings_3d[i]
        clu = str(row.get("cluster", "unknown"))
        era = str(row.get("era", "unknown"))
        if rgb_lookup is None:
            r, g, b = 0.85, 0.87, 0.92
        else:
            r, g, b = rgb_lookup.get(str(row[color_field]), [0.85, 0.87, 0.92])
        pts.append({
            "x": round(float(x), 4), "y": round(float(y), 4), "z": round(float(z), 4),
            "r": r, "g": g, "b": b,
            "d": round(float(density[i]), 4),
            "ci": int(ci[i]),
            "pi": int(pidx.get(era, 0)),
            "id": str(row.get("id", "")),
            "cat": str(row.get("category", "")),
            "sub": str(row.get("subtype", "")),
            "era": era,
            "clu": clu,
        })
    return pts


def render_html(points, edges, periods, background):
    return (
        CONSTELLATION_TEMPLATE
        .replace("__POINTS_JSON__", json.dumps(points))
        .replace("__EDGES_JSON__", json.dumps(edges))
        .replace("__PERIODS_JSON__", json.dumps(periods))
        .replace("__SETTINGS_JSON__", json.dumps({}))
        .replace("__BG__", background)
    )


# =============================================================================
# UI
# =============================================================================
st.title("✦ Constellation Explorer")
st.caption(
    "A second representation of the same latent space. Positions are the 3D "
    "UMAP; the agreement-density field is re-read as light. Dense regions where "
    "representations agree bloom into a milky core; conflict zones stay sparse "
    "and singular. The worlds-unify slider welds each cluster into forms using "
    "the original viewer's spanning-forest logic, measured from your vantage, so "
    "the forms assemble and dissolve as you move through the space. Click a star "
    "to ignite its cluster."
)

df_valid, valid_features, embeddings_3d, missing = load_data()

if missing:
    st.error("Missing required files. Run the catalog and embedding notebooks first.")
    for m in missing:
        st.code(m)
    st.stop()

with st.sidebar:
    st.header("Field")
    color_field = st.selectbox(
        "Color by", COLOR_FIELDS, index=0,
        help="What hue each star takes. Cluster keeps colours aligned with the "
             "forms you ignite.",
    )
    palette_name = st.selectbox("Palette", PALETTE_NAMES, index=0,
                                disabled=(color_field == "none"))

    st.header("Light")
    background = st.color_picker("Background", "#05060a")
    density_k = st.slider(
        "Density neighbours (k)", 3, 30, 8, 1,
        help="How many nearest neighbours define a star's agreement density. "
             "Higher k smooths the milkiness across broader regions.",
    )

    st.header("Constellation lines")
    show_edges = st.checkbox("Link nearest neighbours", value=True)
    edges_k = st.slider("Links per star (k)", 1, 8, 2, 1, disabled=not show_edges)

    st.divider()
    st.caption(
        "Star size, glow, milkiness, lens, unify, twinkle, drift, cluster "
        "ignition, the period toggle menu, the polygon-worlds overlay, and SVG / "
        "PNG export are all in the overlay inside the view. The polygon layer "
        "runs the original viewer's worlds-unify logic in screen space over the "
        "same clusters, as a sanity check against the fused stars. The sidebar "
        "sets the data-level choices that rebuild the scene."
    )

# ---- prepare data ----
feats_f32 = np.ascontiguousarray(valid_features.astype(np.float32))
density = compute_density(feats_f32.tobytes(), feats_f32.shape[0],
                          feats_f32.shape[1], int(density_k))
ci, _ = cluster_ints(df_valid)
periods = period_order(df_valid)

edges = []
if show_edges:
    edges = compute_knn_edges(feats_f32.tobytes(), feats_f32.shape[0],
                              feats_f32.shape[1], int(edges_k))

t0 = time.time()
points = build_points(df_valid, embeddings_3d, density, ci, periods, color_field, palette_name)
html = render_html(points, edges, periods, background)
build_secs = time.time() - t0

# ---- view ----
components.html(html, height=760, scrolling=False)

# ---- export / info ----
left, right = st.columns([3, 1])
with left:
    st.caption(
        "WebXR is usually blocked inside this embedded view. For the headset, "
        "download the standalone page below and serve it over https or "
        "localhost — the Enter VR button activates there."
    )
with right:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}__constellation_{color_field}.html"
    st.download_button(
        "Download standalone VR page",
        data=html.encode("utf-8"),
        file_name=fname,
        mime="text/html",
        type="primary",
        use_container_width=True,
    )
    if st.button("Save to data/exports/", use_container_width=True):
        out = EXPORTS_DIR / fname
        out.write_text(html, encoding="utf-8")
        st.success(f"Saved: {out.name}")

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Stars", f"{len(points):,}")
c2.metric("Links", f"{len(edges):,}")
c3.metric("Build time", f"{build_secs:.2f}s")
