"""
Latent Explorer — Streamlit web UI.

Sits at the project root, alongside notebooks/ and data/. Run with:

    streamlit run app.py

Reads the manifest, CLIP features, 2D UMAP, and clusters that
latent_embedding.ipynb wrote into data/embeddings/. Computes 3D UMAP
on first run and caches it. All rendering goes through the same code
path as the notebook version.
"""
from __future__ import annotations

import base64
import io
import json
import time
import xml.sax.saxutils as _su
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image as PILImage

import matplotlib
import matplotlib.colors as mcolors


# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Latent Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="◐",
)

# Tighten Streamlit's default padding so the preview gets more room
st.markdown("""
<style>
  .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 100%; }
  [data-testid="stSidebar"] { min-width: 360px; }
  h1 { margin-top: 0; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Paths
# =============================================================================
def _find_project_root(start: Path) -> Path:
    """Walk up from `start` looking for data/catalog/manifest.json.

    Falls back to start.parent if the app lives in a folder named
    `notebooks` (matches the convention the other notebooks use),
    otherwise falls back to `start` itself.
    """
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
EXPORTS_DIR = PROJECT_ROOT / "data" / "exports" / "latent_svg"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

SHAPE_CHOICES = ["circle", "square", "triangle", "diamond", "cross", "image"]
PALETTE_NAMES = [
    "tab10", "tab20", "Set1", "Set2", "Set3", "Dark2", "Paired", "Accent",
    "viridis", "plasma", "magma", "inferno", "cividis", "twilight", "hsv",
    "Greys", "Blues", "Reds", "Greens",
]


# =============================================================================
# Projection
# =============================================================================
def project_3d_to_screen(points_3d, elev_deg, azim_deg, width, height, padding=80):
    P = np.asarray(points_3d, dtype=float)
    P = P - P.mean(axis=0)
    el = np.radians(elev_deg)
    az = np.radians(azim_deg)
    Ry = np.array([[ np.cos(az), 0, np.sin(az)],
                   [ 0,          1, 0         ],
                   [-np.sin(az), 0, np.cos(az)]])
    Rx = np.array([[1, 0,           0          ],
                   [0, np.cos(el), -np.sin(el)],
                   [0, np.sin(el),  np.cos(el)]])
    Q = (Rx @ Ry @ P.T).T
    xy_world = Q[:, :2]
    depth = Q[:, 2]
    mins = xy_world.min(axis=0)
    maxs = xy_world.max(axis=0)
    span = np.where(maxs - mins == 0, 1.0, maxs - mins)
    inner_w = width - 2 * padding
    inner_h = height - 2 * padding
    scale = min(inner_w / span[0], inner_h / span[1])
    centered = xy_world - (mins + maxs) / 2
    scaled = centered * scale
    screen_x = scaled[:, 0] + width / 2
    screen_y = -scaled[:, 1] + height / 2
    return np.column_stack([screen_x, screen_y]), depth, scale


def project_2d_to_screen(points_2d, width, height, padding=80):
    P = np.asarray(points_2d, dtype=float)
    mins = P.min(axis=0)
    maxs = P.max(axis=0)
    span = np.where(maxs - mins == 0, 1.0, maxs - mins)
    inner_w = width - 2 * padding
    inner_h = height - 2 * padding
    scale = min(inner_w / span[0], inner_h / span[1])
    centered = P - (mins + maxs) / 2
    scaled = centered * scale
    screen_x = scaled[:, 0] + width / 2
    screen_y = -scaled[:, 1] + height / 2
    return np.column_stack([screen_x, screen_y]), scale


# =============================================================================
# Palettes
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


# =============================================================================
# SVG writer
# =============================================================================
class SVGCanvas:
    INKSCAPE = "http://www.inkscape.org/namespaces/inkscape"
    SODIPODI = "http://sodipodi.sourceforge.net/DTD/sodipodi-0.0.dtd"

    def __init__(self, width, height, background="#ffffff", include_bg=True):
        self.width = float(width)
        self.height = float(height)
        self.background = background
        self._layers = {}
        self._order = []
        if include_bg:
            self._add_layer("background")
            self._layers["background"].append(
                f'<rect x="0" y="0" width="{self.width:.0f}" '
                f'height="{self.height:.0f}" fill="{background}"/>'
            )

    def _add_layer(self, name):
        if name not in self._layers:
            self._layers[name] = []
            self._order.append(name)

    @staticmethod
    def _id_safe(name):
        out = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(name))
        if not out or not (out[0].isalpha() or out[0] == "_"):
            out = "L_" + out
        return out

    def add_circle(self, layer, x, y, r, fill, stroke=None, sw=0, opacity=1.0):
        self._add_layer(layer)
        s = f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}"'
        if opacity < 1.0: s += f' fill-opacity="{opacity:.2f}"'
        if stroke and sw > 0:
            s += f' stroke="{stroke}" stroke-width="{sw:.2f}"'
        s += "/>"
        self._layers[layer].append(s)

    def add_polygon(self, layer, points, fill, stroke=None, sw=0, opacity=1.0):
        self._add_layer(layer)
        pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        s = f'<polygon points="{pts}" fill="{fill}"'
        if opacity < 1.0: s += f' fill-opacity="{opacity:.2f}"'
        if stroke and sw > 0:
            s += f' stroke="{stroke}" stroke-width="{sw:.2f}"'
        s += "/>"
        self._layers[layer].append(s)

    def add_line(self, layer, x1, y1, x2, y2, stroke, sw=1.0, opacity=1.0):
        self._add_layer(layer)
        s = (f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
             f'stroke="{stroke}" stroke-width="{sw:.2f}"')
        if opacity < 1.0: s += f' stroke-opacity="{opacity:.2f}"'
        s += "/>"
        self._layers[layer].append(s)

    def add_image(self, layer, x, y, w, h, image_b64, mime="image/png", opacity=1.0):
        self._add_layer(layer)
        href = f"data:{mime};base64,{image_b64}"
        s = (f'<image x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
             f'preserveAspectRatio="xMidYMid meet" xlink:href="{href}"')
        if opacity < 1.0: s += f' opacity="{opacity:.2f}"'
        s += "/>"
        self._layers[layer].append(s)

    def add_text(self, layer, x, y, text, fill, size=10, font="sans-serif"):
        self._add_layer(layer)
        safe = _su.escape(str(text))
        s = (f'<text x="{x:.2f}" y="{y:.2f}" font-family="{font}" '
             f'font-size="{size:.1f}" fill="{fill}">{safe}</text>')
        self._layers[layer].append(s)

    def to_string(self, title="latent_explorer"):
        parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (f'<svg xmlns="http://www.w3.org/2000/svg" '
             f'xmlns:xlink="http://www.w3.org/1999/xlink" '
             f'xmlns:inkscape="{self.INKSCAPE}" '
             f'xmlns:sodipodi="{self.SODIPODI}" '
             f'width="{self.width:.0f}" height="{self.height:.0f}" '
             f'viewBox="0 0 {self.width:.0f} {self.height:.0f}">'),
            f'<title>{_su.escape(title)}</title>',
        ]
        for name in self._order:
            parts.append(
                f'<g id="{self._id_safe(name)}" '
                f'inkscape:label="{_su.escape(name)}" '
                f'inkscape:groupmode="layer">'
            )
            parts.extend(self._layers[name])
            parts.append("</g>")
        parts.append("</svg>")
        return "\n".join(parts)


def shape_points(shape, cx, cy, r):
    if shape == "square":
        return [(cx - r, cy - r), (cx + r, cy - r),
                (cx + r, cy + r), (cx - r, cy + r)]
    if shape == "triangle":
        return [(cx, cy - r), (cx + r * 0.866, cy + r * 0.5),
                (cx - r * 0.866, cy + r * 0.5)]
    if shape == "diamond":
        return [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
    return None


def add_marker(svg, layer, shape, x, y, r, fill, stroke, sw, opacity,
               thumb_b64=None, thumb_size=None):
    if shape == "circle":
        svg.add_circle(layer, x, y, r, fill, stroke=stroke, sw=sw, opacity=opacity)
    elif shape == "cross":
        svg.add_line(layer, x - r, y, x + r, y, stroke=fill, sw=max(sw, 1.0), opacity=opacity)
        svg.add_line(layer, x, y - r, x, y + r, stroke=fill, sw=max(sw, 1.0), opacity=opacity)
    elif shape == "image":
        if thumb_b64 is None:
            svg.add_circle(layer, x, y, r, fill, stroke=stroke, sw=sw, opacity=opacity)
            return
        tw, th = thumb_size
        max_dim = 2 * r
        if tw >= th:
            w = max_dim
            h = max_dim * th / tw
        else:
            h = max_dim
            w = max_dim * tw / th
        svg.add_image(layer, x - w / 2, y - h / 2, w, h, thumb_b64, opacity=opacity)
    else:
        pts = shape_points(shape, x, y, r)
        if pts is None:
            svg.add_circle(layer, x, y, r, fill, stroke=stroke, sw=sw, opacity=opacity)
        else:
            svg.add_polygon(layer, pts, fill, stroke=stroke, sw=sw, opacity=opacity)


# =============================================================================
# Data loading
# =============================================================================
@st.cache_data(show_spinner="Loading manifest and features...")
def load_data():
    """Load manifest, features, 2D UMAP, and (compute if missing) 3D UMAP."""
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

    # 3D UMAP
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

    # 2D PCA (cheap, compute now)
    from sklearn.decomposition import PCA
    pca_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(valid_features)
    df_valid["pca_x"] = pca_2d[:, 0]
    df_valid["pca_y"] = pca_2d[:, 1]

    # Normalize categorical columns to strings
    for col in ["category", "subtype", "era", "cluster"]:
        if col in df_valid.columns:
            df_valid[col] = df_valid[col].astype("object").where(df_valid[col].notna(), "unknown")
            df_valid[col] = df_valid[col].astype(str)

    return df_valid, valid_features, embeddings_3d, []


@st.cache_data(show_spinner="Recomputing UMAP layout...")
def recompute_umap_3d(n_neighbors: int, min_dist: float, metric: str,
                      random_state: int, pca_components: int = 50):
    """Recompute the 3D UMAP layout from clip_features.npy with arbitrary
    parameters. Cached by parameter tuple — changing any value triggers a
    fresh computation; same values return the cached result instantly. The
    same PCA → UMAP pipeline as load_data, just with user-set knobs."""
    features_path = EMBEDDINGS_DIR / "clip_features.npy"
    if not features_path.exists():
        return None
    features = np.load(features_path)
    # Apply the same success mask as load_data so row order matches df_valid
    with open(EMBEDDINGS_DIR / "index.json") as f:
        feature_index = json.load(f)
    success_mask = np.array(feature_index["success"], dtype=bool)
    valid_features = features[success_mask]

    from sklearn.decomposition import PCA
    import umap as _umap
    n_pca = min(pca_components, valid_features.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=random_state)
    features_pca = pca.fit_transform(valid_features)
    reducer = _umap.UMAP(
        n_components=3,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric=metric,
        random_state=int(random_state),
    )
    return reducer.fit_transform(features_pca)


# Default UMAP parameters — what load_data uses, and what the sidebar form
# initializes to. Changing any of these in the sidebar triggers recompute.
UMAP_DEFAULTS = dict(
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42,
)


@st.cache_data(show_spinner=False)
def get_thumbnail_b64(rel_path: str, max_edge: int):
    abs_path = PROJECT_ROOT / rel_path
    img = PILImage.open(abs_path).convert("RGB")
    img.thumbnail((max_edge, max_edge), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii"), img.size


@st.cache_data(show_spinner="Computing KNN edges...")
def compute_knn_edges(features_bytes: bytes, n: int, dim: int, k: int):
    """KNN edges. features_bytes makes the cache key stable."""
    from sklearn.neighbors import NearestNeighbors
    features = np.frombuffer(features_bytes, dtype=np.float32).reshape(n, dim)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(features)
    _, indices = nn.kneighbors(features)
    seen = set()
    edges = []
    for i in range(n):
        for j in indices[i, 1:]:
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append((a, b))
    return edges


# =============================================================================
# Concave-hull boundary (closed curve around the outside of a 2D projection)
# =============================================================================
# Algorithm:
#   1. Build a convex hull (scipy).
#   2. For each hull edge, find the nearest interior point near its midpoint.
#      If close enough (controlled by `concavity`) and on the interior side of
#      the edge, insert it into the boundary, splitting the edge.
#   3. Repeat until no edge can be split.
#   4. Smooth the polygon with a moving average.
# Higher concavity = looser, closer to the convex hull.
# Lower concavity = tighter; below ~1.5 the boundary can self-intersect.
def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def fit_concave_hull(points_2d, concavity=2.5, smoothing=3, max_iter=120):
    pts = np.asarray(points_2d, dtype=float)
    if len(pts) < 4:
        return None
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
    except Exception:
        return None
    boundary = [tuple(pts[i]) for i in hull.vertices]  # CCW
    bset = set(boundary)
    interior = [tuple(p) for p in pts if tuple(p) not in bset]
    if len(boundary) < 3:
        return None

    for _ in range(max_iter):
        changed = False
        new_b = []
        for i in range(len(boundary)):
            p1 = boundary[i]
            p2 = boundary[(i + 1) % len(boundary)]
            new_b.append(p1)
            elen = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            threshold = elen / concavity
            best, best_d, best_idx = None, np.inf, -1
            for j, p in enumerate(interior):
                d = np.hypot(p[0] - mx, p[1] - my)
                if d < threshold and d < best_d and _cross(p1, p2, p) > 0:
                    best, best_d, best_idx = p, d, j
            if best is not None:
                new_b.append(best)
                interior.pop(best_idx)
                changed = True
        boundary = new_b
        if not changed:
            break

    arr = np.array(boundary, dtype=float)
    n = len(arr)
    for _ in range(smoothing):
        new = arr.copy()
        for i in range(n):
            prev_p = arr[(i - 1) % n]
            nxt_p = arr[(i + 1) % n]
            new[i] = 0.25 * prev_p + 0.5 * arr[i] + 0.25 * nxt_p
        arr = new
    return arr


def closed_curve_to_svg_path(points):
    """Catmull-Rom closed loop -> cubic-Bezier SVG path with Z."""
    if points is None or len(points) < 3:
        return ""
    pts = np.asarray(points)
    n = len(pts)
    cmds = [f"M {pts[0, 0]:.2f},{pts[0, 1]:.2f}"]
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_curr = pts[i]
        p_next = pts[(i + 1) % n]
        p_next2 = pts[(i + 2) % n]
        c1 = p_curr + (p_next - p_prev) / 6.0
        c2 = p_next - (p_next2 - p_curr) / 6.0
        cmds.append(
            f"C {c1[0]:.2f},{c1[1]:.2f} {c2[0]:.2f},{c2[1]:.2f} "
            f"{p_next[0]:.2f},{p_next[1]:.2f}"
        )
    cmds.append("Z")
    return " ".join(cmds)


# =============================================================================
# Render
# =============================================================================
def render_to_svg(settings, df_valid, valid_features):
    s = settings
    W, H, pad = s["width"], s["height"], s["padding"]
    svg = SVGCanvas(W, H, background=s["background"])

    if s["projection"] == "umap_3d":
        pts = df_valid[["umap_x_3d", "umap_y_3d", "umap_z_3d"]].to_numpy()
        screen, depth, _ = project_3d_to_screen(
            pts, s["elev"], s["azim"], W, H, padding=pad
        )
    elif s["projection"] == "pca_2d":
        pts = df_valid[["pca_x", "pca_y"]].to_numpy()
        screen, _ = project_2d_to_screen(pts, W, H, padding=pad)
        depth = np.zeros(len(df_valid))
    else:
        pts = df_valid[["umap_x", "umap_y"]].to_numpy()
        screen, _ = project_2d_to_screen(pts, W, H, padding=pad)
        depth = np.zeros(len(df_valid))

    if s["color_field"] == "none":
        fills = [s["override_color"]] * len(df_valid)
    else:
        vals = df_valid[s["color_field"]].astype(str)
        color_lookup = categorical_palette(vals, s["palette_name"])
        fills = [color_lookup[v] for v in vals]

    order = np.argsort(depth) if s["projection"] == "umap_3d" else np.arange(len(df_valid))

    if s["projection"] == "umap_3d" and s["depth_size_scale"] > 0:
        d = depth - depth.min()
        d = d / d.max() if d.max() > 0 else d
        size_scale = 1.0 + s["depth_size_scale"] * (d - 0.5)
    else:
        size_scale = np.ones(len(df_valid))

    if s["show_edges"]:
        feats_f32 = np.ascontiguousarray(valid_features.astype(np.float32))
        edges = compute_knn_edges(
            feats_f32.tobytes(), feats_f32.shape[0], feats_f32.shape[1], s["edges_k"]
        )
        for a, b in edges:
            xa, ya = screen[a]; xb, yb = screen[b]
            svg.add_line("edges", xa, ya, xb, yb,
                         stroke=s["edges_color"], sw=s["edges_width"],
                         opacity=s["edges_opacity"])

    use_image = (s["shape"] == "image")
    base_r = s["point_size"]
    for k in order:
        row = df_valid.iloc[k]
        x, y = screen[k]
        r = base_r * size_scale[k]
        fill = fills[k]
        if s["stratify_by"] == "none":
            layer = "points"
        else:
            v = row[s["stratify_by"]]
            label = "noise" if (s["stratify_by"] == "cluster" and str(v) == "-1") else str(v)
            layer = f'{s["stratify_by"]}_{label}'

        if use_image:
            try:
                b64, sz = get_thumbnail_b64(row["path"], s["image_max_edge"])
            except Exception:
                b64, sz = None, None
            add_marker(svg, layer, "image", x, y, r,
                       fill=fill, stroke=s["stroke_color"],
                       sw=s["stroke_width"], opacity=s["opacity"],
                       thumb_b64=b64, thumb_size=sz)
        else:
            add_marker(svg, layer, s["shape"], x, y, r,
                       fill=fill, stroke=s["stroke_color"],
                       sw=s["stroke_width"], opacity=s["opacity"])

    # Concave-hull boundary around the projected points
    if s.get("show_curve"):
        curve_ctrls = fit_concave_hull(
            screen,
            concavity=float(s.get("curve_concavity", 2.5)),
            smoothing=int(s.get("curve_smoothing", 3)),
        )
        if curve_ctrls is not None:
            path_d = closed_curve_to_svg_path(curve_ctrls)
            dash_attr = ""
            if s.get("curve_dashed", True):
                dash_attr = (f' stroke-dasharray="{s.get("curve_dash_on", 8.0):.1f} '
                             f'{s.get("curve_dash_off", 4.0):.1f}"')
            svg._add_layer("boundary_curve")
            svg._layers["boundary_curve"].append(
                f'<path d="{path_d}" fill="none" '
                f'stroke="{s.get("curve_color", "#000000")}" '
                f'stroke-width="{float(s.get("curve_width", 2.0)):.2f}" '
                f'stroke-linecap="round" stroke-linejoin="round"{dash_attr}/>'
            )

    if s["show_labels"]:
        for k in range(len(df_valid)):
            x, y = screen[k]
            svg.add_text("labels", x + base_r + 2, y + 3,
                         df_valid.iloc[k]["id"],
                         fill=s["labels_color"], size=s["labels_size"])

    return svg


# =============================================================================
# three.js 3D viewer HTML template
# =============================================================================
# Self-contained HTML page that renders the latent space as image sprites
# using three.js. Substituted at runtime: __POINTS_JSON__, __BG__.
THREEJS_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<style>
  html, body { margin: 0; padding: 0; overflow: hidden; background: __BG__; }
  canvas { display: block; }
  #overlay {
    position: absolute; top: 10px; left: 10px;
    z-index: 10;
    color: #ddd; font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    background: rgba(0,0,0,0.55);
    padding: 8px 12px;
    border-radius: 4px;
    backdrop-filter: blur(4px);
    user-select: none;
  }
  #overlay .row { margin: 2px 0; }
  #overlay button {
    background: #2a2a2a; color: #ddd;
    border: 1px solid #555;
    padding: 4px 10px; cursor: pointer;
    font-family: inherit; font-size: 11px;
    border-radius: 3px;
    margin-top: 4px; margin-right: 4px;
  }
  #overlay button:hover { background: #3a3a3a; }
  #overlay select, #overlay input[type="range"] { vertical-align: middle; }
  #overlay .modeRow label { margin-right: 6px; cursor: pointer; }
  #overlay hr.sep {
    border: 0; border-top: 1px solid #333;
    margin: 8px 0 6px 0;
  }
  #overlayCanvas {
    position: absolute; top: 0; left: 0;
    pointer-events: none;
    display: block;
  }
  #tooltip {
    position: absolute; pointer-events: none;
    z-index: 11;
    color: #fff; font-family: ui-monospace, monospace;
    font-size: 11px;
    background: rgba(0,0,0,0.8);
    padding: 4px 8px; border-radius: 3px;
    display: none;
  }
</style>
</head>
<body>
<div id="overlay">
  <div class="row">elev <b><span id="elev">0</span>°</b> &nbsp; azim <b><span id="azim">0</span>°</b></div>
  <div class="row">sprite <span id="scale">1.0</span></div>
  <div>
    <button id="export" title="Exports a layered SVG of exactly this 3D camera angle, using the current opacity and blend mode.">Download this view as SVG</button>
    <button id="copy">Copy angles</button>
    <button id="reset">Reset</button>
  </div>
  <div class="row" style="margin-top:6px;color:#999;font-size:10px;">
    sprite size: <input id="sizeRange" type="range" min="0.3" max="3.0" step="0.1" value="1.0" style="vertical-align:middle">
  </div>
  <div class="row" style="margin-top:4px;color:#999;font-size:10px;">
    lens: <input id="lensRange" type="range" min="15" max="100" step="1" value="45" style="vertical-align:middle;width:90px">
    <span id="lensValue">45°</span> <span style="color:#666;">(wide = strong perspective)</span>
  </div>
  <div class="row" style="margin-top:4px;color:#999;font-size:10px;">
    opacity: <input id="opacityRange" type="range" min="0.05" max="1.0" step="0.05" value="1.0" style="vertical-align:middle;width:90px">
    <span id="opacityValue">1.00</span>
  </div>
  <div class="row" style="margin-top:4px;color:#999;font-size:10px;">
    blend: <select id="blendMode" style="background:#1a1a1a;color:#ccc;border:1px solid #555;font-size:11px;padding:1px 3px;">
      <option value="normal">normal</option>
      <option value="additive">additive (Linear Dodge)</option>
      <option value="screen">screen</option>
      <option value="lighten">lighten (max)</option>
      <option value="multiply">multiply</option>
      <option value="darken">darken (min)</option>
      <option value="difference">difference</option>
    </select>
  </div>
  <div class="row" style="margin-top:6px;color:#888;font-size:10px;">
    last export: <span id="lastExport">none</span>
  </div>
  <hr class="sep">
  <div class="row modeRow" style="color:#999;font-size:11px;">
    <b style="color:#ddd;">render mode</b><br>
    <label><input type="radio" name="renderMode" value="images" checked>images</label>
    <label><input type="radio" name="renderMode" value="heatmap">heatmap</label>
    <label><input type="radio" name="renderMode" value="linework">linework</label>
  </div>
  <div id="heatmapControls" class="row" style="display:none;margin-top:4px;color:#999;font-size:10px;">
    bandwidth: <input id="bandwidthRange" type="range" min="10" max="160" step="2" value="40" style="width:80px">
    <span id="bandwidthValue">40</span>px<br>
    levels: <input id="levelsRange" type="range" min="2" max="12" step="1" value="6" style="width:80px">
    <span id="levelsValue">6</span><br>
    style:
    <select id="heatmapStyle" style="background:#1a1a1a;color:#ccc;border:1px solid #555;font-size:11px;padding:1px 3px;">
      <option value="filled">filled isobands</option>
      <option value="lines" selected>contour lines</option>
      <option value="both">both</option>
    </select>
  </div>
  <div id="lineworkControls" class="row" style="display:none;margin-top:4px;color:#999;font-size:10px;">
    style:
    <select id="lineworkStyle" style="background:#1a1a1a;color:#ccc;border:1px solid #555;font-size:11px;padding:1px 3px;">
      <option value="segmentation" selected>image segmentation (watershed-ish)</option>
      <option value="collapsed_field">worlds (recognizable exemplar per cluster)</option>
      <option value="composition">swatch composition (rectangles + segmentation)</option>
      <option value="stacked_perspectives">stacked perspectives (Oehlen-ish, binary pixel)</option>
      <option value="all">all edges (rectangle accumulation)</option>
      <option value="ghosted">ghosted (visible solid, hidden dashed)</option>
      <option value="occluded">occluded (visible rectangle edges only)</option>
    </select><br>
    line weight: <input id="lineweightRange" type="range" min="0.3" max="3.0" step="0.1" value="0.8" style="width:80px">
    <span id="lineweightValue">0.8</span>px<br>
    <span style="color:#666;">— segmentation —</span><br>
    sensitivity: <input id="segSensRange" type="range" min="0.35" max="0.95" step="0.02" value="0.75" style="width:80px">
    <span id="segSensValue">0.75</span><br>
    smoothness: <input id="segBlurRange" type="range" min="0.5" max="3.5" step="0.1" value="1.4" style="width:80px">
    <span id="segBlurValue">1.4</span><br>
    worlds complexity: <input id="worldsComplexityRange" type="range" min="0" max="1" step="0.02" value="1" style="width:80px">
    <span id="worldsComplexityValue">1.00</span><br>
    worlds unify: <input id="worldsUnifyRange" type="range" min="0" max="1" step="0.02" value="0" style="width:80px">
    <span id="worldsUnifyValue">0.00</span> <span style="color:#666;">(boolean union → figures)</span><br>
    articulation: <select id="articulationSignalSelect" style="width:140px">
      <option value="autoencoder">autoencoder (typical↔surprising)</option>
      <option value="vision">vision (edge density)</option>
    </select> <span style="color:#666;">(what turns gestural)</span><br>
    <span style="color:#666;">— rectangle modes —</span><br>
    depth fade: <input id="lineworkFadeRange" type="range" min="0" max="0.9" step="0.05" value="0.5" style="width:80px">
    <span id="lineworkFadeValue">0.50</span>
  </div>
  <hr class="sep">
  <div class="row" style="color:#999;font-size:11px;">
    <b style="color:#ddd;">3D enclosing volume</b><br>
    <label><input type="checkbox" id="showVolume">show metaball surface</label>
  </div>
  <div id="volumeControls" class="row" style="display:none;margin-top:4px;color:#999;font-size:10px;">
    resolution: <input id="volumeResRange" type="range" min="20" max="64" step="2" value="40" style="width:80px">
    <span id="volumeResValue">40</span><br>
    metaball size: <input id="volumeStrRange" type="range" min="0.05" max="1.5" step="0.05" value="0.3" style="width:80px">
    <span id="volumeStrValue">0.30</span><br>
    opacity: <input id="volumeOpRange" type="range" min="0.15" max="1.0" step="0.05" value="0.45" style="width:80px">
    <span id="volumeOpValue">0.45</span><br>
    color: <input id="volumeColor" type="color" value="#e6e6e6" style="vertical-align:middle;width:40px"><br>
    style:
    <select id="volumeStyle" style="background:#1a1a1a;color:#ccc;border:1px solid #555;font-size:11px;padding:1px 3px;">
      <option value="solid" selected>solid (lit)</option>
      <option value="wireframe">wireframe</option>
      <option value="both">both</option>
    </select><br>
    <button id="exportVolumeObj" style="margin-top:6px;">Download volume as OBJ</button>
    <div id="exportVolumeStatus" style="font-size:10px;color:#888;margin-top:2px;"></div>
  </div>
  <hr class="sep">
  <div class="row" style="color:#999;font-size:11px;">
    <b style="color:#ddd;">latent contours</b><br>
    <label><input type="checkbox" id="showContours" checked>trace the space</label>
  </div>
  <div id="contourControls" class="row" style="margin-top:4px;color:#999;font-size:10px;">
    density: <input id="contourLevelRange" type="range" min="0.06" max="0.4" step="0.02" value="0.16" style="width:80px">
    <span id="contourLevelValue">0.16</span><br>
    presence: <input id="contourOpacityRange" type="range" min="0.05" max="0.6" step="0.01" value="0.22" style="width:80px">
    <span id="contourOpacityValue">0.22</span><br>
  </div>
</div>
<canvas id="overlayCanvas"></canvas>
<div id="tooltip"></div>

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
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js';

const POINTS = __POINTS_JSON__;
const EDGES = __EDGES_JSON__;
const SETTINGS = __SETTINGS_JSON__;

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color('__BG__');

// Compute centroid + radius for camera framing
let cx=0, cy=0, cz=0;
for (const p of POINTS) { cx+=p.x; cy+=p.y; cz+=p.z; }
cx /= POINTS.length; cy /= POINTS.length; cz /= POINTS.length;
let radius = 0;
for (const p of POINTS) {
  const dx=p.x-cx, dy=p.y-cy, dz=p.z-cz;
  radius = Math.max(radius, Math.sqrt(dx*dx + dy*dy + dz*dz));
}
if (radius === 0) radius = 1;

const W = () => window.innerWidth;
const H = () => window.innerHeight;

const camera = new THREE.PerspectiveCamera(45, W()/H(), 0.01, radius * 200);
camera.position.set(cx, cy + radius * 0.5, cz + radius * 2.5);

const renderer = new THREE.WebGLRenderer({
  antialias: true, alpha: false,
  // preserveDrawingBuffer is needed so the segmentation linework style can
  // read the rendered pixels back via drawImage(...) onto a 2D canvas
  preserveDrawingBuffer: true,
});
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(W(), H());
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(cx, cy, cz);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.enablePan = true;
controls.screenSpacePanning = true;   // pan parallel to the screen (right-drag)
controls.update();

// Scene lighting. Sprites use unlit SpriteMaterial so they don't react to
// these — only the marching-cubes enclosing volume (added later, on demand)
// will be lit. Directional light comes from above-right; ambient fills shadows
// just enough to keep the dark side readable.
const ambientLight = new THREE.AmbientLight(0xffffff, 0.55);
scene.add(ambientLight);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.75);
dirLight.position.set(1.2, 1.6, 1.0);
scene.add(dirLight);
const rimLight = new THREE.DirectionalLight(0xa0c8ff, 0.20);
rimLight.position.set(-1.0, -0.5, -0.8);
scene.add(rimLight);

// ---------------------------------------------------------------------------
// Latent contour lattice. Three families of density level-set lines, one per
// world axis, tracing the shape of the point cloud. They orient the viewer the
// way a grid would, but follow the latent terrain rather than imposing a
// rectilinear frame, and they rotate with the scene so the three-dimensional
// positioning of the space reads as you move through it. The scalar field is
// built once; the density slider only re-runs the cheap contour extraction.
// ---------------------------------------------------------------------------
let showContours = true;
let contourLevelFrac = 0.16;     // primary iso-level as a fraction of peak density
let contourOpacity = 0.22;
let latentContoursObj = null;
let contourDensity = null;       // cached field { dens, G, minx, miny, minz, h, maxD }
let contourGeomCache = null;     // cached 3D segment positions (Float32Array)

function contourLineColor() {
  const bg = (scene.background && typeof scene.background.r === 'number')
    ? scene.background : new THREE.Color('#101014');
  const lum = 0.2126 * bg.r + 0.7152 * bg.g + 0.0722 * bg.b;
  return (lum < 0.5) ? new THREE.Color(0.62, 0.70, 0.80) : new THREE.Color(0.20, 0.24, 0.30);
}

function mcSquaresSlice(grid, G, level, out, to3) {
  const interp = (x0, y0, v0, x1, y1, v1) => {
    const d = v1 - v0; const t = (Math.abs(d) < 1e-12) ? 0.5 : (level - v0) / d;
    return [x0 + t * (x1 - x0), y0 + t * (y1 - y0)];
  };
  for (let y = 0; y < G - 1; y++) for (let x = 0; x < G - 1; x++) {
    const v00 = grid[y*G+x], v10 = grid[y*G+x+1], v01 = grid[(y+1)*G+x], v11 = grid[(y+1)*G+x+1];
    let c = 0; if (v00>=level)c|=1; if (v10>=level)c|=2; if (v11>=level)c|=4; if (v01>=level)c|=8;
    if (c === 0 || c === 15) continue;
    const x0=x, y0=y, x1=x+1, y1=y+1;
    const eT=()=>interp(x0,y0,v00,x1,y0,v10), eR=()=>interp(x1,y0,v10,x1,y1,v11),
          eB=()=>interp(x1,y1,v11,x0,y1,v01), eL=()=>interp(x0,y1,v01,x0,y0,v00);
    const seg=(a,b)=>{ const A=to3(a[0],a[1]), B=to3(b[0],b[1]); out.push(A[0],A[1],A[2],B[0],B[1],B[2]); };
    switch (c) {
      case 1: case 14: seg(eL(),eT()); break;
      case 2: case 13: seg(eT(),eR()); break;
      case 3: case 12: seg(eL(),eR()); break;
      case 4: case 11: seg(eR(),eB()); break;
      case 6: case 9:  seg(eT(),eB()); break;
      case 7: case 8:  seg(eL(),eB()); break;
      case 5:  seg(eL(),eT()); seg(eR(),eB()); break;
      case 10: seg(eL(),eB()); seg(eT(),eR()); break;
    }
  }
}

function buildContourDensity() {
  const G = 44, pad = 1.05, R = radius * pad;
  const minx = cx - R, miny = cy - R, minz = cz - R, h = 2 * R / G;
  const dens = new Float32Array(G*G*G);
  const sigma = radius * 0.13, sc = sigma / h, rad = Math.max(1, Math.ceil(3 * sc)), inv = 1 / (2 * sc * sc);
  const idx = (i,j,k) => i + G*(j + G*k);
  for (const p of POINTS) {
    const gi = Math.round((p.x-minx)/h), gj = Math.round((p.y-miny)/h), gk = Math.round((p.z-minz)/h);
    for (let dk=-rad;dk<=rad;dk++) for (let dj=-rad;dj<=rad;dj++) for (let di=-rad;di<=rad;di++) {
      const i=gi+di, j=gj+dj, k=gk+dk; if (i<0||j<0||k<0||i>=G||j>=G||k>=G) continue;
      dens[idx(i,j,k)] += Math.exp(-(di*di + dj*dj + dk*dk) * inv);
    }
  }
  let maxD = 0; for (let i=0;i<dens.length;i++) if (dens[i]>maxD) maxD=dens[i];
  contourDensity = { dens, G, minx, miny, minz, h, maxD: maxD || 1 };
}

function buildContourGeometry() {
  if (contourGeomCache) return contourGeomCache;
  if (!contourDensity) buildContourDensity();
  const { dens, G, minx, miny, minz, h, maxD } = contourDensity;
  const idx = (i,j,k) => i + G*(j + G*k);
  const W = (i,j,k) => [minx + i*h, miny + j*h, minz + k*h];
  const levels = [contourLevelFrac * maxD, (contourLevelFrac + 0.14) * maxD];
  const nSlices = 6;
  const out = [];
  for (let axis=0; axis<3; axis++) {
    for (let s=0; s<nSlices; s++) {
      const f = (s + 0.5) / nSlices;
      const fixed = Math.round((0.16 + 0.68 * f) * (G - 1));
      const grid = new Float32Array(G*G);
      for (let b=0;b<G;b++) for (let a=0;a<G;a++) {
        let i,j,k;
        if (axis===0){i=fixed;j=a;k=b;} else if (axis===1){i=a;j=fixed;k=b;} else {i=a;j=b;k=fixed;}
        grid[b*G+a] = dens[idx(i,j,k)];
      }
      const to3 = (a,b) => { let i,j,k; if(axis===0){i=fixed;j=a;k=b;}else if(axis===1){i=a;j=fixed;k=b;}else{i=a;j=b;k=fixed;} return W(i,j,k); };
      for (const lv of levels) mcSquaresSlice(grid, G, lv, out, to3);
    }
  }
  contourGeomCache = new Float32Array(out);
  return contourGeomCache;
}

function rebuildLatentContours() {
  if (latentContoursObj) {
    scene.remove(latentContoursObj);
    latentContoursObj.geometry.dispose();
    latentContoursObj.material.dispose();
    latentContoursObj = null;
  }
  if (!showContours) return;
  const positions = buildContourGeometry();
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const mat = new THREE.LineBasicMaterial({
    color: contourLineColor(), transparent: true, opacity: contourOpacity, depthWrite: false,
  });
  const seg = new THREE.LineSegments(geom, mat);
  seg.renderOrder = -1;   // sits behind the sprites and any volume
  scene.add(seg);
  latentContoursObj = seg;
}

rebuildLatentContours();

// Texture loader (decodes base64 data URLs synchronously)
const loader = new THREE.TextureLoader();
let baseScale = radius * 0.10;
let userScale = 1.0;
document.getElementById('scale').textContent = baseScale.toFixed(2);

// Blend mode tables: three.js side and SVG mix-blend-mode side.
//
// "screen" uses CustomBlending for true Porter-Duff screen.
// "lighten" uses MaxEquation: the brighter source/dest wins per channel.
// "darken"  uses MinEquation: the darker  source/dest wins per channel.
// "additive" uses three.js's built-in AdditiveBlending (saturates faster
//   than screen on bright images). SVG side falls back to 'screen' since
//   true additive ('plus-lighter') is not standard mix-blend-mode yet.
const THREE_BLENDS = {
  normal:     { mode: THREE.NormalBlending },
  additive:   { mode: THREE.AdditiveBlending },
  multiply:   { mode: THREE.MultiplyBlending },
  difference: { mode: THREE.SubtractiveBlending },
  screen: {
    mode: THREE.CustomBlending,
    eq: THREE.AddEquation,
    src: THREE.OneMinusDstColorFactor,
    dst: THREE.OneFactor,
  },
  lighten: {
    mode: THREE.CustomBlending,
    eq: THREE.MaxEquation,
    src: THREE.OneFactor,
    dst: THREE.OneFactor,
  },
  darken: {
    mode: THREE.CustomBlending,
    eq: THREE.MinEquation,
    src: THREE.OneFactor,
    dst: THREE.OneFactor,
  },
};
const SVG_BLENDS = {
  normal:     'normal',
  additive:   'screen',
  screen:     'screen',
  lighten:    'lighten',
  multiply:   'multiply',
  darken:     'darken',
  difference: 'difference',
};

let currentOpacity = (SETTINGS.opacity != null) ? SETTINGS.opacity : 1.0;
let currentBlend = 'normal';

// Render mode controls which view the 3D viewport shows, and which SVG
// builder runs when the export button is clicked.
//   'images'   — the textured sprites (original behavior)
//   'heatmap'  — kernel-density contours of the projected positions
//   'linework' — hidden-line-removed outlines of the projected image footprints
let currentMode = 'images';
let heatmapBandwidth = 40;
let heatmapLevels = 6;
let heatmapStyle = 'lines';
let lineworkWeight = 0.8;
let lineworkStyle = 'segmentation';   // 'segmentation' | 'all' | 'ghosted' | 'occluded'
let lineworkFade = 0.5;
let worldsComplexity = 1.0;   // 1 = dense field, lower = conjoin via chords
let worldsUnify = 0.0;        // 0 = fragmented field, higher = boolean-union figures
let segSensitivity = 0.75;            // 0..1; higher = more detail (lower threshold)
let segBlur = 1.4;                    // Gaussian sigma in pixels for pre-blur

// The overlay canvas hosts whatever the non-image modes draw. It sits on
// top of the WebGL canvas with pointer-events disabled so OrbitControls
// still receives mouse input through it.
const overlayCanvas = document.getElementById('overlayCanvas');

// Debounced redraw scheduler. Segmentation linework reads the full WebGL
// canvas back and runs Sobel — too expensive to fire on every camera-change
// event during a drag. Lighter modes (heatmap, geometric linework) also use
// the same scheduler for consistency.
let overlayTimer = null;
let lastOverlayDraw = 0;
function markOverlayDirty(delayMs) {
  // Segmentation linework reads the whole WebGL canvas back and runs Sobel, so
  // it stays on a trailing debounce and only redraws once the camera settles.
  // The lighter styles (worlds, heatmap, geometric linework) redraw live on a
  // throttle, so panning, rotating and zooming show continuous movement in the
  // overlay instead of staying frozen until the motion stops.
  const heavy = (currentMode === 'linework' && lineworkStyle === 'segmentation');
  const schedule = (ms) => {
    if (overlayTimer !== null) clearTimeout(overlayTimer);
    overlayTimer = setTimeout(() => {
      overlayTimer = null;
      lastOverlayDraw = (typeof performance !== 'undefined' ? performance.now() : Date.now());
      redrawOverlay();
    }, ms);
  };
  if (delayMs != null) { schedule(delayMs); return; }
  if (heavy) { schedule(70); return; }
  const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const gap = 90;                    // ~11 fps live feedback during a drag
  const since = now - lastOverlayDraw;
  if (since >= gap) {
    if (overlayTimer !== null) { clearTimeout(overlayTimer); overlayTimer = null; }
    lastOverlayDraw = now;
    redrawOverlay();
  } else {
    schedule(gap - since);
  }
}

function resizeOverlayCanvas() {
  const w = W(), h = H();
  const dpr = window.devicePixelRatio || 1;
  overlayCanvas.width = w * dpr;
  overlayCanvas.height = h * dpr;
  overlayCanvas.style.width = w + 'px';
  overlayCanvas.style.height = h + 'px';
  const ctx = overlayCanvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  markOverlayDirty(0);
}
resizeOverlayCanvas();

function applyBlendTo(material, name) {
  const b = THREE_BLENDS[name] || THREE_BLENDS.normal;
  material.blending = b.mode;
  if (b.mode === THREE.CustomBlending) {
    material.blendEquation = b.eq;
    material.blendSrc = b.src;
    material.blendDst = b.dst;
  }
  material.needsUpdate = true;
}

const sprites = [];
const meta = [];

for (const p of POINTS) {
  if (p.img) {
    const tex = loader.load('data:image/png;base64,' + p.img);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.minFilter = THREE.LinearFilter;
    tex.generateMipmaps = false;
    const mat = new THREE.SpriteMaterial({
      map: tex,
      transparent: true,
      opacity: currentOpacity,
      depthWrite: false,
    });
    applyBlendTo(mat, currentBlend);
    const sprite = new THREE.Sprite(mat);
    sprite.position.set(p.x, p.y, p.z);
    const s = baseScale * userScale;
    sprite.scale.set(s * p.ar, s, 1);
    sprite.userData = p;
    scene.add(sprite);
    sprites.push(sprite);
    meta.push(p);
  } else {
    // Fallback marker if no image
    const geo = new THREE.SphereGeometry(baseScale * 0.25, 8, 6);
    const mat = new THREE.MeshBasicMaterial({ color: 0x888888 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(p.x, p.y, p.z);
    scene.add(mesh);
  }
}

function applyUserScale() {
  for (const sp of sprites) {
    const s = baseScale * userScale;
    sp.scale.set(s * sp.userData.ar, s, 1);
  }
}

document.getElementById('opacityRange').value = currentOpacity.toFixed(2);
document.getElementById('opacityValue').textContent = currentOpacity.toFixed(2);

document.getElementById('sizeRange').addEventListener('input', (e) => {
  userScale = parseFloat(e.target.value);
  applyUserScale();
});

document.getElementById('opacityRange').addEventListener('input', (e) => {
  currentOpacity = parseFloat(e.target.value);
  document.getElementById('opacityValue').textContent = currentOpacity.toFixed(2);
  for (const sp of sprites) {
    sp.material.opacity = currentOpacity;
    sp.material.needsUpdate = true;
  }
});

document.getElementById('blendMode').addEventListener('change', (e) => {
  currentBlend = e.target.value;
  for (const sp of sprites) {
    applyBlendTo(sp.material, currentBlend);
  }
});

// Render-mode handlers ------------------------------------------------------
function setSpritesVisible(visible) {
  for (const sp of sprites) sp.visible = visible;
}

function refreshModeControls() {
  document.getElementById('heatmapControls').style.display =
    (currentMode === 'heatmap') ? 'block' : 'none';
  document.getElementById('lineworkControls').style.display =
    (currentMode === 'linework') ? 'block' : 'none';
}

document.querySelectorAll('input[name="renderMode"]').forEach(el => {
  el.addEventListener('change', (e) => {
    if (!e.target.checked) return;
    currentMode = e.target.value;
    refreshModeControls();
    // Linework no longer reads the live WebGL sprites (it uses precomputed
    // edge maps and content polygons), so hide the sprites in linework mode.
    // With 1000+ sprites this makes orbiting dramatically faster, since the
    // GPU isn't redrawing every card each frame. Heatmap still overlays the
    // visible sprites, and images mode obviously needs them.
    setSpritesVisible(currentMode !== 'linework');
    markOverlayDirty();
  });
});

document.getElementById('bandwidthRange').addEventListener('input', (e) => {
  heatmapBandwidth = parseFloat(e.target.value);
  document.getElementById('bandwidthValue').textContent = heatmapBandwidth.toFixed(0);
  markOverlayDirty();
});
document.getElementById('levelsRange').addEventListener('input', (e) => {
  heatmapLevels = parseInt(e.target.value, 10);
  document.getElementById('levelsValue').textContent = heatmapLevels;
  markOverlayDirty();
});
document.getElementById('heatmapStyle').addEventListener('change', (e) => {
  heatmapStyle = e.target.value;
  markOverlayDirty();
});
document.getElementById('lineworkRange') &&
  document.getElementById('lineworkRange').addEventListener('input', () => { markOverlayDirty(); });
document.getElementById('lineweightRange').addEventListener('input', (e) => {
  lineworkWeight = parseFloat(e.target.value);
  document.getElementById('lineweightValue').textContent = lineworkWeight.toFixed(1);
  markOverlayDirty();
});
document.getElementById('lineworkStyle').addEventListener('change', (e) => {
  lineworkStyle = e.target.value;
  markOverlayDirty();
});
document.getElementById('lineworkFadeRange').addEventListener('input', (e) => {
  lineworkFade = parseFloat(e.target.value);
  document.getElementById('lineworkFadeValue').textContent = lineworkFade.toFixed(2);
  markOverlayDirty();
});
document.getElementById('worldsComplexityRange').addEventListener('input', (e) => {
  worldsComplexity = parseFloat(e.target.value);
  document.getElementById('worldsComplexityValue').textContent = worldsComplexity.toFixed(2);
  markOverlayDirty();
});
document.getElementById('worldsUnifyRange').addEventListener('input', (e) => {
  worldsUnify = parseFloat(e.target.value);
  document.getElementById('worldsUnifyValue').textContent = worldsUnify.toFixed(2);
  markOverlayDirty();
});
document.getElementById('articulationSignalSelect').addEventListener('change', (e) => {
  ARTICULATION_SIGNAL = e.target.value;
  markOverlayDirty();
});
document.getElementById('lensRange').addEventListener('input', (e) => {
  // Dolly-zoom: change FOV and move the camera in/out to preserve the current
  // framing, so only the perspective strength changes. Wide FOV pulls the
  // camera close and exaggerates near/far scale differences; narrow FOV pulls
  // back and flattens them. This is the lens-length control for composition.
  const newFov = parseFloat(e.target.value);
  document.getElementById('lensValue').textContent = newFov + '\u00B0';
  const oldFov = camera.fov;
  const offset = camera.position.clone().sub(controls.target);
  const oldDist = offset.length();
  const newDist = oldDist * Math.tan(oldFov * Math.PI / 360) / Math.tan(newFov * Math.PI / 360);
  offset.setLength(Math.max(camera.near * 2, newDist));
  camera.position.copy(controls.target).add(offset);
  camera.fov = newFov;
  camera.updateProjectionMatrix();
  controls.update();
  markOverlayDirty();
});
document.getElementById('segSensRange').addEventListener('input', (e) => {
  segSensitivity = parseFloat(e.target.value);
  document.getElementById('segSensValue').textContent = segSensitivity.toFixed(2);
  markOverlayDirty();
});
document.getElementById('segBlurRange').addEventListener('input', (e) => {
  segBlur = parseFloat(e.target.value);
  document.getElementById('segBlurValue').textContent = segBlur.toFixed(1);
  markOverlayDirty();
});

controls.addEventListener('change', () => { markOverlayDirty(); });
window.addEventListener('resize', () => {
  resizeOverlayCanvas();
  markOverlayDirty();
});

// Hover tooltip via raycasting
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tooltip = document.getElementById('tooltip');
let hoveredId = null;
renderer.domElement.addEventListener('mousemove', (e) => {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(sprites);
  if (hits.length > 0) {
    const p = hits[0].object.userData;
    tooltip.textContent = `${p.id} · ${p.cat}/${p.sub} · ${p.era} · cluster ${p.clu}`;
    tooltip.style.left = (e.clientX + 12) + 'px';
    tooltip.style.top  = (e.clientY + 12) + 'px';
    tooltip.style.display = 'block';
    hoveredId = p.id;
  } else {
    tooltip.style.display = 'none';
    hoveredId = null;
  }
});
renderer.domElement.addEventListener('mouseleave', () => {
  tooltip.style.display = 'none';
});

// Camera-angle overlay readout
function updateOverlay() {
  const offset = camera.position.clone().sub(controls.target);
  const r = offset.length();
  const azim = Math.atan2(offset.x, offset.z) * 180 / Math.PI;
  const elev = Math.asin(offset.y / r) * 180 / Math.PI;
  document.getElementById('elev').textContent = elev.toFixed(0);
  document.getElementById('azim').textContent = azim.toFixed(0);
  window._currentElev = elev;
  window._currentAzim = azim;
}

// Copy button: clipboard first, fallback to selectable prompt
document.getElementById('copy').onclick = async () => {
  const elev = window._currentElev.toFixed(1);
  const azim = window._currentAzim.toFixed(1);
  const text = `elev=${elev}  azim=${azim}`;
  const btn = document.getElementById('copy');
  const original = btn.textContent;
  let copied = false;
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      copied = true;
    }
  } catch (e) { /* fall through */ }
  if (!copied) {
    // Fallback: execCommand on hidden textarea
    try {
      const ta = document.createElement('textarea');
      ta.value = text;
      ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta); ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
      copied = true;
    } catch (e) { /* fall through */ }
  }
  if (copied) {
    btn.textContent = 'Copied: ' + text;
  } else {
    // Last resort: prompt the user to copy manually
    window.prompt('Copy these values into the sidebar sliders:', text);
  }
  setTimeout(() => { btn.textContent = original; }, 2000);
};

// ---------------------------------------------------------------------------
// SVG export at current camera angle.
//
// Replicates the live three.js view exactly:
//   - perspective projection via the live camera's matrices
//   - NDC -> SVG canvas directly (no autofit), so the user's zoom and pan
//     transfer to the file
//   - marker pixel sizes computed from sprite world size + perspective
//     formula, matching what three.js renders on screen
// ---------------------------------------------------------------------------
// Shared projection: project all POINTS through the live camera at a given
// aspect ratio, returning the survivors plus angle/depth metadata. Used by
// the image-mode SVG export, the heatmap canvas/SVG paths, and the linework
// canvas/SVG paths.
// Project all POINTS through an arbitrary PerspectiveCamera (not necessarily
// the scene camera). Used by the stacked-perspectives composition, where
// the same data is projected through multiple synthetic cameras at
// different focal lengths and orbit positions.
function projectThroughCamera(cam, W, H) {
  cam.aspect = W / H;
  cam.updateMatrixWorld(true);
  cam.updateProjectionMatrix();
  const camForward = new THREE.Vector3();
  cam.getWorldDirection(camForward);
  const camPos = cam.position.clone();
  const projected = [];
  const tmp = new THREE.Vector3();
  for (let i = 0; i < POINTS.length; i++) {
    const p = POINTS[i];
    tmp.set(p.x, p.y, p.z);
    const distAlongForward = tmp.clone().sub(camPos).dot(camForward);
    if (distAlongForward <= cam.near) continue;
    const worldDist = tmp.distanceTo(camPos);
    tmp.project(cam);
    if (Math.abs(tmp.x) > 1 || Math.abs(tmp.y) > 1) continue;
    projected.push({
      sx: (tmp.x + 1) * 0.5 * W,
      sy: (1 - tmp.y) * 0.5 * H,
      worldDist: worldDist,
      p: p,
      idx: i,
    });
  }
  return { projected, exportCam: cam };
}

// Generate the synthetic camera set for the stacked-perspectives composition.
// Each variant uses a different focal length + distance combination (a dolly
// zoom variation, so the framing stays similar but perspective compression
// changes) and/or a small orbit offset around the target. The result is
// five distinct projections of the same data that can be layered together
// to make a composition recording the *operation* of perspective rather
// than any single viewpoint.
function generateStackedPerspectiveVariants(W, H) {
  const target = controls.target.clone();
  const offset = camera.position.clone().sub(target);
  const baseDist = offset.length();
  const baseDir = offset.clone().normalize();
  const variants = [];

  function makeCam(pos, fovDeg) {
    const c = new THREE.PerspectiveCamera(
      Math.max(8, Math.min(120, fovDeg)),
      W / H,
      Math.max(0.01, camera.near * 0.1),
      camera.far * 3
    );
    c.position.copy(pos);
    c.up.copy(camera.up);
    c.lookAt(target);
    c.updateMatrixWorld(true);
    c.updateProjectionMatrix();
    return c;
  }

  // 1. Current view — the baseline. Foreground register.
  variants.push({
    label: 'current',
    register: 'solid',
    camera: makeCam(camera.position.clone(), camera.fov),
  });

  // 2. Telephoto — long lens, camera much further back. Perspective compressed,
  //    so foreshortening flattens; cards stay close to uniform in size.
  variants.push({
    label: 'telephoto',
    register: 'halftone',
    camera: makeCam(
      target.clone().add(baseDir.clone().multiplyScalar(baseDist * 2.5)),
      camera.fov * 0.45
    ),
  });

  // 3. Wide angle — short lens, camera pulled close. Strong foreshortening,
  //    cards near camera become huge, far cards tiny.
  variants.push({
    label: 'wide',
    register: 'hatch',
    camera: makeCam(
      target.clone().add(baseDir.clone().multiplyScalar(baseDist * 0.55)),
      Math.min(camera.fov * 2.0, 95)
    ),
  });

  // 4. Orbit left by ~22° — same lens, different vantage angle.
  const offsetLeft = offset.clone().applyAxisAngle(camera.up, Math.PI * 22 / 180);
  variants.push({
    label: 'orbit_left',
    register: 'vstripe',
    camera: makeCam(target.clone().add(offsetLeft), camera.fov),
  });

  // 5. Orbit right by ~22°.
  const offsetRight = offset.clone().applyAxisAngle(camera.up, -Math.PI * 22 / 180);
  variants.push({
    label: 'orbit_right',
    register: 'outline',
    camera: makeCam(target.clone().add(offsetRight), camera.fov),
  });

  return variants;
}

// Quantize a coordinate to a pixel grid — gives the binary/digital aesthetic.
const STACKED_GRID = 4;
function snapPx(n) { return Math.round(n / STACKED_GRID) * STACKED_GRID; }

// Pre-rendered tile patterns for the stacked-perspectives composition.
// Each variant's rectangles get filled with one of these textures so the
// layered projections stay legible against each other.
function makeStackedPatternCanvas(type) {
  const c = document.createElement('canvas');
  if (type === 'hatch') { c.width = 6; c.height = 6; }
  else if (type === 'dense_hatch') { c.width = 3; c.height = 3; }
  else { c.width = 4; c.height = 4; }
  const ctx = c.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, c.width, c.height);
  ctx.fillStyle = '#000000';
  ctx.strokeStyle = '#000000';
  ctx.lineCap = 'square';
  if (type === 'halftone') {
    // 50% density: one 2x2 dot per 4x4 cell
    ctx.fillRect(1, 1, 2, 2);
  } else if (type === 'hatch') {
    // Diagonal hatching
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(-1, 7); ctx.lineTo(7, -1);
    ctx.stroke();
  } else if (type === 'dense_hatch') {
    // Tighter diagonal — for larger fills that need more weight
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(-1, 4); ctx.lineTo(4, -1);
    ctx.stroke();
  } else if (type === 'vstripe') {
    ctx.fillRect(0, 0, 2, 4);
  }
  return c;
}

const _stackedPatternCache = {};
function getStackedPattern(ctx, type) {
  const key = type + '_' + (ctx.canvas.width || 'x');
  if (!_stackedPatternCache[key]) {
    _stackedPatternCache[key] = ctx.createPattern(
      makeStackedPatternCanvas(type), 'repeat'
    );
  }
  return _stackedPatternCache[key];
}

// SVG <pattern> definitions matching the canvas patterns above. Pasted into
// <defs> when the stacked-perspectives SVG is built.
const STACKED_SVG_PATTERNS = (
  '<pattern id="sp_halftone" patternUnits="userSpaceOnUse" width="4" height="4">' +
  '<rect width="4" height="4" fill="#ffffff"/>' +
  '<rect x="1" y="1" width="2" height="2" fill="#000000"/></pattern>' +
  '<pattern id="sp_hatch" patternUnits="userSpaceOnUse" width="6" height="6">' +
  '<rect width="6" height="6" fill="#ffffff"/>' +
  '<path d="M-1,7 L7,-1" stroke="#000000" stroke-width="1"/></pattern>' +
  '<pattern id="sp_dense_hatch" patternUnits="userSpaceOnUse" width="3" height="3">' +
  '<rect width="3" height="3" fill="#ffffff"/>' +
  '<path d="M-1,4 L4,-1" stroke="#000000" stroke-width="1"/></pattern>' +
  '<pattern id="sp_vstripe" patternUnits="userSpaceOnUse" width="4" height="4">' +
  '<rect width="4" height="4" fill="#ffffff"/>' +
  '<rect x="0" y="0" width="2" height="4" fill="#000000"/></pattern>'
);

function projectVisiblePoints(W, H) {
  camera.updateMatrixWorld(true);
  const exportCam = new THREE.PerspectiveCamera(
    camera.fov, W / H, camera.near, camera.far
  );
  exportCam.position.copy(camera.position);
  exportCam.quaternion.copy(camera.quaternion);
  exportCam.updateMatrixWorld(true);
  exportCam.updateProjectionMatrix();

  const offset = camera.position.clone().sub(controls.target);
  const r = offset.length();
  const azim_deg = Math.atan2(offset.x, offset.z) * 180 / Math.PI;
  const elev_deg = Math.asin(offset.y / Math.max(r, 1e-9)) * 180 / Math.PI;

  const camForward = new THREE.Vector3();
  exportCam.getWorldDirection(camForward);
  const camPos = camera.position.clone();

  const projected = [];
  const tmp = new THREE.Vector3();
  let distMin = Infinity, distMax = -Infinity;
  let culledBehind = 0, culledOutside = 0;
  for (let i = 0; i < POINTS.length; i++) {
    const p = POINTS[i];
    tmp.set(p.x, p.y, p.z);
    const distAlongForward = tmp.clone().sub(camPos).dot(camForward);
    if (distAlongForward <= exportCam.near) { culledBehind++; continue; }
    const worldDist = tmp.distanceTo(camPos);
    tmp.project(exportCam);
    if (Math.abs(tmp.x) > 1 || Math.abs(tmp.y) > 1) { culledOutside++; continue; }
    if (worldDist < distMin) distMin = worldDist;
    if (worldDist > distMax) distMax = worldDist;
    projected.push({
      sx: (tmp.x + 1) * 0.5 * W,
      sy: (1 - tmp.y) * 0.5 * H,
      worldDist: worldDist,
      p: p,
      idx: i,
    });
  }
  return {
    projected, exportCam,
    elev_deg, azim_deg,
    distMin: (distMin === Infinity) ? 0 : distMin,
    distMax: (distMax === -Infinity) ? 0 : distMax,
    culledBehind, culledOutside,
  };
}

// Footprint rectangle for a projected point, matching the live sprite size
function projectedRect(q, W, H, tanHalfFov, worldSize) {
  const pixelHeight = (worldSize * H) / (2 * q.worldDist * tanHalfFov);
  const ar = (q.p.exp_ar != null) ? q.p.exp_ar : 1.0;
  const w = (ar >= 1) ? pixelHeight : pixelHeight * ar;
  const h = (ar >= 1) ? pixelHeight / ar : pixelHeight;
  return {
    xmin: q.sx - w / 2, xmax: q.sx + w / 2,
    ymin: q.sy - h / 2, ymax: q.sy + h / 2,
    depth: q.worldDist,
  };
}

// Project the latent-contour segments (3D) into 2D screen space using the same
// camera as the points, so "trace the space" appears in the linework / worlds
// path (canvas + SVG), not only in the live 3D scene. Returns [{x1,y1,x2,y2}].
function projectedContourSegments(exportCam, W, H) {
  if (!showContours) return [];
  const pos = buildContourGeometry();          // 6 floats per segment (two 3D points)
  const camPos = exportCam.position.clone();
  const camForward = new THREE.Vector3();
  exportCam.getWorldDirection(camForward);
  const a = new THREE.Vector3(), b = new THREE.Vector3();
  const out = [];
  for (let i = 0; i + 5 < pos.length; i += 6) {
    a.set(pos[i], pos[i+1], pos[i+2]);
    b.set(pos[i+3], pos[i+4], pos[i+5]);
    const da = a.clone().sub(camPos).dot(camForward);
    const db = b.clone().sub(camPos).dot(camForward);
    if (da <= exportCam.near || db <= exportCam.near) continue;   // skip near-plane crossers
    a.project(exportCam); b.project(exportCam);
    if ((Math.abs(a.x) > 1.3 && Math.abs(b.x) > 1.3) ||
        (Math.abs(a.y) > 1.3 && Math.abs(b.y) > 1.3)) continue;   // both well off-frame
    out.push({
      x1: (a.x + 1) * 0.5 * W, y1: (1 - a.y) * 0.5 * H,
      x2: (b.x + 1) * 0.5 * W, y2: (1 - b.y) * 0.5 * H,
    });
  }
  return out;
}

function buildSVGFromCurrentView() {
  const W = SETTINGS.width, H = SETTINGS.height;
  const pv = projectVisiblePoints(W, H);
  const { projected, exportCam, elev_deg, azim_deg, distMin, distMax,
          culledBehind, culledOutside } = pv;

  if (projected.length === 0) {
    throw new Error('no points visible — try zooming out in the 3D viewer');
  }

  // Back-to-front for painter's algorithm
  const order = projected.slice().sort((a, b) => b.worldDist - a.worldDist);

  // ---- Build layer map ----
  const layers = new Map();
  const layerOrder = [];
  function addLayer(name) {
    if (!layers.has(name)) { layers.set(name, []); layerOrder.push(name); }
  }
  addLayer("background");
  layers.get("background").push(
    `<rect x="0" y="0" width="${W}" height="${H}" fill="${SETTINGS.background}"/>`
  );

  // Edges between points that survived culling
  if (SETTINGS.show_edges && EDGES && EDGES.length > 0) {
    addLayer("edges");
    const op = SETTINGS.edges_opacity < 1
      ? ` stroke-opacity="${SETTINGS.edges_opacity.toFixed(2)}"` : '';
    const byIdx = new Map();
    for (const q of projected) byIdx.set(q.idx, q);
    for (const e of EDGES) {
      const a = byIdx.get(e[0]), b = byIdx.get(e[1]);
      if (!a || !b) continue;
      layers.get("edges").push(
        `<line x1="${a.sx.toFixed(2)}" y1="${a.sy.toFixed(2)}" ` +
        `x2="${b.sx.toFixed(2)}" y2="${b.sy.toFixed(2)}" ` +
        `stroke="${SETTINGS.edges_color}" ` +
        `stroke-width="${SETTINGS.edges_width.toFixed(2)}"${op}/>`
      );
    }
  }

  // Markers — same formula three.js uses for sprite screen size.
  // At distance d, a sprite of world-size s projects to pixel height:
  //   s * H / (2 * d * tan(fov/2))
  // This makes the SVG marker sizes match what the live viewer shows.
  const worldSize = baseScale * userScale;
  const tanHalfFov = Math.tan((exportCam.fov * Math.PI / 180) / 2);
  const opacity = currentOpacity;
  const opAttr = opacity < 1 ? ` fill-opacity="${opacity.toFixed(2)}"` : '';
  const strokeAttr = (SETTINGS.stroke_width > 0)
    ? ` stroke="${SETTINGS.stroke_color}" stroke-width="${SETTINGS.stroke_width.toFixed(2)}"`
    : '';

  for (const q of order) {
    const p = q.p;
    addLayer(p.layer);
    const pixelHeight = (worldSize * H) / (2 * q.worldDist * tanHalfFov);
    // Clamp so a point that gets very close to the camera doesn't blow
    // up the whole canvas
    const r = Math.max(0.5, Math.min(H * 0.5, pixelHeight / 2));
    layers.get(p.layer).push(markerSVG(q.sx, q.sy, r, p, opAttr, strokeAttr, opacity));
  }

  // Concave-hull boundary around the projected (and culled) points
  if (SETTINGS.show_curve) {
    const curveCtrls = fitConcaveHull(
      projected,
      Math.max(1.5, +SETTINGS.curve_concavity || 2.5),
      Math.max(0, SETTINGS.curve_smoothing | 0),
      120
    );
    if (curveCtrls) {
      const pathD = closedCurveToSvgPath(curveCtrls);
      let dashAttr = '';
      if (SETTINGS.curve_dashed) {
        dashAttr = ` stroke-dasharray="${SETTINGS.curve_dash_on.toFixed(1)} ` +
                   `${SETTINGS.curve_dash_off.toFixed(1)}"`;
      }
      addLayer("boundary_curve");
      layers.get("boundary_curve").push(
        `<path d="${pathD}" fill="none" ` +
        `stroke="${SETTINGS.curve_color}" ` +
        `stroke-width="${SETTINGS.curve_width.toFixed(2)}" ` +
        `stroke-linecap="round" stroke-linejoin="round"${dashAttr}/>`
      );
    }
  }

  // Diagnostic stamp (its own layer; hide in Illustrator if unwanted)
  addLayer("view_info");
  const infoFontSize = Math.max(10, Math.min(W, H) * 0.012);
  const distSpan = distMax - distMin;
  layers.get("view_info").push(
    `<text x="${SETTINGS.padding}" y="${H - SETTINGS.padding * 0.4}" ` +
    `font-family="ui-monospace, monospace" font-size="${infoFontSize.toFixed(1)}" ` +
    `fill="#666" opacity="0.7">` +
    `elev ${elev_deg.toFixed(0)}° · azim ${azim_deg.toFixed(0)}° · ` +
    `persp · ${currentBlend} · ${projected.length}/${POINTS.length} pts ` +
    `(${culledBehind} behind, ${culledOutside} off-frame) · ` +
    `depth ${distSpan.toFixed(2)} · sprite ${worldSize.toFixed(2)}` +
    `</text>`
  );

  const svgString = assembleSVG(W, H, layerOrder, layers, SVG_BLENDS[currentBlend] || 'normal');
  return {
    svg: svgString,
    elev: elev_deg,
    azim: azim_deg,
    kept: projected.length,
    total: POINTS.length,
    distSpan: distSpan,
    culledBehind: culledBehind,
    culledOutside: culledOutside,
  };
}

// ---------------------------------------------------------------------------
// Heatmap SVG export. Computes a KDE over the projected screen positions,
// then for each density level emits an isocontour as its own SVG layer.
// ---------------------------------------------------------------------------
function buildHeatmapSVG() {
  const W = SETTINGS.width, H = SETTINGS.height;
  const pv = projectVisiblePoints(W, H);
  if (pv.projected.length === 0) {
    throw new Error('no points visible — try zooming out in the 3D viewer');
  }
  // Grid resolution: keep it proportional to canvas size, capped for speed
  const gridW = Math.min(220, Math.max(60, Math.round(W / 8)));
  const gridH = Math.min(220, Math.max(60, Math.round(H / 8)));
  const cellW = W / gridW, cellH = H / gridH;
  const grid = computeDensityGrid(pv.projected, W, H, gridW, gridH, heatmapBandwidth);

  let maxV = 0;
  for (let i = 0; i < grid.length; i++) if (grid[i] > maxV) maxV = grid[i];
  if (maxV <= 0) maxV = 1;

  // Levels: skip 0 and the max to avoid degenerate contours
  const levels = [];
  for (let k = 1; k <= heatmapLevels; k++) {
    levels.push((k / (heatmapLevels + 1)) * maxV);
  }

  const layers = new Map();
  const layerOrder = [];
  function addLayer(name) {
    if (!layers.has(name)) { layers.set(name, []); layerOrder.push(name); }
  }
  addLayer("background");
  layers.get("background").push(
    `<rect x="0" y="0" width="${W}" height="${H}" fill="${SETTINGS.background}"/>`
  );

  // One layer per density level, lower levels drawn first
  for (let li = 0; li < levels.length; li++) {
    const lvl = levels[li];
    const segs = marchingSquaresAt(grid, gridW, gridH, lvl, cellW, cellH);
    const paths = stitchSegmentsToPaths(segs, Math.min(cellW, cellH) * 0.6);
    if (paths.length === 0) continue;
    const layerName = `iso_${(li + 1).toString().padStart(2, '0')}`;
    addLayer(layerName);
    // Higher levels (denser) get more opacity / a darker stroke
    const t = (li + 1) / levels.length;
    const fillAlpha = (0.06 + 0.10 * t).toFixed(3);
    const strokeAlpha = (0.30 + 0.55 * t).toFixed(3);
    const wantFill = (heatmapStyle === 'filled' || heatmapStyle === 'both');
    const wantStroke = (heatmapStyle === 'lines'  || heatmapStyle === 'both');
    for (const pts of paths) {
      const closed = (Math.hypot(pts[0][0] - pts[pts.length - 1][0],
                                  pts[0][1] - pts[pts.length - 1][1]) < cellW);
      const d = pathToSvgD(pts, closed);
      const fillAttr   = wantFill   ? `fill="#ffffff" fill-opacity="${fillAlpha}"` : `fill="none"`;
      const strokeAttr = wantStroke ? `stroke="#ffffff" stroke-opacity="${strokeAlpha}" stroke-width="1.0"` : ``;
      layers.get(layerName).push(`<path d="${d}" ${fillAttr} ${strokeAttr}/>`);
    }
  }

  // Diagnostic stamp
  addLayer("view_info");
  const fontSize = Math.max(10, Math.min(W, H) * 0.012);
  layers.get("view_info").push(
    `<text x="${SETTINGS.padding}" y="${H - SETTINGS.padding * 0.4}" ` +
    `font-family="ui-monospace, monospace" font-size="${fontSize.toFixed(1)}" ` +
    `fill="#666" opacity="0.7">` +
    `heatmap · elev ${pv.elev_deg.toFixed(0)}° azim ${pv.azim_deg.toFixed(0)}° · ` +
    `${pv.projected.length}/${POINTS.length} pts · bandwidth ${heatmapBandwidth}px · ` +
    `${heatmapLevels} levels · ${heatmapStyle}` +
    `</text>`
  );

  const svgString = assembleSVG(W, H, layerOrder, layers, 'normal');
  return {
    svg: svgString,
    elev: pv.elev_deg, azim: pv.azim_deg,
    kept: pv.projected.length, total: POINTS.length,
    distSpan: pv.distMax - pv.distMin,
    culledBehind: pv.culledBehind, culledOutside: pv.culledOutside,
  };
}

// ---------------------------------------------------------------------------
// Linework SVG export. Projects each image to its screen-space footprint
// rectangle, sorts back-to-front, then for every rectangle's 4 edges removes
// any portion behind a rectangle in front. The visible segments are the
// architectural drawing of the dataset as stacked, occluding cards.
// ---------------------------------------------------------------------------
function buildLineworkSVG() {
  const W = SETTINGS.width, H = SETTINGS.height;
  const pv = projectVisiblePoints(W, H);
  if (pv.projected.length === 0) {
    throw new Error('no points visible — try zooming out in the 3D viewer');
  }
  const worldSize = baseScale * userScale;
  const tanHalfFov = Math.tan((pv.exportCam.fov * Math.PI / 180) / 2);
  const rects = [];
  for (const q of pv.projected) rects.push(projectedRect(q, W, H, tanHalfFov, worldSize));

  const layers = new Map();
  const layerOrder = [];
  function addLayer(name) {
    if (!layers.has(name)) { layers.set(name, []); layerOrder.push(name); }
  }
  // White paper background (this is a linework export, meant for print)
  addLayer("background");
  layers.get("background").push(
    `<rect x="0" y="0" width="${W}" height="${H}" fill="#ffffff"/>`
  );

  // Latent-field contours as their own bottom layer, faint, so the export
  // carries the same sense of the space the worlds are drawn from.
  if (showContours) {
    const csegs = projectedContourSegments(pv.exportCam, W, H);
    if (csegs.length) {
      addLayer("latent_field");
      let cd = '';
      for (const s of csegs) cd += `M${s.x1.toFixed(1)},${s.y1.toFixed(1)}L${s.x2.toFixed(1)},${s.y2.toFixed(1)}`;
      layers.get("latent_field").push(
        `<path d="${cd}" fill="none" stroke="rgb(64,78,100)" stroke-width="0.6" stroke-opacity="${contourOpacity.toFixed(3)}"/>`
      );
    }
  }

  let dMin = Infinity, dMax = -Infinity;
  for (const r of rects) {
    if (r.depth < dMin) dMin = r.depth;
    if (r.depth > dMax) dMax = r.depth;
  }
  const dSpan = Math.max(dMax - dMin, 1e-6);
  function alphaForDepth(d) {
    const t = (d - dMin) / dSpan;
    return 1.0 - lineworkFade * t;
  }

  let visibleCount = 0, hiddenCount = 0;
  let defsXml = '';

  if (lineworkStyle === 'segmentation') {
    // Embed each photograph's edge map (black-on-transparent PNG) at its
    // projected position. Robust per-image content; composites over the
    // white background as black linework. (For pure-vector output use the
    // stacked_perspectives or composition styles, which trace polygons.)
    if (edgeMaps.size > 0) {
      addLayer("linework_segmentation");
      const worldSize = baseScale * userScale;
      const tanHalfFov = Math.tan((pv.exportCam.fov * Math.PI / 180) / 2);
      const sorted = pv.projected.slice().sort((a, b) => b.worldDist - a.worldDist);
      for (const q of sorted) {
        const edge = edgeMaps.get(q.p.id);
        if (!edge) continue;
        const r = projectedRect(q, W, H, tanHalfFov, worldSize);
        const rw = r.xmax - r.xmin, rh = r.ymax - r.ymin;
        if (rw < 2 || rh < 2) continue;
        let href;
        try { href = edge.toDataURL('image/png'); } catch (e) { continue; }
        layers.get("linework_segmentation").push(
          `<image x="${r.xmin.toFixed(2)}" y="${r.ymin.toFixed(2)}" ` +
          `width="${rw.toFixed(2)}" height="${rh.toFixed(2)}" ` +
          `preserveAspectRatio="none" xlink:href="${href}"/>`
        );
        visibleCount++;
      }
    }
  } else if (lineworkStyle === 'collapsed_field') {
    // Worlds: bounded latent clusters with content inside, projected through
    // the live camera. Two layers: world_content (quiet) and world_boundaries.
    if (contentPolygons.size > 0) {
      const { forms, links, bridges, unions, depthRange } = computeWorlds(W, H);
      const minDim = Math.min(W, H);
      defsXml += STACKED_SVG_PATTERNS;
      addLayer("world_links");
      const linkWt = Math.max(0.5, lineworkWeight * 0.9).toFixed(2);
      for (const link of links) {
        if (!link || link.length < 2) continue;
        const parts = [`M ${link[0][0].toFixed(2)},${link[0][1].toFixed(2)}`];
        for (let i = 1; i < link.length; i++) parts.push(`L ${link[i][0].toFixed(2)},${link[i][1].toFixed(2)}`);
        layers.get("world_links").push(
          `<path d="${parts.join(' ')}" fill="none" stroke="#000000" stroke-width="${linkWt}" ` +
          `stroke-opacity="0.3" stroke-linecap="round" stroke-linejoin="round"/>`
        );
      }
      const fillOf = { solid: '#ffffff', halftone: 'url(#sp_halftone)', hatch: 'url(#sp_hatch)', vstripe: 'url(#sp_vstripe)' };
      if (worldsUnify > 0.001 && unions.length) {
        // Each world becomes its own Inkscape layer, emitted far → near, so its
        // fill can be edited in one place: the figure paths sit inside a nested
        // <g fill="..."> and inherit it, while their depth-weighted strokes stay
        // on the paths. Near worlds read heavy and black, far worlds light and
        // grey (architectural line hierarchy).
        const byWorld = new Map();
        for (const u of unions) { if (!byWorld.has(u.lab)) byWorld.set(u.lab, []); byWorld.get(u.lab).push(u); }
        const medoidByWorld = new Map();
        for (const f of forms) { if (f.isMedoid) medoidByWorld.set(f.lab, f); }
        const worldList = [];
        for (const [lab, figs] of byWorld) {
          let ds = 0; for (const g of figs) ds += g.depth;
          worldList.push({ lab, figs, depth: ds / figs.length });
        }
        worldList.sort((a, b) => b.depth - a.depth);   // far first (painter's algorithm)
        const pad2 = (n) => (n < 10 ? '0' + n : '' + n);
        for (const wd of worldList) {
          const pat = WORLD_PATTERNS[wd.lab % WORLD_PATTERNS.length];
          const layerName = `world_${pad2(wd.lab)}_${pat}`;
          addLayer(layerName);
          const layer = layers.get(layerName);
          const figPaths = [];
          for (const g of wd.figs) {
            const segs = [];
            for (const c of g.contours) {
              if (!c || c.length < 3) continue;
              const parts = [`M ${c[0][0].toFixed(2)},${c[0][1].toFixed(2)}`];
              for (let i = 1; i < c.length; i++) parts.push(`L ${c[i][0].toFixed(2)},${c[i][1].toFixed(2)}`);
              parts.push('Z'); segs.push(parts.join(' '));
            }
            if (!segs.length) continue;
            const dw = worldsDepthWeight(g.depth, depthRange, lineworkWeight);
            figPaths.push(
              `<path d="${segs.join(' ')}" fill-rule="evenodd" ` +
              `stroke="${dw.stroke}" stroke-width="${dw.w.toFixed(2)}" stroke-linejoin="round"/>`
            );
            visibleCount++;
          }
          if (figPaths.length) {
            layer.push(`<g fill="${fillOf[pat] || '#ffffff'}">\n${figPaths.join('\n')}\n</g>`);
          }
          // medoid character lines for this world, depth-weighted and lighter
          const md = medoidByWorld.get(wd.lab);
          if (md && md.inner && md.inner.length) {
            const dw = worldsDepthWeight(md.depth, depthRange, lineworkWeight);
            const iwt = Math.max(0.18, dw.w * 0.45).toFixed(2);
            for (const c of md.inner) {
              if (!c || c.length < 2) continue;
              const ip = [`M ${(md.cx + (c[0][0]-0.5)*md.size).toFixed(2)},${(md.cy + (c[0][1]-0.5)*md.size).toFixed(2)}`];
              for (let i = 1; i < c.length; i++) ip.push(`L ${(md.cx + (c[i][0]-0.5)*md.size).toFixed(2)},${(md.cy + (c[i][1]-0.5)*md.size).toFixed(2)}`);
              layer.push(`<path d="${ip.join(' ')}" fill="none" stroke="${dw.stroke}" stroke-width="${iwt}" stroke-linecap="round" stroke-linejoin="round"/>`);
            }
          }
        }
        // Gestural guides: the machine-flagged painterly arcs, isolated as open
        // strokes in their own layer so they can be followed by hand or hidden.
        const gestPaths = [];
        for (const u of unions) {
          if (!u.gestures) continue;
          for (const g of u.gestures) {
            if (!g || g.length < 2) continue;
            const gp = [`M ${g[0][0].toFixed(2)},${g[0][1].toFixed(2)}`];
            for (let i = 1; i < g.length; i++) gp.push(`L ${g[i][0].toFixed(2)},${g[i][1].toFixed(2)}`);
            gestPaths.push(`<path d="${gp.join(' ')}" fill="none" stroke="#000000" stroke-width="${Math.max(1.0, lineworkWeight * 2.0).toFixed(2)}" stroke-opacity="0.85" stroke-linecap="round" stroke-linejoin="round"/>`);
          }
        }
        if (gestPaths.length) {
          addLayer("gestural_guides");
          for (const p of gestPaths) layers.get("gestural_guides").push(p);
        }
      } else {
      addLayer("world_field");
      const items = [];
      for (const f of forms) items.push({ t: 'f', depth: f.depth, f });
      for (const br of bridges) items.push({ t: 'b', depth: br.depth, br });
      items.sort((a, b) => b.depth - a.depth);
      for (const it of items) {
        if (it.t === 'b') {
          const br = it.br;
          const dx = br.bx - br.ax, dy = br.by - br.ay; const L = Math.hypot(dx, dy) || 1;
          const px = -dy / L * br.w / 2, py = dx / L * br.w / 2;
          const d = `M ${(br.ax+px).toFixed(2)},${(br.ay+py).toFixed(2)} L ${(br.bx+px).toFixed(2)},${(br.by+py).toFixed(2)} ` +
                    `L ${(br.bx-px).toFixed(2)},${(br.by-py).toFixed(2)} L ${(br.ax-px).toFixed(2)},${(br.ay-py).toFixed(2)} Z`;
          const dw = worldsDepthWeight(br.depth, depthRange, lineworkWeight);
          layers.get("world_field").push(
            `<path d="${d}" fill="${fillOf[br.pattern] || '#ffffff'}" stroke="${dw.stroke}" stroke-width="${(dw.w*0.7).toFixed(2)}" stroke-linejoin="round"/>`
          );
          continue;
        }
        const f = it.f, S = f.size, cx = f.cx, cy = f.cy;
        if (!f.dom || f.dom.length < 3) continue;
        const parts = [`M ${(cx + (f.dom[0][0]-0.5)*S).toFixed(2)},${(cy + (f.dom[0][1]-0.5)*S).toFixed(2)}`];
        for (let i = 1; i < f.dom.length; i++) parts.push(`L ${(cx + (f.dom[i][0]-0.5)*S).toFixed(2)},${(cy + (f.dom[i][1]-0.5)*S).toFixed(2)}`);
        const dw = worldsDepthWeight(f.depth, depthRange, lineworkWeight);
        const ow = Math.max(0.18, dw.w * (f.isMedoid ? 1.0 : 0.55)).toFixed(2);
        layers.get("world_field").push(
          `<path d="${parts.join(' ')} Z" fill="${fillOf[f.pattern] || '#ffffff'}" ` +
          `stroke="${dw.stroke}" stroke-width="${ow}" stroke-linejoin="round"/>`
        );
        if (f.isMedoid && f.inner && f.inner.length) {
          const iw = Math.max(0.18, dw.w * 0.45).toFixed(2);
          for (const c of f.inner) {
            if (!c || c.length < 2) continue;
            const ip = [`M ${(cx + (c[0][0]-0.5)*S).toFixed(2)},${(cy + (c[0][1]-0.5)*S).toFixed(2)}`];
            for (let i = 1; i < c.length; i++) ip.push(`L ${(cx + (c[i][0]-0.5)*S).toFixed(2)},${(cy + (c[i][1]-0.5)*S).toFixed(2)}`);
            layers.get("world_field").push(
              `<path d="${ip.join(' ')}" fill="none" stroke="${dw.stroke}" stroke-width="${iw}" stroke-linecap="round" stroke-linejoin="round"/>`
            );
          }
        }
        visibleCount++;
      }
      }
    }
  } else if (lineworkStyle === 'stacked_perspectives') {
    // Project per-image content polygons (extracted from each photograph's
    // segmentation field at load time) through 5 synthetic cameras, then
    // render each variant's polygons in a different visual register. The
    // primitives here are SHAPES traced from the image content — not
    // rectangles — so each layer reads as the photographic substance
    // re-rendered under a different lens. Coordinates snap to a 4px grid
    // for the binary/digital aesthetic.
    defsXml += STACKED_SVG_PATTERNS;
    const variants = generateStackedPerspectiveVariants(W, H);
    const haveContent = contentPolygons.size > 0;

    for (const v of variants) {
      const layerName = `perspective_${v.label}`;
      addLayer(layerName);
      const pv = projectThroughCamera(v.camera, W, H);
      const tanHalfFov = Math.tan((v.camera.fov * Math.PI / 180) / 2);
      const projSorted = pv.projected.slice().sort((a, b) => b.worldDist - a.worldDist);
      let attrs;
      switch (v.register) {
        case 'solid':
          attrs = 'fill="#000000" stroke="none" fill-rule="evenodd"'; break;
        case 'halftone':
          attrs = 'fill="url(#sp_halftone)" stroke="#000000" stroke-width="0.8" fill-rule="evenodd"'; break;
        case 'hatch':
          attrs = 'fill="url(#sp_hatch)" stroke="#000000" stroke-width="0.8" fill-rule="evenodd"'; break;
        case 'vstripe':
          attrs = 'fill="url(#sp_vstripe)" stroke="#000000" stroke-width="0.8" fill-rule="evenodd"'; break;
        case 'outline':
          attrs = 'fill="none" stroke="#000000" stroke-width="1.4"'; break;
        default:
          attrs = 'fill="none" stroke="#000000" stroke-width="1"';
      }
      for (const q of projSorted) {
        const polys = haveContent ? contentPolygons.get(q.p.id) : null;
        if (!polys || polys.length === 0) continue;
        // Compute the sprite's projected footprint in this view — we use it
        // only to position and scale the content polygons, never to draw
        // its rectangle.
        const r = projectedRect(q, W, H, tanHalfFov, worldSize);
        const rw = r.xmax - r.xmin, rh = r.ymax - r.ymin;
        if (rw < STACKED_GRID * 2 || rh < STACKED_GRID * 2) continue;
        for (const poly of polys) {
          if (poly.length < 3) continue;
          const ptsScreen = poly.map(p => [
            snapPx(r.xmin + p[0] * rw),
            snapPx(r.ymin + p[1] * rh)
          ]);
          // Build path with closure
          const d = 'M ' + ptsScreen.map(p => `${p[0]},${p[1]}`).join(' L ') + ' Z';
          layers.get(layerName).push(`<path d="${d}" ${attrs}/>`);
          visibleCount++;
        }
      }
    }
  } else if (lineworkStyle === 'composition') {
    // Mirrors the swatch-study composition logic but with the geometry
    // derived from the projection rather than a regular grid.
    addLayer("protocol_swatches");
    const sortedC = rects.slice().sort((a, b) => b.depth - a.depth);
    for (const r of sortedC) {
      const rw = (r.xmax - r.xmin).toFixed(2);
      const rh = (r.ymax - r.ymin).toFixed(2);
      layers.get("protocol_swatches").push(
        `<rect x="${r.xmin.toFixed(2)}" y="${r.ymin.toFixed(2)}" ` +
        `width="${rw}" height="${rh}" fill="#ffffff" ` +
        `stroke="#000000" stroke-width="${(lineworkWeight * 0.7).toFixed(2)}"/>`
      );
      visibleCount++;
    }
    const segC = computeSegmentationPaths(900, W, H);
    if (segC && segC.paths.length > 0) {
      addLayer("amalgamation_lines");
      const sxc = W / segC.width, syc = H / segC.height;
      for (const pts of segC.paths) {
        if (pts.length < 2) continue;
        const parts = [`M ${(pts[0][0] * sxc).toFixed(2)},${(pts[0][1] * syc).toFixed(2)}`];
        for (let i = 1; i < pts.length; i++) {
          parts.push(`L ${(pts[i][0] * sxc).toFixed(2)},${(pts[i][1] * syc).toFixed(2)}`);
        }
        layers.get("amalgamation_lines").push(
          `<path d="${parts.join(' ')}" fill="none" stroke="#000000" ` +
          `stroke-width="${(lineworkWeight * 1.2).toFixed(2)}" ` +
          `stroke-linecap="round" stroke-linejoin="round"/>`
        );
      }
    }
  } else if (lineworkStyle === 'all') {
    const sorted = rects.slice().sort((a, b) => b.depth - a.depth);
    for (const r of sorted) {
      const a = alphaForDepth(r.depth);
      const opAttr = a < 0.999 ? ` stroke-opacity="${a.toFixed(3)}"` : '';
      for (const e of rectEdges(r)) {
        layers.get("linework_all").push(
          `<line x1="${e.x1.toFixed(2)}" y1="${e.y1.toFixed(2)}" ` +
          `x2="${e.x2.toFixed(2)}" y2="${e.y2.toFixed(2)}" ` +
          `stroke="#000000" stroke-width="${lineworkWeight.toFixed(2)}" ` +
          `stroke-linecap="square"${opAttr}/>`
        );
        visibleCount++;
      }
    }
  } else if (lineworkStyle === 'ghosted') {
    addLayer("linework_hidden");
    addLayer("linework_visible");
    const sorted = rects.slice().sort((a, b) => b.depth - a.depth);
    const dashOn = 5.0, dashOff = 3.0;
    for (let i = 0; i < sorted.length; i++) {
      const r = sorted[i];
      const occluders = sorted.slice(i + 1);
      const a = alphaForDepth(r.depth);
      for (const e of rectEdges(r)) {
        const { visible, hidden } = visibleHiddenIntervals(e, occluders);
        const vAttr = a < 0.999 ? ` stroke-opacity="${a.toFixed(3)}"` : '';
        for (const [t0, t1] of visible) {
          const s = intervalToSegment(e, t0, t1);
          if (Math.hypot(s.x2 - s.x1, s.y2 - s.y1) < 0.5) continue;
          layers.get("linework_visible").push(
            `<line x1="${s.x1.toFixed(2)}" y1="${s.y1.toFixed(2)}" ` +
            `x2="${s.x2.toFixed(2)}" y2="${s.y2.toFixed(2)}" ` +
            `stroke="#000000" stroke-width="${lineworkWeight.toFixed(2)}" ` +
            `stroke-linecap="square"${vAttr}/>`
          );
          visibleCount++;
        }
        const hAlpha = (a * 0.6).toFixed(3);
        const hWeight = Math.max(0.3, lineworkWeight * 0.7).toFixed(2);
        for (const [t0, t1] of hidden) {
          const s = intervalToSegment(e, t0, t1);
          if (Math.hypot(s.x2 - s.x1, s.y2 - s.y1) < 0.5) continue;
          layers.get("linework_hidden").push(
            `<line x1="${s.x1.toFixed(2)}" y1="${s.y1.toFixed(2)}" ` +
            `x2="${s.x2.toFixed(2)}" y2="${s.y2.toFixed(2)}" ` +
            `stroke="#8c8c8c" stroke-width="${hWeight}" ` +
            `stroke-opacity="${hAlpha}" ` +
            `stroke-dasharray="${dashOn} ${dashOff}"/>`
          );
          hiddenCount++;
        }
      }
    }
  } else {
    addLayer("linework");
    const survivors = hiddenLineRemoval(rects);
    for (const s of survivors) {
      if (Math.hypot(s.x2 - s.x1, s.y2 - s.y1) < 0.5) continue;
      layers.get("linework").push(
        `<line x1="${s.x1.toFixed(2)}" y1="${s.y1.toFixed(2)}" ` +
        `x2="${s.x2.toFixed(2)}" y2="${s.y2.toFixed(2)}" ` +
        `stroke="#000000" stroke-width="${lineworkWeight.toFixed(2)}" ` +
        `stroke-linecap="square"/>`
      );
      visibleCount++;
    }
  }

  addLayer("view_info");
  const fontSize = Math.max(10, Math.min(W, H) * 0.012);
  layers.get("view_info").push(
    `<text x="${SETTINGS.padding}" y="${H - SETTINGS.padding * 0.4}" ` +
    `font-family="ui-monospace, monospace" font-size="${fontSize.toFixed(1)}" ` +
    `fill="#777" opacity="0.7">` +
    `linework · ${lineworkStyle} · elev ${pv.elev_deg.toFixed(0)}° azim ${pv.azim_deg.toFixed(0)}° · ` +
    `${rects.length} cards · ${visibleCount} visible` +
    (hiddenCount > 0 ? ` · ${hiddenCount} hidden` : '') +
    `</text>`
  );

  const svgString = assembleSVG(W, H, layerOrder, layers, 'normal', defsXml);
  return {
    svg: svgString,
    elev: pv.elev_deg, azim: pv.azim_deg,
    kept: pv.projected.length, total: POINTS.length,
    distSpan: pv.distMax - pv.distMin,
    culledBehind: pv.culledBehind, culledOutside: pv.culledOutside,
  };
}

// ---------------------------------------------------------------------------
// Canvas overlay rendering for live preview of heatmap and linework modes.
// Mirrors the SVG builders above but draws to 2D canvas for speed.
// ---------------------------------------------------------------------------
function renderHeatmapToCanvas(ctx, projected, W, H) {
  if (projected.length === 0) return;
  const gridW = Math.min(180, Math.max(40, Math.round(W / 10)));
  const gridH = Math.min(180, Math.max(40, Math.round(H / 10)));
  const cellW = W / gridW, cellH = H / gridH;
  const grid = computeDensityGrid(projected, W, H, gridW, gridH, heatmapBandwidth);
  let maxV = 0;
  for (let i = 0; i < grid.length; i++) if (grid[i] > maxV) maxV = grid[i];
  if (maxV <= 0) return;

  const levels = [];
  for (let k = 1; k <= heatmapLevels; k++) {
    levels.push((k / (heatmapLevels + 1)) * maxV);
  }

  ctx.save();
  ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  const wantFill   = (heatmapStyle === 'filled' || heatmapStyle === 'both');
  const wantStroke = (heatmapStyle === 'lines'  || heatmapStyle === 'both');

  for (let li = 0; li < levels.length; li++) {
    const lvl = levels[li];
    const segs = marchingSquaresAt(grid, gridW, gridH, lvl, cellW, cellH);
    const paths = stitchSegmentsToPaths(segs, Math.min(cellW, cellH) * 0.6);
    if (paths.length === 0) continue;
    const t = (li + 1) / levels.length;

    if (wantFill) {
      ctx.beginPath();
      for (const pts of paths) {
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
        ctx.closePath();
      }
      ctx.fillStyle = `rgba(255, 240, 200, ${(0.05 + 0.10 * t).toFixed(3)})`;
      ctx.fill('evenodd');
    }
    if (wantStroke) {
      ctx.beginPath();
      for (const pts of paths) {
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      }
      ctx.strokeStyle = `rgba(255, 240, 200, ${(0.30 + 0.55 * t).toFixed(3)})`;
      ctx.lineWidth = 1.0;
      ctx.stroke();
    }
  }
  ctx.restore();
}

function renderLineworkToCanvas(ctx, projected, exportCam, W, H) {
  if (projected.length === 0) return;
  const worldSize = baseScale * userScale;
  const tanHalfFov = Math.tan((exportCam.fov * Math.PI / 180) / 2);
  const rects = [];
  for (const q of projected) rects.push(projectedRect(q, W, H, tanHalfFov, worldSize));

  // White paper background
  ctx.save();
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);
  ctx.lineCap = 'square';

  // Latent-field contours, faint, behind everything (same "trace the space"
  // signal as the 3D scene, projected into the linework view).
  if (showContours) {
    const csegs = projectedContourSegments(exportCam, W, H);
    if (csegs.length) {
      ctx.save();
      ctx.strokeStyle = 'rgba(64,78,100,' + contourOpacity + ')';
      ctx.lineWidth = 0.7;
      ctx.beginPath();
      for (const s of csegs) { ctx.moveTo(s.x1, s.y1); ctx.lineTo(s.x2, s.y2); }
      ctx.stroke();
      ctx.restore();
    }
  }

  if (rects.length === 0) { ctx.restore(); return; }
  let dMin = Infinity, dMax = -Infinity;
  for (const r of rects) {
    if (r.depth < dMin) dMin = r.depth;
    if (r.depth > dMax) dMax = r.depth;
  }
  const dSpan = Math.max(dMax - dMin, 1e-6);
  // For each rect, compute a depth-attenuation factor 0..1 (1 = closest)
  function alphaForDepth(d) {
    const t = (d - dMin) / dSpan;  // 0 closest, 1 farthest
    return 1.0 - lineworkFade * t;
  }

  if (lineworkStyle === 'segmentation') {
    // Draw each photograph's edge map (black-on-transparent linework) at its
    // projected screen position. This is the per-image content — rooflines,
    // windows, building silhouettes — stacked at UMAP-determined positions.
    // Using the cached raster directly is robust: it always shows content as
    // soon as the edge maps load, with no dependency on polygon extraction.
    if (edgeMaps.size === 0) {
      ctx.fillStyle = '#888';
      ctx.font = '12px ui-monospace, monospace';
      ctx.fillText(
        edgeMapsTotal === 0
          ? 'preparing edge maps...'
          : `precomputing edge maps... ${edgeMapsReady}/${edgeMapsTotal}`,
        20, 30);
    } else {
      const pv = projectVisiblePoints(W, H);
      const worldSize = baseScale * userScale;
      const tanHalfFov = Math.tan((pv.exportCam.fov * Math.PI / 180) / 2);
      const sorted = pv.projected.slice().sort((a, b) => b.worldDist - a.worldDist);
      ctx.imageSmoothingEnabled = true;
      // Depth fade: far images fainter when lineworkFade > 0.
      let dMin = Infinity, dMax = -Infinity;
      for (const q of pv.projected) {
        if (q.worldDist < dMin) dMin = q.worldDist;
        if (q.worldDist > dMax) dMax = q.worldDist;
      }
      const dSpan = Math.max(dMax - dMin, 1e-6);
      let drawn = 0;
      for (const q of sorted) {
        const edge = edgeMaps.get(q.p.id);
        if (!edge) continue;
        const r = projectedRect(q, W, H, tanHalfFov, worldSize);
        const rw = r.xmax - r.xmin, rh = r.ymax - r.ymin;
        if (rw < 2 || rh < 2) continue;
        const tDepth = (q.worldDist - dMin) / dSpan;   // 0 near, 1 far
        ctx.globalAlpha = Math.max(0.04, 1.0 - lineworkFade * tDepth);
        ctx.drawImage(edge, r.xmin, r.ymin, rw, rh);
        drawn++;
      }
      ctx.globalAlpha = 1.0;
      if (drawn === 0) {
        ctx.fillStyle = '#888';
        ctx.font = '12px ui-monospace, monospace';
        ctx.fillText('no images visible at this view angle', 20, 30);
      }
    }
  } else if (lineworkStyle === 'collapsed_field') {
    // Worlds: each latent cluster drawn as a concrete bounded region with its
    // image content inside, projected through the live camera so it rotates.
    if (contentPolygons.size === 0) {
      ctx.fillStyle = '#888';
      ctx.font = '12px ui-monospace, monospace';
      ctx.fillText(
        edgeMapsTotal === 0 ? 'preparing content polygons...'
          : `precomputing content polygons... ${edgeMapsReady}/${edgeMapsTotal}`,
        20, 30);
    } else {
      const { forms, links, bridges, unions, depthRange } = computeWorlds(W, H);
      const minDim = Math.min(W, H);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      // --- interconnections (under everything) ---
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = Math.max(0.5, lineworkWeight * 0.9);
      ctx.globalAlpha = 0.3;
      for (const link of links) {
        if (!link || link.length < 2) continue;
        ctx.beginPath();
        for (let i = 0; i < link.length; i++) { if (i === 0) ctx.moveTo(link[i][0], link[i][1]); else ctx.lineTo(link[i][0], link[i][1]); }
        ctx.stroke();
      }
      ctx.globalAlpha = 1.0;
      if (worldsUnify > 0.001 && unions.length) {
        // --- boolean-union figures: connected components, far → near ---
        for (const u of unions) {
          ctx.fillStyle = (u.pattern === 'solid') ? '#ffffff' : getStackedPattern(ctx, u.pattern);
          ctx.beginPath();
          for (const c of u.contours) {
            if (!c || c.length < 3) continue;
            for (let i = 0; i < c.length; i++) { if (i === 0) ctx.moveTo(c[i][0], c[i][1]); else ctx.lineTo(c[i][0], c[i][1]); }
            ctx.closePath();
          }
          ctx.fill('evenodd');
          const dw = worldsDepthWeight(u.depth, depthRange, lineworkWeight);
          ctx.strokeStyle = dw.stroke; ctx.lineWidth = dw.w; ctx.stroke();
        }
        // medoid inner detail on top for character, depth-weighted + lighter
        for (const f of forms) {
          if (!f.isMedoid || !f.inner || !f.inner.length) continue;
          const dw = worldsDepthWeight(f.depth, depthRange, lineworkWeight);
          ctx.strokeStyle = dw.stroke; ctx.lineWidth = Math.max(0.18, dw.w * 0.45);
          for (const c of f.inner) {
            if (!c || c.length < 2) continue;
            ctx.beginPath();
            for (let i = 0; i < c.length; i++) { const x = f.cx + (c[i][0] - 0.5) * f.size, y = f.cy + (c[i][1] - 0.5) * f.size; if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y); }
            ctx.stroke();
          }
        }
      } else {
      // --- merge bridges + forms, draw far→near (occlusion) ---
      const items = [];
      for (const f of forms) items.push({ t: 'f', depth: f.depth, f });
      for (const br of bridges) items.push({ t: 'b', depth: br.depth, br });
      items.sort((a, b) => b.depth - a.depth);
      for (const it of items) {
        if (it.t === 'b') {
          const br = it.br;
          const dx = br.bx - br.ax, dy = br.by - br.ay; const L = Math.hypot(dx, dy) || 1;
          const px = -dy / L * br.w / 2, py = dx / L * br.w / 2;
          ctx.beginPath();
          ctx.moveTo(br.ax + px, br.ay + py); ctx.lineTo(br.bx + px, br.by + py);
          ctx.lineTo(br.bx - px, br.by - py); ctx.lineTo(br.ax - px, br.ay - py);
          ctx.closePath();
          ctx.fillStyle = (br.pattern === 'solid') ? '#ffffff' : getStackedPattern(ctx, br.pattern);
          ctx.fill();
          const dwb = worldsDepthWeight(br.depth, depthRange, lineworkWeight);
          ctx.strokeStyle = dwb.stroke; ctx.lineWidth = Math.max(0.18, dwb.w * 0.7); ctx.stroke();
          continue;
        }
        const f = it.f, S = f.size, cx = f.cx, cy = f.cy;
        if (!f.dom || f.dom.length < 3) continue;
        ctx.beginPath();
        for (let i = 0; i < f.dom.length; i++) {
          const dx2 = cx + (f.dom[i][0] - 0.5) * S, dy2 = cy + (f.dom[i][1] - 0.5) * S;
          if (i === 0) ctx.moveTo(dx2, dy2); else ctx.lineTo(dx2, dy2);
        }
        ctx.closePath();
        ctx.fillStyle = (f.pattern === 'solid') ? '#ffffff' : getStackedPattern(ctx, f.pattern);
        ctx.fill();
        const dwf = worldsDepthWeight(f.depth, depthRange, lineworkWeight);
        if (f.isMedoid) {
          // primary: bold outline + full inner structure (recognizable, rich)
          ctx.strokeStyle = dwf.stroke;
          ctx.lineWidth = dwf.w;
          ctx.stroke();
          if (f.inner && f.inner.length) {
            ctx.lineWidth = Math.max(0.18, dwf.w * 0.45);
            for (const c of f.inner) {
              if (!c || c.length < 2) continue;
              ctx.beginPath();
              for (let i = 0; i < c.length; i++) {
                const dx2 = cx + (c[i][0] - 0.5) * S, dy2 = cy + (c[i][1] - 0.5) * S;
                if (i === 0) ctx.moveTo(dx2, dy2); else ctx.lineTo(dx2, dy2);
              }
              ctx.stroke();
            }
          }
        } else {
          // secondary texture: thin outline only
          ctx.strokeStyle = dwf.stroke;
          ctx.lineWidth = Math.max(0.18, dwf.w * 0.55);
          ctx.stroke();
        }
        continue;
      }
      }
    }
  } else if (lineworkStyle === 'stacked_perspectives') {
    if (contentPolygons.size === 0) {
      ctx.fillStyle = '#888';
      ctx.font = '12px ui-monospace, monospace';
      ctx.fillText('precomputing per-image content polygons...', 20, 30);
      ctx.restore(); return;
    }
    const variants = generateStackedPerspectiveVariants(W, H);
    ctx.lineCap = 'square';
    ctx.lineJoin = 'miter';
    ctx.imageSmoothingEnabled = false;
    for (const v of variants) {
      const pv = projectThroughCamera(v.camera, W, H);
      const tanHalfFov = Math.tan((v.camera.fov * Math.PI / 180) / 2);
      const projSorted = pv.projected.slice().sort((a, b) => b.worldDist - a.worldDist);
      // Set up the variant's fill/stroke style once
      let fillStyle, strokeStyle, strokeWidth;
      switch (v.register) {
        case 'solid':
          fillStyle = '#000000'; strokeStyle = null; break;
        case 'halftone':
          fillStyle = getStackedPattern(ctx, 'halftone');
          strokeStyle = '#000000'; strokeWidth = 0.8; break;
        case 'hatch':
          fillStyle = getStackedPattern(ctx, 'hatch');
          strokeStyle = '#000000'; strokeWidth = 0.8; break;
        case 'vstripe':
          fillStyle = getStackedPattern(ctx, 'vstripe');
          strokeStyle = '#000000'; strokeWidth = 0.8; break;
        case 'outline':
          fillStyle = null; strokeStyle = '#000000'; strokeWidth = 1.4; break;
      }
      for (const q of projSorted) {
        const polys = contentPolygons.get(q.p.id);
        if (!polys || polys.length === 0) continue;
        const r = projectedRect(q, W, H, tanHalfFov, worldSize);
        const rw = r.xmax - r.xmin, rh = r.ymax - r.ymin;
        if (rw < STACKED_GRID * 2 || rh < STACKED_GRID * 2) continue;
        for (const poly of polys) {
          if (poly.length < 3) continue;
          ctx.beginPath();
          for (let i = 0; i < poly.length; i++) {
            const px = snapPx(r.xmin + poly[i][0] * rw);
            const py = snapPx(r.ymin + poly[i][1] * rh);
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }
          ctx.closePath();
          if (fillStyle !== null) { ctx.fillStyle = fillStyle; ctx.fill('evenodd'); }
          if (strokeStyle !== null) {
            ctx.strokeStyle = strokeStyle;
            ctx.lineWidth = strokeWidth;
            ctx.stroke();
          }
        }
      }
    }
  } else if (lineworkStyle === 'composition') {
    const sortedC = rects.slice().sort((a, b) => b.depth - a.depth);
    ctx.lineCap = 'square';
    ctx.lineJoin = 'miter';
    for (const r of sortedC) {
      const w = r.xmax - r.xmin, h = r.ymax - r.ymin;
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(r.xmin, r.ymin, w, h);
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = lineworkWeight * 0.7;
      ctx.strokeRect(r.xmin, r.ymin, w, h);
    }
    const segC = computeSegmentationPaths(800);
    if (segC && segC.paths.length > 0) {
      const sxc = W / segC.width, syc = H / segC.height;
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = lineworkWeight * 1.2;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      for (const pts of segC.paths) {
        if (pts.length < 2) continue;
        ctx.moveTo(pts[0][0] * sxc, pts[0][1] * syc);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i][0] * sxc, pts[i][1] * syc);
        }
      }
      ctx.stroke();
    }
  } else if (lineworkStyle === 'all') {
    const sorted = rects.slice().sort((a, b) => b.depth - a.depth);
    for (const r of sorted) {
      const a = alphaForDepth(r.depth);
      ctx.strokeStyle = `rgba(0, 0, 0, ${a.toFixed(3)})`;
      ctx.lineWidth = lineworkWeight;
      ctx.beginPath();
      for (const e of rectEdges(r)) {
        ctx.moveTo(e.x1, e.y1); ctx.lineTo(e.x2, e.y2);
      }
      ctx.stroke();
    }
  } else if (lineworkStyle === 'ghosted') {
    // Visible parts solid black, hidden parts dashed mid-grey.
    const sorted = rects.slice().sort((a, b) => b.depth - a.depth);
    for (let i = 0; i < sorted.length; i++) {
      const r = sorted[i];
      const occluders = sorted.slice(i + 1);
      const a = alphaForDepth(r.depth);
      for (const e of rectEdges(r)) {
        const { visible, hidden } = visibleHiddenIntervals(e, occluders);
        if (visible.length > 0) {
          ctx.setLineDash([]);
          ctx.strokeStyle = `rgba(0, 0, 0, ${a.toFixed(3)})`;
          ctx.lineWidth = lineworkWeight;
          ctx.beginPath();
          for (const [t0, t1] of visible) {
            const s = intervalToSegment(e, t0, t1);
            ctx.moveTo(s.x1, s.y1); ctx.lineTo(s.x2, s.y2);
          }
          ctx.stroke();
        }
        if (hidden.length > 0) {
          ctx.setLineDash([5, 3]);
          ctx.strokeStyle = `rgba(140, 140, 140, ${(a * 0.6).toFixed(3)})`;
          ctx.lineWidth = Math.max(0.3, lineworkWeight * 0.7);
          ctx.beginPath();
          for (const [t0, t1] of hidden) {
            const s = intervalToSegment(e, t0, t1);
            ctx.moveTo(s.x1, s.y1); ctx.lineTo(s.x2, s.y2);
          }
          ctx.stroke();
        }
      }
    }
    ctx.setLineDash([]);
  } else {
    // 'occluded' — strict hidden-line removal, current behavior
    const survivors = hiddenLineRemoval(rects);
    ctx.strokeStyle = '#000000';
    ctx.lineWidth = lineworkWeight;
    ctx.beginPath();
    for (const s of survivors) {
      if (Math.hypot(s.x2 - s.x1, s.y2 - s.y1) < 0.5) continue;
      ctx.moveTo(s.x1, s.y1); ctx.lineTo(s.x2, s.y2);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function redrawOverlay() {
  const ctx = overlayCanvas.getContext('2d');
  const w = W(), h = H();
  ctx.clearRect(0, 0, w, h);
  if (currentMode === 'images') return;

  const pv = projectVisiblePoints(w, h);
  if (currentMode === 'heatmap') {
    renderHeatmapToCanvas(ctx, pv.projected, w, h);
  } else if (currentMode === 'linework') {
    renderLineworkToCanvas(ctx, pv.projected, pv.exportCam, w, h);
  }
}

function markerSVG(x, y, r, p, opAttr, strokeAttr, opacity) {
  const shape = SETTINGS.shape;
  if (shape === "image" && p.exp_img) {
    const max_dim = 2 * r;
    let w, h;
    if (p.exp_ar >= 1) { w = max_dim; h = max_dim / p.exp_ar; }
    else { h = max_dim; w = max_dim * p.exp_ar; }
    const opImg = opacity < 1 ? ` opacity="${opacity.toFixed(2)}"` : '';
    return `<image x="${(x - w/2).toFixed(2)}" y="${(y - h/2).toFixed(2)}" ` +
           `width="${w.toFixed(2)}" height="${h.toFixed(2)}" ` +
           `preserveAspectRatio="xMidYMid meet" ` +
           `xlink:href="data:image/png;base64,${p.exp_img}"${opImg}/>`;
  }
  if (shape === "square") {
    return `<rect x="${(x-r).toFixed(2)}" y="${(y-r).toFixed(2)}" ` +
           `width="${(2*r).toFixed(2)}" height="${(2*r).toFixed(2)}" ` +
           `fill="${p.color}"${opAttr}${strokeAttr}/>`;
  }
  if (shape === "triangle") {
    const a = `${x.toFixed(2)},${(y - r).toFixed(2)}`;
    const b = `${(x + r * 0.866).toFixed(2)},${(y + r * 0.5).toFixed(2)}`;
    const c = `${(x - r * 0.866).toFixed(2)},${(y + r * 0.5).toFixed(2)}`;
    return `<polygon points="${a} ${b} ${c}" fill="${p.color}"${opAttr}${strokeAttr}/>`;
  }
  if (shape === "diamond") {
    const pts = `${x.toFixed(2)},${(y - r).toFixed(2)} ` +
                `${(x + r).toFixed(2)},${y.toFixed(2)} ` +
                `${x.toFixed(2)},${(y + r).toFixed(2)} ` +
                `${(x - r).toFixed(2)},${y.toFixed(2)}`;
    return `<polygon points="${pts}" fill="${p.color}"${opAttr}${strokeAttr}/>`;
  }
  if (shape === "cross") {
    const sw = Math.max(SETTINGS.stroke_width, 1).toFixed(2);
    const opLine = opacity < 1 ? ` stroke-opacity="${opacity.toFixed(2)}"` : '';
    return `<line x1="${(x-r).toFixed(2)}" y1="${y.toFixed(2)}" ` +
           `x2="${(x+r).toFixed(2)}" y2="${y.toFixed(2)}" ` +
           `stroke="${p.color}" stroke-width="${sw}"${opLine}/>` +
           `<line x1="${x.toFixed(2)}" y1="${(y-r).toFixed(2)}" ` +
           `x2="${x.toFixed(2)}" y2="${(y+r).toFixed(2)}" ` +
           `stroke="${p.color}" stroke-width="${sw}"${opLine}/>`;
  }
  // default: circle (also fallback when shape=image but no exp_img)
  return `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="${r.toFixed(2)}" ` +
         `fill="${p.color}"${opAttr}${strokeAttr}/>`;
}

function safeId(name) {
  let out = String(name).replace(/[^a-zA-Z0-9_.\-]/g, '_');
  if (!/^[a-zA-Z_]/.test(out)) out = 'L_' + out;
  return out;
}
function xmlEscape(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
                  .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ---------------------------------------------------------------------------
// Concave-hull boundary (JS port of fit_concave_hull in app.py).
// Convex hull via Andrew's monotone chain, then iterative edge refinement
// pulls the boundary inward where interior points are close to edge
// midpoints. Output is a closed, ordered ring of {x, y} points.
// ---------------------------------------------------------------------------
function _cross(o, a, b) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

function convexHullMonotone(pts) {
  if (pts.length < 3) return pts.slice();
  const sorted = pts.slice().sort((a, b) => a.x - b.x || a.y - b.y);
  const lower = [];
  for (const p of sorted) {
    while (lower.length >= 2 && _cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i];
    while (upper.length >= 2 && _cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  lower.pop(); upper.pop();
  return lower.concat(upper);  // CCW ring
}

function fitConcaveHull(points, concavity, smoothing, maxIter) {
  if (!points || points.length < 4) return null;
  // Work on a copy with {x, y} so we don't fight the input shape
  const work = points.map(p => ({ x: p.sx !== undefined ? p.sx : p.x,
                                   y: p.sy !== undefined ? p.sy : p.y }));
  let boundary = convexHullMonotone(work);
  if (boundary.length < 3) return null;
  const onBoundary = new Set(boundary);
  let interior = work.filter(p => !onBoundary.has(p));

  const iters = Math.max(1, maxIter | 0);
  for (let it = 0; it < iters; it++) {
    let changed = false;
    const next = [];
    for (let i = 0; i < boundary.length; i++) {
      const p1 = boundary[i];
      const p2 = boundary[(i + 1) % boundary.length];
      next.push(p1);
      const elen = Math.hypot(p2.x - p1.x, p2.y - p1.y);
      const mx = (p1.x + p2.x) / 2, my = (p1.y + p2.y) / 2;
      const threshold = elen / concavity;
      let best = null, bestD = Infinity, bestIdx = -1;
      for (let j = 0; j < interior.length; j++) {
        const p = interior[j];
        const d = Math.hypot(p.x - mx, p.y - my);
        if (d < threshold && d < bestD && _cross(p1, p2, p) > 0) {
          best = p; bestD = d; bestIdx = j;
        }
      }
      if (best !== null) {
        next.push(best);
        interior.splice(bestIdx, 1);
        changed = true;
      }
    }
    boundary = next;
    if (!changed) break;
  }

  // Smooth (closed loop)
  let arr = boundary.map(p => ({ x: p.x, y: p.y }));
  const sm = Math.max(0, smoothing | 0);
  for (let it = 0; it < sm; it++) {
    const n = arr.length;
    const next = arr.map(p => ({ x: p.x, y: p.y }));
    for (let i = 0; i < n; i++) {
      const prev = arr[(i - 1 + n) % n];
      const nxt = arr[(i + 1) % n];
      next[i].x = 0.25 * prev.x + 0.5 * arr[i].x + 0.25 * nxt.x;
      next[i].y = 0.25 * prev.y + 0.5 * arr[i].y + 0.25 * nxt.y;
    }
    arr = next;
  }
  return arr;
}

// Catmull-Rom closed loop -> cubic-Bezier SVG path with Z
function closedCurveToSvgPath(pts) {
  if (!pts || pts.length < 3) return "";
  const n = pts.length;
  const parts = [`M ${pts[0].x.toFixed(2)},${pts[0].y.toFixed(2)}`];
  for (let i = 0; i < n; i++) {
    const pPrev = pts[(i - 1 + n) % n];
    const pCurr = pts[i];
    const pNext = pts[(i + 1) % n];
    const pNext2 = pts[(i + 2) % n];
    const c1x = pCurr.x + (pNext.x - pPrev.x) / 6;
    const c1y = pCurr.y + (pNext.y - pPrev.y) / 6;
    const c2x = pNext.x - (pNext2.x - pCurr.x) / 6;
    const c2y = pNext.y - (pNext2.y - pCurr.y) / 6;
    parts.push(
      `C ${c1x.toFixed(2)},${c1y.toFixed(2)} ` +
      `${c2x.toFixed(2)},${c2y.toFixed(2)} ` +
      `${pNext.x.toFixed(2)},${pNext.y.toFixed(2)}`
    );
  }
  parts.push("Z");
  return parts.join(' ');
}

// ---------------------------------------------------------------------------
// Heatmap helpers: histogram + separable Gaussian blur (a.k.a. fast KDE),
// then marching squares to extract isocontours from the density grid.
// All operate in screen pixel space.
// ---------------------------------------------------------------------------
function computeDensityGrid(points, W, H, gridW, gridH, bandwidthPx) {
  // 1) bin points into a 2D histogram
  const hist = new Float32Array(gridW * gridH);
  const cellW = W / gridW, cellH = H / gridH;
  for (const p of points) {
    const gx = Math.floor(p.sx / cellW);
    const gy = Math.floor(p.sy / cellH);
    if (gx >= 0 && gx < gridW && gy >= 0 && gy < gridH) {
      hist[gy * gridW + gx] += 1;
    }
  }
  // 2) separable Gaussian kernel in cell units
  const sigmaCells = Math.max(0.5, bandwidthPx / Math.min(cellW, cellH));
  const radius = Math.min(40, Math.ceil(3 * sigmaCells));
  const kernel = new Float32Array(2 * radius + 1);
  let kSum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigmaCells * sigmaCells));
    kernel[i + radius] = v; kSum += v;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= kSum;

  // horizontal pass
  const tmp = new Float32Array(gridW * gridH);
  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) {
      let s = 0;
      for (let k = -radius; k <= radius; k++) {
        let sx = x + k;
        if (sx < 0) sx = 0; else if (sx >= gridW) sx = gridW - 1;
        s += hist[y * gridW + sx] * kernel[k + radius];
      }
      tmp[y * gridW + x] = s;
    }
  }
  // vertical pass
  const out = new Float32Array(gridW * gridH);
  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) {
      let s = 0;
      for (let k = -radius; k <= radius; k++) {
        let sy = y + k;
        if (sy < 0) sy = 0; else if (sy >= gridH) sy = gridH - 1;
        s += tmp[sy * gridW + x] * kernel[k + radius];
      }
      out[y * gridW + x] = s;
    }
  }
  return out;
}

// Marching squares: returns array of line segments at a given iso-level.
// Each segment is {x1, y1, x2, y2} in pixel coordinates.
function marchingSquaresAt(grid, gridW, gridH, level, cellW, cellH) {
  const segs = [];
  function interp(x0, y0, v0, x1, y1, v1) {
    // Preserve the sign of the denominator. Using Math.max(v1-v0, 1e-12) was
    // a bug: on a descending edge (v1 < v0) it forced the denominator to a
    // tiny positive number, producing astronomically large t and off-canvas
    // garbage coordinates that broke contour stitching.
    const d = v1 - v0;
    const t = (Math.abs(d) < 1e-12) ? 0.5 : (level - v0) / d;
    return [x0 + t * (x1 - x0), y0 + t * (y1 - y0)];
  }
  for (let y = 0; y < gridH - 1; y++) {
    for (let x = 0; x < gridW - 1; x++) {
      const v00 = grid[y * gridW + x];
      const v10 = grid[y * gridW + x + 1];
      const v01 = grid[(y + 1) * gridW + x];
      const v11 = grid[(y + 1) * gridW + x + 1];
      let c = 0;
      if (v00 >= level) c |= 1;
      if (v10 >= level) c |= 2;
      if (v11 >= level) c |= 4;
      if (v01 >= level) c |= 8;
      if (c === 0 || c === 15) continue;
      const x0 = x * cellW, y0 = y * cellH;
      const x1 = (x + 1) * cellW, y1 = (y + 1) * cellH;
      // Edge crossings: top, right, bottom, left
      const eT = () => interp(x0, y0, v00, x1, y0, v10);
      const eR = () => interp(x1, y0, v10, x1, y1, v11);
      const eB = () => interp(x1, y1, v11, x0, y1, v01);
      const eL = () => interp(x0, y1, v01, x0, y0, v00);
      const push = (a, b) => segs.push({ x1: a[0], y1: a[1], x2: b[0], y2: b[1] });
      switch (c) {
        case 1: case 14: push(eL(), eT()); break;
        case 2: case 13: push(eT(), eR()); break;
        case 3: case 12: push(eL(), eR()); break;
        case 4: case 11: push(eR(), eB()); break;
        case 6: case  9: push(eT(), eB()); break;
        case 7: case  8: push(eL(), eB()); break;
        case 5: push(eL(), eT()); push(eR(), eB()); break;
        case 10: push(eL(), eB()); push(eT(), eR()); break;
      }
    }
  }
  return segs;
}

// Stitch line segments at a level into ordered polylines. Two segments
// connect at a shared endpoint (within a small tolerance). Closed loops
// come back closed; open chains come back open.
function stitchSegmentsToPaths(segs, tol) {
  if (segs.length === 0) return [];
  const t = tol || 0.5;
  const key = (x, y) => `${Math.round(x / t)}|${Math.round(y / t)}`;
  // build adjacency from point-key to segment indices
  const adj = new Map();
  const used = new Uint8Array(segs.length);
  for (let i = 0; i < segs.length; i++) {
    const s = segs[i];
    for (const k of [key(s.x1, s.y1), key(s.x2, s.y2)]) {
      if (!adj.has(k)) adj.set(k, []);
      adj.get(k).push(i);
    }
  }
  const paths = [];
  for (let i = 0; i < segs.length; i++) {
    if (used[i]) continue;
    used[i] = 1;
    const s = segs[i];
    const pts = [[s.x1, s.y1], [s.x2, s.y2]];
    // extend forward
    while (true) {
      const last = pts[pts.length - 1];
      const cands = adj.get(key(last[0], last[1])) || [];
      let nxt = -1;
      for (const ci of cands) {
        if (used[ci]) continue;
        nxt = ci; break;
      }
      if (nxt < 0) break;
      used[nxt] = 1;
      const ns = segs[nxt];
      const matchA = Math.abs(ns.x1 - last[0]) < t && Math.abs(ns.y1 - last[1]) < t;
      pts.push(matchA ? [ns.x2, ns.y2] : [ns.x1, ns.y1]);
    }
    // extend backward
    while (true) {
      const first = pts[0];
      const cands = adj.get(key(first[0], first[1])) || [];
      let nxt = -1;
      for (const ci of cands) {
        if (used[ci]) continue;
        nxt = ci; break;
      }
      if (nxt < 0) break;
      used[nxt] = 1;
      const ns = segs[nxt];
      const matchA = Math.abs(ns.x1 - first[0]) < t && Math.abs(ns.y1 - first[1]) < t;
      pts.unshift(matchA ? [ns.x2, ns.y2] : [ns.x1, ns.y1]);
    }
    paths.push(pts);
  }
  return paths;
}

function pathToSvgD(pts, closed) {
  if (pts.length < 2) return "";
  const parts = [`M ${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`];
  for (let i = 1; i < pts.length; i++) {
    parts.push(`L ${pts[i][0].toFixed(1)},${pts[i][1].toFixed(1)}`);
  }
  if (closed) parts.push('Z');
  return parts.join(' ');
}

// ---------------------------------------------------------------------------
// Hidden-line removal for the linework mode.
// Each card is an axis-aligned rectangle in screen space at a depth.
// We sort back-to-front; for each card's 4 edges we clip away any portion
// that lies behind a card in front of it. What survives is "visible".
// ---------------------------------------------------------------------------
function clipSegmentByRect(seg, rect) {
  // Returns the parts of seg OUTSIDE rect (Liang-Barsky inside, then subtract)
  const dx = seg.x2 - seg.x1, dy = seg.y2 - seg.y1;
  const p = [-dx, dx, -dy, dy];
  const q = [seg.x1 - rect.xmin, rect.xmax - seg.x1,
             seg.y1 - rect.ymin, rect.ymax - seg.y1];
  let tEnter = 0, tExit = 1;
  for (let i = 0; i < 4; i++) {
    if (p[i] === 0) {
      if (q[i] < 0) return [seg];   // parallel and outside
      continue;
    }
    const t = q[i] / p[i];
    if (p[i] < 0) { if (t > tEnter) tEnter = t; }
    else          { if (t < tExit)  tExit  = t; }
  }
  if (tEnter > tExit) return [seg]; // no intersection
  if (tEnter <= 0 && tExit >= 1) return []; // entirely inside, no survivors
  const out = [];
  if (tEnter > 1e-6) {
    out.push({
      x1: seg.x1, y1: seg.y1,
      x2: seg.x1 + tEnter * dx, y2: seg.y1 + tEnter * dy,
    });
  }
  if (tExit < 1 - 1e-6) {
    out.push({
      x1: seg.x1 + tExit * dx, y1: seg.y1 + tExit * dy,
      x2: seg.x2, y2: seg.y2,
    });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Image-segmentation linework. The key insight: don't read the rendered
// WebGL canvas (its sprite-card BORDERS dominate gradient and produce a
// grid of axis-aligned crosshairs at any view angle). Instead, pre-compute
// the Sobel edge map of each photograph once, at load time. These edge
// maps contain only INTERIOR photo edges (rooflines, window mullions, etc.)
// because Sobel responds to pixel changes, not to "off-image background."
//
// At render time we additively composite those edge maps at the sprites'
// projected screen positions and sizes, threshold the accumulated intensity,
// and extract marching-squares contours from THAT. The result follows the
// actual photo content in its overlap pattern at the current camera angle.
// ---------------------------------------------------------------------------
function imageDataToGrayscale(imd) {
  const { data, width, height } = imd;
  const out = new Float32Array(width * height);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    out[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }
  return { gray: out, width, height };
}

function gaussianBlurFloat(input, w, h, sigma) {
  const radius = Math.max(1, Math.ceil(3 * sigma));
  const k = new Float32Array(2 * radius + 1);
  let ks = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    k[i + radius] = v; ks += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= ks;
  const tmp = new Float32Array(input.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = -radius; i <= radius; i++) {
        let sx = x + i;
        if (sx < 0) sx = 0; else if (sx >= w) sx = w - 1;
        s += input[y * w + sx] * k[i + radius];
      }
      tmp[y * w + x] = s;
    }
  }
  const out = new Float32Array(input.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let i = -radius; i <= radius; i++) {
        let sy = y + i;
        if (sy < 0) sy = 0; else if (sy >= h) sy = h - 1;
        s += tmp[sy * w + x] * k[i + radius];
      }
      out[y * w + x] = s;
    }
  }
  return out;
}

function sobelMagnitude(g, w, h) {
  const out = new Float32Array(w * h);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const tl = g[i - w - 1], tc = g[i - w], tr = g[i - w + 1];
      const ml = g[i - 1],                       mr = g[i + 1];
      const bl = g[i + w - 1], bc = g[i + w], br = g[i + w + 1];
      const gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl);
      const gy = (bl + 2 * bc + br) - (tl + 2 * tc + tr);
      out[i] = Math.sqrt(gx * gx + gy * gy);
    }
  }
  return out;
}

const edgeMaps = new Map();           // p.id -> black-on-transparent edge Canvas
let edgeMapsReady = 0, edgeMapsTotal = 0;
const EDGE_MAP_BLUR_SIGMA = 0.8;      // pre-blur sigma when computing per-image edges

// Per-member visual-complexity signal from machine vision: the fraction of a
// member's image that carries significant Sobel edge energy, computed once at
// load. High = densely, intricately articulated content (a busy drawing or
// detailed photo); low = a flat, sparse image. This is the signal that decides
// whether a member's contribution to a union edge is rendered as intricate
// digital articulation or a gestural painterly move.
const edgeComplexity = new Map();     // p.id -> raw edge density in [0,1]
let complexityRankCache = null;       // id -> percentile rank in [0,1], lazily built
// Fraction of members (the least visually complex) whose edges turn gestural;
// the rest stay intricately articulated. Kept a minority by design.
const GESTURAL_FRACTION = 0.3;
function complexityRankOf(id) {
  if (!complexityRankCache) {
    const entries = [...edgeComplexity.entries()];
    if (entries.length === 0) return 0.5;
    entries.sort((a, b) => a[1] - b[1]);
    complexityRankCache = new Map();
    const n = entries.length;
    for (let i = 0; i < n; i++) complexityRankCache.set(entries[i][0], n > 1 ? i / (n - 1) : 0.5);
  }
  const r = complexityRankCache.get(id);
  return (r === undefined) ? 0.5 : r;   // unknown members default to articulated side
}

// Which signal decides articulation: 'vision' (Sobel edge density) or
// 'autoencoder' (linear-autoencoder reconstruction error on the CLIP corpus,
// passed in as ae_surprise per member). Defaults to the autoencoder when that
// signal carries real variation, otherwise falls back to vision.
let ARTICULATION_SIGNAL = 'autoencoder';
let surpriseRankCache = null;           // id -> percentile rank of ae_surprise
let surpriseUsable = null;              // whether ae_surprise varies enough to use
function buildSurpriseRank() {
  const entries = [];
  for (const p of POINTS) if (typeof p.ae_surprise === 'number') entries.push([p.id, p.ae_surprise]);
  surpriseRankCache = new Map();
  if (entries.length < 2) { surpriseUsable = false; return; }
  let lo = Infinity, hi = -Infinity;
  for (const e of entries) { if (e[1] < lo) lo = e[1]; if (e[1] > hi) hi = e[1]; }
  surpriseUsable = (hi - lo) > 1e-9;
  entries.sort((a, b) => a[1] - b[1]);
  const n = entries.length;
  for (let i = 0; i < n; i++) surpriseRankCache.set(entries[i][0], i / (n - 1));
}
function surpriseRankOf(id) {
  if (!surpriseRankCache) buildSurpriseRank();
  const r = surpriseRankCache.get(id);
  return (r === undefined) ? 0.5 : r;
}
// Per-member articulation scalar, low = gestural. For the autoencoder signal a
// SURPRISING work (high reconstruction error, high rank) maps to LOW articulation
// so the gesture falls where the corpus reconstructs the work least well. For
// the vision signal a visually flat work maps low.
function articulationSignalOf(id) {
  if (ARTICULATION_SIGNAL === 'autoencoder') {
    if (surpriseRankCache === null) buildSurpriseRank();
    if (surpriseUsable) return 1 - surpriseRankOf(id);
  }
  return complexityRankOf(id);
}

// Latent-space bounds, computed once. Used by the collapsed-perspective mode
// to lay images out by their latent coordinates rather than a 3D camera.
let _latentBounds = null;
function getLatentBounds() {
  if (_latentBounds) return _latentBounds;
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity,
      zmin = Infinity, zmax = -Infinity;
  for (const p of POINTS) {
    if (p.x < xmin) xmin = p.x; if (p.x > xmax) xmax = p.x;
    if (p.y < ymin) ymin = p.y; if (p.y > ymax) ymax = p.y;
    if (p.z < zmin) zmin = p.z; if (p.z > zmax) zmax = p.z;
  }
  _latentBounds = { xmin, xmax, ymin, ymax, zmin, zmax };
  return _latentBounds;
}

// --- Curve synthesizer ----------------------------------------------------
// The missing Oehlen "gesture" layer: a few bold, smooth, sweeping curves that
// connect structure in the field and draw a sense of space, sitting over the
// quieted texture of image content. The curves are derived, not random: we
// cluster the latent positions, then run smooth splines through the cluster
// centroids (a dominant spine ordered along the principal axis, a wandering
// nearest-neighbour tour, and short connecting arcs between near centroids).

function kmeans2d(pts, k, seed) {
  // Lightweight k-means. pts is an array of [x,y]. Deterministic given seed.
  const n = pts.length;
  if (n === 0 || k <= 0) return { centroids: [], labels: [] };
  k = Math.min(k, n);
  let s = seed >>> 0;
  const rand = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  // k-means++ style spread-out seeding
  const centroids = [pts[Math.floor(rand() * n)].slice()];
  while (centroids.length < k) {
    let best = null, bestD = -1;
    for (let i = 0; i < n; i++) {
      let dm = Infinity;
      for (const c of centroids) {
        const dx = pts[i][0] - c[0], dy = pts[i][1] - c[1];
        const d = dx * dx + dy * dy; if (d < dm) dm = d;
      }
      // weight pick by distance, with a little randomness
      const w = dm * (0.5 + rand());
      if (w > bestD) { bestD = w; best = i; }
    }
    centroids.push(pts[best].slice());
  }
  const labels = new Array(n).fill(0);
  for (let iter = 0; iter < 30; iter++) {
    for (let i = 0; i < n; i++) {
      let bd = Infinity, bl = 0;
      for (let j = 0; j < centroids.length; j++) {
        const dx = pts[i][0] - centroids[j][0], dy = pts[i][1] - centroids[j][1];
        const d = dx * dx + dy * dy; if (d < bd) { bd = d; bl = j; }
      }
      labels[i] = bl;
    }
    const sx = new Array(centroids.length).fill(0);
    const sy = new Array(centroids.length).fill(0);
    const cnt = new Array(centroids.length).fill(0);
    for (let i = 0; i < n; i++) { sx[labels[i]] += pts[i][0]; sy[labels[i]] += pts[i][1]; cnt[labels[i]]++; }
    for (let j = 0; j < centroids.length; j++) {
      if (cnt[j] > 0) { centroids[j][0] = sx[j] / cnt[j]; centroids[j][1] = sy[j] / cnt[j]; }
    }
  }
  return { centroids, labels };
}

// Convert an ordered list of points into a smooth path string of cubic Beziers
// using the Catmull-Rom -> Bezier conversion. Returns SVG path data.
function catmullRomPath(pts, closed) {
  if (pts.length < 2) return '';
  const P = pts.map(p => [p[0], p[1]]);
  if (closed) { P.unshift(pts[pts.length - 1]); P.push(pts[0]); P.push(pts[1]); }
  else { P.unshift(pts[0]); P.push(pts[pts.length - 1]); }
  let d = `M ${P[1][0].toFixed(2)},${P[1][1].toFixed(2)}`;
  for (let i = 1; i < P.length - 2; i++) {
    const p0 = P[i - 1], p1 = P[i], p2 = P[i + 1], p3 = P[i + 2];
    const c1x = p1[0] + (p2[0] - p0[0]) / 6, c1y = p1[1] + (p2[1] - p0[1]) / 6;
    const c2x = p2[0] - (p3[0] - p1[0]) / 6, c2y = p2[1] - (p3[1] - p1[1]) / 6;
    d += ` C ${c1x.toFixed(2)},${c1y.toFixed(2)} ${c2x.toFixed(2)},${c2y.toFixed(2)} ${p2[0].toFixed(2)},${p2[1].toFixed(2)}`;
  }
  return d;
}

// Same Catmull-Rom but sampled to a flat polyline (for canvas stroking).
function catmullRomPoly(pts, closed, samplesPer) {
  if (pts.length < 2) return pts.slice();
  const P = pts.map(p => [p[0], p[1]]);
  if (closed) { P.unshift(pts[pts.length - 1]); P.push(pts[0]); P.push(pts[1]); }
  else { P.unshift(pts[0]); P.push(pts[pts.length - 1]); }
  const out = [];
  const sp = samplesPer || 18;
  for (let i = 1; i < P.length - 2; i++) {
    const p0 = P[i - 1], p1 = P[i], p2 = P[i + 1], p3 = P[i + 2];
    for (let s = 0; s < sp; s++) {
      const t = s / sp, t2 = t * t, t3 = t2 * t;
      const x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3);
      const y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3);
      out.push([x, y]);
    }
  }
  out.push(P[P.length - 2]);
  return out;
}

let _gestureCache = { key: null, gestures: null };
// Compute the bold gesture curves for a set of 2D anchor positions. Returns
// [{ path, poly, weight }]. Cached by (count of positions, K, arcDensity).
function computeCurveGestures(positions, K, arcDensity) {
  const key = positions.length + ':' + K + ':' + arcDensity;
  if (_gestureCache.key === key && _gestureCache.gestures) return _gestureCache.gestures;
  const gestures = [];
  if (positions.length >= 3) {
    const { centroids } = kmeans2d(positions, K, 12345);
    const C = centroids;
    if (C.length >= 3) {
      // mean
      let mx = 0, my = 0; for (const c of C) { mx += c[0]; my += c[1]; } mx /= C.length; my /= C.length;
      // principal axis via 2x2 covariance eigenvector
      let sxx = 0, sxy = 0, syy = 0;
      for (const c of C) { const dx = c[0] - mx, dy = c[1] - my; sxx += dx * dx; sxy += dx * dy; syy += dy * dy; }
      const tr = sxx + syy, det = sxx * syy - sxy * sxy;
      const lam = tr / 2 + Math.sqrt(Math.max(0, tr * tr / 4 - det));
      let ax = sxy, ay = lam - sxx;
      if (Math.abs(ax) < 1e-9 && Math.abs(ay) < 1e-9) { ax = 1; ay = 0; }
      const an = Math.hypot(ax, ay); ax /= an; ay /= an;
      // dominant spine: order centroids by projection onto principal axis
      const order = C.map((c, i) => i).sort((a, b) =>
        ((C[a][0] - mx) * ax + (C[a][1] - my) * ay) - ((C[b][0] - mx) * ax + (C[b][1] - my) * ay));
      const spine = order.map(i => C[i]);
      gestures.push({ path: catmullRomPath(spine, false), poly: catmullRomPoly(spine, false, 20), weight: 1.0 });
      // wandering nearest-neighbour tour from one extreme
      const used = new Array(C.length).fill(false);
      const tour = [order[0]]; used[order[0]] = true;
      while (tour.length < C.length) {
        const last = C[tour[tour.length - 1]]; let bn = -1, bd = Infinity;
        for (let j = 0; j < C.length; j++) {
          if (used[j]) continue;
          const d = (C[j][0] - last[0]) ** 2 + (C[j][1] - last[1]) ** 2;
          if (d < bd) { bd = d; bn = j; }
        }
        tour.push(bn); used[bn] = true;
      }
      const tourPts = tour.map(i => C[i]);
      gestures.push({ path: catmullRomPath(tourPts, false), poly: catmullRomPoly(tourPts, false, 20), weight: 0.55 });
      // short connecting arcs between near centroids (count scaled by arcDensity)
      const nbCount = Math.max(0, Math.min(3, arcDensity));
      for (let i = 0; i < C.length; i++) {
        const d = C.map((c, j) => [(c[0] - C[i][0]) ** 2 + (c[1] - C[i][1]) ** 2, j])
          .sort((a, b) => a[0] - b[0]).slice(1, 1 + nbCount);
        for (const [, j] of d) {
          if (j < i) continue; // each pair once
          const midx = (C[i][0] + C[j][0]) / 2 - (C[j][1] - C[i][1]) * 0.16;
          const midy = (C[i][1] + C[j][1]) / 2 + (C[j][0] - C[i][0]) * 0.16;
          const arc = [C[i], [midx, midy], C[j]];
          gestures.push({ path: catmullRomPath(arc, false), poly: catmullRomPoly(arc, false, 14), weight: 0.32 });
        }
      }
    }
  }
  _gestureCache = { key, gestures };
  return gestures;
}

// --- Worlds composition ---------------------------------------------------
// The latent space holds clusters of similar buildings — "worlds." Each world
// holds its own images — the "perspectives and vibes" within it. This draws
// each world as a concrete bounded region (a bold smooth hull) with its image
// content inside, projected through the live camera so the whole thing rotates.

let _worldLabels = { key: null, labels: null };
function worldClusterLabels(K) {
  // k-means on the LATENT 3D positions (stable worlds, independent of camera).
  if (_worldLabels.key === K && _worldLabels.labels) return _worldLabels.labels;
  const X = POINTS.map(p => [p.x, p.y, p.z]);
  const n = X.length;
  let s = 9871 >>> 0;
  const rand = () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; };
  const k = Math.max(1, Math.min(K, n));
  const C = [X[Math.floor(rand() * n)].slice()];
  while (C.length < k) {
    let best = 0, bd = -1;
    for (let i = 0; i < n; i++) {
      let dm = Infinity;
      for (const c of C) {
        const dx = X[i][0]-c[0], dy = X[i][1]-c[1], dz = X[i][2]-c[2];
        const d = dx*dx+dy*dy+dz*dz; if (d < dm) dm = d;
      }
      const w = dm * (0.5 + rand()); if (w > bd) { bd = w; best = i; }
    }
    C.push(X[best].slice());
  }
  const labels = new Array(n).fill(0);
  for (let it = 0; it < 25; it++) {
    for (let i = 0; i < n; i++) {
      let bd = Infinity, bl = 0;
      for (let j = 0; j < C.length; j++) {
        const dx = X[i][0]-C[j][0], dy = X[i][1]-C[j][1], dz = X[i][2]-C[j][2];
        const d = dx*dx+dy*dy+dz*dz; if (d < bd) { bd = d; bl = j; }
      }
      labels[i] = bl;
    }
    const sx=new Array(C.length).fill(0), sy=new Array(C.length).fill(0),
          sz=new Array(C.length).fill(0), cnt=new Array(C.length).fill(0);
    for (let i = 0; i < n; i++){ const l=labels[i]; sx[l]+=X[i][0];sy[l]+=X[i][1];sz[l]+=X[i][2];cnt[l]++; }
    for (let j=0;j<C.length;j++) if(cnt[j]){C[j][0]=sx[j]/cnt[j];C[j][1]=sy[j]/cnt[j];C[j][2]=sz[j]/cnt[j];}
  }
  _worldLabels = { key: K, labels };
  return labels;
}

// Polar profile of a contour: for N angles around its centroid, the outer
// radius. Returns Float64Array of length N, or null if too sparse. This is
// the representation we average across a world's members to get a mean shape.
function polarProfile(poly, N) {
  if (!poly || poly.length < 6) return null;
  let cx = 0, cy = 0;
  for (const p of poly) { cx += p[0]; cy += p[1]; }
  cx /= poly.length; cy /= poly.length;
  const prof = new Float64Array(N).fill(NaN);
  for (const p of poly) {
    const dx = p[0] - cx, dy = p[1] - cy;
    const a = (Math.atan2(dy, dx) + Math.PI * 2) % (Math.PI * 2);
    const r = Math.hypot(dx, dy);
    const i = Math.min(N - 1, Math.floor(a / (Math.PI * 2) * N));
    if (isNaN(prof[i]) || r > prof[i]) prof[i] = r;   // outer silhouette
  }
  // fill empty angular bins by circular interpolation
  let valid = 0; for (let i = 0; i < N; i++) if (!isNaN(prof[i])) valid++;
  if (valid < 6) return null;
  for (let i = 0; i < N; i++) {
    if (!isNaN(prof[i])) continue;
    let lo = i, hi = i, ld = 0, hd = 0;
    while (isNaN(prof[(lo - 1 + N) % N]) && ld < N) { lo = (lo - 1 + N) % N; ld++; }
    while (isNaN(prof[(hi + 1) % N]) && hd < N) { hi = (hi + 1) % N; hd++; }
    const a = prof[(lo - 1 + N) % N], b = prof[(hi + 1) % N];
    const t = (ld + 1) / (ld + hd + 2);
    prof[i] = a + (b - a) * t;
  }
  return prof;
}

// Mean silhouette of a set of contours: average their polar profiles, then
// emit closed-curve offset points (from centroid) in contour-space units.
// Returns { offsets: [[dx,dy],...], maxR } or null.
function meanSilhouette(contours, N) {
  const profs = [];
  for (const c of contours) { const pr = polarProfile(c, N); if (pr) profs.push(pr); }
  if (profs.length === 0) return null;
  const mean = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let s = 0; for (const pr of profs) s += pr[i]; mean[i] = s / profs.length;
  }
  // light circular smoothing of the mean profile
  const sm = new Float64Array(N);
  for (let i = 0; i < N; i++) sm[i] = (mean[(i-1+N)%N] + 2*mean[i] + mean[(i+1)%N]) / 4;
  let maxR = 0; for (let i = 0; i < N; i++) if (sm[i] > maxR) maxR = sm[i];
  if (maxR < 1e-6) return null;
  const offsets = [];
  for (let i = 0; i < N; i++) {
    const a = i / N * Math.PI * 2;
    offsets.push([sm[i] * Math.cos(a), sm[i] * Math.sin(a)]);
  }
  return { offsets, maxR };
}

// Register a contour: center on its centroid and scale so its mean radius is 1.
// This is the alignment that lets a world's heterogeneous members reinforce
// when overlaid, regardless of original size/position.
function registerContour(poly) {
  if (!poly || poly.length < 6) return null;
  let cx = 0, cy = 0; for (const p of poly) { cx += p[0]; cy += p[1]; } cx /= poly.length; cy /= poly.length;
  let r = 0; for (const p of poly) r += Math.hypot(p[0] - cx, p[1] - cy); r /= poly.length;
  if (r < 1e-6) return null;
  return poly.map(p => [(p[0] - cx) / r, (p[1] - cy) / r]);
}

function sampleEvenly(arr, n) {
  if (arr.length <= n) return arr;
  const out = []; const step = arr.length / n;
  for (let i = 0; i < n; i++) out.push(arr[Math.floor(i * step)]);
  return out;
}

// Density consensus: rasterize registered contours into a grid, blur, and
// trace the iso-contour where many agree — the world's shared structure
// emerging from the superposition. Returns normalized points or null.
function densityConsensus(regList) {
  const G = 96, span = 5.0, half = span / 2;
  const acc = new Float32Array(G * G);
  const toG = (v) => (v + half) / span * (G - 1);
  for (const c of regList) {
    for (let i = 0; i < c.length; i++) {
      const a = c[i], b = c[(i + 1) % c.length];
      const ax = toG(a[0]), ay = toG(a[1]), bx = toG(b[0]), by = toG(b[1]);
      const steps = Math.max(2, Math.ceil(Math.hypot(bx - ax, by - ay)));
      for (let s = 0; s <= steps; s++) {
        const x = Math.round(ax + (bx - ax) * s / steps), y = Math.round(ay + (by - ay) * s / steps);
        if (x >= 0 && x < G && y >= 0 && y < G) acc[y * G + x] += 1;
      }
    }
  }
  const blurred = gaussianBlurFloat(acc, G, G, 2.2);
  let mx = 0; for (let i = 0; i < blurred.length; i++) if (blurred[i] > mx) mx = blurred[i];
  if (mx < 1e-6) return null;
  const segs = marchingSquaresAt(blurred, G, G, 0.32 * mx, 1, 1);
  const paths = stitchSegmentsToPaths(segs, 0.5);
  let best = null, bestA = -1;
  for (const p of paths) {
    if (p.length < 6) continue;
    let xn = Infinity, xx = -Infinity, yn = Infinity, yx = -Infinity;
    for (const q of p) { if (q[0] < xn) xn = q[0]; if (q[0] > xx) xx = q[0]; if (q[1] < yn) yn = q[1]; if (q[1] > yx) yx = q[1]; }
    const A = (xx - xn) * (yx - yn); if (A > bestA) { bestA = A; best = p; }
  }
  if (!best) return null;
  const norm = best.map(q => [q[0] / (G - 1) * span - half, q[1] / (G - 1) * span - half]);
  return simplifyPolyline(norm, span / G * 1.4);
}

// Per-world shapes, cached by K. Each world is represented by a recognizable
// EXEMPLAR — the medoid member's full contour set (silhouette + internal
// edges), which reads as an actual artifact — plus a few satellite members
// (their dominant contour) showing the other perspectives in that world.
// All in image-relative [0,1] coords; placed/scaled per camera at render.
let _worldShapes = { key: null, shapes: null };
function computeWorldShapes(K) {
  if (_worldShapes.key === K && _worldShapes.shapes) return _worldShapes.shapes;
  const labels = worldClusterLabels(K);
  const groups = new Map();
  for (let i = 0; i < POINTS.length; i++) {
    const polys = contentPolygons.get(POINTS[i].id);
    if (!polys || !polys.length) continue;
    const prof = polarProfile(polys[0], 48);
    if (!prof) continue;
    const lab = labels[i];
    if (!groups.has(lab)) groups.set(lab, []);
    groups.get(lab).push({ dom: polys[0], full: polys, prof, idx: i });
  }
  const shapes = new Map();
  for (const [lab, mem] of groups) {
    if (mem.length === 0) continue;
    // medoid: member whose dominant contour is most typical of the world
    let mi = 0, mbest = Infinity;
    for (let i = 0; i < mem.length; i++) {
      let s = 0;
      for (let j = 0; j < mem.length; j++) {
        if (i === j) continue;
        let d = 0; for (let a = 0; a < 48; a++) { const dd = mem[i].prof[a] - mem[j].prof[a]; d += dd * dd; }
        s += Math.sqrt(d);
      }
      if (s < mbest) { mbest = s; mi = i; }
    }
    const others = []; for (let i = 0; i < mem.length; i++) if (i !== mi) others.push(mem[i].dom);
    shapes.set(lab, { medoid: mem[mi].full, medoidIdx: mem[mi].idx, satellites: sampleEvenly(others, 5), count: mem.length });
  }
  _worldShapes = { key: K, shapes };
  return shapes;
}

function convexHull(points) {
  // Andrew's monotone chain. points: [[x,y],...]. Returns hull CCW.
  if (points.length < 3) return points.slice();
  const P = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0]);
  const lo = [];
  for (const p of P) { while (lo.length >= 2 && cross(lo[lo.length-2], lo[lo.length-1], p) <= 0) lo.pop(); lo.push(p); }
  const up = [];
  for (let i = P.length - 1; i >= 0; i--) { const p = P[i]; while (up.length >= 2 && cross(up[up.length-2], up[up.length-1], p) <= 0) up.pop(); up.push(p); }
  lo.pop(); up.pop();
  return lo.concat(up);
}

function expandHull(hull, frac) {
  if (hull.length === 0) return hull;
  let cx = 0, cy = 0; for (const h of hull) { cx += h[0]; cy += h[1]; } cx /= hull.length; cy /= hull.length;
  return hull.map(h => [cx + (h[0]-cx)*(1+frac), cy + (h[1]-cy)*(1+frac)]);
}

// Project all front-of-camera points through the live camera, then fit the
// result to the frame so the composition stays framed and full as you orbit.
// Returns { pos: [{x,y,wd,p,idx}], distMed }.
function projectAndFitWorlds(W, H) {
  camera.updateMatrixWorld(true);
  const exportCam = new THREE.PerspectiveCamera(camera.fov, W / H, camera.near, camera.far);
  exportCam.position.copy(camera.position);
  exportCam.quaternion.copy(camera.quaternion);
  exportCam.updateMatrixWorld(true);
  exportCam.updateProjectionMatrix();
  const camForward = new THREE.Vector3(); exportCam.getWorldDirection(camForward);
  const camPos = camera.position.clone();
  const tmp = new THREE.Vector3();
  const raw = [];
  for (let i = 0; i < POINTS.length; i++) {
    const p = POINTS[i]; tmp.set(p.x, p.y, p.z);
    const along = tmp.clone().sub(camPos).dot(camForward);
    if (along <= exportCam.near) continue;
    const wd = tmp.distanceTo(camPos);
    tmp.project(exportCam);
    raw.push({ nx: tmp.x, ny: tmp.y, wd, p, idx: i });
  }
  if (raw.length === 0) return { pos: [], distMed: 1 };
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (const r of raw) { if (r.nx<xmin)xmin=r.nx; if(r.nx>xmax)xmax=r.nx; if(r.ny<ymin)ymin=r.ny; if(r.ny>ymax)ymax=r.ny; }
  const pad = Math.min(W, H) * 0.07;
  const bw = Math.max(xmax - xmin, 1e-6), bh = Math.max(ymax - ymin, 1e-6);
  const sc = Math.min((W - 2*pad) / bw, (H - 2*pad) / bh);
  const cxN = (xmin+xmax)/2, cyN = (ymin+ymax)/2;
  const pos = raw.map(r => ({
    x: W/2 + (r.nx - cxN) * sc,
    y: H/2 - (r.ny - cyN) * sc,
    wd: r.wd, p: r.p, idx: r.idx,
  }));
  const ds = raw.map(r => r.wd).sort((a, b) => a - b);
  const distMed = ds[Math.floor(ds.length/2)] || 1;
  return { pos, distMed };
}

// Slider mapping for the worlds composition:
//   sensitivity → number of worlds K
//   smoothness  → boundary looseness (hull expansion)
//   line weight → boundary stroke weight
//   depth fade  → content texture quietness
function worldsParams() {
  const K = Math.round(3 + ((segSensitivity - 0.35) / 0.6) * 9);   // 3..12
  const expand = 0.06 + ((segBlur - 0.5) / 3.0) * 0.26;            // 0.06..0.32
  return { K: Math.max(2, K), expand: Math.max(0.04, Math.min(0.4, expand)) };
}

// True perspective projection of every point through the live camera — NO
// fit-to-frame. This is what lets the camera actually drive the composition:
// zooming changes depth (so sizes change), orbiting changes which worlds are
// near or far. Returns { proj: [{x,y,depth}|null per POINTS index], refDist }.
function projectWorldsPerspective(W, H) {
  camera.updateMatrixWorld(true);
  const exportCam = new THREE.PerspectiveCamera(camera.fov, W / H, camera.near, camera.far);
  exportCam.position.copy(camera.position);
  exportCam.quaternion.copy(camera.quaternion);
  exportCam.updateMatrixWorld(true);
  exportCam.updateProjectionMatrix();
  const camForward = new THREE.Vector3(); exportCam.getWorldDirection(camForward);
  const camPos = camera.position.clone();
  const tmp = new THREE.Vector3();
  const proj = new Array(POINTS.length).fill(null);
  const depths = [];
  for (let i = 0; i < POINTS.length; i++) {
    const p = POINTS[i]; tmp.set(p.x, p.y, p.z);
    const along = tmp.clone().sub(camPos).dot(camForward);
    if (along <= exportCam.near) continue;   // behind camera
    const wd = tmp.distanceTo(camPos);
    tmp.project(exportCam);
    proj[i] = { x: (tmp.x * 0.5 + 0.5) * W, y: (1 - (tmp.y * 0.5 + 0.5)) * H, depth: wd };
    depths.push(wd);
  }
  depths.sort((a, b) => a - b);
  const refDist = depths.length ? depths[Math.floor(depths.length / 2)] : 1;
  return { proj, refDist };
}

// ---------------------------------------------------------------------------
// Worlds boolean union, rewritten to GUARANTEE non-self-intersecting figures.
// A world's members are rasterized into a binary mask, joined by a minimum
// spanning forest of straight chord necks whose width and reach grow with the
// unify amount, welded with a small morphological close, split into connected
// components, and each component's boundary traced by crack-following along the
// mask lattice. A crack-followed boundary is a simple closed loop by
// construction, so no figure can self-cross. Members too far to reach stay as
// separate components, so a world resolves from a fragmented field into one
// continuous figure as unify rises. This is the de l'Orme / Evans boolean union
// of the drawing methodology: members fused by chords cutting across the field,
// kept angular and specific rather than rounded to a blob.
// ---------------------------------------------------------------------------

// Signed area of a ring (screen coords). Magnitude orders outer vs holes.
function polyAreaSigned(p) {
  let a = 0;
  for (let i = 0; i < p.length; i++) { const j = (i + 1) % p.length; a += p[i][0]*p[j][1] - p[j][0]*p[i][1]; }
  return a / 2;
}

// Drop collinear vertices (the crack walk emits one point per unit step).
function ringDropCollinear(poly) {
  const n = poly.length; if (n < 3) return poly;
  const out = [];
  for (let i = 0; i < n; i++) {
    const a = poly[(i-1+n)%n], b = poly[i], c = poly[(i+1)%n];
    const cr = (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]);
    if (Math.abs(cr) > 1e-9) out.push(b);
  }
  return out.length >= 3 ? out : poly;
}

// Does a closed ring self-intersect? O(n^2), fine for boundary sizes.
function ringSelfIntersects(poly) {
  const n = poly.length;
  const ci = (a,b,d) => (d[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(d[0]-a[0]);
  const si = (p1,p2,p3,p4) => ci(p1,p3,p4)*ci(p2,p3,p4) < 0 && ci(p1,p2,p3)*ci(p1,p2,p4) < 0;
  for (let i = 0; i < n; i++) for (let j = i + 2; j < n; j++) {
    if (i === 0 && j === n - 1) continue;
    if (si(poly[i], poly[(i+1)%n], poly[j], poly[(j+1)%n])) return true;
  }
  return false;
}

// Douglas-Peucker on a closed ring, backing off the tolerance if simplification
// would introduce a crossing, so the output stays a simple polygon.
function ringSimplifySafe(ring, eps) {
  if (ring.length < 4) return ring;
  const perp = (p,a,b) => { const dx=b[0]-a[0], dy=b[1]-a[1], L=Math.hypot(dx,dy)||1;
    return Math.abs((p[0]-a[0])*dy - (p[1]-a[1])*dx) / L; };
  function rdp(pts, e) {
    const keep = new Uint8Array(pts.length); keep[0] = 1; keep[pts.length-1] = 1;
    const st = [[0, pts.length-1]];
    while (st.length) { const seg = st.pop(); const s = seg[0], en = seg[1];
      let mx = 0, mi = -1;
      for (let i = s + 1; i < en; i++) { const d = perp(pts[i], pts[s], pts[en]); if (d > mx) { mx = d; mi = i; } }
      if (mx > e && mi > 0) { keep[mi] = 1; st.push([s, mi], [mi, en]); }
    }
    const o = []; for (let i = 0; i < pts.length; i++) if (keep[i]) o.push(pts[i]);
    return o;
  }
  for (let f = 1; f <= 4; f++) {
    const open = ring.slice(); open.push(ring[0]);
    let s = rdp(open, eps / f); s = s.slice(0, s.length - 1);
    if (s.length >= 3 && !ringSelfIntersects(s)) return s;
  }
  return ringDropCollinear(ring);
}

// Scanline-fill a polygon (grid coords) into a binary mask.
function fillPolyMask(mask, gw, gh, pts) {
  if (!pts || pts.length < 3) return;
  let ymin = Infinity, ymax = -Infinity;
  for (const p of pts) { if (p[1] < ymin) ymin = p[1]; if (p[1] > ymax) ymax = p[1]; }
  ymin = Math.max(0, Math.floor(ymin)); ymax = Math.min(gh - 1, Math.ceil(ymax));
  for (let y = ymin; y <= ymax; y++) {
    const yc = y + 0.5; const xs = [];
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i+1) % pts.length];
      if ((a[1] <= yc && b[1] > yc) || (b[1] <= yc && a[1] > yc)) {
        xs.push(a[0] + (yc - a[1]) / (b[1] - a[1]) * (b[0] - a[0]));
      }
    }
    xs.sort((p, q) => p - q);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const x0 = Math.max(0, Math.ceil(xs[k] - 0.5)), x1 = Math.min(gw - 1, Math.floor(xs[k+1] - 0.5));
      for (let x = x0; x <= x1; x++) mask[y * gw + x] = 1;
    }
  }
}

// One pass of a square morphological dilate (max) or erode (min).
function maxMinFilter(mask, gw, gh, r, dilate) {
  const out = new Uint8Array(gw * gh);
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
    let v = dilate ? 0 : 1;
    for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
      const xx = x + dx, yy = y + dy;
      const s = (xx >= 0 && yy >= 0 && xx < gw && yy < gh) ? mask[yy*gw+xx] : 0;
      v = dilate ? Math.max(v, s) : Math.min(v, s);
      if (dilate && v) break;
    }
    out[y*gw+x] = v;
  }
  return out;
}

// Morphological close: weld neck/member junctions and remove single-cell
// pinches that would make boundary tracing ambiguous. Small radius only; this
// is a weld, not the rejected blob-blur.
function closeMaskBin(mask, gw, gh, r) {
  if (r <= 0) return mask;
  return maxMinFilter(maxMinFilter(mask, gw, gh, r, true), gw, gh, r, false);
}

// 4-connected component labels for a binary mask.
function labelMask(mask, gw, gh) {
  const lab = new Int32Array(gw * gh).fill(-1);
  let n = 0; const stack = [];
  for (let i = 0; i < gw * gh; i++) {
    if (!mask[i] || lab[i] >= 0) continue;
    lab[i] = n; stack.length = 0; stack.push(i);
    while (stack.length) {
      const p = stack.pop(); const x = p % gw, y = (p / gw) | 0;
      if (x > 0      && mask[p-1]  && lab[p-1]  < 0) { lab[p-1]  = n; stack.push(p-1); }
      if (x < gw - 1 && mask[p+1]  && lab[p+1]  < 0) { lab[p+1]  = n; stack.push(p+1); }
      if (y > 0      && mask[p-gw] && lab[p-gw] < 0) { lab[p-gw] = n; stack.push(p-gw); }
      if (y < gh - 1 && mask[p+gw] && lab[p+gw] < 0) { lab[p+gw] = n; stack.push(p+gw); }
    }
    n++;
  }
  return { lab, n };
}

// Crack-following contour tracer. Each filled cell emits directed boundary
// edges (interior kept on the right, clockwise in screen coords); chaining
// picks the most-clockwise successor at each lattice corner, hugging the
// interior. Output is a set of simple closed loops (outer + hole boundaries),
// guaranteed not to self-cross.
function traceMaskContours(mask, gw, gh) {
  const at = (x, y) => (x >= 0 && y >= 0 && x < gw && y < gh) ? mask[y*gw+x] : 0;
  const cw = gw + 1, cid = (x, y) => y * cw + x;
  const ea = [], eb = [], eax = [], eay = [], ebx = [], eby = [];
  const push = (ax, ay, bx, by) => { ea.push(cid(ax,ay)); eb.push(cid(bx,by)); eax.push(ax); eay.push(ay); ebx.push(bx); eby.push(by); };
  for (let y = 0; y < gh; y++) for (let x = 0; x < gw; x++) {
    if (!at(x, y)) continue;
    if (!at(x, y-1)) push(x, y, x+1, y);
    if (!at(x+1, y)) push(x+1, y, x+1, y+1);
    if (!at(x, y+1)) push(x+1, y+1, x, y+1);
    if (!at(x-1, y)) push(x, y+1, x, y);
  }
  const m = ea.length;
  const outMap = new Map();
  for (let i = 0; i < m; i++) { const k = ea[i]; if (!outMap.has(k)) outMap.set(k, []); outMap.get(k).push(i); }
  const used = new Uint8Array(m);
  const loops = [];
  for (let s = 0; s < m; s++) {
    if (used[s]) continue;
    const loop = []; let ei = s, guard = 0;
    while (ei >= 0 && !used[ei] && guard++ < m + 5) {
      used[ei] = 1; loop.push([eax[ei], eay[ei]]);
      const inx = Math.sign(ebx[ei]-eax[ei]), iny = Math.sign(eby[ei]-eay[ei]);
      const cands = outMap.get(eb[ei]) || [];
      let best = -1, bs = Infinity;
      for (const cidx of cands) {
        if (used[cidx]) continue;
        const dx = Math.sign(ebx[cidx]-eax[cidx]), dy = Math.sign(eby[cidx]-eay[cidx]);
        const cross = inx*dy - iny*dx, dot = inx*dx + iny*dy;
        let score = cross; if (dot < 0 && Math.abs(cross) < 1e-9) score = 1e6;
        if (score < bs) { bs = score; best = cidx; }
      }
      ei = best;
    }
    if (loop.length >= 4) loops.push(loop);
  }
  return loops;
}

// Disjoint-set for the spanning-forest necks.
function unionFindMake(n) {
  const p = new Int32Array(n); for (let i = 0; i < n; i++) p[i] = i;
  const find = (i) => { while (p[i] !== i) { p[i] = p[p[i]]; i = p[i]; } return i; };
  return { find, union: (a, b) => { a = find(a); b = find(b); if (a !== b) { p[a] = b; return true; } return false; } };
}

// Open-polyline Douglas-Peucker (endpoints pinned). Thins a generated run
// before it is smoothed.
function rdpOpen(pts, eps) {
  if (pts.length < 3) return pts.slice();
  const perp = (p, a, b) => { const dx = b[0]-a[0], dy = b[1]-a[1], L = Math.hypot(dx, dy) || 1;
    return Math.abs((p[0]-a[0])*dy - (p[1]-a[1])*dx) / L; };
  const keep = new Uint8Array(pts.length); keep[0] = 1; keep[pts.length-1] = 1;
  const st = [[0, pts.length-1]];
  while (st.length) { const seg = st.pop(); const s = seg[0], e = seg[1];
    let mx = 0, mi = -1;
    for (let i = s + 1; i < e; i++) { const d = perp(pts[i], pts[s], pts[e]); if (d > mx) { mx = d; mi = i; } }
    if (mx > eps && mi > 0) { keep[mi] = 1; st.push([s, mi], [mi, e]); }
  }
  const o = []; for (let i = 0; i < pts.length; i++) if (keep[i]) o.push(pts[i]);
  return o;
}

// Chaikin corner-cutting on an OPEN run with endpoints pinned. Turns a stepped
// run into a smooth polycurve while staying inside the run's own corridor, so
// it cannot introduce a crossing.
function chaikinOpen(run, iters) {
  let pts = run.slice();
  for (let it = 0; it < iters; it++) {
    if (pts.length < 3) break;
    const out = [pts[0]];
    for (let i = 0; i < pts.length - 1; i++) {
      const a = pts[i], b = pts[i + 1];
      out.push([a[0]*0.75 + b[0]*0.25, a[1]*0.75 + b[1]*0.25]);
      out.push([a[0]*0.25 + b[0]*0.75, a[1]*0.25 + b[1]*0.75]);
    }
    out.push(pts[pts.length - 1]);
    pts = out;
  }
  return pts;
}

// Scanline-fill a polygon (grid coords), tagging each filled cell with the
// member that owns it. Used to trace each boundary stretch back to the member
// whose silhouette produced it.
function fillPolyMaskOwner(owner, gw, gh, pts, id) {
  if (!pts || pts.length < 3) return;
  let ymin = Infinity, ymax = -Infinity;
  for (const p of pts) { if (p[1] < ymin) ymin = p[1]; if (p[1] > ymax) ymax = p[1]; }
  ymin = Math.max(0, Math.floor(ymin)); ymax = Math.min(gh - 1, Math.ceil(ymax));
  for (let y = ymin; y <= ymax; y++) {
    const yc = y + 0.5; const xs = [];
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i+1) % pts.length];
      if ((a[1] <= yc && b[1] > yc) || (b[1] <= yc && a[1] > yc)) xs.push(a[0] + (yc - a[1]) / (b[1] - a[1]) * (b[0] - a[0]));
    }
    xs.sort((p, q) => p - q);
    for (let k = 0; k + 1 < xs.length; k += 2) {
      const x0 = Math.max(0, Math.ceil(xs[k] - 0.5)), x1 = Math.min(gw - 1, Math.floor(xs[k+1] - 0.5));
      for (let x = x0; x <= x1; x++) owner[y * gw + x] = id;
    }
  }
}

// Boolean union per world: one or more connected figures, each a simple,
// non-self-intersecting contour set (outer boundary + interior holes) in screen
// space, with its own mean depth. Members within reach are linked by a minimum
// spanning forest of chord necks; reach, neck width and weld radius grow with
// the unify amount, so a world reads as a fragmented field at low unify and
// solidifies into one fuller figure as unify rises.
//
// Articulation is decided by machine vision, at the scale of a boundary REGION
// rather than a single member. Each boundary vertex inherits the complexity of
// the member it belongs to (Sobel edge density measured at load); that signal
// is filled across the generated necks and smoothed along the boundary so it
// forms contiguous regions. Where smoothed complexity stays high the edge keeps
// its intricate, stepped, digital articulation. Where it drops below
// GESTURAL_FRACTION the whole contiguous region (which may span several quiet
// members and the necks between them) is simplified with a span-scaled epsilon
// into a few control points and smoothed, becoming one large painterly move.
// Those gestural arcs are also returned separately so they can be drawn as a
// guide for hand-painting. A graduated self-intersection guard keeps every
// figure simple. Returns [{ contours, depth, gestures }].
function rasterUnion(mem, unify, minDim) {
  let xmn = Infinity, xmx = -Infinity, ymn = Infinity, ymx = -Infinity, avg = 0;
  for (const m of mem) {
    const h = m.size * 0.6; avg += m.size;
    if (m.cx - h < xmn) xmn = m.cx - h; if (m.cx + h > xmx) xmx = m.cx + h;
    if (m.cy - h < ymn) ymn = m.cy - h; if (m.cy + h > ymx) ymx = m.cy + h;
  }
  avg /= Math.max(1, mem.length);
  const pad = avg * 0.8;
  xmn -= pad; xmx += pad; ymn -= pad; ymx += pad;
  const bw = xmx - xmn, bh = ymx - ymn;
  if (!(bw > 0) || !(bh > 0)) return [];
  const GMAX = 260;
  const gscale = Math.min(GMAX / bw, GMAX / bh, 2.5);
  const gw = Math.max(8, Math.ceil(bw * gscale)), gh = Math.max(8, Math.ceil(bh * gscale));
  const mask = new Uint8Array(gw * gh);
  const owner = new Int32Array(gw * gh).fill(-1);   // member index per cell, -1 = generated
  const toG = (m, p) => [ (m.cx + (p[0]-0.5)*m.size - xmn) * gscale, (m.cy + (p[1]-0.5)*m.size - ymn) * gscale ];
  for (let mi = 0; mi < mem.length; mi++) {
    const m = mem[mi], d = m.dom; if (!d || d.length < 3) continue;
    const poly = d.map(p => toG(m, p));
    fillPolyMask(mask, gw, gh, poly); fillPolyMaskOwner(owner, gw, gh, poly, mi);
  }
  const reach = Math.pow(unify, 1.5) * minDim * 0.7;
  if (reach > 1 && mem.length > 1) {
    const edges = [];
    const lim = Math.min(mem.length, 400);
    for (let i = 0; i < lim; i++) for (let j = i + 1; j < lim; j++) {
      const d = Math.hypot(mem[i].cx - mem[j].cx, mem[i].cy - mem[j].cy);
      if (d <= reach) edges.push([d, i, j]);
    }
    edges.sort((a, b) => a[0] - b[0]);
    const uf = unionFindMake(mem.length);
    for (const e of edges) {
      if (!uf.union(e[1], e[2])) continue;
      const a = mem[e[1]], b = mem[e[2]];
      const ax = (a.cx - xmn) * gscale, ay = (a.cy - ymn) * gscale;
      const bx = (b.cx - xmn) * gscale, by = (b.cy - ymn) * gscale;
      const vx = bx - ax, vy = by - ay, L = Math.hypot(vx, vy) || 1;
      const px = -vy / L, py = vx / L;
      const wScreen = Math.max(minDim * 0.004 * unify, Math.min(a.size, b.size) * (0.12 + unify * 1.1));
      const w = wScreen * gscale;
      fillPolyMask(mask, gw, gh, [
        [ax + px*w/2, ay + py*w/2], [bx + px*w/2, by + py*w/2],
        [bx - px*w/2, by - py*w/2], [ax - px*w/2, ay - py*w/2],
      ]);
    }
  }
  const closeR = Math.max(0, Math.round(gscale * (unify * unify * 3.2)));
  const closed = closeMaskBin(mask, gw, gh, closeR);
  const lm = labelMask(closed, gw, gh);
  const depthSum = new Float64Array(lm.n), depthCnt = new Int32Array(lm.n);
  for (const m of mem) {
    const gx = Math.round((m.cx - xmn) * gscale), gy = Math.round((m.cy - ymn) * gscale);
    if (gx < 0 || gy < 0 || gx >= gw || gy >= gh) continue;
    const c = lm.lab[gy * gw + gx]; if (c < 0) continue;
    depthSum[c] += m.depth; depthCnt[c]++;
  }
  const cellCnt = new Int32Array(lm.n);
  for (let i = 0; i < gw * gh; i++) if (lm.lab[i] >= 0) cellCnt[lm.lab[i]]++;
  const minCells = Math.max(12, gscale * gscale * 6);
  const minDimGrid = minDim * gscale;
  const ownerAt = (vx, vy) => {
    for (const cc of [[vx-1,vy-1],[vx,vy-1],[vx-1,vy],[vx,vy]]) {
      const cx = cc[0], cy = cc[1];
      if (cx < 0 || cy < 0 || cx >= gw || cy >= gh) continue;
      if (closed[cy*gw+cx]) { const o = owner[cy*gw+cx]; if (o >= 0) return o; }
    }
    return -1;
  };
  // assemble a contour: high-complexity regions crisp, low-complexity regions
  // collapsed into large painterly moves. Returns the points and the gestural
  // arcs (grid coords). gstr scales the gestural epsilon for the guard.
  const buildContour = (lp, gstr) => {
    const n = lp.length;
    const cv = new Array(n);
    for (let i = 0; i < n; i++) { const o = ownerAt(lp[i][0], lp[i][1]); cv[i] = (o >= 0) ? mem[o].cplx : null; }
    if (cv.some(v => v !== null)) {
      let last = null;
      for (let s = 0; s < 2 * n; s++) { const i = s % n; if (cv[i] !== null) last = cv[i]; else if (last !== null) cv[i] = last; }
    } else { for (let i = 0; i < n; i++) cv[i] = 0.5; }
    const win = Math.max(2, Math.round(n * 0.04));
    const sm = new Array(n);
    for (let i = 0; i < n; i++) { let acc = 0, cnt = 0; for (let k = -win; k <= win; k++) { acc += cv[(i+k+n)%n]; cnt++; } sm[i] = acc / cnt; }
    const gest = sm.map(v => v < GESTURAL_FRACTION);
    let start = 0;
    for (let i = 0; i < n; i++) { if (gest[i] !== gest[(i-1+n)%n]) { start = i; break; } }
    const out = []; const gestures = [];
    let i = 0;
    while (i < n) {
      const k = gest[(start + i) % n];
      const run = [];
      let j = i;
      while (j < n && gest[(start + j) % n] === k) { run.push(lp[(start + j) % n]); j++; }
      run.push(lp[(start + (j % n)) % n]);
      let seg;
      if (!k) { seg = run; }                                            // articulated: crisp digital steps
      else {                                                            // gestural region: large painterly move
        let xn = Infinity, xx = -Infinity, yn = Infinity, yx = -Infinity;
        for (const p of run) { if (p[0]<xn) xn=p[0]; if (p[0]>xx) xx=p[0]; if (p[1]<yn) yn=p[1]; if (p[1]>yx) yx=p[1]; }
        const diag = Math.hypot(xx - xn, yx - yn);
        const eps = Math.max(minDimGrid * 0.016, diag * 0.09) * gstr;
        seg = chaikinOpen(rdpOpen(run, eps), 3);
        if (seg.length >= 2) gestures.push(seg);
      }
      for (let t = 0; t < seg.length - 1; t++) out.push(seg[t]);
      i = j;
    }
    return { out, gestures };
  };
  const figures = [];
  for (let c = 0; c < lm.n; c++) {
    if (cellCnt[c] < minCells) continue;
    const sub = new Uint8Array(gw * gh);
    for (let i = 0; i < gw * gh; i++) if (lm.lab[i] === c) sub[i] = 1;
    const loops = traceMaskContours(sub, gw, gh);
    const contours = []; const allGest = [];
    for (const lpRaw of loops) {
      const lp = ringDropCollinear(lpRaw);
      if (lp.length < 3) continue;
      let out = null, gst = [];
      for (const gstr of [1.0, 0.6, 0.35]) {
        const cand = buildContour(lp, gstr);
        if (!ringSelfIntersects(cand.out)) { out = cand.out; gst = cand.gestures; break; }
      }
      if (!out) { out = lp.slice(); gst = []; }   // crisp fallback
      if (out.length < 3) continue;
      contours.push(out.map(q => [xmn + q[0] / gscale, ymn + q[1] / gscale]));
      for (const g of gst) allGest.push(g.map(q => [xmn + q[0] / gscale, ymn + q[1] / gscale]));
    }
    if (!contours.length) continue;
    const ord = contours.map((c, i) => i).sort((a, b) => Math.abs(polyAreaSigned(contours[b])) - Math.abs(polyAreaSigned(contours[a])));
    const oc = ord.map(i => contours[i]);
    const depth = depthCnt[c] > 0 ? depthSum[c] / depthCnt[c] : 1;
    figures.push({ contours: oc, depth, gestures: allGest });
  }
  return figures;
}

// Build the dense worlds field for the current camera. Every member is drawn
// at its true projected position and perspective scale, depth-ordered, so the
// canvas fills with overlapping forms at varying scales. Each world carries a
// fill register (solid / halftone / hatch / vstripe) so clusters read as
// distinct families. Returns { forms: [...sorted far→near], links }.
const WORLD_PATTERNS = ['solid', 'halftone', 'hatch', 'vstripe'];

// Architectural depth → lineweight. Near objects read heavy and black; far
// objects read light and grey — the standard line hierarchy of a measured
// drawing, where the foreground is cut with the heaviest pen and depth recedes
// into progressively finer, greyer line. t runs 0 (nearest) to 1 (farthest)
// across the view's depth range; both stroke width and value track it.
function worldsDepthWeight(depth, dr, base) {
  const span = Math.max(dr.far - dr.near, 1e-6);
  let t = (depth - dr.near) / span;
  t = t < 0 ? 0 : (t > 1 ? 1 : t);
  const wfac = 2.6 + (0.5 - 2.6) * t;        // 2.6x near → 0.5x far
  const g = Math.round(175 * t);             // black near → light grey far
  return { w: Math.max(0.18, base * wfac), stroke: `rgb(${g},${g},${g})` };
}
function computeWorlds(W, H) {
  const wp = worldsParams();
  const labels = worldClusterLabels(wp.K);
  const { proj, refDist } = projectWorldsPerspective(W, H);
  if (refDist <= 0) return { forms: [], links: [], bridges: [], unions: [], depthRange: { near: 0, far: 1 } };

  // Hierarchical scale. Each world contributes one PRIMARY form — its medoid,
  // drawn large and scaled by the world's mass — and its other members appear
  // only as smaller SECONDARY texture. This breaks the near-uniform field into
  // a few dominant forms with supporting detail, giving real scale variety and
  // far less clutter. Complexity governs how much secondary texture survives.
  const shapes = computeWorldShapes(wp.K);
  const medoidSet = new Set();
  let minCount = Infinity, maxCount = 0;
  for (const sh of shapes.values()) {
    medoidSet.add(sh.medoidIdx);
    if (sh.count < minCount) minCount = sh.count;
    if (sh.count > maxCount) maxCount = sh.count;
  }
  const countOf = new Map();
  for (const [lab, sh] of shapes) countOf.set(lab, sh.count);
  const cspan = Math.max(1, maxCount - minCount);

  const minDim = Math.min(W, H);
  const base = minDim * 0.06;
  const minS = minDim * 0.022, maxS = minDim * 0.55;
  const cullS = minDim * 0.02;

  const forms = [];
  const sumByWorld = new Map();   // for link anchors (projected centroid)
  const allByWorld = new Map();   // every member's screen silhouette, for union
  for (let i = 0; i < POINTS.length; i++) {
    const pr = proj[i];
    if (!pr) continue;
    const polys = contentPolygons.get(POINTS[i].id);
    if (!polys || !polys.length) continue;
    const lab = labels[i];
    const isMedoid = medoidSet.has(i);
    const cnt = countOf.get(lab) || 1;
    const massF = 0.8 + ((cnt - minCount) / cspan) * 1.7;   // 0.8..2.5 by world mass
    const central = isMedoid ? 1.9 : 0.55;                  // primaries dominate
    let size = base * massF * central * (refDist / pr.depth);
    size = Math.max(minS, Math.min(maxS, size));
    // record every member for the union pass (independent of complexity cull)
    if (worldsUnify > 0.001) {
      if (!allByWorld.has(lab)) allByWorld.set(lab, []);
      allByWorld.get(lab).push({ cx: pr.x, cy: pr.y, size, dom: polys[0], depth: pr.depth, cplx: articulationSignalOf(POINTS[i].id) });
    }
    // complexity controls only the secondary texture population
    const r = ((i * 2654435761) >>> 0) / 4294967296;
    if (!isMedoid && r >= worldsComplexity) continue;
    if (!isMedoid && size < cullS) continue;                // drop tiny far texture
    forms.push({
      cx: pr.x, cy: pr.y, size, depth: pr.depth, lab, idx: i, isMedoid,
      dom: polys[0], full: isMedoid ? polys : null, inner: polys.slice(1, 4),
      pattern: WORLD_PATTERNS[lab % WORLD_PATTERNS.length],
    });
    const acc = sumByWorld.get(lab) || { x: 0, y: 0, n: 0 };
    acc.x += pr.x; acc.y += pr.y; acc.n++; sumByWorld.set(lab, acc);
  }
  forms.sort((a, b) => b.depth - a.depth);   // far first (painter's algorithm)

  // Boolean union per world. rasterUnion rasterizes the world's member
  // silhouettes, links those within reach by a spanning forest of chord necks,
  // welds with a small close, and traces each connected component as a simple
  // (non-self-intersecting) figure. The unify amount sets reach and neck width,
  // so the world resolves from a fragmented field of separate components into
  // one continuous figure as unify rises. This is the de l'Orme / Evans
  // boolean-union operation applied to the projected world. Each component is
  // its own figure carrying its own mean depth, tagged with the world label so
  // the SVG can group a world's figures into one editable layer.
  const unions = [];
  if (worldsUnify > 0.001) {
    for (const [lab, mem] of allByWorld) {
      if (mem.length === 0) continue;
      const figs = rasterUnion(mem, worldsUnify, minDim);
      for (const f of figs) {
        if (!f.contours || !f.contours.length) continue;
        unions.push({
          contours: f.contours, depth: f.depth, lab,
          pattern: WORLD_PATTERNS[lab % WORLD_PATTERNS.length],
        });
      }
    }
    unions.sort((a, b) => b.depth - a.depth);   // far first (painter's algorithm)
  }

  // Within-world chord bridges. As complexity drops, members of the SAME world
  // join their nearest same-world neighbours with straight filled chords, so
  // each world coalesces into one conjoined regional mass. Bridges only form
  // inside a world — the regions thicken into distinct characters before any
  // cross-region fusion. Grid-accelerated.
  const bridges = [];
  const mergeRadius = (1 - worldsComplexity) * minDim * 0.28;
  if (mergeRadius > 1 && forms.length > 1) {
    const cell = mergeRadius;
    const grid = new Map();
    const keyOf = (x, y) => Math.floor(x / cell) + ',' + Math.floor(y / cell);
    for (let i = 0; i < forms.length; i++) {
      const k = keyOf(forms[i].cx, forms[i].cy);
      if (!grid.has(k)) grid.set(k, []);
      grid.get(k).push(i);
    }
    const seen = new Set();
    for (let i = 0; i < forms.length; i++) {
      const f = forms[i];
      const gx = Math.floor(f.cx / cell), gy = Math.floor(f.cy / cell);
      const cand = [];
      for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
        const arr = grid.get((gx + dx) + ',' + (gy + dy));
        if (!arr) continue;
        for (const j of arr) {
          if (j === i || forms[j].lab !== f.lab) continue;   // same world only
          const d = Math.hypot(forms[j].cx - f.cx, forms[j].cy - f.cy);
          if (d <= mergeRadius) cand.push([d, j]);
        }
      }
      cand.sort((a, b) => a[0] - b[0]);
      for (let n = 0; n < Math.min(2, cand.length); n++) {
        const j = cand[n][1];
        const key = i < j ? i + '_' + j : j + '_' + i;
        if (seen.has(key)) continue; seen.add(key);
        const g = forms[j];
        const w = Math.max(2, Math.min(f.size, g.size) * 0.42);
        const depth = Math.max(f.depth, g.depth);
        bridges.push({ ax: f.cx, ay: f.cy, bx: g.cx, by: g.cy, w, pattern: f.pattern, depth });
      }
    }
    bridges.sort((a, b) => b.depth - a.depth);
  }

  // Interconnect world centroids (projected), each to its 2 nearest.
  const centers = [...sumByWorld.values()].filter(a => a.n > 0).map(a => ({ x: a.x / a.n, y: a.y / a.n }));
  const links = [];
  const seen = new Set();
  for (let i = 0; i < centers.length; i++) {
    const d = centers.map((c, j) => [(c.x - centers[i].x) ** 2 + (c.y - centers[i].y) ** 2, j])
      .sort((a, b) => a[0] - b[0]).slice(1, 3);
    for (const [, j] of d) {
      const key = i < j ? i + '-' + j : j + '-' + i;
      if (seen.has(key)) continue; seen.add(key);
      const A = centers[i], B = centers[j];
      const mx = (A.x + B.x) / 2 - (B.y - A.y) * 0.12;
      const my = (A.y + B.y) / 2 + (B.x - A.x) * 0.12;
      links.push(catmullRomPoly([[A.x, A.y], [mx, my], [B.x, B.y]], false, 14));
    }
  }
  // Depth range across the rendered worlds objects, for the architectural
  // depth → lineweight mapping (near = heavy/dark, far = light/thin).
  let near = Infinity, far = -Infinity;
  for (const f of forms) { if (f.depth < near) near = f.depth; if (f.depth > far) far = f.depth; }
  for (const u of unions) { if (u.depth < near) near = u.depth; if (u.depth > far) far = u.depth; }
  if (!isFinite(near) || !isFinite(far)) { near = 0; far = 1; }
  const depthRange = { near, far };
  return { forms, links, bridges, unions, depthRange };
}

// Compute the collapsed-perspective layout: a list of placements
// {fx, fy, size, alpha, id} where each image's content is positioned by its
// latent (x,y) and rendered at two gentle global scales for a quiet sense of
// multi-scale depth. This is the TEXTURE layer; the bold synthesized gesture
// curves are drawn over it (see computeCurveGestures). The texture is kept
// deliberately quiet so the gestures carry the composition, the way Oehlen's
// spray-painted curves sit over a halftone diagram field.
function collapsedMappers(W, H) {
  const b = getLatentBounds();
  const pad = Math.min(W, H) * 0.06;
  const spanX = Math.max(b.xmax - b.xmin, 1e-6);
  const spanY = Math.max(b.ymax - b.ymin, 1e-6);
  const spanZ = Math.max(b.zmax - b.zmin, 1e-6);
  return {
    mapX: (x) => pad + ((x - b.xmin) / spanX) * (W - 2 * pad),
    mapY: (y) => pad + ((y - b.ymin) / spanY) * (H - 2 * pad),
    normZ: (z) => (z - b.zmin) / spanZ,
  };
}

// Base 2D anchor positions (one per point, no octave scaling) — the input to
// the curve synthesizer.
function collapsedBasePositions(W, H) {
  const m = collapsedMappers(W, H);
  return POINTS.map(p => [m.mapX(p.x), m.mapY(p.y)]);
}

function computeCollapsedField(W, H) {
  const m = collapsedMappers(W, H);
  const cx = W / 2, cy = H / 2;
  const variation = Math.max(0, Math.min(1, (segSensitivity - 0.35) / 0.6));
  // Two gentle octaves: a slightly compressed background pass and a normal
  // foreground pass. Subtle, not the busy 5-octave stack that read as mud.
  const octs = [{ g: 0.78, a: 0.6 }, { g: 1.12, a: 1.0 }];
  const baseImg = Math.min(W, H) * 0.06;
  const items = [];
  for (const oc of octs) {
    for (const p of POINTS) {
      const bx = m.mapX(p.x), by = m.mapY(p.y);
      const fx = cx + (bx - cx) * oc.g;
      const fy = cy + (by - cy) * oc.g;
      const zs = 1.0 + variation * (m.normZ(p.z) - 0.5) * 1.6;
      const size = baseImg * oc.g * Math.max(0.3, zs);
      if (size < 6) continue;
      if (fx + size < 0 || fx - size > W || fy + size < 0 || fy - size > H) continue;
      items.push({ fx, fy, size, alpha: oc.a, id: p.id });
    }
  }
  items.sort((a, b2) => a.size - b2.size);
  return items;
}

// Slider mapping for the collapsed/Oehlen field:
//   sensitivity → cluster count K (gesture complexity)
//   smoothness  → connecting-arc density
//   line weight → gesture stroke weight
//   depth fade  → how quiet the texture is (higher = quieter)
function collapsedGestureParams() {
  const K = Math.round(5 + ((segSensitivity - 0.35) / 0.6) * 13);   // 5..18
  const arcDensity = Math.round((segBlur - 0.5) / 3.0 * 3);          // 0..3
  return { K: Math.max(3, K), arcDensity: Math.max(0, Math.min(3, arcDensity)) };
}

// Per-image content polygons. For each photograph, we trace closed contour
// shapes from its Sobel gradient — these are the "segmentation regions"
// expressed as polygons rather than lines, so they can be filled with
// patterns. Stored in image-relative [0,1] × [0,1] coordinates and projected
// at render time at each sprite's current screen rect.
const contentPolygons = new Map();   // p.id -> [polygon, polygon, ...]

function simplifyPolyline(pts, tolerance) {
  // Douglas–Peucker polyline simplification. Reduces a noisy contour to its
  // essential vertices so we don't carry hundreds of points per shape.
  if (pts.length <= 2) return pts.slice();
  function distToLine(p, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const len2 = dx * dx + dy * dy;
    if (len2 < 1e-12) return Math.hypot(p[0] - a[0], p[1] - a[1]);
    let t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2;
    if (t < 0) t = 0; else if (t > 1) t = 1;
    const projX = a[0] + t * dx, projY = a[1] + t * dy;
    return Math.hypot(p[0] - projX, p[1] - projY);
  }
  const keep = new Uint8Array(pts.length);
  keep[0] = 1; keep[pts.length - 1] = 1;
  // Iterative stack-based DP (avoids recursion depth issues on long paths)
  const stack = [[0, pts.length - 1]];
  while (stack.length) {
    const [s, e] = stack.pop();
    let maxD = 0, maxI = -1;
    for (let i = s + 1; i < e; i++) {
      const d = distToLine(pts[i], pts[s], pts[e]);
      if (d > maxD) { maxD = d; maxI = i; }
    }
    if (maxD > tolerance && maxI > 0) {
      keep[maxI] = 1;
      stack.push([s, maxI]); stack.push([maxI, e]);
    }
  }
  const out = [];
  for (let i = 0; i < pts.length; i++) if (keep[i]) out.push(pts[i]);
  return out;
}

function extractContentPolygonsFromGrad(grad, w, h) {
  // Given an already-computed gradient field (with border margin zeroed),
  // trace closed contour polygons in image-relative [0,1] × [0,1] coords,
  // rejecting frame-hugging polygons. Shared by the combined preprocessor.
  let maxG = 0;
  for (let i = 0; i < grad.length; i++) if (grad[i] > maxG) maxG = grad[i];
  if (maxG < 1) return [];
  const level = 0.30 * maxG;
  const segs = marchingSquaresAt(grad, w, h, level, 1, 1);
  const paths = stitchSegmentsToPaths(segs, 0.5);
  const margin = Math.max(3, Math.round(Math.min(w, h) * 0.07));
  const polys = [];
  for (const pts of paths) {
    if (pts.length < 6) continue;
    let bxmin = Infinity, bymin = Infinity, bxmax = -Infinity, bymax = -Infinity;
    let nearBoundary = 0;
    const bx = margin + 2, by = margin + 2;
    for (const p of pts) {
      if (p[0] < bxmin) bxmin = p[0];
      if (p[0] > bxmax) bxmax = p[0];
      if (p[1] < bymin) bymin = p[1];
      if (p[1] > bymax) bymax = p[1];
      if (p[0] <= bx || p[0] >= w - bx || p[1] <= by || p[1] >= h - by) nearBoundary++;
    }
    const spanX = (bxmax - bxmin) / w, spanY = (bymax - bymin) / h;
    const boundaryFrac = nearBoundary / pts.length;
    if (spanX > 0.85 && spanY > 0.85 && boundaryFrac > 0.6) continue;  // frame tracer
    const simplified = simplifyPolyline(pts, 0.8);
    if (simplified.length < 3) continue;
    // significance = perimeter span; used to keep only the dominant forms
    const sig = (bxmax - bxmin) + (bymax - bymin);
    polys.push({ pts: simplified.map(p => [p[0] / w, p[1] / h]), sig });
  }
  // Keep only the most significant contours per image. The long tail is noise
  // (window mullions, texture speckle) that turns the stacked field muddy.
  polys.sort((a, b) => b.sig - a.sig);
  return polys.slice(0, 6).map(o => o.pts);
}

// Combined preprocessor: one Sobel pass per image yields BOTH the normalized
// edge-map canvas and the content polygons. Halves the work vs computing the
// gradient twice, and guarantees the two outputs populate together.
function preprocessImage(srcImg) {
  const w = srcImg.naturalWidth || srcImg.width;
  const h = srcImg.naturalHeight || srcImg.height;
  if (w < 8 || h < 8) return { edgeCanvas: null, polys: [], complexity: 0.5 };
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.drawImage(srcImg, 0, 0);
  const imd = ctx.getImageData(0, 0, w, h);
  const { gray } = imageDataToGrayscale(imd);
  const pre = gaussianBlurFloat(gray, w, h, EDGE_MAP_BLUR_SIGMA);
  const grad = sobelMagnitude(pre, w, h);

  // Normalized edge-map canvas: BLACK pixels with alpha = edge intensity.
  // This composites as black linework directly over a white background (no
  // inversion needed) and overlapping edges simply stay black. Used both for
  // the segmentation raster draw and (inverted internally) the gesture
  // overlay.
  let maxG = 0;
  for (let i = 0; i < grad.length; i++) if (grad[i] > maxG) maxG = grad[i];
  const norm = (maxG < 1) ? 1 : maxG;
  const out = ctx.createImageData(w, h);
  let strong = 0;
  const edgeThresh = norm * 0.2;        // "significant edge" relative to this image
  for (let i = 0, j = 0; i < grad.length; i++, j += 4) {
    const v = Math.min(255, Math.floor(255 * grad[i] / norm));
    out.data[j] = 0; out.data[j + 1] = 0; out.data[j + 2] = 0; out.data[j + 3] = v;
    if (grad[i] > edgeThresh) strong++;
  }
  ctx.putImageData(out, 0, 0);
  // visual complexity = share of the image carrying significant edge structure
  const complexity = grad.length > 0 ? strong / grad.length : 0;

  // Content polygons: zero the border margin on a COPY of the gradient so the
  // frame ring never forms, then trace.
  const gradCopy = grad.slice();
  const margin = Math.max(3, Math.round(Math.min(w, h) * 0.07));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x < margin || x >= w - margin || y < margin || y >= h - margin) {
        gradCopy[y * w + x] = 0;
      }
    }
  }
  const polys = extractContentPolygonsFromGrad(gradCopy, w, h);
  return { edgeCanvas: c, polys, complexity };
}

function precomputeAllEdgeMaps() {
  edgeMaps.clear();
  contentPolygons.clear();
  edgeComplexity.clear();
  complexityRankCache = null;
  edgeMapsReady = 0;
  edgeMapsTotal = 0;
  let polyTotal = 0;
  for (const p of POINTS) {
    if (!p.img) continue;
    edgeMapsTotal++;
    const img = new Image();
    img.onload = () => {
      try {
        const { edgeCanvas, polys, complexity } = preprocessImage(img);
        if (edgeCanvas) edgeMaps.set(p.id, edgeCanvas);
        edgeComplexity.set(p.id, complexity);
        complexityRankCache = null;   // rank rebuilds on next worlds render
        if (polys && polys.length > 0) {
          contentPolygons.set(p.id, polys);
          polyTotal += polys.length;
        }
      } catch (e) { console.warn('image preproc failed for', p.id, e); }
      edgeMapsReady++;
      if (currentMode === 'linework' &&
          (lineworkStyle === 'segmentation' ||
           lineworkStyle === 'composition' ||
           lineworkStyle === 'collapsed_field' ||
           lineworkStyle === 'stacked_perspectives')) {
        markOverlayDirty(250);
      }
    };
    img.onerror = () => { edgeMapsReady++; };
    img.src = 'data:image/png;base64,' + p.img;
  }
}

// Composite the cached edge maps at the projected screen positions, then
// threshold + marching-squares to extract contour paths. Output is in the
// processing-canvas coordinate space; callers scale to wherever they need.
function computeSegmentationPaths(processWidth, targetW, targetH) {
  if (edgeMaps.size === 0) {
    // First-ever call — kick off the async precomputation
    if (edgeMapsTotal === 0) precomputeAllEdgeMaps();
    return null;
  }
  const viewW = (targetW != null) ? targetW : W();
  const viewH = (targetH != null) ? targetH : H();
  const scale = Math.min(1, processWidth / viewW);
  const captureW = Math.max(2, Math.floor(viewW * scale));
  const captureH = Math.max(2, Math.floor(viewH * scale));

  // Project at the target aspect ratio so the result matches the canvas
  // (or SVG) we're going to draw it onto.
  const pv = projectVisiblePoints(viewW, viewH);
  if (pv.projected.length === 0) return null;

  // Composite all edge maps onto a fresh canvas. Edge maps are now black
  // pixels with alpha = edge intensity, so we draw over white with normal
  // blending; edges darken the canvas, overlaps stay dark.
  const composite = document.createElement('canvas');
  composite.width = captureW;
  composite.height = captureH;
  const cctx = composite.getContext('2d');
  cctx.fillStyle = 'white';
  cctx.fillRect(0, 0, captureW, captureH);
  cctx.globalCompositeOperation = 'source-over';

  const worldSize = baseScale * userScale;
  const tanHalfFov = Math.tan((pv.exportCam.fov * Math.PI / 180) / 2);
  const sx = captureW / viewW;
  const sy = captureH / viewH;

  let composited = 0;
  for (const q of pv.projected) {
    const edge = edgeMaps.get(q.p.id);
    if (!edge) continue;
    const pixelHeight = (worldSize * viewH) / (2 * q.worldDist * tanHalfFov);
    const ar = (q.p.exp_ar != null) ? q.p.exp_ar : 1.0;
    const pw = (ar >= 1) ? pixelHeight : pixelHeight * ar;
    const ph = (ar >= 1) ? pixelHeight / ar : pixelHeight;
    const x = (q.sx - pw / 2) * sx;
    const y = (q.sy - ph / 2) * sy;
    cctx.drawImage(edge, x, y, pw * sx, ph * sy);
    composited++;
  }
  if (composited === 0) return { paths: [], width: captureW, height: captureH };

  // Read pixels and INVERT: edges are dark (low luminance), background is
  // white (high). The field should be high where edges are, so field = 255 - lum.
  const imd = cctx.getImageData(0, 0, captureW, captureH);
  let field = new Float32Array(captureW * captureH);
  for (let i = 0, j = 0; i < field.length; i++, j += 4) {
    const lum = Math.max(imd.data[j], imd.data[j + 1], imd.data[j + 2]);
    field[i] = 255 - lum;
  }
  if (segBlur > 0.3) {
    field = gaussianBlurFloat(field, captureW, captureH, segBlur);
  }

  let maxV = 0;
  for (let i = 0; i < field.length; i++) if (field[i] > maxV) maxV = field[i];
  if (maxV < 1) return { paths: [], width: captureW, height: captureH };

  // Percentile-based threshold. The previous formula was (1 - sens) * max,
  // which broke when a few pixels had extreme values (the whole rest of the
  // field then sat under the threshold and nothing got traced). Sorting the
  // non-zero distribution and indexing into it gives a threshold that
  // tracks the actual data shape regardless of outliers.
  //   sensitivity 0.95 → 5th percentile (catches most accumulations)
  //   sensitivity 0.50 → 50th percentile (median; balanced)
  //   sensitivity 0.35 → 65th percentile (only stronger agreement)
  const nonzero = [];
  for (let i = 0; i < field.length; i++) {
    if (field[i] > maxV * 0.005) nonzero.push(field[i]);
  }
  if (nonzero.length < 16) return { paths: [], width: captureW, height: captureH };
  nonzero.sort((a, b) => a - b);
  const targetP = Math.max(0.05, Math.min(0.95, 1.0 - segSensitivity));
  const threshold = nonzero[Math.floor(targetP * nonzero.length)];
  const segs = marchingSquaresAt(field, captureW, captureH, threshold, 1, 1);
  const paths = stitchSegmentsToPaths(segs, 0.5);
  return { paths, width: captureW, height: captureH };
}


// Visible vs. hidden intervals of a segment against a set of axis-aligned
// occluder rectangles. Returns both lists as [t0, t1] pairs along the seg
// parameter. Used for 'ghosted' linework — draw visible solid, hidden dashed.
function visibleHiddenIntervals(seg, occluders) {
  const dx = seg.x2 - seg.x1, dy = seg.y2 - seg.y1;
  const hidden = [];
  for (const r of occluders) {
    const p = [-dx, dx, -dy, dy];
    const q = [seg.x1 - r.xmin, r.xmax - seg.x1,
               seg.y1 - r.ymin, r.ymax - seg.y1];
    let tEnter = 0, tExit = 1, parallelOutside = false;
    for (let i = 0; i < 4; i++) {
      if (p[i] === 0) {
        if (q[i] < 0) { parallelOutside = true; break; }
        continue;
      }
      const t = q[i] / p[i];
      if (p[i] < 0) { if (t > tEnter) tEnter = t; }
      else          { if (t < tExit)  tExit  = t; }
    }
    if (parallelOutside || tEnter >= tExit) continue;
    hidden.push([Math.max(0, tEnter), Math.min(1, tExit)]);
  }
  hidden.sort((a, b) => a[0] - b[0]);
  // merge
  const merged = [];
  for (const iv of hidden) {
    if (merged.length === 0 || iv[0] > merged[merged.length - 1][1]) {
      merged.push([iv[0], iv[1]]);
    } else {
      merged[merged.length - 1][1] = Math.max(merged[merged.length - 1][1], iv[1]);
    }
  }
  // visible = [0, 1] - merged
  const visible = [];
  let cursor = 0;
  for (const [h0, h1] of merged) {
    if (h0 > cursor + 1e-6) visible.push([cursor, h0]);
    cursor = Math.max(cursor, h1);
  }
  if (cursor < 1 - 1e-6) visible.push([cursor, 1]);
  return { visible, hidden: merged };
}

function intervalToSegment(seg, t0, t1) {
  const dx = seg.x2 - seg.x1, dy = seg.y2 - seg.y1;
  return {
    x1: seg.x1 + t0 * dx, y1: seg.y1 + t0 * dy,
    x2: seg.x1 + t1 * dx, y2: seg.y1 + t1 * dy,
  };
}

function rectEdges(r) {
  return [
    { x1: r.xmin, y1: r.ymin, x2: r.xmax, y2: r.ymin, depth: r.depth },
    { x1: r.xmax, y1: r.ymin, x2: r.xmax, y2: r.ymax, depth: r.depth },
    { x1: r.xmax, y1: r.ymax, x2: r.xmin, y2: r.ymax, depth: r.depth },
    { x1: r.xmin, y1: r.ymax, x2: r.xmin, y2: r.ymin, depth: r.depth },
  ];
}

function hiddenLineRemoval(rects) {
  // rects is array of {xmin, xmax, ymin, ymax, depth}
  // Sort back-to-front: largest depth first; later in the array = closer to
  // camera = occluder of everything before it.
  const sorted = rects.slice().sort((a, b) => b.depth - a.depth);
  const survivors = [];
  for (let i = 0; i < sorted.length; i++) {
    const r = sorted[i];
    const edges = [
      { x1: r.xmin, y1: r.ymin, x2: r.xmax, y2: r.ymin }, // top
      { x1: r.xmax, y1: r.ymin, x2: r.xmax, y2: r.ymax }, // right
      { x1: r.xmax, y1: r.ymax, x2: r.xmin, y2: r.ymax }, // bottom
      { x1: r.xmin, y1: r.ymax, x2: r.xmin, y2: r.ymin }, // left
    ];
    for (const e of edges) {
      let parts = [e];
      for (let j = i + 1; j < sorted.length; j++) {
        const occluder = sorted[j];
        const next = [];
        for (const part of parts) {
          for (const s of clipSegmentByRect(part, occluder)) next.push(s);
        }
        parts = next;
        if (parts.length === 0) break;
      }
      for (const s of parts) survivors.push(s);
    }
  }
  return survivors;
}

function assembleSVG(W, H, layerOrder, layers, blendMode, defsXml) {
  const parts = [
    '<?xml version="1.0" encoding="UTF-8"?>',
    `<svg xmlns="http://www.w3.org/2000/svg" ` +
    `xmlns:xlink="http://www.w3.org/1999/xlink" ` +
    `xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" ` +
    `width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`,
    `<title>latent_3d_view</title>`,
  ];
  if (defsXml && defsXml.length > 0) {
    parts.push(`<defs>${defsXml}</defs>`);
  }
  for (const name of layerOrder) {
    // mix-blend-mode applies only to the point-bearing layers, not to
    // the background plate or the edges layer (those stay normal).
    const isPointLayer = (name !== "background" && name !== "edges");
    const styleAttr = (isPointLayer && blendMode && blendMode !== 'normal')
      ? ` style="mix-blend-mode:${blendMode}"`
      : '';
    parts.push(`<g id="${safeId(name)}" inkscape:label="${xmlEscape(name)}" inkscape:groupmode="layer"${styleAttr}>`);
    parts.push(layers.get(name).join('\n'));
    parts.push('</g>');
  }
  parts.push('</svg>');
  return parts.join('\n');
}

document.getElementById('export').onclick = () => {
  const btn = document.getElementById('export');
  const original = btn.textContent;
  btn.textContent = 'Rendering...';
  setTimeout(() => {
    try {
      const result = (currentMode === 'heatmap') ? buildHeatmapSVG()
                  : (currentMode === 'linework') ? buildLineworkSVG()
                  :                                 buildSVGFromCurrentView();
      const blob = new Blob([result.svg], { type: 'image/svg+xml' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      const elev = result.elev.toFixed(0);
      const azim = result.azim.toFixed(0);
      a.download = `latent_3d_${currentMode}_${ts}_e${elev}_a${azim}.svg`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      btn.textContent = 'Downloaded';
      const cullStr = (result.culledBehind + result.culledOutside > 0)
        ? ` (${result.culledBehind} behind, ${result.culledOutside} off-frame)`
        : '';
      document.getElementById('lastExport').textContent =
        `${currentMode} · elev=${elev}° azim=${azim}° · ` +
        `${result.kept}/${result.total} pts${cullStr} · depth ${result.distSpan.toFixed(2)}`;
      console.log('[latent_explorer] exported:', {
        mode: currentMode,
        elev: parseFloat(elev), azim: parseFloat(azim),
        blend: currentBlend, opacity: currentOpacity,
        pointsKept: result.kept, pointsTotal: result.total,
        culledBehind: result.culledBehind,
        culledOutside: result.culledOutside,
        worldDepthSpan: result.distSpan,
      });
    } catch (e) {
      console.error(e);
      btn.textContent = 'Error: ' + e.message;
    }
    setTimeout(() => { btn.textContent = original; }, 1800);
  }, 10);
};

// Reset button
document.getElementById('reset').onclick = () => {
  camera.fov = 45;
  camera.position.set(cx, cy + radius * 0.5, cz + radius * 2.5);
  controls.target.set(cx, cy, cz);
  camera.updateProjectionMatrix();
  controls.update();
  const lensEl = document.getElementById('lensRange');
  if (lensEl) { lensEl.value = 45; document.getElementById('lensValue').textContent = '45\u00B0'; }
  markOverlayDirty();
};

// Resize handling
window.addEventListener('resize', () => {
  camera.aspect = W() / H();
  camera.updateProjectionMatrix();
  renderer.setSize(W(), H());
});

// Animate
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  updateOverlay();
  renderer.render(scene, camera);
  // Overlay redraws are debounced via overlayTimer below; the animate loop
  // doesn't trigger them directly anymore.
}
animate();

// Kick off async per-image edge-map precomputation. Each photograph's
// Sobel result is cached as a grayscale canvas. The first time the user
// switches to linework + segmentation, these are ready (or filling in).
precomputeAllEdgeMaps();

// ---------------------------------------------------------------------------
// 3D enclosing volume (metaballs + marching cubes).
//
// Same idea as the 2D boundary curve, but in three dimensions: each data
// point is a small Gaussian "blob" in space; nearby blobs merge into a
// connected isosurface that follows the outer envelope of the cloud, with
// concavities where regions are sparse. three.js's MarchingCubes primitive
// handles the grid evaluation and mesh extraction.
//
// The volume is a regular Object3D, lit by the directional+ambient lights
// above, and translucent — so sprites inside remain visible. As the user
// orbits, the directional light reveals the surface's form.
// ---------------------------------------------------------------------------
let showVolume = false;
let volumeResolution = 40;
let volumeStrength = 0.3;
let volumeOpacity = 0.45;
let volumeStyle = 'solid';
let volumeColorHex = '#e6e6e6';

let volumeObj = null;          // the MarchingCubes mesh in the scene
let volumeWire = null;         // separate wireframe overlay (for 'both' style)
let volumeRebuildTimer = null;

// Padding factor: the MC local space [0,1] is scaled to cover
// (cloud-bbox * pad). pad > 1 leaves room for the surface to extend
// outward without being clipped at the grid edge.
const VOLUME_PAD = 1.25;

function volumeMaterial() {
  const c = new THREE.Color(volumeColorHex);
  return new THREE.MeshPhongMaterial({
    color: c,
    specular: 0x202020,
    shininess: 18,
    flatShading: false,
    transparent: volumeOpacity < 0.999,
    opacity: volumeOpacity,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
}
function volumeWireMaterial() {
  return new THREE.MeshBasicMaterial({
    color: 0x000000,
    wireframe: true,
    transparent: true,
    opacity: 0.35,
    depthWrite: false,
  });
}

function rebuildVolumeObject() {
  // Destroy + recreate the MarchingCubes instance (resolution is fixed at
  // construction time, so any res change goes through this path).
  if (volumeObj) {
    scene.remove(volumeObj);
    volumeObj.material.dispose();
    volumeObj = null;
  }
  if (volumeWire) {
    scene.remove(volumeWire);
    volumeWire.material.dispose();
    volumeWire = null;
  }
  if (!showVolume) return;

  const mat = volumeMaterial();
  // enableUvs=false, enableColors=false, maxPolyCount=200000 (~6.5MB of verts)
  const obj = new MarchingCubes(volumeResolution, mat, true, false, 200000);
  // MC's local vertex space is [-1, +1] per axis. Setting Object3D scale to
  // (radius * VOLUME_PAD) makes local [-1, 1] map to world [centroid ± rp],
  // i.e. a box that's VOLUME_PAD * radius from centroid on each side.
  // addBall uses [0, 1] coords; the inverse mapping happens in
  // updateVolumeBalls via `span = 2 * radius * VOLUME_PAD`.
  const objScale = radius * VOLUME_PAD;
  obj.position.set(cx, cy, cz);
  obj.scale.set(objScale, objScale, objScale);
  obj.isolation = 80;
  // Render the volume after sprites so sprite alpha shows through it.
  obj.renderOrder = 2;
  scene.add(obj);
  volumeObj = obj;

  if (volumeStyle === 'both') {
    const wmat = volumeWireMaterial();
    const wobj = new MarchingCubes(volumeResolution, wmat, true, false, 200000);
    wobj.position.copy(obj.position);
    wobj.scale.copy(obj.scale);
    wobj.isolation = obj.isolation;
    wobj.renderOrder = 3;
    scene.add(wobj);
    volumeWire = wobj;
  }
  updateVolumeBalls();
}

function updateVolumeBalls() {
  if (!showVolume || !volumeObj) return;
  // Populate metaballs into all visible MC instances
  const fillBalls = (mc) => {
    mc.reset();
    const span = 2 * radius * VOLUME_PAD;
    for (const p of POINTS) {
      const lx = (p.x - cx) / span + 0.5;
      const ly = (p.y - cy) / span + 0.5;
      const lz = (p.z - cz) / span + 0.5;
      if (lx < 0 || lx > 1 || ly < 0 || ly > 1 || lz < 0 || lz > 1) continue;
      // subtract value: keep small so distant points don't bleed everything
      // together. Tweak in concert with volumeStrength.
      mc.addBall(lx, ly, lz, volumeStrength, 12);
    }
    mc.update();
  };
  fillBalls(volumeObj);
  if (volumeWire) fillBalls(volumeWire);
}

function applyVolumeStyle() {
  if (!volumeObj) return;
  // Toggle solid mesh visibility based on style
  volumeObj.visible = (volumeStyle !== 'wireframe');
  // If we should have a wire overlay and don't, rebuild; if we shouldn't and do, rebuild
  const wantWire = (volumeStyle === 'wireframe' || volumeStyle === 'both');
  const haveWire = (volumeWire !== null);
  if (wantWire !== haveWire) {
    // Need to rebuild to add/remove the wire layer
    scheduleVolumeRebuild(0);
    return;
  }
  if (volumeWire) {
    volumeWire.visible = true;
  }
  // For pure wireframe mode, hide the solid and show only the wire layer
  if (volumeStyle === 'wireframe' && volumeWire) {
    volumeObj.visible = false;
    volumeWire.visible = true;
  }
}

function applyVolumeMaterial() {
  if (!volumeObj) return;
  const c = new THREE.Color(volumeColorHex);
  volumeObj.material.color.copy(c);
  volumeObj.material.opacity = volumeOpacity;
  volumeObj.material.transparent = volumeOpacity < 0.999;
  volumeObj.material.needsUpdate = true;
}

function scheduleVolumeRebuild(delayMs) {
  const d = (delayMs == null) ? 100 : delayMs;
  if (volumeRebuildTimer !== null) clearTimeout(volumeRebuildTimer);
  volumeRebuildTimer = setTimeout(() => {
    volumeRebuildTimer = null;
    rebuildVolumeObject();
  }, d);
}

// UI hookup -----------------------------------------------------------------
document.getElementById('showVolume').addEventListener('change', (e) => {
  showVolume = e.target.checked;
  document.getElementById('volumeControls').style.display =
    showVolume ? 'block' : 'none';
  scheduleVolumeRebuild(0);
});
document.getElementById('showContours').addEventListener('change', (e) => {
  showContours = e.target.checked;
  document.getElementById('contourControls').style.display = showContours ? 'block' : 'none';
  rebuildLatentContours();
  markOverlayDirty();
});
document.getElementById('contourLevelRange').addEventListener('input', (e) => {
  contourLevelFrac = parseFloat(e.target.value);
  document.getElementById('contourLevelValue').textContent = contourLevelFrac.toFixed(2);
  contourGeomCache = null;
  if (showContours) rebuildLatentContours();
  markOverlayDirty();
});
document.getElementById('contourOpacityRange').addEventListener('input', (e) => {
  contourOpacity = parseFloat(e.target.value);
  document.getElementById('contourOpacityValue').textContent = contourOpacity.toFixed(2);
  if (latentContoursObj) latentContoursObj.material.opacity = contourOpacity;
  markOverlayDirty();
});
document.getElementById('volumeResRange').addEventListener('input', (e) => {
  volumeResolution = parseInt(e.target.value, 10);
  document.getElementById('volumeResValue').textContent = volumeResolution;
  scheduleVolumeRebuild(150);
});
document.getElementById('volumeStrRange').addEventListener('input', (e) => {
  volumeStrength = parseFloat(e.target.value);
  document.getElementById('volumeStrValue').textContent = volumeStrength.toFixed(2);
  // Only need to refill metaballs, not rebuild MC
  if (volumeObj) updateVolumeBalls();
});
document.getElementById('volumeOpRange').addEventListener('input', (e) => {
  volumeOpacity = parseFloat(e.target.value);
  document.getElementById('volumeOpValue').textContent = volumeOpacity.toFixed(2);
  applyVolumeMaterial();
});
document.getElementById('volumeColor').addEventListener('input', (e) => {
  volumeColorHex = e.target.value;
  applyVolumeMaterial();
});
document.getElementById('volumeStyle').addEventListener('change', (e) => {
  volumeStyle = e.target.value;
  applyVolumeStyle();
});

// OBJ export of the marching-cubes surface ----------------------------------
// Reads vertex and normal data directly from the MarchingCubes internal
// buffers (the first `count` triplets are valid; everything past that is
// stale). Applies the Object3D's world matrix so the OBJ comes out in the
// same coordinate system as the data, then writes a plain-text Wavefront
// OBJ with vertices, normals, and triangle faces.
function buildVolumeObj() {
  if (!volumeObj) return null;
  const count = volumeObj.count | 0;
  if (count === 0) return null;

  volumeObj.updateMatrixWorld(true);
  const matrix = volumeObj.matrixWorld;
  const normalMatrix = new THREE.Matrix3().getNormalMatrix(matrix);

  const positions = volumeObj.positionArray;
  const normals = volumeObj.normalArray;

  const out = [];
  out.push('# Latent explorer — 3D enclosing volume');
  out.push('# Marching-cubes isosurface over metaball field');
  out.push(`# resolution: ${volumeResolution}`);
  out.push(`# metaball size: ${volumeStrength.toFixed(3)}`);
  out.push(`# isolation: ${volumeObj.isolation}`);
  if (SETTINGS.umap_params) {
    const u = SETTINGS.umap_params;
    out.push(`# UMAP: n_neighbors=${u.n_neighbors}, ` +
             `min_dist=${Number(u.min_dist).toFixed(3)}, ` +
             `metric=${u.metric}, seed=${u.random_state}`);
  }
  out.push(`# scene centroid: (${cx.toFixed(4)}, ${cy.toFixed(4)}, ${cz.toFixed(4)})`);
  out.push(`# scene radius: ${radius.toFixed(4)}`);
  out.push(`# vertices: ${count}`);
  out.push(`# triangles: ${count / 3}`);
  out.push(`# exported: ${new Date().toISOString()}`);
  out.push('');
  out.push('o latent_volume');
  out.push('');

  const v = new THREE.Vector3();
  for (let i = 0; i < count; i++) {
    v.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
    v.applyMatrix4(matrix);
    out.push(`v ${v.x.toFixed(6)} ${v.y.toFixed(6)} ${v.z.toFixed(6)}`);
  }

  const n = new THREE.Vector3();
  for (let i = 0; i < count; i++) {
    n.set(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
    n.applyMatrix3(normalMatrix).normalize();
    out.push(`vn ${n.x.toFixed(6)} ${n.y.toFixed(6)} ${n.z.toFixed(6)}`);
  }

  out.push('');
  // Every consecutive 3 vertices form one triangle (MarchingCubes emits a
  // non-indexed triangle list). OBJ indices are 1-based.
  for (let i = 0; i < count; i += 3) {
    const a = i + 1, b = i + 2, c = i + 3;
    out.push(`f ${a}//${a} ${b}//${b} ${c}//${c}`);
  }

  return out.join('\n');
}

document.getElementById('exportVolumeObj').addEventListener('click', () => {
  const btn = document.getElementById('exportVolumeObj');
  const status = document.getElementById('exportVolumeStatus');
  const original = btn.textContent;
  if (!showVolume || !volumeObj) {
    status.textContent = 'enable "show metaball surface" first';
    setTimeout(() => { status.textContent = ''; }, 2400);
    return;
  }
  btn.textContent = 'Writing OBJ...';
  status.textContent = '';
  // Defer so the label change paints before the (potentially slow) string build
  setTimeout(() => {
    try {
      const text = buildVolumeObj();
      if (!text) {
        status.textContent = 'no surface — increase metaball size';
        btn.textContent = original;
        return;
      }
      const blob = new Blob([text], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const ts = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      // Pack the parameters that defined this form into the filename so a
      // batch of exports can be sorted/grouped into a catalog.
      let umapTag = '';
      if (SETTINGS.umap_params) {
        const u = SETTINGS.umap_params;
        const md = Number(u.min_dist).toFixed(2).replace('.', 'p');
        umapTag = `_umap-n${u.n_neighbors}-d${md}-${u.metric}-s${u.random_state}`;
      }
      a.download = `latent_volume_${ts}${umapTag}` +
                   `_r${volumeResolution}-m${volumeStrength.toFixed(2)}.obj`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      const triCount = (volumeObj.count / 3) | 0;
      status.textContent = `${triCount.toLocaleString()} triangles, ` +
                           `${(text.length / 1024).toFixed(0)} KB`;
      btn.textContent = 'Downloaded';
    } catch (err) {
      console.error(err);
      status.textContent = 'export failed: ' + err.message;
      btn.textContent = original;
    }
    setTimeout(() => { btn.textContent = original; }, 1800);
  }, 10);
});

</script>
</body>
</html>
"""


# =============================================================================
# App
# =============================================================================
df_valid, valid_features, embeddings_3d, missing = load_data()

if missing:
    st.error("Required data files are missing. Run latent_embedding.ipynb first.")
    st.write(f"Looking under project root: `{PROJECT_ROOT}`")
    st.code("\n".join(missing))
    st.stop()

# Session state for UMAP parameters. The sidebar form writes into this; on
# read we either use precomputed UMAP from load_data (default) or recompute
# with the user's values.
if "umap_params" not in st.session_state:
    st.session_state.umap_params = dict(UMAP_DEFAULTS)

# Apply current UMAP params if they differ from the precomputed file's
# defaults. The recompute itself is cached, so repeated reads with the same
# params are instantaneous after the first compute.
_p = st.session_state.umap_params
if _p != UMAP_DEFAULTS:
    _custom = recompute_umap_3d(**_p)
    if _custom is not None:
        embeddings_3d = _custom
        df_valid["umap_x_3d"] = _custom[:, 0]
        df_valid["umap_y_3d"] = _custom[:, 1]
        df_valid["umap_z_3d"] = _custom[:, 2]

st.title("Latent Explorer")
st.caption(f"{len(df_valid)} valid entries · features {valid_features.shape} · "
           f"root: `{PROJECT_ROOT}`")

# ---- Sidebar controls ----
with st.sidebar:
    with st.expander("Latent space layout (UMAP)", expanded=False):
        st.caption(
            "Each parameter combination produces a different 3D form. "
            "Same data, different reading. Recompute to update the viewer "
            "and the enclosing volume."
        )
        _p = st.session_state.umap_params
        with st.form("umap_form", clear_on_submit=False):
            n_neighbors = st.slider(
                "n_neighbors", 2, 80, _p["n_neighbors"],
                help="Local (low) vs global (high) structure. Low values break "
                     "the cloud into subclusters with gaps; high values smooth "
                     "into one continent.",
            )
            min_dist = st.slider(
                "min_dist", 0.0, 0.99, _p["min_dist"], 0.01,
                help="Cluster compactness. Low = tight knots; high = spread out.",
            )
            metric = st.selectbox(
                "metric",
                ["cosine", "euclidean", "manhattan", "correlation"],
                index=["cosine", "euclidean", "manhattan", "correlation"].index(_p["metric"]),
                help="Different similarity readings of CLIP features.",
            )
            random_state = st.number_input(
                "seed", value=int(_p["random_state"]), min_value=0, max_value=99999,
                step=1, help="UMAP optimization is non-convex — different seeds "
                             "give different local optima, i.e. different shapes.",
            )
            submitted = st.form_submit_button("Recompute layout")
            if submitted:
                new_params = dict(
                    n_neighbors=int(n_neighbors),
                    min_dist=float(min_dist),
                    metric=metric,
                    random_state=int(random_state),
                )
                if new_params != st.session_state.umap_params:
                    st.session_state.umap_params = new_params
                    st.rerun()
        # Live readout of the active parameter set
        active = st.session_state.umap_params
        st.caption(
            f"active: n_neighbors={active['n_neighbors']}, "
            f"min_dist={active['min_dist']}, "
            f"metric={active['metric']}, seed={active['random_state']}"
        )
        if st.button("Reset to defaults"):
            if st.session_state.umap_params != UMAP_DEFAULTS:
                st.session_state.umap_params = dict(UMAP_DEFAULTS)
                st.rerun()

    st.header("View")
    projection = st.selectbox(
        "Projection",
        options=["umap_2d", "umap_3d", "pca_2d"],
        format_func=lambda x: {"umap_2d": "UMAP 2D",
                               "umap_3d": "UMAP 3D",
                               "pca_2d": "PCA 2D"}[x],
        index=0,
    )

    if projection == "umap_3d":
        elev = st.slider("Elevation", -90.0, 90.0, 20.0, 1.0)
        azim = st.slider("Azimuth", -180.0, 180.0, 30.0, 1.0)
        depth_size_scale = st.slider("Depth-size scale", 0.0, 1.0, 0.4, 0.05)
    else:
        elev, azim, depth_size_scale = 20.0, 30.0, 0.0

    with st.expander("Canvas", expanded=False):
        width = st.number_input("Width (px)", 400, 8000, 1600, 100)
        height = st.number_input("Height (px)", 400, 8000, 1200, 100)
        padding = st.slider("Padding", 0, 400, 80, 10)
        background = st.color_picker("Background", "#0e0e0e")

    st.header("Markers")
    shape = st.selectbox("Shape", SHAPE_CHOICES, index=0)
    point_size = st.slider("Point size", 1.0, 80.0, 10.0, 0.5)
    opacity = st.slider("Opacity", 0.05, 1.0, 1.0, 0.05)
    with st.expander("Stroke", expanded=False):
        stroke_color = st.color_picker("Stroke color", "#ffffff")
        stroke_width = st.slider("Stroke width", 0.0, 4.0, 0.0, 0.1)
    if shape == "image":
        image_max_edge = st.slider("Thumbnail max edge (px)", 32, 256, 96, 8)
    else:
        image_max_edge = 96

    st.header("Color")
    color_field = st.selectbox(
        "Color by",
        options=["cluster", "category", "subtype", "era", "none"],
        format_func=lambda x: f"by {x}" if x != "none" else "uniform",
        index=0,
    )
    if color_field == "none":
        override_color = st.color_picker("Uniform fill", "#ffffff")
        palette_name = "tab10"
    else:
        palette_name = st.selectbox("Palette", PALETTE_NAMES, index=0)
        override_color = "#ffffff"

    st.header("Layers")
    stratify_by = st.selectbox(
        "Split layers by",
        options=["cluster", "category", "subtype", "era", "none"],
        format_func=lambda x: f"by {x}" if x != "none" else "single layer",
        index=0,
    )

    with st.expander("KNN edges", expanded=False):
        show_edges = st.checkbox("Show edges", value=False)
        edges_k = st.slider("k (neighbors)", 1, 10, 3, 1)
        edges_color = st.color_picker("Edge color", "#ffffff")
        edges_opacity = st.slider("Edge opacity", 0.02, 1.0, 0.15, 0.02)
        edges_width = st.slider("Edge width", 0.1, 3.0, 0.5, 0.1)

    with st.expander("Labels", expanded=False):
        show_labels = st.checkbox("Show labels", value=False)
        labels_size = st.slider("Label size", 4, 24, 8, 1)
        labels_color = st.color_picker("Label color", "#ffffff")

    with st.expander("Boundary curve", expanded=False):
        show_curve = st.checkbox(
            "Draw boundary around projection", value=False,
            help="Computes a concave hull around the visible points and emits "
                 "it as a closed SVG path on its own layer.",
        )
        curve_concavity = st.slider(
            "Concavity tightness", 1.5, 6.0, 2.5, 0.1,
            help="Lower = boundary pulls inward to follow concavities. "
                 "Higher = looser, closer to a convex hull. Below ~1.7 the "
                 "boundary can self-intersect.",
        )
        curve_smoothing = st.slider("Smoothing iterations", 0, 8, 3, 1)
        curve_color = st.color_picker("Boundary color", "#000000")
        curve_width = st.slider("Boundary width", 0.5, 12.0, 2.0, 0.5)
        curve_dashed = st.checkbox("Dashed", value=True)
        curve_dash_on = st.slider("Dash length", 1.0, 30.0, 8.0, 1.0)
        curve_dash_off = st.slider("Gap length", 1.0, 30.0, 4.0, 1.0)

settings = dict(
    projection=projection,
    width=int(width), height=int(height), padding=int(padding),
    background=background,
    shape=shape, point_size=float(point_size),
    stroke_color=stroke_color, stroke_width=float(stroke_width),
    opacity=float(opacity),
    color_field=color_field, palette_name=palette_name,
    override_color=override_color,
    elev=float(elev), azim=float(azim),
    depth_size_scale=float(depth_size_scale),
    stratify_by=stratify_by,
    show_edges=bool(show_edges), edges_k=int(edges_k),
    edges_color=edges_color, edges_opacity=float(edges_opacity),
    edges_width=float(edges_width),
    show_labels=bool(show_labels),
    labels_size=int(labels_size), labels_color=labels_color,
    image_max_edge=int(image_max_edge),
    show_curve=bool(show_curve),
    curve_concavity=float(curve_concavity),
    curve_smoothing=int(curve_smoothing),
    curve_color=curve_color,
    curve_width=float(curve_width),
    curve_dashed=bool(curve_dashed),
    curve_dash_on=float(curve_dash_on),
    curve_dash_off=float(curve_dash_off),
)

# ---- Render ----
t0 = time.time()
svg = render_to_svg(settings, df_valid, valid_features)
svg_string = svg.to_string()
render_secs = time.time() - t0

n_layers = len(svg._order)
n_elements = sum(len(v) for v in svg._layers.values())
size_kb = len(svg_string) / 1024

# ---- Main layout ----
left, right = st.columns([4, 1])

with right:
    st.subheader("Export preview")
    st.caption(
        f"Source: matplotlib preview · projection **{settings['projection']}** · "
        f"elev {settings['elev']:.0f}° azim {settings['azim']:.0f}°"
    )
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_name = f"{settings['projection']}_{settings['color_field']}_{settings['shape']}"
    name_stub = st.text_input("Filename stub", value=default_name)
    safe_stub = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name_stub)
    fname = f"{ts}__{safe_stub}.svg"

    st.download_button(
        label="Download preview SVG",
        data=svg_string.encode("utf-8"),
        file_name=fname,
        mime="image/svg+xml",
        type="primary",
        use_container_width=True,
        help="Exports the matplotlib preview shown to the left. "
             "For a 3D view from the live camera angle, use the Download SVG "
             "button inside the 3D viewer below.",
    )
    st.download_button(
        label="Download settings (.json)",
        data=json.dumps(settings, indent=2).encode("utf-8"),
        file_name=fname.replace(".svg", ".json"),
        mime="application/json",
        use_container_width=True,
    )
    if st.button("Save to data/exports/", use_container_width=True):
        out_path = EXPORTS_DIR / fname
        out_path.write_text(svg_string, encoding="utf-8")
        out_path.with_suffix(".json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
        st.success(f"Saved: {out_path.name}")

    st.divider()
    st.caption("Render info")
    st.text(f"layers      {n_layers}")
    st.text(f"elements    {n_elements:,}")
    st.text(f"file size   {size_kb:,.1f} KB")
    st.text(f"render time {render_secs:.2f}s")

    if shape == "image" and size_kb > 5000:
        st.warning("Large file (image markers). May render slowly in browser.")

with left:
    # Render SVG inside an iframe so it stays isolated and scales cleanly
    iframe_html = f"""<!doctype html>
<html><head><style>
  html, body {{ margin: 0; padding: 0; background: #1a1a1a; }}
  .wrap {{ width: 100%; height: 100vh; display: flex; align-items: center;
         justify-content: center; padding: 12px; box-sizing: border-box; }}
  svg {{ max-width: 100%; max-height: 100%; height: auto; display: block;
         box-shadow: 0 0 0 1px rgba(255,255,255,0.05); }}
</style></head>
<body><div class="wrap">{svg_string}</div></body></html>"""

    iframe_height = min(int(height * 0.65), 900) + 40
    components.html(iframe_html, height=iframe_height, scrolling=False)

# ---- three.js 3D image-sprite viewer below ----
st.divider()
with st.expander("Fluid 3D navigation (with image markers)", expanded=True):
    st.caption(
        "Drag to rotate, scroll to zoom, right-click drag to pan. Adjust "
        "opacity and blend mode in the overlay to find the look you want — "
        "**additive**, **screen**, and **lighten** all intensify on overlap; "
        "**multiply** and **darken** do the inverse. The **Download this view "
        "as SVG** button in the overlay exports the live 3D angle (separate "
        "from the sidebar's preview-based export button on the right)."
    )
    threejs_thumb = st.slider(
        "3D thumbnail size (px)", 32, 192, 64, 8,
        help="Higher values render sharper sprites but increase load time.",
        key="threejs_thumb_size",
    )

    # Resolve colors and layers once for all points, mirroring render_to_svg
    if settings["color_field"] == "none":
        color_for_point = [settings["override_color"]] * len(df_valid)
    else:
        vals = df_valid[settings["color_field"]].astype(str)
        cmap_lookup = categorical_palette(vals, settings["palette_name"])
        color_for_point = [cmap_lookup[v] for v in vals]

    def _layer_for(row):
        if settings["stratify_by"] == "none":
            return "points"
        v = row[settings["stratify_by"]]
        label = ("noise" if (settings["stratify_by"] == "cluster" and str(v) == "-1")
                 else str(v))
        return f'{settings["stratify_by"]}_{label}'

    # Build per-point data: position, display thumb, and (if shape=image) an
    # export thumb at the user-specified export resolution
    # Autoencoder-style typicality signal. A low-rank linear autoencoder (PCA
    # with a 32-dim bottleneck, matching the conv-autoencoders' bottleneck) is
    # fit on the L2-normalized CLIP corpus; the per-work reconstruction error is
    # how poorly the corpus's principal structure reproduces that work. High
    # error means surprising / off-manifold, low error means typical / deep in
    # the vibe. The worlds view can let this decide where an edge stays
    # articulated and where it breaks into gesture. Swap in per-period
    # convolutional-autoencoder errors here later by overwriting ae_surprise.
    try:
        from sklearn.decomposition import PCA as _PCA
        _X = np.asarray(valid_features, dtype=np.float32)
        _Xn = _X / np.clip(np.linalg.norm(_X, axis=1, keepdims=True), 1e-8, None)
        _k = int(min(32, max(2, _Xn.shape[0] - 1), _Xn.shape[1]))
        _ae = _PCA(n_components=_k, random_state=RANDOM_STATE)
        _Xr = _ae.inverse_transform(_ae.fit_transform(_Xn))
        _recon_err = np.linalg.norm(_Xn - _Xr, axis=1).astype(float)
    except Exception:
        _recon_err = np.zeros(len(df_valid), dtype=float)

    sprites_data = []
    want_export_img = (settings["shape"] == "image")
    for i in range(len(df_valid)):
        row = df_valid.iloc[i]
        x, y, z = embeddings_3d[i]

        disp_b64, disp_ar = None, 1.0
        try:
            db, (diw, dih) = get_thumbnail_b64(row["path"], threejs_thumb)
            disp_b64 = db
            disp_ar = float(diw) / float(dih) if dih > 0 else 1.0
        except Exception:
            pass

        exp_b64, exp_ar = None, 1.0
        if want_export_img:
            try:
                eb, (eiw, eih) = get_thumbnail_b64(row["path"], settings["image_max_edge"])
                exp_b64 = eb
                exp_ar = float(eiw) / float(eih) if eih > 0 else 1.0
            except Exception:
                pass

        sprites_data.append({
            "x": float(x), "y": float(y), "z": float(z),
            "img": disp_b64,                  # display sprite
            "ar": disp_ar,
            "exp_img": exp_b64,               # export thumbnail (if shape=image)
            "exp_ar": exp_ar,
            "color": color_for_point[i],
            "layer": _layer_for(row),
            "id": str(row["id"]),
            "cat": str(row.get("category", "")),
            "sub": str(row.get("subtype", "")),
            "era": str(row.get("era", "")),
            "clu": str(row.get("cluster", "")),
            "ae_surprise": float(_recon_err[i]) if i < len(_recon_err) else 0.0,
        })

    # Pre-compute KNN edges so JS doesn't have to
    edges_data = []
    if settings["show_edges"]:
        feats_f32 = np.ascontiguousarray(valid_features.astype(np.float32))
        for a, b in compute_knn_edges(
            feats_f32.tobytes(), feats_f32.shape[0], feats_f32.shape[1],
            settings["edges_k"]
        ):
            edges_data.append([int(a), int(b)])

    export_settings = {
        "width": int(settings["width"]),
        "height": int(settings["height"]),
        "padding": int(settings["padding"]),
        "background": settings["background"],
        "shape": settings["shape"],
        "point_size": float(settings["point_size"]),
        "stroke_color": settings["stroke_color"],
        "stroke_width": float(settings["stroke_width"]),
        "opacity": float(settings["opacity"]),
        "show_edges": bool(settings["show_edges"]),
        "edges_color": settings["edges_color"],
        "edges_opacity": float(settings["edges_opacity"]),
        "edges_width": float(settings["edges_width"]),
        "show_curve": bool(settings["show_curve"]),
        "curve_concavity": float(settings["curve_concavity"]),
        "curve_smoothing": int(settings["curve_smoothing"]),
        "curve_color": settings["curve_color"],
        "curve_width": float(settings["curve_width"]),
        "curve_dashed": bool(settings["curve_dashed"]),
        "curve_dash_on": float(settings["curve_dash_on"]),
        "curve_dash_off": float(settings["curve_dash_off"]),
        # UMAP parameters that produced this layout — used by the OBJ
        # exporter so each saved volume records its provenance.
        "umap_params": st.session_state.get("umap_params", dict(UMAP_DEFAULTS)),
    }

    threejs_html = (
        THREEJS_TEMPLATE
        .replace("__POINTS_JSON__", json.dumps(sprites_data))
        .replace("__EDGES_JSON__", json.dumps(edges_data))
        .replace("__SETTINGS_JSON__", json.dumps(export_settings))
        .replace("__BG__", background)
    )
    components.html(threejs_html, height=720, scrolling=False)
