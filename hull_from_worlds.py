"""
hull_from_worlds.py — Histories projection tool.

Moves the latent field from the worlds-mode linework viewer into a 3D solid by
shape-from-silhouette. Each silhouette is extruded back through the volume and
the extrusions are intersected. The surviving solid is the visual hull, the
shape that casts those shadows.

Two ways in.

  --masks DIR   Connect to the worlds view. Reads a folder of silhouette PNGs
                exported from the worlds-mode viewer, each paired with the
                camera JSON the copy-camera bridge already serializes
                (position, target, fov, w, h). Carves in the viewer's own
                perspective, in the raw UMAP coordinates the camera lives in,
                so the hull registers exactly with what you saw on screen. The
                object becomes the reconstruction from the stations you chose
                to occupy while navigating, and the phantom is what those
                viewpoints could not disambiguate.

  --axes xyz    Render orthographic silhouettes down the cube's standard axes
                directly from the points. The cube is the orthographic
                antagonist, the primitive that insists it can be fully known
                from its plans. Use this when you want the projection frame
                itself to be the thing strained.

The payload either way is the phantom volume. A hull from a handful of views
over-estimates: it holds the field plus mass wherever the silhouettes align
but no member sits. That surplus is registration error given volume. The OBJ
splits into two groups, core and phantom, each with its own material, so the
surplus can carry the unstable-metal or pigment treatment while the core stays
matte.

Dataset note: the points are heterogeneous representations. The silhouettes are
footprints of that mixed field, never building outlines.

  # connect to the worlds view
  python hull_from_worlds.py --masks data\\exports\\worlds_set --core-radius 0.05

  # cube / orthographic
  python hull_from_worlds.py --axes xyz --view iso0 --res 160 --unify 6
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from skimage import measure
from skimage.morphology import dilation, disk
from skimage.draw import disk as draw_disk


# =============================================================================
# Paths
# =============================================================================
def find_project_root(start: Path) -> Path:
    for c in [start, *start.parents][:6]:
        if (c / "data" / "embeddings").exists():
            return c
    return start


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(HERE)
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
EXPORTS_DIR = PROJECT_ROOT / "data" / "exports" / "hull"

NAMED_VIEWS = {
    "x": ((1, 0, 0), (0, 0, 1)),
    "y": ((0, 1, 0), (0, 0, 1)),
    "z": ((0, 0, 1), (0, 1, 0)),
    "iso0": ((1, 1, 1), (0, 0, 1)),
    "iso1": ((-1, 1, 1), (0, 0, 1)),
    "iso2": ((1, -1, 1), (0, 0, 1)),
    "iso3": ((-1, -1, 1), (0, 0, 1)),
}


# =============================================================================
# Data
# =============================================================================
def load_points(npy_path: Path):
    pts = np.load(npy_path).astype(np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"expected (N,3) in {npy_path.name}, got {pts.shape}")
    return pts


def _vec3(v):
    """Accept [x,y,z] or {'x':..,'y':..,'z':..} from the camera JSON."""
    if isinstance(v, dict):
        return np.array([v["x"], v["y"], v["z"]], float)
    return np.array(v, float)


# =============================================================================
# Views. Each implements sample(centers) -> bool "inside silhouette", in one
# shared coordinate space so silhouette and carve cannot drift.
# =============================================================================
class OrthoView:
    """Axis-aligned or oblique orthographic view, rendered from the points."""
    def __init__(self, name, direction, up, cube_pts, sil_res, dot_radius, unify):
        self.name = name
        self.res = sil_res
        d = np.asarray(direction, float); d /= np.linalg.norm(d)
        up = np.asarray(up, float)
        if abs(np.dot(d, up / np.linalg.norm(up))) > 0.99:
            up = np.array([1.0, 0.0, 0.0])
        right = np.cross(up, d); right /= np.linalg.norm(right)
        true_up = np.cross(d, right)
        self.right, self.up = right, true_up
        corners = np.array([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], float)
        u, v = corners @ right, corners @ true_up
        self.umin, self.umax = float(u.min()), float(u.max())
        self.vmin, self.vmax = float(v.min()), float(v.max())
        self.mask = self._render(cube_pts, dot_radius, unify)

    def _px(self, p3):
        u, v = p3 @ self.right, p3 @ self.up
        col = (u - self.umin) / (self.umax - self.umin) * (self.res - 1)
        row = (1.0 - (v - self.vmin) / (self.vmax - self.vmin)) * (self.res - 1)
        return col, row

    def _render(self, cube_pts, dot_radius, unify):
        mask = np.zeros((self.res, self.res), bool)
        col, row = self._px(cube_pts)
        r = max(1, int(round(dot_radius)))
        for c, rr in zip(col, row):
            yy, xx = draw_disk((int(round(rr)), int(round(c))), r, shape=mask.shape)
            mask[yy, xx] = True
        if unify > 0:
            mask = dilation(mask, disk(int(unify)))
        return mask

    def sample(self, centers):
        col, row = self._px(centers)
        ci, ri = np.round(col).astype(int), np.round(row).astype(int)
        inb = (ci >= 0) & (ci < self.res) & (ri >= 0) & (ri < self.res)
        out = np.zeros(len(centers), bool)
        out[inb] = self.mask[np.clip(ri[inb], 0, self.res - 1), np.clip(ci[inb], 0, self.res - 1)]
        return out


class PerspView:
    """Perspective view from the worlds viewer: a silhouette mask plus the
    camera JSON the copy-camera bridge serializes. Matches three.js
    PerspectiveCamera conventions (vertical fov in degrees, aspect = w/h,
    camera looking down -Z, y-down pixels)."""
    def __init__(self, name, mask, cam, up=(0, 1, 0)):
        self.name = name
        self.mask = mask  # bool (H, W), True = inside figure
        self.h, self.w = mask.shape
        self.C = _vec3(cam["position"])
        T = _vec3(cam["target"])
        self.fov = float(cam["fov"])
        self.aspect = float(cam.get("w", self.w)) / float(cam.get("h", self.h))
        f = T - self.C; f /= np.linalg.norm(f)
        up = np.asarray(up, float)
        right = np.cross(f, up); right /= np.linalg.norm(right)
        true_up = np.cross(right, f)
        self.f, self.right, self.up = f, right, true_up
        self.tan_half = np.tan(np.radians(self.fov) / 2.0)

    def sample(self, centers):
        rel = centers - self.C
        depth = rel @ self.f                      # >0 in front of camera
        xv = rel @ self.right
        yv = rel @ self.up
        out = np.zeros(len(centers), bool)
        valid = depth > 1e-9
        ndc_x = np.zeros(len(centers)); ndc_y = np.zeros(len(centers))
        ndc_x[valid] = xv[valid] / (depth[valid] * self.tan_half * self.aspect)
        ndc_y[valid] = yv[valid] / (depth[valid] * self.tan_half)
        col = (ndc_x * 0.5 + 0.5) * self.w
        row = (1.0 - (ndc_y * 0.5 + 0.5)) * self.h
        ci, ri = np.round(col).astype(int), np.round(row).astype(int)
        inb = valid & (ci >= 0) & (ci < self.w) & (ri >= 0) & (ri < self.h)
        out[inb] = self.mask[np.clip(ri[inb], 0, self.h - 1), np.clip(ci[inb], 0, self.w - 1)]
        return out


# =============================================================================
# Mask loading (PNG, optional SVG via cairosvg)
# =============================================================================
def load_mask(path: Path, threshold: int, invert: bool) -> np.ndarray:
    from PIL import Image
    if path.suffix.lower() == ".svg":
        try:
            import cairosvg, io
            png = cairosvg.svg2png(url=str(path))
            img = Image.open(io.BytesIO(png))
        except Exception as e:
            raise SystemExit(f"SVG needs cairosvg ({e}); export worlds as PNG instead")
    else:
        img = Image.open(path)
    if img.mode == "RGBA":
        alpha = np.array(img.split()[-1])
        gray = np.array(img.convert("L"))
        fg = (alpha > 8) & (gray < 250) if not invert else (alpha > 8) & (gray >= 250)
        if fg.sum() == 0:  # opaque export: fall back to luminance
            fg = (gray < threshold) if not invert else (gray > 255 - threshold)
    else:
        gray = np.array(img.convert("L"))
        fg = (gray < threshold) if not invert else (gray > 255 - threshold)
    return fg


def load_worlds_set(mask_dir: Path, threshold: int, invert: bool, up):
    """Folder of stemNN.png + stemNN.json pairs. The JSON is the camera bridge
    output: {position, target, fov, w, h}."""
    paths = sorted(glob.glob(str(mask_dir / "*.png")) + glob.glob(str(mask_dir / "*.svg")))
    views = []
    for p in paths:
        p = Path(p)
        j = p.with_suffix(".json")
        if not j.exists():
            print(f"  skip {p.name}: no matching {j.name} camera file")
            continue
        cam = json.loads(j.read_text())
        mask = load_mask(p, threshold, invert)
        views.append(PerspView(p.stem, mask, cam, up=up))
    if not views:
        raise SystemExit(f"no (png, json) pairs in {mask_dir}")
    return views


# =============================================================================
# Carve / phantom / mesh / OBJ
# =============================================================================
def voxel_grid(lo, hi, res, pad_frac):
    span = (hi - lo).max()
    pad = span * pad_frac
    lo = lo - pad; hi = hi + pad
    size = (hi - lo)
    idx = (np.arange(res) + 0.5) / res
    gx, gy, gz = np.meshgrid(idx, idx, idx, indexing="ij")
    centers = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1) * size + lo
    return centers, lo, size


def carve(views, centers, res, mode):
    sampled = [v.sample(centers).reshape(res, res, res) for v in views]
    if mode == "intersect":
        occ = np.ones((res, res, res), bool)
        for s in sampled: occ &= s
    elif mode == "union":
        occ = np.zeros((res, res, res), bool)
        for s in sampled: occ |= s
    elif mode == "difference":
        occ = sampled[0].copy()
        for s in sampled[1:]: occ &= ~s
    else:
        raise ValueError(mode)
    return occ


def split_core_phantom(occ, centers, pts, core_world, res):
    tree = cKDTree(pts)
    flat = occ.ravel()
    dist = np.full(centers.shape[0], np.inf)
    if flat.any():
        dist[flat], _ = tree.query(centers[flat])
    near = (dist <= core_world).reshape(res, res, res)
    core, phantom = occ & near, occ & ~near
    frac = float(phantom.sum()) / float(max(occ.sum(), 1))
    return core, phantom, frac


def mesh_hull(occ, lo, size, res):
    padded = np.pad(occ.astype(np.float32), 1)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    verts = (verts - 1.0) / res * size + lo
    return verts, faces


def write_obj(path: Path, verts, faces, vert_is_phantom, name):
    mtl = path.with_suffix(".mtl")
    core_f, phantom_f = [], []
    for f in faces:
        n = int(vert_is_phantom[f[0]]) + int(vert_is_phantom[f[1]]) + int(vert_is_phantom[f[2]])
        (phantom_f if n >= 2 else core_f).append(f)
    lines = [f"mtllib {mtl.name}", f"o {name}"]
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for grp, mat, fl in (("core", "core", core_f), ("phantom", "phantom", phantom_f)):
        lines.append(f"g {grp}"); lines.append(f"usemtl {mat}")
        for f in fl:
            lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")
    path.write_text("\n".join(lines), encoding="utf-8")
    mtl.write_text("newmtl core\nKd 0.62 0.34 0.32\nillum 1\n\n"
                   "newmtl phantom\nKd 0.78 0.80 0.82\nillum 1\n", encoding="utf-8")
    return len(core_f), len(phantom_f)


# =============================================================================
# Build
# =============================================================================
def fit_cube(pts, pad=0.06):
    lo, hi = pts.min(0), pts.max(0)
    span = float((hi - lo).max()) or 1.0
    center = (lo + hi) / 2.0
    half = span * (0.5 + pad)
    return np.clip((pts - center) / (2 * half) + 0.5, 0, 1)


def build(pts, args, up):
    if args.masks:
        # perspective carve in raw UMAP coords (where the camera lives)
        views = load_worlds_set(Path(args.masks), args.mask_threshold, args.invert, up)
        work_pts = pts
        centers, lo, size = voxel_grid(pts.min(0), pts.max(0), args.res, 0.06)
        span = (pts.max(0) - pts.min(0)).max()
        label = "worlds:" + ",".join(v.name for v in views)
    else:
        # orthographic carve in the unit cube
        cube = fit_cube(pts)
        keys = [c for c in (args.axes or "") if c in NAMED_VIEWS] + \
               [v for v in args.view if v in NAMED_VIEWS]
        if not keys:
            keys = ["x", "y", "z"]
        views = [OrthoView(k, *NAMED_VIEWS[k], cube, args.sil_res, args.dot_radius, args.unify)
                 for k in keys]
        work_pts = cube
        centers, lo, size = voxel_grid(np.zeros(3), np.ones(3), args.res, 0.0)
        span = 1.0
        label = "cube:" + "".join(keys)

    core_world = args.core_radius * span
    occ = carve(views, centers, args.res, args.mode)
    if occ.sum() == 0:
        raise SystemExit("empty hull: raise --unify/--dot-radius, lower --res, or check mask polarity (--invert)")
    core, phantom, frac = split_core_phantom(occ, centers, work_pts, core_world, args.res)
    verts, faces = mesh_hull(occ, lo, size, args.res)
    tree = cKDTree(work_pts)
    d, _ = tree.query(verts)
    vert_is_phantom = d > core_world
    return views, label, occ, frac, verts, faces, vert_is_phantom


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npy", default=str(EMBEDDINGS_DIR / "clip_umap_3d.npy"))
    ap.add_argument("--masks", default="", help="folder of worlds silhouette PNG + camera JSON pairs")
    ap.add_argument("--mask-threshold", type=int, default=128, help="luminance cutoff for figure vs paper")
    ap.add_argument("--invert", action="store_true", help="figure is light on dark instead of dark on light")
    ap.add_argument("--up", default="0,1,0", help="camera up vector for the worlds carve")
    ap.add_argument("--axes", default="xyz")
    ap.add_argument("--view", nargs="*", default=[])
    ap.add_argument("--mode", choices=["intersect", "union", "difference"], default="intersect")
    ap.add_argument("--res", type=int, default=160)
    ap.add_argument("--sil-res", type=int, default=512)
    ap.add_argument("--dot-radius", type=float, default=3.0)
    ap.add_argument("--unify", type=float, default=6.0)
    ap.add_argument("--core-radius", type=float, default=0.05,
                    help="as a fraction of the field's max extent")
    ap.add_argument("--name", default="hull")
    args = ap.parse_args()

    npy = Path(args.npy)
    if not npy.exists():
        raise SystemExit(f"not found: {npy}\nPoint --npy at clip_umap_3d.npy")
    up = np.array([float(x) for x in args.up.split(",")], float)
    pts = load_points(npy)

    views, label, occ, frac, verts, faces, vph = build(pts, args, up)

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    src = "worlds" if args.masks else "cube"
    stub = EXPORTS_DIR / f"{ts}__{args.name}_{src}_{args.mode}"
    obj = stub.with_suffix(".obj")
    nc, npf = write_obj(obj, verts, faces, vph, args.name)
    stub.with_suffix(".json").write_text(json.dumps(dict(
        source=label, mode=args.mode, res=args.res, n_points=int(len(pts)),
        n_views=len(views), hull_voxels=int(occ.sum()), phantom_fraction=round(frac, 4),
        core_faces=nc, phantom_faces=npf, vertices=int(len(verts)),
        core_radius=args.core_radius), indent=2), encoding="utf-8")

    print(f"source          {label}")
    print(f"views           {len(views)}  ({args.mode})")
    print(f"hull voxels     {int(occ.sum()):,} / {args.res**3:,}")
    print(f"phantom volume  {frac*100:.1f}%  of the hull is registration surplus")
    print(f"mesh            {len(verts):,} verts, {len(faces):,} faces ({nc:,} core / {npf:,} phantom)")
    print(f"wrote           {obj}")


if __name__ == "__main__":
    main()
