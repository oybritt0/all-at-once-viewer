"""
unify_volume_gen.py  --  run from the project root (Histories\\).

Builds an OBJ solid from the 3D latent point cloud using the same field logic
as the viewer's swatch composition "unify" slider, carried into three
dimensions. The composition unify runs four steps in the image plane: deposit
the linework as a stroke, blur it by a reach radius so neighbours fuse,
threshold the blurred field into a filled figure, and trace the boundary with
marching squares. This script runs the identical four steps in a voxel grid:
deposit each point as a ball, blur by reach, threshold, and trace the boundary
with marching cubes.

The articulate boundary character transfers as well. In 2D the dense stretches
(dogs, high DINO density) quantize to a grid and the sparse stretches (rabbits,
gap-adjacent) get Chaikin smoothed. Here each surface vertex is classified by
the nearest point's density: dog vertices snap to the quantization grid, which
reads as hard faceting, and rabbit vertices are Taubin smoothed, which reads as
gesture. The dog surface is the accumulated deposit rendered as arrested matter.
The rabbit surface is the phantom edge that will not resolve, rendered as flow.

Reads the same files the viewer loads:
  data/embeddings/index.json          ids + success mask
  data/embeddings/clip_umap_3d.npy    3D UMAP coordinates (viewer's frame)
  data/embeddings/point_signals.json  per-point density (1 dog core, 0 rabbit)
  data/catalog/manifest.json          optional, only for --split-by

Writes one OBJ per body to data/exports/, with faces grouped g_dog / g_rabbit so
media assignment downstream stays trivial (quantized dog fields to acrylic, the
smoothed rabbit register to oil).

Dependencies: numpy, scipy, scikit-image.
  pip install numpy scipy scikit-image
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------------
# The four constants below are lifted directly from the 2D unify in app.py
# (computeSegmentationPaths). Reusing them is what keeps the 3D character
# matched to what the composition slider produces on screen.
#   stroke width  = 1 + unify * minDim * 0.02      -> deposit radius
#   reach sigma   = max(0.8, unify * minDim * 0.03) -> blur radius
#   level frac    = 0.45 - 0.22 * unify             -> isolevel as frac of max
UNIFY_STROKE_K = 0.02
UNIFY_REACH_K = 0.03
UNIFY_LEVEL_A = 0.45
UNIFY_LEVEL_B = 0.22


def load_points(root: Path, split_by: str):
    emb = root / "data" / "embeddings"
    idx_path = emb / "index.json"
    umap_path = emb / "clip_umap_3d.npy"
    sig_path = emb / "point_signals.json"

    for p in (idx_path, umap_path):
        if not p.exists():
            sys.exit(f"[abort] missing {p}. Run this from the project root (Histories\\).")

    with open(idx_path, encoding="utf-8") as f:
        index = json.load(f)
    ids = index["ids"]
    success = np.asarray(index["success"], dtype=bool)

    coords_all = np.load(umap_path).astype(np.float64)
    if coords_all.shape[0] == success.sum():
        coords = coords_all
        ids = [i for i, ok in zip(ids, success) if ok]
    elif coords_all.shape[0] == len(ids):
        coords = coords_all[success]
        ids = [i for i, ok in zip(ids, success) if ok]
    else:
        sys.exit(f"[abort] coordinate rows {coords_all.shape[0]} do not align with "
                 f"index ({len(ids)} ids, {int(success.sum())} successful)")

    # density: 1 dog core, 0 rabbit gap-adjacent. absent file leaves it None,
    # which turns articulate off (uniform smooth) with a warning.
    density = None
    if sig_path.exists():
        with open(sig_path, encoding="utf-8") as f:
            sig = json.load(f)
        dmap = dict(zip(sig["ids"], sig["density"]))
        density = np.array([dmap.get(i, np.nan) for i in ids], dtype=np.float64)
    else:
        print("[warn] point_signals.json not found. Articulate disabled, surface uniform.")

    # optional stratify column from the manifest
    layer = np.array(["points"] * len(ids), dtype=object)
    if split_by != "none":
        man_path = root / "data" / "catalog" / "manifest.json"
        if not man_path.exists():
            sys.exit(f"[abort] --split-by {split_by} needs {man_path}")
        with open(man_path, encoding="utf-8") as f:
            manifest = json.load(f)
        field = {str(m["id"]): str(m.get(split_by, "unknown")) for m in manifest}
        layer = np.array([f"{split_by}_{field.get(i, 'unknown')}" for i in ids], dtype=object)

    return np.asarray(ids, dtype=object), coords, density, layer


def select(coords, density, layer, args):
    keep = np.ones(len(coords), dtype=bool)

    if args.layers:
        wanted = set(s.strip() for s in args.layers.split(","))
        keep &= np.array([l in wanted for l in layer], dtype=bool)

    # density focus gate: the dog/rabbit percentile band, mirroring the viewer's
    # pointShownP focus filter. density here is already a 0..1 percentile.
    if density is not None and (args.focus_min > 0.0 or args.focus_max < 1.0):
        d = np.where(np.isnan(density), 0.0, density)
        keep &= (d >= args.focus_min) & (d <= args.focus_max)

    return keep


def build_grid(coords, res, pad):
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    longest = span.max()
    voxel = (longest * pad) / res
    origin = lo - (longest * pad - span) / 2.0 - voxel  # one voxel of margin
    dims = np.ceil((hi - origin + voxel) / voxel).astype(int) + 1
    dims = np.maximum(dims, 4)
    return origin, voxel, tuple(int(d) for d in dims)


def deposit_balls(coords, origin, voxel, dims, radius_vox):
    """Rasterize each point as a solid ball of value 1. Overlaps stay 1, which
    matches the 2D stroke that stays white where strokes cross."""
    field = np.zeros(dims, dtype=np.float32)
    r = max(1, int(np.ceil(radius_vox)))
    # local ball stamp
    ax = np.arange(-r, r + 1)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    stamp = (gx * gx + gy * gy + gz * gz) <= (radius_vox * radius_vox + 1e-6)

    ijk = np.floor((coords - origin) / voxel).astype(int)
    nx, ny, nz = dims
    for (i, j, k) in ijk:
        i0, i1 = i - r, i + r + 1
        j0, j1 = j - r, j + r + 1
        k0, k1 = k - r, k + r + 1
        si0, si1 = max(0, -i0), stamp.shape[0] - max(0, i1 - nx)
        sj0, sj1 = max(0, -j0), stamp.shape[1] - max(0, j1 - ny)
        sk0, sk1 = max(0, -k0), stamp.shape[2] - max(0, k1 - nz)
        ci0, cj0, ck0 = max(0, i0), max(0, j0), max(0, k0)
        ci1, cj1, ck1 = min(nx, i1), min(ny, j1), min(nz, k1)
        if ci1 <= ci0 or cj1 <= cj0 or ck1 <= ck0:
            continue
        sub = stamp[si0:si1, sj0:sj1, sk0:sk1]
        np.maximum(field[ci0:ci1, cj0:cj1, ck0:ck1], sub, out=field[ci0:ci1, cj0:cj1, ck0:ck1])
    return field


def build_adjacency(n_verts, faces):
    from scipy.sparse import coo_matrix
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    data = np.ones(len(e), dtype=np.float64)
    A = coo_matrix((data, (e[:, 0], e[:, 1])), shape=(n_verts, n_verts)).tocsr()
    A.data[:] = 1.0  # dedupe weights
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    from scipy.sparse import diags
    Dinv = diags(1.0 / deg)
    return Dinv @ A  # row-normalized neighbour averaging operator


def taubin_smooth(verts, faces, move_mask, iters, lam, mu):
    """Taubin lambda/mu smoothing. Only vertices in move_mask move, so the
    quantized dog vertices stay pinned as anchors and the rabbit surface flows
    off them. Two passes per iteration keep the volume from shrinking."""
    if iters <= 0 or not move_mask.any():
        return verts
    Anorm = build_adjacency(len(verts), faces)
    V = verts.copy()
    m = move_mask.astype(np.float64)[:, None]
    for _ in range(iters):
        L = Anorm @ V - V
        V = V + m * (lam * L)
        L = Anorm @ V - V
        V = V + m * (mu * L)
    return V


def articulate(verts, faces, coords, density, quant, dog_thresh, taubin_iters, lam, mu):
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    _, nn = tree.query(verts, k=1)
    vdens = density[nn]
    dog = np.isfinite(vdens) & (vdens >= dog_thresh)
    rabbit = ~dog

    # dogs quantize to the grid: hard facets, arrested deposit
    if quant > 0:
        verts = verts.copy()
        verts[dog] = np.round(verts[dog] / quant) * quant

    # rabbits smooth: gestural phantom edge, dogs pinned
    verts = taubin_smooth(verts, faces, rabbit, taubin_iters, lam, mu)
    return verts, dog


def marching(field, level):
    from skimage.measure import marching_cubes
    verts, faces, _, _ = marching_cubes(field, level=level)
    return verts, faces


def write_obj(path, verts, faces, dog_vert_mask, header):
    lines = [f"# {h}" for h in header]
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    if dog_vert_mask is not None:
        vote = dog_vert_mask[faces].sum(axis=1)  # 0..3 dog verts per face
        dog_faces = faces[vote >= 2]
        rabbit_faces = faces[vote < 2]
        lines.append("g g_dog")
        for f in dog_faces:
            lines.append(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}")
        lines.append("g g_rabbit")
        for f in rabbit_faces:
            lines.append(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}")
    else:
        for f in faces:
            lines.append(f"f {f[0] + 1} {f[1] + 1} {f[2] + 1}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_body(coords, density, origin, voxel, dims, args, name, out_dir):
    minGrid = min(dims)

    deposit_r = args.deposit if args.deposit is not None else \
        0.5 * (1.0 + args.unify * minGrid * UNIFY_STROKE_K)
    reach = args.reach if args.reach is not None else \
        max(0.8, args.unify * minGrid * UNIFY_REACH_K)
    level_frac = args.level if args.level is not None else \
        (UNIFY_LEVEL_A - UNIFY_LEVEL_B * args.unify)

    from scipy.ndimage import gaussian_filter

    deposit = deposit_balls(coords, origin, voxel, dims, deposit_r)

    if args.mode == "solid":
        field = gaussian_filter(deposit, sigma=reach)
    else:
        # gap / phantom: a wide envelope of the occupancy minus the cores, so the
        # surface encloses the emptiness between deposits. Mirrors the 2D gap
        # contour (envelope minus mask).
        env = gaussian_filter(deposit, sigma=reach * 3.0)
        core = gaussian_filter(deposit, sigma=reach)
        field = env - args.carve * core
        field = np.clip(field, 0.0, None)

    fmax = float(field.max())
    if fmax < 1e-6:
        print(f"[skip] {name}: field is empty at these settings")
        return None
    level = fmax * level_frac
    if level >= fmax:
        level = fmax * 0.5

    verts_idx, faces = marching(field, level)
    if len(verts_idx) == 0 or len(faces) == 0:
        print(f"[skip] {name}: no surface at level {level:.4f}")
        return None

    # index space -> world (viewer's raw UMAP frame)
    verts = origin + verts_idx * voxel

    dog_mask = None
    if args.articulate and density is not None and np.isfinite(density).any():
        quant = args.quant if args.quant is not None else voxel * 2.0
        verts, dog_mask = articulate(
            verts, faces, coords, density, quant, args.dog_thresh,
            args.taubin_iters, args.taubin_lambda, args.taubin_mu,
        )

    if args.up == "z":
        verts = verts[:, [0, 2, 1]]

    header = [
        f"unify_volume_gen  body={name}  mode={args.mode}",
        f"unify={args.unify}  deposit_r={deposit_r:.3f}vox  reach={reach:.3f}vox  "
        f"level_frac={level_frac:.3f}",
        f"grid={dims}  voxel={voxel:.5f}  points={len(coords)}",
        f"articulate={'on' if dog_mask is not None else 'off'}  "
        f"dog_thresh={args.dog_thresh}  up={args.up}",
        f"verts={len(verts)}  faces={len(faces)}",
    ]
    out_path = out_dir / f"{name}.obj"
    write_obj(out_path, verts, faces, dog_mask, header)
    dog_faces = int((dog_mask[faces].sum(axis=1) >= 2).sum()) if dog_mask is not None else 0
    print(f"[ok] {out_path.name}: {len(verts)} verts, {len(faces)} faces "
          f"({dog_faces} dog / {len(faces) - dog_faces} rabbit)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="3D unify volume OBJ generator")
    ap.add_argument("--root", default=".", help="project root (Histories\\)")
    ap.add_argument("--mode", choices=["solid", "gap"], default="solid")
    ap.add_argument("--split-by", default="none",
                    choices=["none", "period", "category", "mode_label", "cluster", "subject"])
    ap.add_argument("--layers", default="", help="comma list of layer labels to keep")
    ap.add_argument("--focus-min", type=float, default=0.0, help="density gate low (0..1)")
    ap.add_argument("--focus-max", type=float, default=1.0, help="density gate high (0..1)")
    ap.add_argument("--res", type=int, default=100, help="voxels along the longest axis")
    ap.add_argument("--pad", type=float, default=1.25, help="bbox padding (viewer VOLUME_PAD)")
    ap.add_argument("--unify", type=float, default=0.5,
                    help="0..1, drives deposit/reach/threshold like the 2D slider")
    ap.add_argument("--deposit", type=float, default=None, help="override deposit radius (vox)")
    ap.add_argument("--reach", type=float, default=None, help="override reach sigma (vox)")
    ap.add_argument("--level", type=float, default=None, help="override isolevel fraction (0..1)")
    ap.add_argument("--carve", type=float, default=1.0, help="gap mode core subtraction")
    ap.add_argument("--articulate", dest="articulate", action="store_true", default=True)
    ap.add_argument("--no-articulate", dest="articulate", action="store_false")
    ap.add_argument("--quant", type=float, default=None,
                    help="dog quantization grid in world units (default 2 voxels)")
    ap.add_argument("--dog-thresh", type=float, default=0.5)
    ap.add_argument("--taubin-iters", type=int, default=12)
    ap.add_argument("--taubin-lambda", type=float, default=0.5)
    ap.add_argument("--taubin-mu", type=float, default=-0.53)
    ap.add_argument("--up", choices=["y", "z"], default="y",
                    help="y matches the viewer; z for Blender/Rhino import")
    ap.add_argument("--out", default=None, help="output dir (default data/exports)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    ids, coords, density, layer = load_points(root, args.split_by)
    keep = select(coords, density, layer, args)
    if keep.sum() == 0:
        sys.exit("[abort] no points pass the layer / focus filters")

    coords, layer = coords[keep], layer[keep]
    density = density[keep] if density is not None else None

    out_dir = Path(args.out) if args.out else (root / "data" / "exports")
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = f"{args.mode}_u{args.unify:.2f}"
    written = []
    if args.split_by == "none":
        origin, voxel, dims = build_grid(coords, args.res, args.pad)
        p = generate_body(coords, density, origin, voxel, dims, args,
                          f"unify_{stamp}", out_dir)
        if p:
            written.append(p)
    else:
        # per-layer bodies share one grid so they register against each other,
        # the way the viewer's per-layer volumes meet and stand apart.
        origin, voxel, dims = build_grid(coords, args.res, args.pad)
        for lab in sorted(set(layer)):
            m = (layer == lab)
            dsub = density[m] if density is not None else None
            safe = "".join(c if c.isalnum() else "_" for c in str(lab))
            p = generate_body(coords[m], dsub, origin, voxel, dims, args,
                              f"unify_{stamp}_{safe}", out_dir)
            if p:
                written.append(p)

    print(f"\n[done] wrote {len(written)} OBJ file(s) to {out_dir}")


if __name__ == "__main__":
    main()
