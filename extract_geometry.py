<<<<<<< HEAD
r"""
extract_geometry.py — figural geometry sidecar for the latent viewer.

Regenerates data/embeddings/content_geometry.json keyed by the CURRENT manifest
ids, so the worlds silhouettes and the quote system work again after a re-embed
(a re-embed regenerates every id, orphaning any previous sidecar).

Per canonical image it writes:
  sil    — outer figure silhouette polygon(s), [0,1]-relative, from a
           border-flood figure/ground segmentation (background grown inward
           from the frame edge across low-gradient, border-coloured pixels).
  inner  — a few strong interior contours (rooflines, profiles, mullions),
           [0,1]-relative, from gradient isolines within the figure.
  frags  — open curve fragments cut from sil+inner at high-curvature corners,
           each with a 16-bin turning-angle descriptor. This is the quote
           library: worlds boundaries get rebuilt from these fragments.

Schema (matches app.py exactly):
  { "items": { "<id>": { "sil": [[[x,y],...]], "inner": [...], "frags":
      [ { "p": [[x,y],...], "d": [16 floats], "c": int, "t": 0, "k": "free" } ] } } }
  All coordinates are image-relative in [0,1]. DESC_N must equal the viewer's
  QUOTE_DESC_N (16).

Run from the project root:
    python extract_geometry.py
    python extract_geometry.py --limit 50      # quick sample
=======
"""
extract_geometry.py — offline figural geometry extraction for the latent viewer.

Replaces the browser-side Sobel/marching-squares polygon extraction with a
proper pipeline, run once over the corpus. Writes a sidecar the viewer loads
at startup:

    data/embeddings/content_geometry.json

Per image (keyed by manifest id) the sidecar carries three registers, all in
normalized [0,1] x [0,1] image coordinates:

    sil    figural silhouettes (SAM masks if available, else the dominant
           closed contour of the learned edge map). These become the worlds
           form geometry (dom + full).
    inner  secondary closed contours (internal structure) drawn inside
           medoid forms.
    frags  open curve fragments traced from the thinned edge skeleton —
           arches, profiles, sweeps — each classified (line / arc / free)
           and carrying a turning-angle descriptor. These feed the
           curve-quoting boundary pass in the viewer.

Edge backends, best first:
    pidinet / hed   learned perceptual edges via controlnet_aux (pip install
                    controlnet-aux). Downloads weights from HF hub on first
                    run. Dramatically better than Sobel on drawings and
                    paintings, media-agnostic.
    xdog            classical fallback, no downloads (DoG + soft threshold).
    canny           last resort.

Silhouette backend (optional, biggest fidelity win):
    --sam PATH      segment-anything ViT-B checkpoint (sam_vit_b_01ec64.pth,
                    ~375 MB, pip install segment-anything). Runs on the
                    A3000 in fp32; ~1–2 s per image.

Usage (from project root, same place you run streamlit):

    python extract_geometry.py                          # pidinet edges only
    python extract_geometry.py --sam sam_vit_b_01ec64.pth
    python extract_geometry.py --edges xdog --limit 20  # quick smoke test

Re-run any time the corpus changes; the viewer picks the sidecar up on next
reload and falls back to the browser Sobel path for any id it doesn't cover.
>>>>>>> 803bd82c5ea402c3619f61257a4e90b232ff03a1
"""
from __future__ import annotations

import argparse
import json
import math
<<<<<<< HEAD
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

Image.MAX_IMAGE_PIXELS = None

# --- config ------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\Users\rbritain\Documents\All At Once\Histories")
MANIFEST_PATH = PROJECT_ROOT / "data" / "catalog" / "manifest.json"
INDEX_PATH = PROJECT_ROOT / "data" / "embeddings" / "index.json"
OUT_PATH = PROJECT_ROOT / "data" / "embeddings" / "content_geometry.json"

WORK = 256           # analysis resolution (long edge); silhouettes are relative
DESC_N = 16          # MUST equal app.py QUOTE_DESC_N
GRAD_TOL_FRAC = 0.22  # flood crosses pixels with gradient below this * maxGrad
COLOR_TOL = 70       # and within this L1 colour distance of the border palette
MAX_SIL = 2
MAX_INNER = 4
MAX_FRAGS = 10
CORNER_ANGLE = 0.9   # radians; split a contour into fragments at sharper turns
MIN_FRAG_PTS = 5
# -----------------------------------------------------------------------------


def long(p) -> str:
    s = os.path.abspath(str(p))
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        return "\\\\?\\" + s
    return s


def resolve_path(p: str) -> Path:
    return PROJECT_ROOT / p


def load_rgb_small(path):
    im = Image.open(long(path))
    try:
        im.draft("RGB", (WORK * 2, WORK * 2))
    except Exception:
        pass
    im = im.convert("RGB")
    im.thumbnail((WORK, WORK), Image.BILINEAR)
    return im


def sobel_mag(gray: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    return np.hypot(gx, gy)


def marching_squares(field: np.ndarray, level: float):
    """Minimal marching squares -> list of open/closed point paths (px coords)."""
    h, w = field.shape
    segs = []
    f = field
    for y in range(h - 1):
        for x in range(w - 1):
            tl = f[y, x] >= level
            tr = f[y, x + 1] >= level
            br = f[y + 1, x + 1] >= level
            bl = f[y + 1, x] >= level
            idx = (tl << 3) | (tr << 2) | (br << 1) | bl
            if idx == 0 or idx == 15:
                continue

            def ip(ax, ay, bx, by):
                fa, fb = f[ay, ax], f[by, bx]
                t = 0.5 if abs(fb - fa) < 1e-9 else (level - fa) / (fb - fa)
                return (ax + (bx - ax) * t, ay + (by - ay) * t)

            top = ip(x, y, x + 1, y)
            rgt = ip(x + 1, y, x + 1, y + 1)
            bot = ip(x, y + 1, x + 1, y + 1)
            lft = ip(x, y, x, y + 1)
            table = {
                1: [(lft, bot)], 2: [(bot, rgt)], 3: [(lft, rgt)],
                4: [(top, rgt)], 5: [(lft, top), (bot, rgt)], 6: [(top, bot)],
                7: [(lft, top)], 8: [(top, lft)], 9: [(top, bot)],
                10: [(top, rgt), (lft, bot)], 11: [(top, rgt)],
                12: [(rgt, lft)], 13: [(bot, rgt)], 14: [(lft, bot)],
            }
            for a, b in table.get(idx, []):
                segs.append((a, b))
    return stitch(segs)


def stitch(segs, tol=0.9):
    paths = []
    used = [False] * len(segs)
    key = lambda p: (round(p[0] / tol), round(p[1] / tol))
    start_map = {}
    for i, (a, b) in enumerate(segs):
        start_map.setdefault(key(a), []).append((i, False))
        start_map.setdefault(key(b), []).append((i, True))
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        a, b = segs[i]
        path = [a, b]
        # extend forward
        while True:
            k = key(path[-1])
            nxt = None
            for j, rev in start_map.get(k, []):
                if not used[j]:
                    nxt = (j, rev)
                    break
            if nxt is None:
                break
            j, rev = nxt
            used[j] = True
            sa, sb = segs[j]
            path.append(sa if rev else sb)
        if len(path) >= 3:
            paths.append(path)
    return paths


def poly_area_frac(pts, w, h):
    a = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return abs(a / 2) / (w * h)


def border_flood_silhouette(im: Image.Image):
    arr = np.asarray(im, dtype=np.float32)
    h, w, _ = arr.shape
    gray = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    blur = np.asarray(im.convert("L").filter(ImageFilter.GaussianBlur(1.2)),
                      dtype=np.float32)
    grad = sobel_mag(blur)
    gmax = float(grad.max()) or 1.0
    grad_tol = gmax * GRAD_TOL_FRAC

    ring = np.concatenate([
        arr[:2, :, :].reshape(-1, 3), arr[-2:, :, :].reshape(-1, 3),
        arr[:, :2, :].reshape(-1, 3), arr[:, -2:, :].reshape(-1, 3)])
    med = np.median(ring, axis=0)
    coldist = np.abs(arr - med).sum(axis=2)

    bg = np.zeros((h, w), dtype=bool)
    from collections import deque
    q = deque()
    for x in range(w):
        for y in (0, h - 1):
            if coldist[y, x] < COLOR_TOL * 1.4:
                bg[y, x] = True
                q.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not bg[y, x] and coldist[y, x] < COLOR_TOL * 1.4:
                bg[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not bg[ny, nx]:
                if grad[ny, nx] < grad_tol and coldist[ny, nx] < COLOR_TOL:
                    bg[ny, nx] = True
                    q.append((ny, nx))

    fig = (~bg).astype(np.float32)
    frac = float(fig.mean())
    if frac < 0.02 or frac > 0.96:
        return [], grad, gmax
    fig_img = Image.fromarray((fig * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(1.4))
    fig = np.asarray(fig_img, dtype=np.float32) / 255.0

    paths = marching_squares(fig, 0.5)
    sils = []
    for pts in paths:
        if len(pts) < 8:
            continue
        if poly_area_frac(pts, w, h) < 0.015:
            continue
        sils.append(simplify_any(pts, 0.8))
    sils.sort(key=lambda p: -poly_area_frac(p, w, h))
    rel = [[[float(x) / w, float(y) / h] for (x, y) in s] for s in sils[:MAX_SIL]]
    return rel, grad, gmax


def inner_contours(grad, gmax, w, h):
    level = 0.30 * gmax
    if level <= 0:
        return []
    paths = marching_squares(grad, level)
    margin = max(3, int(min(w, h) * 0.07))
    out = []
    for pts in paths:
        if len(pts) < 6:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        spanx = (max(xs) - min(xs)) / w
        spany = (max(ys) - min(ys)) / h
        nb = sum(1 for (x, y) in pts
                 if x <= margin or x >= w - margin or y <= margin or y >= h - margin)
        if spanx > 0.85 and spany > 0.85 and nb / len(pts) > 0.6:
            continue
        sp = simplify_any(pts, 0.8)
        if len(sp) < 3:
            continue
        out.append(((max(xs) - min(xs)) + (max(ys) - min(ys)),
                    [[float(x) / w, float(y) / h] for (x, y) in sp]))
    out.sort(key=lambda o: -o[0])
    return [o[1] for o in out[:MAX_INNER]]


def simplify(pts, eps):
    """Ramer-Douglas-Peucker on a polyline of (x,y)."""
    if len(pts) < 3:
        return pts

    def rdp(a, b, lo, hi):
        dmax, idx = 0.0, -1
        ax, ay = pts[a]
        bx, by = pts[b]
        dx, dy = bx - ax, by - ay
        nlen = math.hypot(dx, dy) or 1e-9
        for i in range(lo, hi):
            px, py = pts[i]
            d = abs((px - ax) * dy - (py - ay) * dx) / nlen
            if d > dmax:
                dmax, idx = d, i
        if dmax > eps and idx != -1:
            left = rdp(a, idx, a + 1, idx)
            right = rdp(idx, b, idx + 1, b)
            return left[:-1] + right
        return [pts[a], pts[b]]

    return rdp(0, len(pts) - 1, 1, len(pts) - 1)


def _is_closed(pts):
    return (len(pts) > 3 and
            math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 2.0)


def simplify_any(pts, eps):
    """RDP that handles closed contours (plain RDP collapses them: coincident
    endpoints make a degenerate baseline). Splits a closed loop at its farthest
    point and RDPs each arc."""
    if len(pts) < 4:
        return pts
    if not _is_closed(pts):
        return simplify(pts, eps)
    p = pts[:-1] if pts[0] == pts[-1] else pts[:]
    n = len(p)
    far = max(range(n), key=lambda i: (p[i][0] - p[0][0]) ** 2 + (p[i][1] - p[0][1]) ** 2)
    arc1 = p[0:far + 1]
    arc2 = p[far:] + [p[0]]
    s1 = simplify(arc1, eps)
    s2 = simplify(arc2, eps)
    return s1[:-1] + s2[:-1]


def resample_open(pts, n):
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + math.hypot(pts[i][0] - pts[i - 1][0],
                                        pts[i][1] - pts[i - 1][1]))
    total = cum[-1]
    if total < 1e-9:
        return [list(pts[0]) for _ in range(n)]
    out, j = [], 0
    for i in range(n):
        t = total * i / (n - 1)
        while j < len(pts) - 2 and cum[j + 1] < t:
            j += 1
        u = (t - cum[j]) / max(1e-9, cum[j + 1] - cum[j])
        out.append([pts[j][0] + (pts[j + 1][0] - pts[j][0]) * u,
                    pts[j][1] + (pts[j + 1][1] - pts[j][1]) * u])
    return out


def turning_descriptor(pts):
    """16-bin turning-angle descriptor — identical math to app.py turningDescOf."""
    rs = resample_open(pts, DESC_N + 2)
    d = [0.0] * DESC_N
    prev = math.atan2(rs[1][1] - rs[0][1], rs[1][0] - rs[0][0])
    for i in range(1, DESC_N + 1):
        a = math.atan2(rs[i + 1][1] - rs[i][1], rs[i + 1][0] - rs[i][0])
        t = a - prev
        while t > math.pi:
            t -= 2 * math.pi
        while t < -math.pi:
            t += 2 * math.pi
        d[i - 1] = t
        prev = a
    return d


def split_fragments(poly):
    """Cut a (relative-coord) polygon into open fragments at sharp corners."""
    n = len(poly)
    if n < MIN_FRAG_PTS:
        return []
    cuts = [0]
    for i in range(1, n - 1):
        ax, ay = poly[i - 1]
        bx, by = poly[i]
        cx, cy = poly[i + 1]
        a1 = math.atan2(by - ay, bx - ax)
        a2 = math.atan2(cy - by, cx - bx)
        t = a2 - a1
        while t > math.pi:
            t -= 2 * math.pi
        while t < -math.pi:
            t += 2 * math.pi
        if abs(t) > CORNER_ANGLE:
            cuts.append(i)
    cuts.append(n - 1)
    # smooth loop with no corners: fall back to even arcs so domes / smooth
    # sculpture still contribute quotable fragments
    if len(cuts) == 2 and n >= 3 * MIN_FRAG_PTS:
        arcs = 3
        cuts = [round(j * (n - 1) / arcs) for j in range(arcs + 1)]
    frags = []
    for a, b in zip(cuts, cuts[1:]):
        seg = poly[a:b + 1]
        if len(seg) >= MIN_FRAG_PTS:
            frags.append(seg)
    return frags


def build_frags(sil, inner):
    frags = []
    for poly in sil + inner:
        for seg in split_fragments(poly):
            frags.append({
                "p": [[round(float(x), 5), round(float(y), 5)] for x, y in seg],
                "d": [round(float(v), 5) for v in turning_descriptor(seg)],
                "c": 1, "t": 0, "k": "free",
            })
            if len(frags) >= MAX_FRAGS:
                return frags
    return frags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)
    rows = [m for m in manifest if m.get("canonical")]
    path_of = {m["id"]: m["path"] for m in rows}

    if INDEX_PATH.exists():
        with open(INDEX_PATH) as f:
            ids = json.load(f)["ids"]
    else:
        ids = [m["id"] for m in rows]
    if args.limit:
        ids = ids[:args.limit]

    items = {}
    n_sil = 0
    for k, i in enumerate(ids):
        p = path_of.get(i)
        if not p:
            continue
        try:
            im = load_rgb_small(str(resolve_path(p)))
            sil, grad, gmax = border_flood_silhouette(im)
            w, h = im.size
            inner = inner_contours(grad, gmax, w, h) if sil else []
            frags = build_frags(sil, inner)
            if sil or inner:
                items[i] = {"sil": sil, "inner": inner, "frags": frags}
                if sil:
                    n_sil += 1
        except Exception as exc:
            print(f"  skip {i}: {exc}")
        if (k + 1) % 200 == 0:
            print(f"  {k + 1}/{len(ids)}  ({n_sil} with silhouette)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"items": items, "desc_n": DESC_N}, f, ensure_ascii=False)
    print(f"\nwrote {OUT_PATH}")
    print(f"{len(items)}/{len(ids)} items, {n_sil} with a figure silhouette")
    print("restart Streamlit + hard refresh; set worlds silhouettes = auto or sidecar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
=======
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    sys.exit("extract_geometry.py requires opencv: pip install opencv-contrib-python")

from PIL import Image

# ----------------------------------------------------------------------------
# Paths (mirrors app.py's project-root convention)
# ----------------------------------------------------------------------------

def find_project_root() -> Path:
    """Prefer the invocation directory (matches how streamlit is run), then
    the script's own directory. Overridable with --root."""
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for candidate in [start, *start.parents][:5]:
            if (candidate / "data" / "catalog" / "manifest.json").exists():
                return candidate
    return Path.cwd()

PROJECT_ROOT = find_project_root()
MANIFEST_PATH = PROJECT_ROOT / "data" / "catalog" / "manifest.json"
OUT_PATH = PROJECT_ROOT / "data" / "embeddings" / "content_geometry.json"

MAX_EDGE = 768          # working resolution for extraction
Q = 4                   # coordinate quantization (decimal places)
DESC_N = 16             # turning-angle descriptor length
MIN_FRAG_LEN = 0.055    # min fragment length as fraction of image diagonal
MAX_FRAGS = 14          # fragments kept per image
MAX_SIL = 3             # silhouettes kept per image
MAX_INNER = 4           # inner contours kept per image


# ----------------------------------------------------------------------------
# Edge backends
# ----------------------------------------------------------------------------

class EdgeBackend:
    def __init__(self, name: str, device: str):
        self.name = name
        self.detector = None
        if name in ("pidinet", "hed"):
            try:
                if name == "pidinet":
                    from controlnet_aux import PidiNetDetector
                    self.detector = PidiNetDetector.from_pretrained("lllyasviel/Annotators")
                else:
                    from controlnet_aux import HEDdetector
                    self.detector = HEDdetector.from_pretrained("lllyasviel/Annotators")
                try:
                    self.detector.to(device)
                except Exception:
                    pass
                print(f"[edges] {name} loaded on {device}")
            except Exception as e:
                print(f"[edges] {name} unavailable ({e}); falling back to xdog")
                self.name = "xdog"
        if self.name == "xdog":
            print("[edges] xdog (classical DoG) backend")
        elif self.name == "canny":
            print("[edges] canny backend")

    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        """Return float32 edge probability in [0,1], same HxW as input."""
        h, w = bgr.shape[:2]
        if self.detector is not None:
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            out = self.detector(pil, safe=False)  # PIL, white edges on black
            e = np.asarray(out.convert("L"), dtype=np.float32) / 255.0
            if e.shape[:2] != (h, w):
                e = cv2.resize(e, (w, h), interpolation=cv2.INTER_LINEAR)
            return e
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if self.name == "xdog":
            s = max(1.0, min(h, w) / 400.0)
            g1 = cv2.GaussianBlur(gray, (0, 0), 0.8 * s)
            g2 = cv2.GaussianBlur(gray, (0, 0), 1.6 * s)
            d = g1 - 0.97 * g2
            e = 1.0 - (1.0 + np.tanh(40.0 * np.minimum(d, 0)))  # dark-side response
            e = np.clip(e, 0, 1)
            e = (e - e.min()) / max(1e-6, e.max() - e.min())
            return e.astype(np.float32)
        edges = cv2.Canny((gray * 255).astype(np.uint8), 60, 160)
        return (edges.astype(np.float32) / 255.0)


# ----------------------------------------------------------------------------
# SAM silhouettes (optional)
# ----------------------------------------------------------------------------

class SamBackend:
    def __init__(self, ckpt: Path, device: str):
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        model_type = "vit_b" if "vit_b" in ckpt.name else ("vit_l" if "vit_l" in ckpt.name else "vit_h")
        sam = sam_model_registry[model_type](checkpoint=str(ckpt))
        sam.to(device)
        self.gen = SamAutomaticMaskGenerator(
            sam,
            points_per_side=16,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.90,
            min_mask_region_area=int(0.004 * MAX_EDGE * MAX_EDGE),
        )
        print(f"[sam] {model_type} loaded on {device}")

    def silhouettes(self, bgr: np.ndarray) -> list[np.ndarray]:
        """Return up to MAX_SIL binary masks, figural, largest first."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        masks = self.gen.generate(rgb)
        area_img = h * w
        keep = []
        for m in masks:
            a = m["area"] / area_img
            if a < 0.02 or a > 0.88:
                continue  # speckle or whole-frame
            seg = m["segmentation"].astype(np.uint8)
            # reject frame-huggers: high share of mask perimeter on the border
            border = np.zeros_like(seg)
            t = max(2, min(h, w) // 60)
            border[:t, :] = 1; border[-t:, :] = 1; border[:, :t] = 1; border[:, -t:] = 1
            if (seg & border).sum() > 0.30 * max(1, cv2.countNonZero(cv2.Canny(seg * 255, 0, 1))):
                # cheap proxy; fall through to a softer check on bbox
                x, y, bw, bh = cv2.boundingRect(seg)
                if bw > 0.96 * w and bh > 0.96 * h:
                    continue
            keep.append((m["area"], seg))
        keep.sort(key=lambda t: -t[0])
        # drop masks nearly contained in an already-kept larger mask
        out = []
        for _, seg in keep:
            dup = False
            for prev in out:
                inter = np.logical_and(seg, prev).sum()
                if inter > 0.85 * seg.sum():
                    dup = True
                    break
            if not dup:
                out.append(seg)
            if len(out) >= MAX_SIL:
                break
        return out


# ----------------------------------------------------------------------------
# Vectorization helpers
# ----------------------------------------------------------------------------

def contour_to_norm(cnt: np.ndarray, w: int, h: int) -> list[list[float]]:
    pts = cnt.reshape(-1, 2).astype(np.float64)
    return [[round(x / w, Q), round(y / h, Q)] for x, y in pts]


def mask_to_polys(mask: np.ndarray, w: int, h: int, eps_frac=0.006) -> list[list[list[float]]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diag = math.hypot(w, h)
    out = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(c) < 0.004 * w * h:
            continue
        ap = cv2.approxPolyDP(c, eps_frac * diag, True)
        if len(ap) >= 4:
            out.append(contour_to_norm(ap, w, h))
    return out


def skeletonize(binary: np.ndarray) -> np.ndarray:
    """Thin to 1-px skeleton. Uses ximgproc if present, else morphological."""
    if hasattr(cv2, "ximgproc"):
        try:
            return cv2.ximgproc.thinning(binary)
        except Exception:
            pass
    img = binary.copy()
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, opened))
        img = eroded
        if cv2.countNonZero(img) == 0:
            return skel


_NBRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def trace_fragments(skel: np.ndarray) -> list[np.ndarray]:
    """Walk the skeleton into open polylines. Chains run node-to-node, where
    a node is any pixel of degree != 2 (endpoint or junction); leftover pure
    cycles are traced afterwards. Edges are marked visited, never pixels, so
    every skeleton segment lands in exactly one chain."""
    on = skel > 0
    ys, xs = np.nonzero(on)
    coords = set(zip(ys.tolist(), xs.tolist()))

    def neighbors(p):
        y, x = p
        for dy, dx in _NBRS:
            q = (y + dy, x + dx)
            if q in coords:
                yield q

    deg = {p: sum(1 for _ in neighbors(p)) for p in coords}
    nodes = {p for p in coords if deg[p] != 2}
    visited: set[frozenset] = set()

    def edge(a, b):
        return frozenset((a, b))

    frags = []

    def emit(path):
        if len(path) >= 4:
            frags.append(np.array([[c[1], c[0]] for c in path], dtype=np.float64))

    # node-to-node chains
    for p in nodes:
        for q in neighbors(p):
            if edge(p, q) in visited:
                continue
            path = [p, q]
            visited.add(edge(p, q))
            prev, cur = p, q
            while cur not in nodes:
                nxt = None
                for r in neighbors(cur):
                    if r != prev and edge(cur, r) not in visited:
                        nxt = r
                        break
                if nxt is None:
                    break
                visited.add(edge(cur, nxt))
                path.append(nxt)
                prev, cur = cur, nxt
            emit(path)

    # leftover cycles (closed curves with no junction anywhere)
    for p in coords:
        if deg[p] != 2:
            continue
        for q in neighbors(p):
            if edge(p, q) in visited:
                continue
            path = [p, q]
            visited.add(edge(p, q))
            prev, cur = p, q
            while cur != p:
                nxt = None
                for r in neighbors(cur):
                    if r != prev and edge(cur, r) not in visited:
                        nxt = r
                        break
                if nxt is None:
                    break
                visited.add(edge(cur, nxt))
                path.append(nxt)
                prev, cur = cur, nxt
            emit(path)
    return frags


def merge_chains(frags: list[np.ndarray], gap_tol=4.0, angle_tol=math.radians(38)) -> list[np.ndarray]:
    """Merge skeleton chains across noisy junctions. Two chains join when
    their endpoints nearly touch and their tangents continue through the
    junction; genuine corners (pillar meets arch) stay split."""
    chains = [f.copy() for f in frags]

    def outward(f, at_start):
        k = min(4, len(f) - 1)
        v = (f[0] - f[k]) if at_start else (f[-1] - f[-1 - k])
        n = np.linalg.norm(v)
        return v / n if n > 1e-9 else v

    merged = True
    while merged and len(chains) > 1:
        merged = False
        best = None  # (score, i, ei, j, ej)
        for i in range(len(chains)):
            for j in range(i + 1, len(chains)):
                for ei in (0, 1):
                    for ej in (0, 1):
                        p = chains[i][0] if ei == 0 else chains[i][-1]
                        q = chains[j][0] if ej == 0 else chains[j][-1]
                        d = np.linalg.norm(p - q)
                        if d > gap_tol:
                            continue
                        ti = outward(chains[i], ei == 0)
                        tj = outward(chains[j], ej == 0)
                        # continuing straight through: outward tangents opposed
                        ang = math.acos(max(-1.0, min(1.0, float(np.dot(ti, tj)))))
                        dev = math.pi - ang
                        if dev > angle_tol:
                            continue
                        score = dev + d * 0.05
                        if best is None or score < best[0]:
                            best = (score, i, ei, j, ej)
        if best is not None:
            _, i, ei, j, ej = best
            a = chains[i] if ei == 1 else chains[i][::-1]   # a ends at junction
            b = chains[j] if ej == 0 else chains[j][::-1]   # b starts at junction
            chains[i] = np.vstack([a, b])
            del chains[j]
            merged = True
    return chains


def resample(pts: np.ndarray, n: int) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-9:
        return np.repeat(pts[:1], n, axis=0)
    t = np.linspace(0, total, n)
    x = np.interp(t, cum, pts[:, 0])
    y = np.interp(t, cum, pts[:, 1])
    return np.stack([x, y], axis=1)


def turning_descriptor(pts: np.ndarray, n: int = DESC_N) -> tuple[list[float], float, float]:
    """Resampled turning-angle vector, total absolute turning, arclen/chord."""
    rs = resample(pts, n + 2)
    v = np.diff(rs, axis=0)
    ang = np.arctan2(v[:, 1], v[:, 0])
    turn = np.diff(ang)
    turn = np.arctan2(np.sin(turn), np.cos(turn))  # wrap to [-pi, pi]
    arclen = np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
    chord = np.linalg.norm(pts[-1] - pts[0])
    ratio = float(arclen / max(chord, 1e-9))
    return [round(float(a), 4) for a in turn], round(float(np.abs(turn).sum()), 4), round(ratio, 4)


def classify(pts: np.ndarray) -> str:
    """line / arc / free by fit residuals (Kasa circle fit)."""
    rs = resample(pts, 24)
    # line: PCA residual
    c = rs - rs.mean(axis=0)
    _, s, _ = np.linalg.svd(c, full_matrices=False)
    scale = max(1e-9, np.linalg.norm(rs[-1] - rs[0]))
    line_res = s[1] / scale
    if line_res < 0.015:
        return "line"
    # circle: Kasa
    x, y = rs[:, 0], rs[:, 1]
    A = np.stack([x, y, np.ones_like(x)], axis=1)
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy = sol[0] / 2, sol[1] / 2
        r = math.sqrt(max(1e-12, sol[2] + cx * cx + cy * cy))
        res = np.abs(np.hypot(x - cx, y - cy) - r).mean() / scale
        if res < 0.02:
            return "arc"
    except Exception:
        pass
    return "free"


# ----------------------------------------------------------------------------
# Per-image extraction
# ----------------------------------------------------------------------------

def extract_one(bgr: np.ndarray, edges: EdgeBackend, sam: SamBackend | None) -> dict:
    h, w = bgr.shape[:2]
    diag = math.hypot(w, h)
    e = edges(bgr)

    # --- fragments: skeleton of the strong-edge set --------------------------
    # cap the adaptive threshold: learned/xdog maps saturate near 1.0 on clean
    # drawings, and a quantile alone would then keep almost nothing
    q80 = float(np.quantile(e[e > 0.02], 0.80)) if (e > 0.02).any() else 1.0
    thresh = float(np.clip(q80, 0.12, 0.35))
    binary = (e >= thresh).astype(np.uint8) * 255
    # Line art (drawings, prints, construction documents): edge detectors
    # respond on both sides of a stroke, and skeletonizing that double band
    # traces the stroke's outline instead of its centerline. When the image
    # reads as high-contrast line art, OR the dark strokes in directly so
    # strokes are solid and the skeleton is the true centerline.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dark_frac = float((gray < 96).mean())
    if 0.0005 < dark_frac < 0.30:
        binary = cv2.bitwise_or(binary, ((gray < 128).astype(np.uint8) * 255))
    kclose = max(3, int(diag * 0.009) | 1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kclose, kclose)))
    skel = skeletonize(binary)
    raw = trace_fragments(skel)
    # drop micro-chains (skeleton noise) before merging, then rejoin chains
    # that continue through junctions
    raw = [f for f in raw
           if np.linalg.norm(np.diff(f, axis=0), axis=1).sum() > 0.012 * diag]
    raw = merge_chains(raw, gap_tol=max(3.0, diag * 0.006))
    scored = []
    for f in raw:
        arclen = np.linalg.norm(np.diff(f, axis=0), axis=1).sum()
        if arclen < MIN_FRAG_LEN * diag:
            continue
        scored.append((arclen, f))
    scored.sort(key=lambda t: -t[0])
    frags = []
    for arclen, f in scored[:MAX_FRAGS]:
        kind = classify(f)
        if kind == "line":
            pts = np.stack([f[0], f[-1]])
        else:
            # keep curve character: resample the raw chain, density by length
            n = int(np.clip(round(arclen / (diag * 0.012)), 8, 24))
            pts = resample(f, n)
        desc, tot, ratio = turning_descriptor(f)   # descriptor from the raw chain
        frags.append({
            "p": [[round(px / w, Q), round(py / h, Q)] for px, py in pts],
            "k": kind,
            "d": desc,
            "t": tot,
            "c": ratio,
        })

    # --- silhouettes ----------------------------------------------------------
    sils: list[list[list[float]]] = []
    if sam is not None:
        try:
            for mask in sam.silhouettes(bgr):
                sils.extend(mask_to_polys(mask * 255, w, h))
        except Exception as ex:
            print(f"  [sam] failed on image ({ex}); edge-map silhouette fallback")
    if not sils:
        # closed regions of the blurred edge field — better than nothing,
        # and still learned-edge quality when pidinet/hed is active
        soft = cv2.GaussianBlur(e, (0, 0), diag * 0.004)
        m = (soft >= max(0.10, float(np.quantile(soft, 0.86)))).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        margin = max(3, int(min(w, h) * 0.05))
        m[:margin, :] = 0; m[-margin:, :] = 0; m[:, :margin] = 0; m[:, -margin:] = 0
        sils = mask_to_polys(m, w, h)
    sils = sils[:MAX_SIL]

    # --- inner contours: next closed contours of the edge field ---------------
    inner: list[list[list[float]]] = []
    binary_in = (e >= max(0.14, thresh * 0.7)).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(binary_in, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts[1:1 + MAX_INNER * 2]:
        if cv2.contourArea(c) < 0.002 * w * h:
            break
        ap = cv2.approxPolyDP(c, 0.004 * diag, True)
        if len(ap) >= 4:
            inner.append(contour_to_norm(ap, w, h))
        if len(inner) >= MAX_INNER:
            break

    return {"sil": sils, "inner": inner, "frags": frags}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", choices=["pidinet", "hed", "xdog", "canny"], default="pidinet")
    ap.add_argument("--sam", type=str, default=None, help="path to SAM checkpoint (vit_b recommended)")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N images (smoke test)")
    ap.add_argument("--root", type=str, default=None, help="project root (contains data/catalog/manifest.json)")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    global PROJECT_ROOT, MANIFEST_PATH, OUT_PATH
    if args.root:
        PROJECT_ROOT = Path(args.root).resolve()
        MANIFEST_PATH = PROJECT_ROOT / "data" / "catalog" / "manifest.json"
        OUT_PATH = PROJECT_ROOT / "data" / "embeddings" / "content_geometry.json"
    if args.out is None:
        args.out = str(OUT_PATH)

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if not MANIFEST_PATH.exists():
        sys.exit(f"manifest not found at {MANIFEST_PATH} — run from the project root")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if args.limit:
        manifest = manifest[: args.limit]

    edges = EdgeBackend(args.edges, device)
    sam = SamBackend(Path(args.sam), device) if args.sam else None
    if sam is None:
        print("[sam] disabled — silhouettes fall back to edge-field regions. "
              "Pass --sam sam_vit_b_01ec64.pth for figural masks.")

    items: dict[str, dict] = {}
    t0 = time.time()
    n_ok = n_fail = 0
    for i, row in enumerate(manifest):
        pid = str(row.get("id"))
        rel = row.get("path")
        if not rel:
            continue
        p = PROJECT_ROOT / rel
        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail((MAX_EDGE, MAX_EDGE), Image.LANCZOS)
            bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            items[pid] = extract_one(bgr, edges, sam)
            n_ok += 1
        except Exception as ex:
            n_fail += 1
            print(f"  [skip] {pid}: {ex}")
        if (i + 1) % 25 == 0:
            rate = (i + 1) / max(1e-6, time.time() - t0)
            eta = (len(manifest) - i - 1) / max(1e-6, rate)
            print(f"  {i + 1}/{len(manifest)}  ({rate:.1f} img/s, eta {eta / 60:.1f} min)")

    out = {
        "version": 1,
        "backend": {"edges": edges.name, "sam": bool(sam)},
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "items": items,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    mb = out_path.stat().st_size / 1e6
    print(f"\nwrote {out_path}  ({n_ok} images, {n_fail} skipped, {mb:.1f} MB)")
    print("reload the streamlit app; worlds mode will pick the sidecar up automatically.")


if __name__ == "__main__":
    main()
>>>>>>> 803bd82c5ea402c3619f61257a4e90b232ff03a1
