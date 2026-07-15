"""
articulate_mesh.py  --  run from the project root (Histories\\).

Post-processes an OBJ from unify_volume_gen.py to give the surface drawn
character: straight lines where the deposit is dense, gestural sweeps where the
phantom is. Operates on the g_dog / g_rabbit groups the generator writes.

    python articulate_mesh.py data\\exports\\unify_solid_u0.24.obj

Why this is a separate pass. The generator's articulate step quantizes dog
vertices to a grid, which does not produce straight lines: on a mesh already
triangulated at voxel scale it produces a stair-step staircase plus thousands of
zero-area slivers (a 4449-point bake came out with 10,171 degenerate faces and
5,138 duplicate vertices, 21 percent of the mesh). Straight lines come from
consolidating coplanar triangles into large facets, which is a mesh operation,
not a vertex-snapping operation.

The pass runs in this order, and the order is the point:

  weld     merge coincident vertices, drop the degenerate slivers. Nothing
           downstream is meaningful on a torn mesh.
  destair  a few smoothing iterations on the dog region only, to erase the
           voxel staircase left by the bake's quantization. The staircase is
           high-frequency noise at voxel scale, not form.
  facet    snap each dog face normal to one of a few canonical directions AND
           quantize its plane offset, then move vertices onto those planes.
           Quantizing the offset is what makes neighbouring parallel faces land
           on a SHARED plane and fuse into one large facet. Snapping the normal
           alone leaves hundreds of tiny parallel steps.
  sweep    Taubin-smooth the rabbit region with the dogs pinned as anchors, so
           the phantom edge flows off the arrested deposit.
  decimate LAST, not first. Quadric error is near zero across a flat plane, so
           the faceted dog planes collapse into a few big triangles with long
           straight edges, while the curved rabbit sweeps keep the triangles
           they need to stay smooth. Decimating before the character treatment
           destroys both.

Dogs are the accumulated deposit, so they arrest into planes and hard creases.
Rabbits are the phantom that will not resolve, so they stay in motion. The
distinction survives into the geometry rather than only the file's group tags.

Writes <name>_articulated.obj with g_dog / g_rabbit preserved and vertex normals
written per group (flat across dog facets, smoothed across rabbit sweeps) so the
character reads immediately on import instead of needing shading tweaks.

Dependencies: numpy, scipy, fast-simplification
  pip install numpy scipy fast-simplification
"""

import argparse
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- io ---------
def load_obj(path):
    V, F, groups, cur = [], [], {}, None
    for ln in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if ln.startswith("v "):
            V.append([float(x) for x in ln[2:].split()[:3]])
        elif ln.startswith("g "):
            cur = ln[2:].strip()
            groups.setdefault(cur, [])
        elif ln.startswith("f "):
            idx = [int(t.split("/")[0]) - 1 for t in ln[2:].split()]
            for k in range(1, len(idx) - 1):            # fan-triangulate
                F.append([idx[0], idx[k], idx[k + 1]])
                if cur is not None:
                    groups[cur].append(len(F) - 1)
    if not V or not F:
        sys.exit(f"[abort] {path} has no geometry")
    return np.array(V, float), np.array(F, np.int64), groups


def write_obj(path, V, F, dog_face, header, smooth_normals=True):
    N = face_normals(V, F)
    lines = [f"# {h}" for h in header]
    for v in V:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    if smooth_normals:
        # rabbit vertices get an averaged normal (sweeps read smooth); dog
        # faces keep their own face normal (facets read flat and crisp).
        vn = np.zeros_like(V)
        rab = ~dog_face
        if rab.any():
            for j in range(3):
                np.add.at(vn, F[rab][:, j], N[rab])
        ln_ = np.linalg.norm(vn, axis=1, keepdims=True)
        ln_[ln_ == 0] = 1
        vn = vn / ln_
        normals = [vn[i] for i in range(len(V))]        # 1..len(V)
        base = len(normals)
        face_n_idx = []
        for fi, f in enumerate(F):
            if dog_face[fi]:
                normals.append(N[fi])                   # flat normal per facet
                face_n_idx.append([base + len(normals) - base] * 3)
            else:
                face_n_idx.append([f[0] + 1, f[1] + 1, f[2] + 1])
        for n in normals:
            lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")

        def emit(sel, name):
            lines.append(f"g {name}")
            for fi in np.where(sel)[0]:
                f = F[fi]
                ni = face_n_idx[fi]
                lines.append(f"f {f[0]+1}//{ni[0]} {f[1]+1}//{ni[1]} {f[2]+1}//{ni[2]}")
        emit(dog_face, "g_dog")
        emit(~dog_face, "g_rabbit")
    else:
        for name, sel in (("g_dog", dog_face), ("g_rabbit", ~dog_face)):
            lines.append(f"g {name}")
            for f in F[sel]:
                lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")

    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------- geometry ------
def face_normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    l = np.linalg.norm(n, axis=1, keepdims=True)
    l[l == 0] = 1
    return n / l


def weld(V, F, tol):
    """Merge coincident vertices, drop degenerate faces."""
    key = np.round(V / tol).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    nV = np.zeros((inv.max() + 1, 3))
    cnt = np.zeros(inv.max() + 1)
    np.add.at(nV, inv, V)
    np.add.at(cnt, inv, 1)
    nV /= cnt[:, None]
    nF = inv[F]
    ok = (nF[:, 0] != nF[:, 1]) & (nF[:, 1] != nF[:, 2]) & (nF[:, 0] != nF[:, 2])
    return nV, nF[ok], ok


def adjacency(n, F):
    from scipy.sparse import coo_matrix, diags
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    A = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    d = np.asarray(A.sum(1)).ravel()
    d[d == 0] = 1
    return diags(1.0 / d) @ A


def taubin(V, F, vmask, iters, lam=0.5, mu=-0.53):
    """Volume-preserving smoothing. Only vmask vertices move."""
    if iters <= 0 or not vmask.any():
        return V
    A = adjacency(len(V), F)
    V = V.copy()
    m = vmask.astype(float)[:, None]
    for _ in range(iters):
        V = V + m * (lam * (A @ V - V))
        V = V + m * (mu * (A @ V - V))
    return V


def canonical_dirs(k):
    d = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
    if k >= 14:
        d += [[a,b,c] for a in (1,-1) for b in (1,-1) for c in (1,-1)]
    if k >= 26:
        d += [[a,b,0] for a in (1,-1) for b in (1,-1)]
        d += [[a,0,c] for a in (1,-1) for c in (1,-1)]
        d += [[0,b,c] for b in (1,-1) for c in (1,-1)]
    d = np.array(d, float)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def facet(V, F, fmask, iters, k_dirs, step, strength=1.0):
    """Snap dog face normals to canonical directions and quantize their plane
    offsets, then pull vertices onto those planes. The offset quantization is
    what fuses parallel neighbours onto one shared plane."""
    if iters <= 0 or not fmask.any() or step <= 0:
        return V
    D = canonical_dirs(k_dirs)
    V = V.copy()
    sel = np.where(fmask)[0]
    for _ in range(iters):
        N = face_normals(V, F)
        tgt = D[np.argmax(N[sel] @ D.T, axis=1)]
        C = V[F[sel]].mean(axis=1)
        dq = np.round(np.einsum("ij,ij->i", tgt, C) / step) * step
        disp = np.zeros_like(V)
        cnt = np.zeros(len(V))
        for j in range(3):
            vi = F[sel][:, j]
            delta = dq - np.einsum("ij,ij->i", tgt, V[vi])
            np.add.at(disp, vi, tgt * delta[:, None])
            np.add.at(cnt, vi, 1)
        cnt[cnt == 0] = 1
        V = V + strength * disp / cnt[:, None]
    return V


def decimate(V, F, reduction):
    import fast_simplification
    v, f = fast_simplification.simplify(V.astype(np.float32),
                                        F.astype(np.int32),
                                        target_reduction=float(reduction))
    return np.asarray(v, float), np.asarray(f, np.int64)


def stats(V, F, tag):
    e = np.vstack([F[:, [0,1]], F[:, [1,2]], F[:, [2,0]]])
    L = np.linalg.norm(V[e[:,0]] - V[e[:,1]], axis=1)
    a = np.cross(V[F[:,1]] - V[F[:,0]], V[F[:,2]] - V[F[:,0]])
    ar = 0.5 * np.linalg.norm(a, axis=1)
    print(f"  {tag:<26} V={len(V):6d} F={len(F):6d} "
          f"edge_med={np.median(L):.4f} degenerate={int((ar < 1e-12).sum())}")


# ----------------------------------------------------------------- main ------
def main():
    ap = argparse.ArgumentParser(description="give a unify volume drawn character")
    ap.add_argument("obj", help="input OBJ from unify_volume_gen.py")
    ap.add_argument("--out", default=None)
    ap.add_argument("--weld-tol", type=float, default=1e-5)
    ap.add_argument("--destair", type=int, default=4,
                    help="smoothing iters on dogs to erase the bake's voxel staircase")
    ap.add_argument("--facet-step", type=float, default=None,
                    help="plane offset quantum in world units. Larger = fewer, "
                         "bigger planes. Default = 5x median edge length.")
    ap.add_argument("--facet-dirs", type=int, default=6, choices=[6, 14, 26],
                    help="6 = axis planes (most crystalline), 14/26 = more freedom")
    ap.add_argument("--facet-iters", type=int, default=25)
    ap.add_argument("--sweep-iters", type=int, default=30,
                    help="Taubin iters on rabbits. Higher = more gestural.")
    ap.add_argument("--decimate", type=float, default=0.92,
                    help="0..1 face reduction. 0.92 = keep 8%%. Main 'less meshy' lever.")
    ap.add_argument("--flat-shade", action="store_true",
                    help="omit vertex normals (write geometry only)")
    args = ap.parse_args()

    src = Path(args.obj)
    if not src.exists():
        sys.exit(f"[abort] {src} not found")

    V, F, groups = load_obj(src)
    if "g_dog" not in groups:
        sys.exit("[abort] no g_dog group. Bake with articulate on (the default).")

    dog_face = np.zeros(len(F), bool)
    dog_face[groups["g_dog"]] = True
    print(f"[in ] {src.name}")
    stats(V, F, "loaded")

    # 1. weld away the quantization damage
    V, F, keep = weld(V, F, args.weld_tol)
    dog_face = dog_face[keep]
    stats(V, F, "welded")

    dog_vert = np.zeros(len(V), bool)
    dog_vert[F[dog_face].ravel()] = True

    e = np.vstack([F[:, [0,1]], F[:, [1,2]], F[:, [2,0]]])
    med_edge = float(np.median(np.linalg.norm(V[e[:,0]] - V[e[:,1]], axis=1)))
    step = args.facet_step if args.facet_step is not None else med_edge * 5.0

    # 2. erase the voxel staircase, 3. crystallize, 4. sweep
    V = taubin(V, F, dog_vert, args.destair)
    V = facet(V, F, dog_face, args.facet_iters, args.facet_dirs, step)
    stats(V, F, f"faceted (step={step:.3f})")
    V = taubin(V, F, ~dog_vert, args.sweep_iters)
    stats(V, F, "swept")

    # 5. decimate last so flat planes collapse and sweeps keep their triangles
    if args.decimate > 0:
        from scipy.spatial import cKDTree
        Vd, Fd = decimate(V, F, args.decimate)
        t1 = cKDTree(V[dog_vert]) if dog_vert.any() else None
        t2 = cKDTree(V[~dog_vert]) if (~dog_vert).any() else None
        if t1 is None:
            dv = np.ones(len(Vd), bool)
        elif t2 is None:
            dv = np.zeros(len(Vd), bool)
        else:
            d1, _ = t1.query(Vd)
            d2, _ = t2.query(Vd)
            dv = d1 < d2
        V, F = Vd, Fd
        dog_face = dv[F].sum(1) >= 2
        stats(V, F, f"decimated {args.decimate:.0%}")

    out = Path(args.out) if args.out else src.with_name(src.stem + "_articulated.obj")
    header = [
        f"articulate_mesh from {src.name}",
        f"weld_tol={args.weld_tol} destair={args.destair} "
        f"facet_step={step:.4f} facet_dirs={args.facet_dirs} "
        f"facet_iters={args.facet_iters} sweep_iters={args.sweep_iters} "
        f"decimate={args.decimate}",
        f"verts={len(V)} faces={len(F)} "
        f"dog={int(dog_face.sum())} rabbit={int((~dog_face).sum())}",
    ]
    write_obj(out, V, F, dog_face, header, smooth_normals=not args.flat_shade)
    print(f"[out] {out}")
    print(f"      {len(F)} faces  ({int(dog_face.sum())} dog / "
          f"{int((~dog_face).sum())} rabbit)")


if __name__ == "__main__":
    main()
