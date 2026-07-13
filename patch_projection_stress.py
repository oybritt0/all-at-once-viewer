"""
patch_projection_stress.py
==========================
Adds the projection-stress reading to the latent viewer. Standalone, anchored,
self-validating. Run from the project root where app.py lives:

    python patch_projection_stress.py            # patches ./app.py in place
    python patch_projection_stress.py --check    # validate only, no write

What it adds
------------
1. Loads data/embeddings/projection_stress.json (from projection_stress.py) and
   merges crush / tear / proj_stress onto df_valid by id, row-safe under any
   visibility filtering.
2. Three new color axes in the sidebar "Color by":
     proj_stress  where the projection distorts most (inferno)
     crush        false overlap — plane-neighbors that are feature strangers (Reds)
     tear         false gap — feature-neighbors the plane pulled apart (Blues)
3. A "Projection chords" overlay: the torn pairs drawn as line segments across
   the manufactured gaps, width and opacity scaled by tear strength. Renders in
   the SVG export (the drawable artifact) and honors the current camera and the
   dog/rabbit / layer / category visibility filters.

Every added element references real members and the real projection. Nothing is
synthesized. Missing sidecar -> the fields are NaN and the overlay is empty; the
app runs unchanged.

Validation before write: full AST parse, THREEJS_TEMPLATE brace balance (delta
must be 0), and presence of every injected JS anchor. File is written back with
CRLF line endings to match the repo convention.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

APP = Path("app.py")


# ---- each edit is (label, anchor, replacement); anchor must occur exactly once
EDITS: list[tuple[str, str, str]] = []


def edit(label, anchor, replacement):
    EDITS.append((label, anchor, replacement))


# --- A. load + merge the sidecar (after mode_label cleanup, before return) ---
edit(
    "A: sidecar load/merge",
    '    df_valid.loc[df_valid["mode_label"].isin(["", "None", "nan"]), "mode_label"] = "unassigned"\n\n'
    "    return df_valid, valid_features, embeddings_3d, []",
    '    df_valid.loc[df_valid["mode_label"].isin(["", "None", "nan"]), "mode_label"] = "unassigned"\n\n'
    "    # Projection-stress signals (projection_stress.py): per-point crush\n"
    "    # (false overlap), tear (false gap), and combined proj_stress. Each is a\n"
    "    # 0..1 percentile measuring where the 2D layout distorts the feature\n"
    "    # space. Aligned by id; absent file -> NaN (fields simply inert).\n"
    '    ps_path = EMBEDDINGS_DIR / "projection_stress.json"\n'
    "    if ps_path.exists():\n"
    '        with open(ps_path, encoding="utf-8") as f:\n'
    "            _ps = json.load(f)\n"
    "        _ps_df = pd.DataFrame({\n"
    '            "id": _ps["ids"],\n'
    '            "crush": _ps["crush"],\n'
    '            "tear": _ps["tear"],\n'
    '            "proj_stress": _ps["stress"],\n'
    "        })\n"
    '        df_valid = df_valid.merge(_ps_df, on="id", how="left")\n'
    "    else:\n"
    '        df_valid["crush"] = np.nan\n'
    '        df_valid["tear"] = np.nan\n'
    '        df_valid["proj_stress"] = np.nan\n'
    "\n"
    "    return df_valid, valid_features, embeddings_3d, []",
)

# --- B. register the three fields as numeric color axes ---
edit(
    "B: NUMERIC_COLOR_FIELDS",
    'NUMERIC_COLOR_FIELDS = {"density", "surprise_pct"}',
    'NUMERIC_COLOR_FIELDS = {"density", "surprise_pct", "proj_stress", "crush", "tear"}',
)

# --- C. per-field colormap selection ---
edit(
    "C: cmap selection",
    '        cmap_name = "viridis" if color_field == "density" else "magma"',
    '        cmap_name = {\n'
    '            "density": "viridis",\n'
    '            "surprise_pct": "magma",\n'
    '            "proj_stress": "inferno",\n'
    '            "crush": "Reds",\n'
    '            "tear": "Blues",\n'
    '        }.get(color_field, "magma")',
)

# --- D. dropdown options ---
edit(
    "D: color dropdown options",
    '                 "mode_label", "density", "surprise_pct", "none"],',
    '                 "mode_label", "density", "surprise_pct",\n'
    '                 "proj_stress", "crush", "tear", "none"],',
)

# --- E. CHORDS payload constant in JS ---
edit(
    "E: const CHORDS",
    "const EDGES = __EDGES_JSON__;",
    "const EDGES = __EDGES_JSON__;\nconst CHORDS = __CHORDS_JSON__;",
)

# --- F. chord render in the SVG export path, after the edges block ---
edit(
    "F: chord SVG render",
    "  // Edges between points that survived culling\n"
    "  if (SETTINGS.show_edges && EDGES && EDGES.length > 0) {\n"
    '    addLayer("edges");\n'
    "    const op = SETTINGS.edges_opacity < 1\n"
    "      ? ` stroke-opacity=\"${SETTINGS.edges_opacity.toFixed(2)}\"` : '';\n"
    "    const byIdx = new Map();\n"
    "    for (const q of projected) byIdx.set(q.idx, q);\n"
    "    for (const e of EDGES) {\n"
    "      const a = byIdx.get(e[0]), b = byIdx.get(e[1]);\n"
    "      if (!a || !b) continue;\n"
    '      layers.get("edges").push(\n'
    '        `<line x1="${a.sx.toFixed(2)}" y1="${a.sy.toFixed(2)}" ` +\n'
    '        `x2="${b.sx.toFixed(2)}" y2="${b.sy.toFixed(2)}" ` +\n'
    '        `stroke="${SETTINGS.edges_color}" ` +\n'
    '        `stroke-width="${SETTINGS.edges_width.toFixed(2)}"${op}/>`\n'
    "      );\n"
    "    }\n"
    "  }\n",
    "  // Edges between points that survived culling\n"
    "  if (SETTINGS.show_edges && EDGES && EDGES.length > 0) {\n"
    '    addLayer("edges");\n'
    "    const op = SETTINGS.edges_opacity < 1\n"
    "      ? ` stroke-opacity=\"${SETTINGS.edges_opacity.toFixed(2)}\"` : '';\n"
    "    const byIdx = new Map();\n"
    "    for (const q of projected) byIdx.set(q.idx, q);\n"
    "    for (const e of EDGES) {\n"
    "      const a = byIdx.get(e[0]), b = byIdx.get(e[1]);\n"
    "      if (!a || !b) continue;\n"
    '      layers.get("edges").push(\n'
    '        `<line x1="${a.sx.toFixed(2)}" y1="${a.sy.toFixed(2)}" ` +\n'
    '        `x2="${b.sx.toFixed(2)}" y2="${b.sy.toFixed(2)}" ` +\n'
    '        `stroke="${SETTINGS.edges_color}" ` +\n'
    '        `stroke-width="${SETTINGS.edges_width.toFixed(2)}"${op}/>`\n'
    "      );\n"
    "    }\n"
    "  }\n"
    "\n"
    "  // Chords — torn pairs. Feature-space neighbors the projection pulled\n"
    "  // apart, drawn straight across the gap the layout manufactured. This is\n"
    "  // chord-not-flow made literal: an adjacency cutting across the manifold,\n"
    "  // not an interpolated skin. Width and opacity scale with tear strength.\n"
    "  if (SETTINGS.show_chords && CHORDS && CHORDS.length > 0) {\n"
    '    addLayer("chords");\n'
    "    const byIdC = new Map();\n"
    "    for (const q of projected) byIdC.set(q.p.id, q);\n"
    "    for (const c of CHORDS) {\n"
    "      const a = byIdC.get(c[0]), b = byIdC.get(c[1]);\n"
    "      if (!a || !b) continue;\n"
    "      const s = c[2];\n"
    "      const w = (SETTINGS.chords_width * (0.35 + 0.65 * s)).toFixed(2);\n"
    "      const o = (0.15 + 0.6 * s).toFixed(2);\n"
    '      layers.get("chords").push(\n'
    '        `<line x1="${a.sx.toFixed(2)}" y1="${a.sy.toFixed(2)}" ` +\n'
    '        `x2="${b.sx.toFixed(2)}" y2="${b.sy.toFixed(2)}" ` +\n'
    '        `stroke="${SETTINGS.chords_color}" stroke-width="${w}" ` +\n'
    '        `stroke-opacity="${o}"/>`\n'
    "      );\n"
    "    }\n"
    "  }\n",
)

# --- G. Streamlit UI: a "Projection chords" expander after "KNN edges" ---
edit(
    "G: chords expander UI",
    '        edges_width = st.slider("Edge width", 0.1, 3.0, 0.5, 0.1)\n',
    '        edges_width = st.slider("Edge width", 0.1, 3.0, 0.5, 0.1)\n'
    "\n"
    '    with st.expander("Projection chords", expanded=False):\n'
    '        show_chords = st.checkbox(\n'
    '            "Draw projection chords", value=False,\n'
    '            help="Torn pairs from projection_stress.json: feature-space "\n'
    '                 "neighbors the 2D layout pulled apart. Drawn across the "\n'
    '                 "manufactured gap. Run projection_stress.py first.",\n'
    "        )\n"
    '        chords_max = st.slider("Max chords", 50, 1200, 400, 50)\n'
    '        chords_color = st.color_picker("Chord color", "#ff3b30")\n'
    '        chords_width = st.slider("Chord width", 0.2, 4.0, 1.0, 0.1)\n',
)

# --- H. settings dict: carry the chord controls ---
edit(
    "H: settings dict",
    "    show_edges=bool(show_edges), edges_k=int(edges_k),\n"
    "    edges_color=edges_color, edges_opacity=float(edges_opacity),\n"
    "    edges_width=float(edges_width),\n",
    "    show_edges=bool(show_edges), edges_k=int(edges_k),\n"
    "    edges_color=edges_color, edges_opacity=float(edges_opacity),\n"
    "    edges_width=float(edges_width),\n"
    "    show_chords=bool(show_chords), chords_max=int(chords_max),\n"
    "    chords_color=chords_color, chords_width=float(chords_width),\n",
)

# --- I1. build chords_data next to edges_data ---
edit(
    "I1: chords_data build",
    "            settings[\"edges_k\"]\n"
    "        ):\n"
    "            edges_data.append([int(a), int(b)])\n",
    "            settings[\"edges_k\"]\n"
    "        ):\n"
    "            edges_data.append([int(a), int(b)])\n"
    "\n"
    "    # Chord segments (projection_stress.py). Filtered to sprites actually in\n"
    "    # this view, sorted by tear strength, capped. id pairs -> robust to any\n"
    "    # visibility filtering upstream.\n"
    "    chords_data = []\n"
    '    if settings.get("show_chords"):\n'
    "        try:\n"
    '            _cp = EMBEDDINGS_DIR / "projection_stress.json"\n'
    "            if _cp.exists():\n"
    '                with open(_cp, encoding="utf-8") as _f:\n'
    '                    _ch = json.load(_f).get("chords", [])\n'
    '                _vis = {sp["id"] for sp in sprites_data}\n'
    "                _kept = [c for c in _ch if c[0] in _vis and c[1] in _vis]\n"
    "                _kept.sort(key=lambda c: -c[2])\n"
    "                chords_data = [[str(c[0]), str(c[1]), float(c[2])]\n"
    '                               for c in _kept[:int(settings["chords_max"])]]\n'
    "        except Exception:\n"
    "            chords_data = []\n",
)

# --- I2. export_settings keys ---
edit(
    "I2: export_settings keys",
    '        "edges_width": float(settings["edges_width"]),\n',
    '        "edges_width": float(settings["edges_width"]),\n'
    '        "show_chords": bool(settings["show_chords"]),\n'
    '        "chords_color": settings["chords_color"],\n'
    '        "chords_width": float(settings["chords_width"]),\n'
    '        "chords_max": int(settings["chords_max"]),\n',
)

# --- I3. inject the payload into the template ---
edit(
    "I3: __CHORDS_JSON__ replace",
    '        .replace("__EDGES_JSON__", json.dumps(edges_data))\n',
    '        .replace("__EDGES_JSON__", json.dumps(edges_data))\n'
    '        .replace("__CHORDS_JSON__", json.dumps(chords_data))\n',
)


def apply_edits(src: str) -> str:
    for label, anchor, repl in EDITS:
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(
                f"[FAIL] anchor for '{label}' matched {n} times (need exactly 1). "
                f"app.py may be a different version — aborting, no write."
            )
        src = src.replace(anchor, repl)
    return src


def extract_template(src: str) -> str:
    m = re.search(r'THREEJS_TEMPLATE = r"""(.*?)"""', src, re.DOTALL)
    if not m:
        raise SystemExit("[FAIL] could not locate THREEJS_TEMPLATE for validation.")
    return m.group(1)


def brace_delta(s: str) -> int:
    return s.count("{") - s.count("}")


def validate(src: str) -> None:
    # 1. Python parses
    ast.parse(src)
    # 2. template brace balance unchanged / zero-delta
    tmpl = extract_template(src)
    d = brace_delta(tmpl)
    if d != 0:
        raise SystemExit(f"[FAIL] THREEJS_TEMPLATE brace delta = {d} (must be 0).")
    # 3. every injected JS anchor present
    required = [
        "const CHORDS = __CHORDS_JSON__;",
        "SETTINGS.show_chords",
        '__CHORDS_JSON__", json.dumps(chords_data)',
        'NUMERIC_COLOR_FIELDS = {"density", "surprise_pct", "proj_stress", "crush", "tear"}',
        'ps_path = EMBEDDINGS_DIR / "projection_stress.json"',
        'show_chords = st.checkbox(',
    ]
    for r in required:
        if r not in src:
            raise SystemExit(f"[FAIL] expected injected marker missing: {r!r}")
    print("[ok] AST parse, template brace delta 0, all markers present.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="validate the patched result but do not write")
    ap.add_argument("--path", default="app.py")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"[FAIL] {path} not found. Run from the project root.")

    # universal-newline read -> \n internally; anchors are written with \n
    src = path.read_text(encoding="utf-8")
    patched = apply_edits(src)
    validate(patched)

    if args.check:
        print("[check] validation passed; not writing (--check).")
        return

    # write back with CRLF to match the repo convention
    with open(path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write(patched)
    print(f"[written] {path}  (+{len(patched) - len(src)} chars, {len(EDITS)} edits)")
    print("Restart Streamlit and hard-refresh (Ctrl+F5).")


if __name__ == "__main__":
    main()
