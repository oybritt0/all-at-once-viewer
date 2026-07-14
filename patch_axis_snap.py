"""
patch_axis_snap.py  —  run from the project root (Histories\\), next to app.py.

Adds Rhino-style axis view snapping to the latent viewer:
  - six named views (Front / Back / Right / Left / Top / Bottom) that glide the
    camera onto the corresponding orthographic axis, preserving the current
    pivot (controls.target), distance, and lens
  - a compact button row in the overlay
  - keyboard shortcuts: 1 front, 3 right, 7 top; hold Shift for the opposite face

Purely additive. Validated by: Python AST parse of the whole file, node --check
on the extracted (placeholder-substituted) module script, brace-balance delta,
and single-anchor matching. CRLF line endings are preserved.
"""

import ast
import subprocess
import sys
import tempfile
from pathlib import Path

APP = Path("app.py")

# ---------------------------------------------------------------- edit 1: HTML
HTML_ANCHOR = (
    '    <button id="reset">Reset</button>\r\n'
    '  </div>'
)
HTML_INSERT = (
    '    <button id="reset">Reset</button>\r\n'
    '  </div>\r\n'
    '  <div class="row" id="viewSnap" title="Snap to an orthographic axis view. '
    'Keys: 1 front, 3 right, 7 top; hold Shift for the opposite face.">\r\n'
    '    <button data-view="front"  style="padding:2px 6px;">Front</button>\r\n'
    '    <button data-view="back"   style="padding:2px 6px;">Back</button>\r\n'
    '    <button data-view="right"  style="padding:2px 6px;">Right</button>\r\n'
    '    <button data-view="left"   style="padding:2px 6px;">Left</button>\r\n'
    '    <button data-view="top"    style="padding:2px 6px;">Top</button>\r\n'
    '    <button data-view="bottom" style="padding:2px 6px;">Bottom</button>\r\n'
    '  </div>'
)

# ------------------------------------------------------------------ edit 2: JS
JS_ANCHOR = (
    '  markOverlayDirty();\r\n'
    '};\r\n'
    '\r\n'
    '// Resize handling'
)
JS_BLOCK = (
    '  markOverlayDirty();\r\n'
    '};\r\n'
    '\r\n'
    '// ---- axis view snapping (Rhino-style named views) --------------------------\r\n'
    '// Glide the camera onto one of the six orthographic axis views, preserving\r\n'
    '// the current pivot (controls.target), distance, and lens. The offset vector\r\n'
    '// is slerped along a great-circle arc so the move reads like a ViewCube snap.\r\n'
    '// camera.up is left at world +Y: OrbitControls (r160) caches its up axis at\r\n'
    "// construction, so we never mutate it. The top/bottom poles are handled by\r\n"
    "// OrbitControls' own makeSafe clamp on the next update().\r\n"
    'const VIEW_DIRS = {\r\n'
    '  front:  new THREE.Vector3( 0,  0,  1),\r\n'
    '  back:   new THREE.Vector3( 0,  0, -1),\r\n'
    '  right:  new THREE.Vector3( 1,  0,  0),\r\n'
    '  left:   new THREE.Vector3(-1,  0,  0),\r\n'
    '  top:    new THREE.Vector3( 0,  1,  0),\r\n'
    '  bottom: new THREE.Vector3( 0, -1,  0),\r\n'
    '};\r\n'
    'let _viewSnapToken = 0;\r\n'
    'function snapToView(name) {\r\n'
    '  const dir = VIEW_DIRS[name];\r\n'
    '  if (!dir) return;\r\n'
    '  const target = controls.target;\r\n'
    '  const startOff = camera.position.clone().sub(target);\r\n'
    '  let dist = startOff.length();\r\n'
    '  if (dist < 1e-6) dist = radius * 2.5;\r\n'
    '  const fromN = startOff.clone().normalize();\r\n'
    '  const toN = dir.clone().normalize();\r\n'
    '  const qEnd = new THREE.Quaternion().setFromUnitVectors(fromN, toN);\r\n'
    '  const qStart = new THREE.Quaternion();      // identity\r\n'
    '  const token = ++_viewSnapToken;             // cancels any in-flight snap\r\n'
    '  const DUR = 320;\r\n'
    '  const t0 = performance.now();\r\n'
    '  const ease = (u) => (u < 0.5 ? 4*u*u*u : 1 - Math.pow(-2*u+2, 3)/2);\r\n'
    '  const tmpQ = new THREE.Quaternion();\r\n'
    '  const tmpV = new THREE.Vector3();\r\n'
    '  function stepView(now) {\r\n'
    '    if (token !== _viewSnapToken) return;     // superseded by a newer snap\r\n'
    '    const u = Math.min(1, (now - t0) / DUR);\r\n'
    '    tmpQ.slerpQuaternions(qStart, qEnd, ease(u));\r\n'
    '    tmpV.copy(fromN).applyQuaternion(tmpQ).multiplyScalar(dist);\r\n'
    '    camera.position.copy(target).add(tmpV);\r\n'
    '    controls.update();\r\n'
    '    markOverlayDirty();\r\n'
    '    if (u < 1) requestAnimationFrame(stepView);\r\n'
    '  }\r\n'
    '  requestAnimationFrame(stepView);\r\n'
    '}\r\n'
    '\r\n'
    "document.querySelectorAll('#viewSnap button').forEach((b) => {\r\n"
    "  b.addEventListener('click', () => snapToView(b.dataset.view));\r\n"
    '});\r\n'
    '\r\n'
    '// keyboard: 1 front, 3 right, 7 top; hold Shift for the opposite face.\r\n'
    '// uses e.code so it is layout independent; ignores modifier combos and text fields.\r\n'
    "window.addEventListener('keydown', (e) => {\r\n"
    '  if (e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;\r\n'
    '  const el = document.activeElement;\r\n'
    "  if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;\r\n"
    '  let name = null;\r\n'
    "  if (e.code === 'Digit1') name = e.shiftKey ? 'back'   : 'front';\r\n"
    "  else if (e.code === 'Digit3') name = e.shiftKey ? 'left'   : 'right';\r\n"
    "  else if (e.code === 'Digit7') name = e.shiftKey ? 'bottom' : 'top';\r\n"
    '  if (name) { e.preventDefault(); snapToView(name); }\r\n'
    '});\r\n'
    '\r\n'
    '// Resize handling'
)

EDITS = [("HTML view-snap buttons", HTML_ANCHOR, HTML_INSERT),
         ("JS snapToView + shortcuts", JS_ANCHOR, JS_BLOCK)]


def apply_edits(src: str) -> str:
    out = src
    for label, anchor, repl in EDITS:
        n = out.count(anchor)
        if n != 1:
            sys.exit(f"[abort] anchor for '{label}' matched {n} times (need exactly 1)")
        out = out.replace(anchor, repl, 1)
    return out


def extract_module_js(src: str) -> str:
    open_tag = '<script type="module">'
    i = src.index(open_tag) + len(open_tag)
    j = src.index('</script>', i)
    js = src[i:j]
    # placeholders are injected as bare values at runtime; sub valid dummies so
    # the syntax check sees parseable JS.
    subs = {
        '__POINTS_JSON__': '[]', '__GEOMETRY_JSON__': '{}',
        '__EDGES_JSON__': '[]', '__CHORDS_JSON__': '[]',
        '__SETTINGS_JSON__': '{}', '__BG__': '#000000',
    }
    for k, v in subs.items():
        js = js.replace(k, v)
    return js


def main():
    if not APP.exists():
        sys.exit("[abort] app.py not found. Run this from the project root (Histories\\).")

    with open(APP, "r", encoding="utf-8", newline="") as f:
        original = f.read()
    patched = apply_edits(original)

    # 1) whole-file Python AST parse
    try:
        ast.parse(patched)
    except SyntaxError as e:
        sys.exit(f"[abort] python AST parse failed: {e}")

    # 2) brace balance of the NET inserted JS must be zero. JS_BLOCK re-includes
    #    the anchor (which carries the reset handler's closing '};'), so subtract
    #    the anchor's own delta to isolate the added text.
    block_delta = JS_BLOCK.count("{") - JS_BLOCK.count("}")
    anchor_delta = JS_ANCHOR.count("{") - JS_ANCHOR.count("}")
    delta = block_delta - anchor_delta
    if delta != 0:
        sys.exit(f"[abort] net inserted JS brace delta is {delta} (must be 0)")

    # 3) node --check on the extracted module script
    js = extract_module_js(patched)
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8") as f:
        f.write(js)
        mjs = f.name
    r = subprocess.run(["node", "--check", mjs], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[abort] node --check failed:\n{r.stderr}")

    with open(APP, "w", encoding="utf-8", newline="") as f:
        f.write(patched)
    print("[ok] axis view snapping patched into app.py")
    print("     - overlay buttons: Front / Back / Right / Left / Top / Bottom")
    print("     - keys: 1 front, 3 right, 7 top (+Shift for the opposite face)")
    print("     - python AST ok | node --check ok | brace delta 0")


if __name__ == "__main__":
    main()
