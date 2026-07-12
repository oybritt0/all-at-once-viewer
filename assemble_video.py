"""
assemble_video.py — build an mp4 from a hallucinated camera-path batch.

After queueing a recorded path from the viewer and letting the worker chew
through it (overnight is realistic for 100+ frames on the A3000):

    python assemble_video.py --tag mypath --fps 12
    python assemble_video.py --tag mypath --fps 12 --boomerang

Reads data/hallucinations/<tag>/frame_*.png, writes <tag>.mp4 alongside.
Requires ffmpeg on PATH.

The per-frame flicker is left alone deliberately. Adjacent frames disagree
because each is an independent registration of the same linework; that
disagreement is content, not noise. If a steadier read is ever wanted,
--fps 8 with --hold 2 duplicates frames rather than interpolating them.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--hold", type=int, default=1, help="duplicate each frame N times")
    ap.add_argument("--boomerang", action="store_true", help="append reversed pass")
    args = ap.parse_args()

    src = Path("data") / "hallucinations" / args.tag
    frames = sorted(src.glob("frame_*.png"))
    if not frames:
        sys.exit(f"no frames in {src}")

    seq_dir = src / "_seq"
    seq_dir.mkdir(exist_ok=True)
    order = list(frames)
    if args.boomerang:
        order += list(reversed(frames[1:-1]))
    i = 0
    for f in order:
        for _ in range(max(1, args.hold)):
            link = seq_dir / f"s_{i:06d}.png"
            if link.exists():
                link.unlink()
            link.write_bytes(f.read_bytes())
            i += 1

    out = src / f"{args.tag}.mp4"
    cmd = [
        "ffmpeg", "-y", "-framerate", str(args.fps),
        "-i", str(seq_dir / "s_%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"wrote {out}  ({i} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
