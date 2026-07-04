#!/usr/bin/env python3
"""
Post-process prediction PNGs with mean_shift brightness alignment.

For each prediction PNG:
  target_mean = manifest's original_mean_0_1 for the corresponding NL frame
  pred'  = clamp(pred + (target_mean - pred_mean), 0, 1)

Usage:
  python brightness_align.py --vis_dir <dir> --out_dir <dir> [--method shift|scale]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image


def build_manifest_lookup(manifest_path):
    """Return dict: 'Group-X/NL_video/stem' → target_mean_0_1"""
    data = json.load(open(manifest_path))
    lookup = {}
    for rec in data["records"]:
        p = Path(rec["path"])
        # path = Group-X/NL_video/frame_event/stem.png
        key = str(p.parent.parent / p.stem)
        lookup[key] = rec["original_mean_0_1"]
    return lookup


def parse_scene_folder(folder_name):
    """Extract (group, nl_video) from vis scene folder name.
    Pattern: Group-X-{mapping}-{input_video}-{nl_video}
    Both video names contain a datetime like 2025_03_26_17_47_36.
    NL video = everything after the '-' that immediately follows the FIRST datetime.
    """
    m = re.match(r"(Group-\d+)", folder_name)
    if not m:
        return None, None
    group = m.group(1)

    date_pattern = re.compile(r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}")
    matches = list(date_pattern.finditer(folder_name))
    if len(matches) >= 2:
        # nl_video starts right after the '-' following the first datetime
        first_end = matches[0].end()
        dash_pos = folder_name.find("-", first_end)
        if dash_pos != -1:
            nl_video = folder_name[dash_pos + 1:]
            return group, nl_video

    return group, None


def process_vis_dir(vis_dir, out_dir, lookup, method="shift"):
    vis_dir = Path(vis_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total, matched, skipped = 0, 0, 0

    for scene in sorted(vis_dir.iterdir()):
        if not scene.is_dir():
            continue

        group, nl_video = parse_scene_folder(scene.name)
        if not group or not nl_video:
            print(f"  [WARN] Could not parse scene: {scene.name}")
            skipped += 1
            continue

        out_scene = out_dir / scene.name
        out_scene.mkdir(parents=True, exist_ok=True)

        pngs = sorted(f for f in scene.iterdir() if "_p0_" in f.name and f.suffix == ".png")
        for png in pngs:
            total += 1
            stem = png.name.split("_p0_")[0]
            manifest_key = f"{group}/{nl_video}/{stem}"

            target_mean = lookup.get(manifest_key)
            if target_mean is None:
                # fallback: copy unchanged
                skipped += 1
                img = Image.open(png).convert("RGB")
                img.save(out_scene / png.name)
                continue

            img = np.array(Image.open(png).convert("RGB")).astype(np.float32) / 255.0
            pred_mean = img.mean()

            if method == "shift":
                img_out = img + (target_mean - pred_mean)
            elif method == "scale":
                if pred_mean > 1e-6:
                    img_out = img * (target_mean / pred_mean)
                else:
                    img_out = img
            else:
                raise ValueError(f"Unknown method: {method}")

            img_out = np.clip(img_out, 0.0, 1.0)
            Image.fromarray((img_out * 255).round().astype(np.uint8)).save(out_scene / png.name)
            matched += 1

    print(f"Done: {matched}/{total} aligned, {skipped} skipped/fallback")
    return matched, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis_dir", required=True, help="Input vis directory")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--method", default="shift", choices=["shift", "scale"])
    parser.add_argument("--manifest", required=True, help="Path to mean_prompt_manifest.json in the eval dataset root")
    args = parser.parse_args()

    print(f"Method: {args.method}")
    print(f"Input:  {args.vis_dir}")
    print(f"Output: {args.out_dir}")

    lookup = build_manifest_lookup(args.manifest)
    print(f"Manifest: {len(lookup)} entries")

    matched, total = process_vis_dir(args.vis_dir, args.out_dir, lookup, method=args.method)
    print(f"Alignment rate: {matched}/{total} = {matched/total*100:.1f}%")


if __name__ == "__main__":
    main()
