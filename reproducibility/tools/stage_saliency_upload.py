"""Stage the DHF1K training inputs of the paper for transfer to a remote GPU machine.

Builds <out>/saliency_dataset_ready/{train,val}/video_NNNNN/ with only the files the saliency training
reads from the 200 FPS video (the retina runs' input): video_upsampled_fps_eq_200.mp4, raw_images.npz,
outputs.npz, params.json; plus metadata.json per split, split.json, SHA256SUMS and a dataset card.
Large files are hard links to the originals (no extra disk space); params.json is copied, because the
retina loaders rewrite it. Clip membership is checked against reproducibility/manifests/.

    python -m reproducibility.tools.stage_saliency_upload \
        --ready src/neural_network/data/saliency_dataset_ready --out /path/to/staging
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

LINKED = ("video_upsampled_fps_eq_200.mp4", "raw_images.npz", "outputs.npz")
COPIED = ("params.json",)

CARD = """---
license: other
pretty_name: Silicon retina paper - DHF1K 200 FPS training inputs (private)
---

# DHF1K 200 FPS inputs for "Programmable Silicon Retina on Pixel Processor Array"

**Private working copy, not for redistribution.** Derived from DHF1K
(W. Wang et al., CVPR 2018 / TPAMI 2021, https://github.com/wenguanwang/DHF1K); DHF1K's terms apply.

The 541 clips used in the paper (431 train / 110 val, `split.json`), exactly as read by the paper's
saliency runs (files unchanged since 2025-11-13). Per clip:

* `video_upsampled_fps_eq_200.mp4`: FILM-upsampled 200 FPS video, 320x180
* `raw_images.npz`: source frames (N x 180 x 320 x 3 uint8)
* `outputs.npz`: DHF1K saliency maps (N x 180 x 320 x 3 uint8), the training target
* `params.json`: timestamps (`timestamps_FF`, seconds) and frames per source-frame interval

Clip IDs map to DHF1K clips through `reproducibility/manifests/dhf1k_saliency_split.csv` in the code
repository. Verify with `sha256sum -c SHA256SUMS`.
"""


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ready", required=True, help="saliency_dataset_ready/ used by the paper runs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="reproducibility/manifests/dhf1k_saliency_split.csv")
    args = ap.parse_args()

    ready, out = Path(args.ready), Path(args.out)
    rows = list(csv.DictReader(open(args.manifest)))
    split = json.load(open(ready / "split.json"))
    assert sorted(split["train"]) == sorted(r["processed_id"] for r in rows if r["split"] == "train")
    assert sorted(split["val"]) == sorted(r["processed_id"] for r in rows if r["split"] == "val")

    dst_root = out / "saliency_dataset_ready"
    dst_root.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ready / "split.json", dst_root / "split.json")
    for r in rows:
        src, dst = ready / r["split"] / r["processed_id"], dst_root / r["split"] / r["processed_id"]
        dst.mkdir(parents=True, exist_ok=True)
        for f in LINKED:
            if not (dst / f).exists():
                os.link(src / f, dst / f)
        for f in COPIED:
            shutil.copyfile(src / f, dst / f)
        params = json.load(open(dst / "params.json"))
        assert len(params["timestamps_FF"]) == sum(params["new_images_per_pair_FF"]), r["processed_id"]
    for s in ("train", "val"):
        shutil.copyfile(ready / s / "metadata.json", dst_root / s / "metadata.json")

    (out / "README.md").write_text(CARD)
    files = sorted(p for p in dst_root.rglob("*") if p.is_file())
    with open(out / "SHA256SUMS", "w") as f:
        for p in files:
            f.write(f"{sha256_file(p)}  {p.relative_to(out)}\n")
    size = sum(p.stat().st_size for p in files)
    print(f"staged {len(rows)} clips, {len(files)} files, {size / 1e9:.1f} GB -> {out}")


if __name__ == "__main__":
    main()
