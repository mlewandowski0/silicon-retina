"""Rebuild the DHF1K selection/split manifests from the original lab records.

This is the provenance tool used to *derive* the published files in
reproducibility/manifests/ from the folders that the paper experiments
actually read. Readers do not need to run it; it is kept so the derivation can
be audited. Nothing is written into the data directory.

    python -m reproducibility.tools.build_dhf1k_manifest \
        --data-root src/neural_network/data \
        --upsample-log /path/to/upsample_200fps.txt \
        --out reproducibility/manifests

Inputs it reads:
  <data-root>/saliency_dataset_processed/NNNN/{input.AVI,params.json}
      output of help_scripts/reorder_saliency_map_ds.py; input.AVI is a byte copy
      (shutil.copy) of DHF1K video/NNN.AVI, so its SHA-256 identifies the source
  <data-root>/dhf1k-restructured.zip (optional)
      archive of saliency_dataset_processed, used for folders missing on disk
  <data-root>/saliency_dataset_truncated/
      clips kept for upsampling; video_{i:05d} = i-th folder in sorted order
  <data-root>/saliency_dataset_ready/split.json
      train/val membership written by data_help/partition_saliency.py
  upsample_200fps.txt
      log of the fixed-rate upsampling pass ("Processing index : i  path : .../NNNN")
"""
import argparse
import csv
import hashlib
import json
import os
import re
import zipfile

MAX_FRAMES = 750         # copy_short_videos.py --max-frames 750 (shell history), keeps frames < 750
MANUAL_EXCLUSIONS = {
    "0058": "removed by hand after optical-flow upsampling repeatedly failed on this clip "
            "(upsample_log.txt stalls at index 40 = 0058 while 542 clips were present; "
            "later passes run on 541 clips)",
}


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest(), os.path.getsize(path)


def sha256_zip_member(zf, name, chunk=1 << 20):
    h = hashlib.sha256()
    size = 0
    with zf.open(name) as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
            size += len(block)
    return h.hexdigest(), size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--upsample-log", required=True)
    ap.add_argument("--out", default="reproducibility/manifests")
    args = ap.parse_args()

    D = args.data_root
    processed = os.path.join(D, "saliency_dataset_processed")
    truncated = sorted(d for d in os.listdir(os.path.join(D, "saliency_dataset_truncated"))
                       if os.path.isdir(os.path.join(D, "saliency_dataset_truncated", d)))
    split = json.load(open(os.path.join(D, "saliency_dataset_ready", "split.json")))
    membership = {v: "train" for v in split["train"]}
    membership.update({v: "val" for v in split["val"]})
    assert not set(split["train"]) & set(split["val"]), "train/val overlap in split.json"

    log = open(args.upsample_log, errors="ignore").read()
    logged = {int(i): p for i, p in
              re.findall(r"Processing index : (\d+)\s+path :\s+\S*/saliency_dataset_truncated/(\d{4})", log)}

    zpath = os.path.join(D, "dhf1k-restructured.zip")
    zf = zipfile.ZipFile(zpath) if os.path.exists(zpath) else None

    # every annotated DHF1K clip handled by reorder_saliency_map_ds.py (1..700)
    info = {}
    for n in range(1, 701):
        fid = f"{n:04d}"
        folder = os.path.join(processed, fid)
        if os.path.isdir(folder):
            params = json.load(open(os.path.join(folder, "params.json")))
            digest, size = sha256_file(os.path.join(folder, "input.AVI"))
        elif zf is not None:
            params = json.loads(zf.read(f"saliency_dataset_processed/{fid}/params.json"))
            digest, size = sha256_zip_member(zf, f"saliency_dataset_processed/{fid}/input.AVI")
        else:
            raise FileNotFoundError(folder)
        info[fid] = dict(params=params, sha256=digest, bytes=size)

    os.makedirs(args.out, exist_ok=True)
    fields = ["dhf1k_id", "source_video", "source_annotation", "source_frames", "source_fps",
              "dhf1k_official_split", "processed_id", "split", "source_video_sha256", "source_video_bytes"]
    rows = []
    for i, fid in enumerate(truncated):
        vid = f"video_{i:05d}"
        if i in logged:
            assert logged[i] == fid, (i, logged[i], fid)
        p = info[fid]["params"]
        assert p["input_frame_count"] == p["maps_frame_count"] == p["fixation_frame_count"] < MAX_FRAMES
        n = int(fid)
        rows.append(dict(dhf1k_id=f"{n:03d}", source_video=f"video/{n:03d}.AVI",
                         source_annotation=f"annotation/{fid}/maps/",
                         source_frames=p["input_frame_count"], source_fps=p["fps"],
                         dhf1k_official_split="train" if n <= 600 else "val",
                         processed_id=vid, split=membership[vid],
                         source_video_sha256=info[fid]["sha256"], source_video_bytes=info[fid]["bytes"]))
    with open(os.path.join(args.out, "dhf1k_saliency_split.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    kept = set(truncated)
    ex_fields = ["dhf1k_id", "source_frames", "source_video_sha256", "stage", "reason"]
    ex_rows = []
    for fid, d in info.items():
        if fid in kept:
            continue
        frames = d["params"]["input_frame_count"]
        if fid in MANUAL_EXCLUSIONS:
            assert frames < MAX_FRAMES
            stage, reason = "upsampling", MANUAL_EXCLUSIONS[fid]
        else:
            assert frames >= MAX_FRAMES, fid
            stage, reason = "length_filter", f"{frames} frames >= {MAX_FRAMES} (copy_short_videos.py --max-frames {MAX_FRAMES})"
        ex_rows.append(dict(dhf1k_id=f"{int(fid):03d}", source_frames=frames,
                            source_video_sha256=d["sha256"], stage=stage, reason=reason))
    with open(os.path.join(args.out, "dhf1k_saliency_exclusions.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ex_fields, lineterminator="\n")
        w.writeheader()
        w.writerows(ex_rows)

    n_train = sum(r["split"] == "train" for r in rows)
    print(f"selected {len(rows)} (train {n_train}, val {len(rows) - n_train}), excluded {len(ex_rows)}, "
          f"index/log agreement on {sum(i in logged for i in range(len(truncated)))} clips")


if __name__ == "__main__":
    main()
