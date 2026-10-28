"""Rebuild the ECOCO sequence manifest from the extracted ecoco_depthmaps_test archive.

Provenance tool (readers do not need to run it). It records, for every sequence the
intensity-reconstruction runs read, the processed ID assigned by the upsampling code
(VideoDatasetFromPictures: i-th folder of sorted(<root>/<split>) -> video_{i:05d}), the frame
count, the source timestamps and a content hash. Nothing is written into the data directory.

    python -m reproducibility.tools.build_ecoco_manifest \
        --ecoco-root src/neural_network/data/ecoco_depthmaps_test --out reproducibility/manifests

frames_sha256 is SHA-256 over the lines "<file name> <sha256 of file bytes>\\n" for
frames/*.png and frames/timestamps.txt, in sorted name order. It identifies the byte content
of the frames the pipeline reads and can be recomputed with
reproducibility.regenerate.ecoco_frames_sha256().
"""
import argparse
import csv
import os

from reproducibility.regenerate import ecoco_frames_sha256

# DVS run FIREnetwork_20250901_173249 logged "Found 947 valid sequences out of 948 items":
# the occupancy check drops this clip (only 20 non-empty DVS grids). It is part of the
# dataset; the drop happens at training time.
TRAINING_NOTES = {("train", "sequence_0000000745"):
                  "dropped at training time by the DVS run's occupancy check (947/948 valid)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecoco-root", required=True)
    ap.add_argument("--out", default="reproducibility/manifests")
    args = ap.parse_args()

    fields = ["split", "sequence_id", "source_frames", "t_first_s", "t_last_s",
              "processed_id", "frames_sha256", "note"]
    rows = []
    for split in ("train", "validation"):
        seqs = sorted(os.listdir(os.path.join(args.ecoco_root, split)))
        for i, seq in enumerate(seqs):
            frames_dir = os.path.join(args.ecoco_root, split, seq, "frames")
            n = len([f for f in os.listdir(frames_dir) if f.endswith(".png")])
            ts = [l.split() for l in open(os.path.join(frames_dir, "timestamps.txt")) if l.strip()]
            assert len(ts) == n, (seq, len(ts), n)
            rows.append(dict(split=split, sequence_id=seq, source_frames=n,
                             t_first_s=f"{float(ts[0][1]):.6f}", t_last_s=f"{float(ts[-1][1]):.6f}",
                             processed_id=f"video_{i:05d}", frames_sha256=ecoco_frames_sha256(frames_dir),
                             note=TRAINING_NOTES.get((split, seq), "")))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "ecoco_intensity_split.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print({s: sum(r["split"] == s for r in rows) for s in ("train", "validation")})


if __name__ == "__main__":
    main()
