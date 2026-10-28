"""Regenerate the paper's derived datasets (upsampled videos, DVS and retina event caches) locally.

Run from the repository root:

    # validate the sources and print the plan, nothing is written
    python -m reproducibility.regenerate --task saliency  --subset smoke --source-root /data/DHF1K --out /data/retina_regen --dry-run
    # smallest real-data run: one training + one validation clip per task
    python -m reproducibility.regenerate --task saliency  --subset smoke --source-root /data/DHF1K --out /data/retina_regen
    python -m reproducibility.regenerate --task intensity --subset smoke --source-root /data/ecoco_depthmaps_test --out /data/retina_regen
    # every clip of the paper manifests (same code; resumable)
    python -m reproducibility.regenerate --task saliency  --subset full  --source-root /data/DHF1K --out /data/retina_regen

Inputs are selected from reproducibility/manifests/ (the clips, splits and IDs of the paper runs).
The stages call the code the paper used:
  prepare   help_scripts/reorder_saliency_map_ds.reorder_clip                 (DHF1K only)
  upsample  dataloaders/VideoDatasetUpsampling (FILM): fixed 200 FPS and optical-flow adaptive
  events    the training scripts' dataset classes (ESIM DVS / silicon-retina CUDA), which
            simulate events, bin them into voxel grids and write cache_*.npy

Everything is written below --out:
  <task>/prepared/NNNN/                              restructured DHF1K clip (DHF1K layout only)
  <task>/upsampled-<frames>-<fp>/<split>/<id>/       same layout as saliency_dataset_ready/ and upsampled_{train,validation}/
  <task>/events/<variant>-<fp>/<split>/<id>/         event caches; videos are symlinks into upsampled-*/
  provenance/<time>_<task>_<subset>.json             inputs, hashes, settings, versions, statistics

<fp> is a hash of every setting that changes the output (source window, upsampling or event settings,
FILM weights, content of the generation code), so runs with different settings never share a cache
directory, and re-running with the same settings resumes from what is on disk.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NN = "src/neural_network"

# Code whose content defines the outputs of each stage (hashed into the fingerprints).
UPSAMPLE_CODE = [f"{NN}/dataloaders/VideoDatasetUpsampling.py", f"{NN}/dataloaders/interpolator.py",
                 f"{NN}/dataloaders/VideoDatasetFromPairs.py", f"{NN}/dataloaders/VideoDataset.py",
                 f"{NN}/help_scripts/upsample_worker_gpu.py", f"{NN}/help_scripts/reorder_saliency_map_ds.py"]
ESIM_SRC = f"{NN}/reusing_videos/esim_torch/src/esim_torch"
EVENT_CODE = [f"{NN}/dataloaders/{f}" for f in (
    "SiliconRetinaEventDatasetFromUpsampledSaliency.py", "EventDatasetFromUpsampledSaliency.py",
    "SiliconRetinaEventDatasetFromUpsampled.py", "EventDatasetFromUpsampled.py", "Encodings.py", "utils.py")] + \
    [f"{NN}/utils/event_utils.py"]

# Files of an upsampled clip that the event loaders only read (symlinked into the event directories).
# params.json is copied, because the retina loaders rewrite it with their cache key.
READ_ONLY_CLIP_FILES = ("video_upsampled_fps_eq_200.mp4", "video_upsampled_using_optical_flow.mp4",
                        "raw_images.npz", "outputs.npz", "optical_flow.npz")

C = f"{NN}/configs"
TASKS = {
    "saliency": dict(
        manifest="reproducibility/manifests/dhf1k_saliency_split.csv",
        splits=("train", "val"), fps=30, smoke_frames=31,
        # upsample_worker_gpu.py: maps.mp4 as target, optical flow not stored
        output_name="maps", return_optical_flows=False, save_optical_flow=False,
        variants={
            # main comparison, retina side (run FIRE_saliency_seq=50_gk5_20251201_014614 was launched with this file)
            "retina_ff200": ("retina", f"{C}/saliency/retina/FIREnetwork_retina_saliency_p2_eq_1.0_seq=30_gk7.json", {}),
            # main comparison, DVS side (ESIM, log input, thresholds 0.2/0.2; reads the optical-flow video)
            "dvs_of": ("dvs", f"{C}/saliency/events/FIREnetwork_event_saliency_seq=50.json", {}),
            # DVS on the retina's 200 FPS video (corrected rerun of the Fig. 12 "DVS (Fixed 200 FPS)" curve)
            "dvs_ff200": ("dvs", f"{C}/saliency/events/FIREnetwork_event_saliency_fixed_200fps.json", {}),
            # same retina on the optical-flow video: shows the two upsampled inputs give different events
            "retina_of": ("retina", f"{C}/saliency/retina/FIREnetwork_retina_saliency_p2_eq_1.0_seq=30_gk7.json",
                          {"use_upsampled_from_OF": True}),
        }),
    "intensity": dict(
        manifest="reproducibility/manifests/ecoco_intensity_split.csv",
        splits=("train", "validation"), fps=27, smoke_frames=None,
        # upsampling_dataset_prototyping.py: no target video, optical flow stored
        output_name=None, return_optical_flows=True, save_optical_flow=True,
        variants={
            # DVS baseline: the events run FIREnetwork_20250901_173249 trained on (log input, 5 bins)
            "dvs_of": ("dvs", f"{C}/FIREnetwork.json", {}),
            # the same with linear input, which that run logged but did not use (its caches were log input)
            "dvs_of_linear": ("dvs", f"{C}/FIREnetwork.json", {"log_input": False}),
            # retina config named in scripts/paper/intensity_reconstruction.sh (plotted runs not identified)
            "retina_ff200": ("retina", f"{C}/FIREnetwork_event_bins_eq2.json", {}),
        }),
}


# ---------------------------------------------------------------------------------------------- hashing

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def ecoco_frames_sha256(frames_dir):
    """Hash of an ECOCO frames/ folder: lines '<name> <sha256(bytes)>' for *.png + timestamps.txt, sorted."""
    names = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png") or f == "timestamps.txt")
    lines = "".join(f"{n} {sha256_file(os.path.join(frames_dir, n))}\n" for n in names)
    return hashlib.sha256(lines.encode()).hexdigest()


def tree_sha256(paths):
    """Hash of the byte content of files (directories are walked), keyed by repo-relative path."""
    files = []
    for p in paths:
        p = REPO / p
        files += sorted(q for q in p.rglob("*") if q.is_file() and "__pycache__" not in q.parts) if p.is_dir() else [p]
    lines = "".join(f"{q.relative_to(REPO)} {sha256_file(q)}\n" for q in files)
    return hashlib.sha256(lines.encode()).hexdigest()


def esim_sources():
    root = REPO / ESIM_SRC
    return [str(q.relative_to(REPO)) for q in sorted(root.iterdir())
            if q.suffix in (".py", ".cu", ".cpp", ".h", ".cuh")]


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------- selection

def load_manifest(task):
    return list(csv.DictReader(open(REPO / TASKS[task]["manifest"])))


def select(task, subset, frames_arg):
    rows = load_manifest(task)
    spec = TASKS[task]
    if subset == "smoke":
        # deterministic: the shortest clip of each split (ties broken by source ID)
        chosen = []
        for split in spec["splits"]:
            cand = [r for r in rows if r["split"] == split]
            chosen.append(min(cand, key=lambda r: (int(r["source_frames"]), source_id(task, r))))
    else:
        chosen = rows
    default = spec["smoke_frames"] if subset == "smoke" else None
    n = default if frames_arg is None else (None if frames_arg == "all" else int(frames_arg))
    clips = []
    for r in chosen:
        total = int(r["source_frames"])
        stop = total if n is None else min(n, total)
        clips.append(dict(row=r, split=r["split"], id=r["processed_id"], source=source_id(task, r),
                          start=0, stop=stop, total=total))
    return clips, ("all" if n is None else f"0-{n}")


def source_id(task, row):
    return row["dhf1k_id"] if task == "saliency" else row["sequence_id"]


# ---------------------------------------------------------------------------------------------- sources

def detect_layout(root):
    if (root / "video").is_dir() and (root / "annotation").is_dir():
        return "dhf1k"
    if (root / "0001" / "input.AVI").exists() or any((root / f"{i:04d}" / "input.AVI").exists() for i in range(1, 30)):
        return "restructured"
    raise SystemExit(f"{root}: expected DHF1K (video/, annotation/) or restructured (NNNN/input.AVI) layout")


def validate_sources(task, clips, args):
    """Check that every selected clip exists and matches the manifest, before any expensive work."""
    import cv2
    root = Path(args.source_root)
    problems, report = [], []
    layout = detect_layout(root) if task == "saliency" else "ecoco"
    for c in clips:
        r = c["row"]
        if task == "saliency":
            n4 = f"{int(r['dhf1k_id']):04d}"
            if layout == "dhf1k":
                video = Path(args.video_dir or root / "video") / f"{r['dhf1k_id']}.AVI"
                maps = Path(args.annotation_dir or root / "annotation") / n4 / "maps"
                n_maps = len([f for f in os.listdir(maps) if f.endswith((".png", ".jpg"))]) if maps.is_dir() else None
            else:
                video, maps = root / n4 / "input.AVI", root / n4 / "maps.mp4"
                n_maps = None
            if not video.exists() or not maps.exists():
                problems.append(f"{c['source']}: missing {video if not video.exists() else maps}")
                continue
            digest = sha256_file(video)
            cap = cv2.VideoCapture(str(video))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            ok = digest == r["source_video_sha256"] and n_frames == c["total"] and n_maps in (None, c["total"])
            c.update(source_path=str(video), annotation_path=str(maps), source_sha256=digest,
                     annotation_sha256=sha256_file(maps) if maps.is_file() else None)
            report.append(dict(id=c["source"], frames=n_frames, maps_frames=n_maps, sha256_match=digest == r["source_video_sha256"]))
        else:
            seq_dir = root / c["split"] / r["sequence_id"]
            frames_dir = seq_dir / "frames"
            if not frames_dir.is_dir():
                problems.append(f"{c['source']}: missing {frames_dir}")
                continue
            digest = ecoco_frames_sha256(frames_dir)
            n_frames = len([f for f in os.listdir(frames_dir) if f.endswith(".png")])
            ok = digest == r["frames_sha256"] and n_frames == c["total"]
            c.update(source_path=str(seq_dir), source_sha256=digest)
            report.append(dict(id=c["source"], frames=n_frames, sha256_match=digest == r["frames_sha256"]))
        if not ok:
            problems.append(f"{c['source']}: does not match the manifest "
                            f"(frames {n_frames} vs {c['total']}, sha256 {digest[:12]}...)")
    return layout, report, problems


# ---------------------------------------------------------------------------------------------- stages

def stage_prepare(clips, layout, out):
    """DHF1K layout -> NNNN/{input.AVI, maps.mp4, fixation.mp4, params.json}, as reorder_saliency_map_ds.py."""
    for c in clips:
        n4 = f"{int(c['source']):04d}"
        if layout == "restructured":
            c["clip_dir"] = str(Path(c["source_path"]).parent)
            continue
        from src.neural_network.help_scripts.reorder_saliency_map_ds import reorder_clip
        dst = out / "saliency" / "prepared" / n4
        if not ((dst / "params.json").exists() and (dst / "input.AVI").exists()
                and sha256_file(dst / "input.AVI") == c["source_sha256"]):
            reorder_clip(c["source_path"], str(Path(c["annotation_path"]).parent), str(dst))
        c["clip_dir"] = str(dst)


def upsample_settings(task, frames_tag, args):
    spec = TASKS[task]
    return dict(task=task, frames=frames_tag, fps=spec["fps"], new_framerate=200, passes=["fixed_200fps", "optical_flow"],
                resize=0.5 if task == "saliency" else None, output_name=spec["output_name"],
                return_optical_flows=spec["return_optical_flows"], save_optical_flow=spec["save_optical_flow"],
                film_weights=tree_sha256_abs(args.film_model), code=tree_sha256(UPSAMPLE_CODE))


def tree_sha256_abs(path):
    files = sorted(q for q in Path(path).rglob("*") if q.is_file())
    return hashlib.sha256("".join(f"{q.relative_to(path)} {sha256_file(q)}\n" for q in files).encode()).hexdigest()


def stage_upsample(task, clips, up_dir, args):
    """FILM upsampling in a subprocess (TensorFlow), one call per split and pass, resumable per clip."""
    for split in TASKS[task]["splits"]:
        todo = [c for c in clips if c["split"] == split]
        if not todo:
            continue
        split_dir = up_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for c in todo:
            marker = split_dir / c["id"] / "_source.json"
            src = dict(source=c["source"], sha256=c["source_sha256"], start=c["start"], stop=c["stop"])
            if marker.exists() and json.load(open(marker)) != src:
                raise SystemExit(f"{marker} was made from a different source/window; refusing to mix outputs")
        for mode in ("fixed_200fps", "optical_flow"):
            job = dict(task=task, mode=mode, split_dir=str(split_dir), model_path=str(args.film_model),
                       clips=[dict(name=c["id"], path=c.get("clip_dir", c["source_path"]),
                                   start=c["start"], stop=c["stop"]) for c in todo])
            t0 = time.time()
            subprocess.run([sys.executable, "-m", "reproducibility.regenerate", "--upsample-job", json.dumps(job)],
                           cwd=REPO, check=True)
            print(f"[upsample] {task}/{split} {mode}: {time.time() - t0:.1f}s")
        for c in todo:
            json.dump(dict(source=c["source"], sha256=c["source_sha256"], start=c["start"], stop=c["stop"]),
                      open(split_dir / c["id"] / "_source.json", "w"))


def run_upsample_job(job):
    """Subprocess body: same dataset classes and settings as the original upsampling scripts."""
    os.chdir(REPO)
    if job["task"] == "saliency":
        # TF environment of the saliency upsampling worker (GPU memory growth, mixed precision policy)
        import src.neural_network.help_scripts.upsample_worker_gpu  # noqa: F401
    else:
        # upsampling_dataset_prototyping.py: no mixed precision policy
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    import tensorflow as tf
    # TensorFlow must create its cuDNN handle before torch initialises CUDA in this process, otherwise
    # (torch 2.10 + TF 2.20) FILM's Conv2D fails with "INTERNAL: an unsupported value or parameter".
    # A dummy convolution does that; it does not affect the interpolated frames.
    tf.nn.conv2d(tf.zeros((1, 8, 8, 3)), tf.zeros((3, 3, 3, 4)), 1, "SAME")
    from src.neural_network.dataloaders.VideoDatasetUpsampling import (VideoDatasetUpsampling,
                                                                        VideoDatasetUpsamplingCacheToFolder)
    from src.neural_network.dataloaders.VideoDatasetFromPairs import VideoDatasetFromVideoPairGlobal
    from src.neural_network.dataloaders.VideoDataset import VideoDatasetFromPictures

    spec = TASKS[job["task"]]
    windows = [(c["start"], c["stop"]) for c in job["clips"]]

    class Windowed:
        def __getitem__(self, idx):
            item = super().__getitem__(idx)
            s, e = windows[idx]
            item["images"] = item["images"][s:e]
            if "maps" in item:
                item["maps"] = item["maps"][s:e]
            return item

    if job["task"] == "saliency":
        class Clips(Windowed, VideoDatasetFromVideoPairGlobal):
            pass
        ds = Clips.__new__(Clips)
        ds.files, ds.fps, ds.suffix = [c["path"] for c in job["clips"]], spec["fps"], ""
        ds.input_video, ds.output_video = "input.AVI", "maps.mp4"
    else:
        class Clips(Windowed, VideoDatasetFromPictures):
            pass
        ds = Clips.__new__(Clips)
        ds.files, ds.fps, ds.suffix = [c["path"] for c in job["clips"]], spec["fps"], "frames"

    of = job["mode"] == "optical_flow"
    up = VideoDatasetUpsampling(ds, model_path=job["model_path"], upsample_to_events=of,
                                new_framerate=False if of else 200, output_name=spec["output_name"],
                                return_optical_flows=spec["return_optical_flows"], debug=False)
    VideoDatasetUpsamplingCacheToFolder(upsamplingDataset=up, output_path=job["split_dir"], override=False,
                                        save_optical_flow=spec["save_optical_flow"],
                                        folder_names=[c["name"] for c in job["clips"]])


def resolve_variant(task, name, overrides):
    kind, cfg_path, fixed = TASKS[task]["variants"][name]
    cfg = json.load(open(REPO / cfg_path))
    for k, v in {**fixed, **overrides}.items():
        node = cfg
        *parents, leaf = k.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = v
    # resolution of the training scripts (saliency_using_FIREnetwork_*.py, reconstruct_intensity_*.py)
    settings = dict(kind=kind, config=cfg_path, config_sha256=sha256_file(REPO / cfg_path),
                    overrides={**fixed, **overrides}, num_bins=cfg.get("num_bins", 5))
    params = dict(cfg.get("event_representation_params", {}))
    params["num_bins"] = settings["num_bins"]
    settings["event_representation_params"] = params
    if kind == "retina":
        from esim_torch.silicon_retina_torch import SiliconRetinaConfig
        settings.update(silicon_retina_config=SiliconRetinaConfig().from_dict(cfg["silicon_retina_config"]).to_dict(),
                        event_representation_class=cfg["event_representation_class"],
                        log_input=cfg.get("log_input", False), use_upsampled_from_OF=cfg.get("use_upsampled_from_OF", False),
                        input_framerate=200)
    else:
        settings.update(contrast_threshold_pos=params.get("contrast_threshold_pos", 0.2),
                        contrast_threshold_neg=params.get("contrast_threshold_neg", 0.2), refractory_period_ns=0,
                        log_input=cfg.get("log_input", True if task == "saliency" else False),
                        dvs_input_video=cfg.get("dvs_input_video", "optical_flow"))
    return settings


def build_dataset(task, settings, directory):
    from src.neural_network.dataloaders.Encodings import EventRepresentation
    params = copy.deepcopy(settings["event_representation_params"])
    if settings["kind"] == "retina":
        from esim_torch.silicon_retina_torch import SiliconRetinaConfig
        se_cfg = SiliconRetinaConfig().from_dict(settings["silicon_retina_config"])
        rep = EventRepresentation(settings["event_representation_class"])
        if task == "saliency":
            from src.neural_network.dataloaders.SiliconRetinaEventDatasetFromUpsampledSaliency import \
                SiliconRetinaEventDatasetFromUpsampledSaliency as D
            return D(directory=directory, config=se_cfg, prediction_video_name="outputs.npz", preprocessing_params=params,
                     preprocessing_type=rep, log_scale=settings["log_input"], force_recomputing=False, limit=None,
                     use_upsampled_from_OF=settings["use_upsampled_from_OF"], save_xypt=True)
        from src.neural_network.dataloaders.SiliconRetinaEventDatasetFromUpsampled import \
            SiliconRetinaEventDatasetFromUpsampled as D
        return D(directory=directory, config=se_cfg, preprocessing_params=params, preprocessing_type=rep,
                 log_scale=settings["log_input"], force_recomputing=False, limit=None,
                 use_upsampled_from_OF=settings["use_upsampled_from_OF"])
    if task == "saliency":
        from src.neural_network.dataloaders.EventDatasetFromUpsampledSaliency import EventDatasetFromUpsampledSaliency as D
        return D(directory=directory, mode="event", prediction_video_name="outputs.npz", preprocessing_params=params,
                 log_scale=settings["log_input"], force_recomputing=False, limit=None, save_xypt=True,
                 input_video=settings["dvs_input_video"])
    from src.neural_network.dataloaders.EventDatasetFromUpsampled import EventDatasetFromUpsampled as D
    return D(directory=directory, mode="event", preprocessing_params=params, log_scale=settings["log_input"],
             force_recomputing=False, limit=None)


def link_clips(src_split_dir, dst_split_dir, ids):
    """Event working directory: symlinks to the read-only upsampled files + a private params.json."""
    dst_split_dir.mkdir(parents=True, exist_ok=True)
    meta = dst_split_dir / "metadata.json"
    if not meta.exists():
        meta.write_bytes((src_split_dir / "metadata.json").read_bytes())
    for cid in ids:
        s, d = src_split_dir / cid, dst_split_dir / cid
        d.mkdir(exist_ok=True)
        for f in READ_ONLY_CLIP_FILES:
            if (s / f).exists() and not (d / f).exists():
                os.symlink((s / f).resolve(), d / f)
        if not (d / "params.json").exists():
            (d / "params.json").write_bytes((s / "params.json").read_bytes())


def stage_events(task, clips, up_dir, ev_root, variants, tag=""):
    import numpy as np
    import torch
    results = {}
    for name, settings in variants.items():
        fp = fingerprint(dict(upsampled=up_dir.name, settings=settings,
                              code=tree_sha256(EVENT_CODE + esim_sources())))
        vdir = ev_root / f"{name}-{fp}{tag}"
        results[name] = dict(dir=str(vdir), fingerprint=fp, settings=settings, clips={})
        for split in TASKS[task]["splits"]:
            ids = [c["id"] for c in clips if c["split"] == split]
            if not ids:
                continue
            link_clips(up_dir / split, vdir / split, ids)
            ds = build_dataset(task, settings, str(vdir / split))
            for i, folder in enumerate(ds.data_files):
                cid = os.path.basename(folder)
                if cid not in ids:
                    continue
                t0 = time.time()
                item = ds[i]
                results[name]["clips"][cid] = dict(split=split, seconds=round(time.time() - t0, 2),
                                                   **item_stats(item, task, settings))
                torch.cuda.empty_cache()
    return results


def item_stats(item, task, settings):
    import numpy as np
    import torch
    g = item["grids"]
    g = g.float() if torch.is_tensor(g) else torch.from_numpy(np.asarray(g, dtype=np.float32))
    st = dict(grids_shape=list(g.shape), grids_finite=bool(torch.isfinite(g).all()),
              grids_nonzero_fraction=float((g != 0).float().mean()),
              grids_abs_sum=float(g.abs().sum()),
              frames_shape=list(np.asarray(item["frames"]).shape))
    if "outputs" in item:
        o = np.asarray(item["outputs"], dtype=np.float32)
        st.update(outputs_shape=list(o.shape), outputs_min=float(o.min()), outputs_max=float(o.max()),
                  outputs_finite=bool(np.isfinite(o).all()))
    ev = item.get("events") or {}
    if ev.get("t") is not None and len(ev["t"]):
        t, p = ev["t"].long(), ev["p"].long()
        H, W = g.shape[-2], g.shape[-1]
        st.update(events=int(t.numel()), events_on=int((p > 0).sum()), events_off=int((p < 0).sum()),
                  t_first_ns=int(t.min()), t_last_ns=int(t.max()), t_sorted=bool((t[1:] >= t[:-1]).all()),
                  x_range=[int(ev["x"].min()), int(ev["x"].max())], y_range=[int(ev["y"].min()), int(ev["y"].max())],
                  pixel_occupancy=float(len(set((ev["y"].long() * W + ev["x"].long()).tolist())) / (H * W)))
    return st


# ---------------------------------------------------------------------------------------------- comparisons

def compare_dirs(a, b):
    """Compare the cache_*.npy files of two event directories with the same layout."""
    import numpy as np
    out = {}
    for f in sorted(Path(a).rglob("cache_*.npy")):
        rel = f.relative_to(a)
        g = Path(b) / rel
        if not g.exists():
            out[str(rel)] = "missing"
            continue
        x, y = np.load(f), np.load(g)
        if x.shape != y.shape:
            out[str(rel)] = f"shape {x.shape} vs {y.shape}"
        elif np.array_equal(x, y):
            out[str(rel)] = "identical"
        else:
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            out[str(rel)] = f"max|diff|={d.max():.4g}, differing={float((d > 0).mean()):.3%}"
    return out


def compare_with_reference(task, clips, up_dir, ref_root):
    """Regenerated upsampled clips vs the original ones (read only), over the frames both cover."""
    import numpy as np
    import imageio.v3 as iio
    fps = TASKS[task]["fps"]
    report = {}
    for c in clips:
        ref = Path(ref_root) / (c["split"] if task == "saliency" else
                                {"train": "upsampled_train", "validation": "upsampled_validation"}[c["split"]]) / c["id"]
        new = up_dir / c["split"] / c["id"]
        r = {}
        a = np.load(new / "raw_images.npz")["raw_images"]
        b = np.load(ref / "raw_images.npz")["raw_images"][c["start"]:c["stop"]]
        r["raw_images_identical"] = bool(a.shape == b.shape and np.array_equal(a, b))
        if task == "saliency":
            a = np.load(new / "outputs.npz")["outputs"]
            b = np.load(ref / "outputs.npz")["outputs"][c["start"]:c["stop"]]
            r["targets_identical"] = bool(a.shape == b.shape and np.array_equal(a, b))
        pn, pr = json.load(open(new / "params.json")), json.load(open(ref / "params.json"))
        n_pairs = c["stop"] - c["start"] - 1
        # fixed rate: frames with t < (stop-1)/fps lie between two frames that both runs interpolate
        ff_n = int(np.sum(np.asarray(pn["timestamps_FF"]) < (c["stop"] - 1) / fps - 1e-12))
        r["ff_timestamps_identical"] = pn["timestamps_FF"][:ff_n] == pr["timestamps_FF"][:ff_n]
        of_new, of_ref = pn["new_images_per_pair_OF"][:n_pairs], pr["new_images_per_pair_OF"][:n_pairs]
        r["of_pairs_compared"] = n_pairs
        r["of_pairs_same_count"] = int(sum(x == y for x, y in zip(of_new, of_ref)))
        for key, video, n in (("ff", "video_upsampled_fps_eq_200.mp4", ff_n),
                              ("of", "video_upsampled_using_optical_flow.mp4",
                               sum(of_new[:next((k for k, (x, y) in enumerate(zip(of_new, of_ref)) if x != y), n_pairs)]))):
            if n == 0:
                continue
            va = iio.imread(new / video)[:n].astype(np.float64)
            vb = iio.imread(ref / video)[:n].astype(np.float64)
            mse = float(((va - vb) ** 2).mean())
            r[f"{key}_video_frames_compared"] = int(n)
            r[f"{key}_video_mean_abs_diff"] = round(float(np.abs(va - vb).mean()), 4)
            r[f"{key}_video_max_abs_diff"] = int(np.abs(va - vb).max())
            r[f"{key}_video_psnr_db"] = round(10 * np.log10(255 ** 2 / mse), 2) if mse > 0 else float("inf")
        report[c["id"]] = r
    return report


def compare_events_with_reference(task, clips, variants, new_results, ref_results):
    """Event grids from regenerated vs original upsampled video, over the intervals both cover."""
    import numpy as np
    out = {}
    for name in variants:
        for c in clips:
            a_dir = Path(new_results[name]["dir"]) / c["split"] / c["id"]
            b_dir = Path(ref_results[name]["dir"]) / c["split"] / c["id"]
            ga = sorted(a_dir.glob("cache_*_grids.npy"))
            gb = sorted(b_dir.glob("cache_*_grids.npy"))
            if not ga or not gb:
                continue
            a, b = np.load(ga[0]).astype(np.float32), np.load(gb[0]).astype(np.float32)
            n = min(len(a), len(b)) - 1          # last regenerated interval is extrapolated in a window
            if n <= 0:
                continue
            a, b = a[:n], b[:n]
            corr = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
            out[f"{name}/{c['id']}"] = dict(intervals=n, identical=bool(np.array_equal(a, b)),
                                            abs_sum_new=float(np.abs(a).sum()), abs_sum_ref=float(np.abs(b).sum()),
                                            nonzero_new=float((a != 0).mean()), nonzero_ref=float((b != 0).mean()),
                                            correlation=round(corr, 4))
    return out


# ---------------------------------------------------------------------------------------------- provenance

def environment():
    env = dict(python=sys.version.split()[0], platform=platform.platform(), packages={})
    for pkg in ("torch", "tensorflow", "numpy", "opencv-python", "imageio", "imageio-ffmpeg", "pillow", "esim_torch"):
        try:
            env["packages"][pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            env["packages"][pkg] = None
    try:
        import imageio_ffmpeg
        env["ffmpeg"] = imageio_ffmpeg.get_ffmpeg_version()
    except Exception:
        pass
    try:
        import torch
        env["torch_cuda"] = torch.version.cuda
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        import esim_torch
        env["esim_torch_module"] = os.path.dirname(esim_torch.__file__)
    except Exception as e:
        env["torch_error"] = repr(e)
    try:
        env["nvidia_driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                              capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        pass
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    env["git_commit"] = git("rev-parse", "HEAD")
    env["git_dirty_files"] = [l[3:] for l in git("status", "--porcelain").splitlines()]
    return env


def disk_usage(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return total


# ---------------------------------------------------------------------------------------------- main

def parse_overrides(items):
    out = {}
    for it in items or []:
        key, value = it.split("=", 1)
        variant, path = key.split(".", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        out.setdefault(variant, {})[path] = value
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=TASKS)
    ap.add_argument("--subset", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--source-root", help="DHF1K root (video/, annotation/) or restructured NNNN/ folders; "
                                          "ECOCO: the extracted ecoco_depthmaps_test/ folder")
    ap.add_argument("--video-dir", help="DHF1K videos if not <source-root>/video")
    ap.add_argument("--annotation-dir", help="DHF1K annotations if not <source-root>/annotation")
    ap.add_argument("--out", help="output root (created if missing); nothing is written elsewhere")
    ap.add_argument("--film-model", default=str(REPO / "models/film_net/Style/saved_model"))
    ap.add_argument("--frames", help="frames per clip from the start ('all' for whole clips); "
                                     "default: smoke 31 (saliency) / all (intensity), full: all")
    ap.add_argument("--stages", default="prepare,upsample,events")
    ap.add_argument("--variants", help="comma-separated event variants (default: all of the task)")
    ap.add_argument("--set", action="append", metavar="VARIANT.KEY=VALUE",
                    help="override a config value, e.g. retina_ff200.silicon_retina_config.p2=1.25")
    ap.add_argument("--repeat", action="store_true", help="generate the events twice and compare (determinism)")
    ap.add_argument("--reference-root", help="original upsampled data to compare with, read only "
                                             "(saliency: saliency_dataset_ready/, intensity: the data/ folder)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--upsample-job", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.upsample_job:
        return run_upsample_job(json.loads(args.upsample_job))
    if not (args.task and args.source_root and args.out):
        ap.error("--task, --source-root and --out are required")
    os.chdir(REPO)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    task, out = args.task, Path(args.out).resolve()
    args.film_model = Path(args.film_model).resolve()
    t_start = time.time()

    clips, frames_tag = select(task, args.subset, args.frames)
    names = args.variants.split(",") if args.variants else list(TASKS[task]["variants"])
    overrides = parse_overrides(args.set)
    stages = set(args.stages.split(","))

    print(f"[select] {task}/{args.subset}: {len(clips)} clips, frames {frames_tag}")
    layout, src_report, problems = validate_sources(task, clips, args)
    if "upsample" in stages and not (args.film_model / "saved_model.pb").exists():
        problems.append(f"FILM weights not found at {args.film_model} (bash scripts/setup/download_model.sh)")
    for r in src_report[:10]:
        print(f"[source] {r}")
    if problems:
        print("\n".join(f"[error] {p}" for p in problems))
        raise SystemExit(f"{len(problems)} problem(s) with the inputs; nothing was generated")

    variants = {n: resolve_variant(task, n, overrides.get(n, {})) for n in names}
    up_settings = upsample_settings(task, frames_tag, args)
    up_dir = out / task / f"upsampled-{frames_tag}-{fingerprint(up_settings)}"
    print(f"[plan] layout={layout}  upsampled -> {up_dir}")
    for n, s in variants.items():
        print(f"[plan] events {n}: {s['kind']}, {s['config']} {s['overrides'] or ''}")
    if args.dry_run:
        frames = sum(c["stop"] - c["start"] for c in clips)
        print(f"[dry-run] {len(clips)} clips, {frames} source frames; inputs validated, nothing written")
        return

    out.mkdir(parents=True, exist_ok=True)
    timings = {}
    if task == "saliency" and "prepare" in stages:
        t0 = time.time()
        stage_prepare(clips, layout, out)
        timings["prepare_s"] = round(time.time() - t0, 1)
    if "upsample" in stages:
        t0 = time.time()
        stage_upsample(task, clips, up_dir, args)
        timings["upsample_s"] = round(time.time() - t0, 1)

    results, repeat, ref_events, comparisons = {}, None, None, {}
    if "events" in stages:
        t0 = time.time()
        results = stage_events(task, clips, up_dir, out / task / "events", variants)
        timings["events_s"] = round(time.time() - t0, 1)
        if args.repeat:
            rep = stage_events(task, clips, up_dir, out / task / "events", variants, tag="-repeat")
            repeat = {n: compare_dirs(results[n]["dir"], rep[n]["dir"]) for n in variants}
    if args.reference_root:
        comparisons["upsampled"] = compare_with_reference(task, clips, up_dir, args.reference_root)
        if "events" in stages:
            ref_up = Path(args.reference_root)
            if task == "intensity":   # same <split>/<id> layout as the regenerated tree
                ref_up = out / task / "reference-layout"
                for split, name in (("train", "upsampled_train"), ("validation", "upsampled_validation")):
                    if not (ref_up / split).exists():
                        (ref_up).mkdir(parents=True, exist_ok=True)
                        os.symlink(Path(args.reference_root).resolve() / name, ref_up / split)
            ref_events = stage_events(task, clips, ref_up, out / task / "reference-events", variants)
            comparisons["events_regenerated_vs_original_upsampling"] = \
                compare_events_with_reference(task, clips, variants, results, ref_events)

    record = dict(
        command=sys.argv, created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), task=task,
        subset=args.subset, frames=frames_tag, manifest=TASKS[task]["manifest"],
        manifest_sha256=sha256_file(REPO / TASKS[task]["manifest"]), source_layout=layout,
        randomness="none: frame selection, upsampling and event generation use no random numbers",
        clips=[dict(id=c["id"], source=c["source"], split=c["split"], frames=[c["start"], c["stop"]],
                    source_frames=c["total"], source_sha256=c["source_sha256"],
                    annotation_sha256=c.get("annotation_sha256")) for c in clips],
        upsampling=dict(dir=str(up_dir), settings=up_settings), events=results, repeat_comparison=repeat,
        reference_comparison=comparisons, environment=environment(), timings=timings,
        disk_bytes={"upsampled": disk_usage(up_dir) if up_dir.exists() else 0,
                    **{f"events/{n}": disk_usage(r["dir"]) for n, r in results.items()}})
    (out / "provenance").mkdir(exist_ok=True)
    prov = out / "provenance" / f"{time.strftime('%Y%m%dT%H%M%S')}_{task}_{args.subset}.json"
    json.dump(record, open(prov, "w"), indent=1)
    summarise(record, up_dir, prov, time.time() - t_start)


def summarise(rec, up_dir, prov, seconds):
    print("\n==================== summary ====================")
    for c in rec["clips"]:
        d = up_dir / c["split"] / c["id"]
        if (d / "params.json").exists():
            p = json.load(open(d / "params.json"))
            ff, of = p.get("timestamps_FF", []), p.get("timestamps_OF", [])
            print(f"{c['split']:>10} {c['id']} <- {c['source']} frames {c['frames'][0]}-{c['frames'][1]}: "
                  f"200 FPS {len(ff)} frames [{ff[0]:.3f}, {ff[-1]:.3f}] s, optical flow {len(of)} frames "
                  f"[{of[0]:.3f}, {of[-1]:.3f}] s")
    for name, r in rec["events"].items():
        print(f"-- {name} ({r['settings']['kind']}) -> {r['dir']}")
        for cid, s in r["clips"].items():
            ev = f"events {s['events']} (+{s['events_on']}/-{s['events_off']}), pixel occupancy {s['pixel_occupancy']:.3f}, " \
                 f"t [{s['t_first_ns'] / 1e9:.4f}, {s['t_last_ns'] / 1e9:.4f}] s" if "events" in s else "no event list"
            print(f"   {cid}: grids {s['grids_shape']} finite={s['grids_finite']} "
                  f"nonzero voxels {s['grids_nonzero_fraction']:.4f}; {ev}")
    if rec["repeat_comparison"]:
        for name, files in rec["repeat_comparison"].items():
            states = sorted(set(v if v in ("identical", "missing") else "different" for v in files.values()))
            print(f"[repeat] {name}: {len(files)} cache files, {states}")
            for f, v in files.items():
                if v != "identical":
                    print(f"         {f}: {v}")
    for k, v in rec["reference_comparison"].items():
        print(f"[reference] {k}:")
        for cid, r in v.items():
            print(f"   {cid}: {r}")
    gib = lambda b: f"{b / 2**20:.1f} MiB"
    print("disk: " + ", ".join(f"{k} {gib(v)}" for k, v in rec["disk_bytes"].items()))
    print(f"provenance: {prov}\ntotal time: {seconds:.0f}s")


if __name__ == "__main__":
    main()
