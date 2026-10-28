import os, sys, site, glob
import gc, torch

# Build LD_LIBRARY_PATH to point at pip CUDA libs inside this venv
libroots = []
for pkg, sub in [
    ("nvidia/cuda_runtime", "lib"),
    ("nvidia/cublas",       "lib"),
    ("nvidia/cudnn",        "lib"),
    ("nvidia/cufft",        "lib"),
]:
    for sp in site.getsitepackages():
        path = os.path.join(sp, pkg, sub)
        if os.path.isdir(path):
            libroots.append(path)

# Prepend (don’t overwrite) so TF finds the right .so’s
os.environ["LD_LIBRARY_PATH"] = ":".join(libroots + [os.environ.get("LD_LIBRARY_PATH","")])

# Usual quality-of-life flags
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf  # <-- import FIRST
print("TF build:", tf.__version__, tf.sysconfig.get_build_info())
print("TF GPUs:", tf.config.list_physical_devices('GPU'))

import json
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from .VideoDataset import VideoDatasetFromPictures
from .interpolator import Interpolator
import tqdm
import imageio
import imageio.v3 as iio                      # ≥ 2.30, pulls imageio-ffmpeg

def write_sequence_to_video(images, filename):
    with imageio.get_writer(filename, fps=30) as writer:
        for i in range(len(images)):
            writer.append_data(np.clip(255. * images[i], 0, 255.).astype(np.uint8)  )


def write_sequence_and_original_to_video(upsampled, timestamps, original, original_fps, filename):
    i = 0
    j = 0
    dt = 1 / original_fps
    with imageio.get_writer(filename, fps = original_fps) as writer:
        for i, t in enumerate(timestamps):
            if t > (j+1) * dt:
                j += 1

            upsampled_new = np.clip(255 *  upsampled[i], 0, 255).astype(np.uint8)
            new_data = np.concatenate([upsampled_new, np.repeat(original[j][..., None], 3, axis=2)  ], axis=1)
            # is of shape
            writer.append_data(new_data)

def to_nhwc3_batched(img):
    """
        img : np.ndarray, shape (H, W) or (H, W, 1) or (H, W, 3) or already batched (1, H, W, C)
        returns : float32 in [0, 1] shape (1, H, W, 3)
    """
    
    x = img 
    
    # remove batch if already present with size 1 ( safe-guard )
    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]
        
    # ensure last 3 channels
    
    # (H, W)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=-1)
        
    elif x.ndim == 3 and x.shape[-1] == 1: # (H, W, 1)
        x = np.repeat(x, 3, axis=-1)
        
    elif x.ndim == 3 and x.shape[-1] == 3: # (H, W, 3)
        pass

    else:
        raise ValueError(f"Unexpected frame shape {x.shape}; expected (H,W),(H,W,1),(H,W,3).")
    
    x = x.astype(np.float32) 
    if x.max() > 1.0:
        x = x / 255.0
        
    x = x[None, ...]
    return x
    
class VideoDatasetUpsampling(Dataset):

    def __init__(self,
                 videoDataset,
                 model_path : str,
                 upsample_to_events : bool = False,
                 new_framerate : Optional[int] = None,
                 debug : bool = False,
                 align: Optional[int] = None,
                 return_optical_flows : bool = True,
                 output_name : Optional[str] = None) -> None:

        super(VideoDatasetUpsampling, self).__init__()
        self.videoDataset = videoDataset
        self.model_path = model_path
        self.upsample_to_events = upsample_to_events
        self.new_framerate = new_framerate
        self.debug = debug
        self.output_name = output_name
        self.return_optical_flows = return_optical_flows

        if not self.upsample_to_events and new_framerate is None:
            raise Exception("Upsampling to events was specified to False and new framerate was not provided !")


        self.interpolation = Interpolator(model_path=model_path,
                                          align=align)


    def _upsample_adaptive(self, I0, I1, t0, t1, num_bisections=-1):
        if num_bisections == 0:
            return [], [], []

        dt = self.batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
        image, F_0_1, F_1_0 = self.interpolation.interpolate(I0, I1, dt)
        was_first = num_bisections < 0

        if num_bisections < 0:
            flow_mag_0_1_max = ((F_0_1 ** 2).sum(-1) ** .5).max()
            flow_mag_1_0_max = ((F_1_0 ** 2).sum(-1) ** .5).max()
            max_flow = max([flow_mag_0_1_max, flow_mag_1_0_max])
            if max_flow < 1.0:
                num_bisections = 0  # No interpolation needed
            else:
                num_bisections = int(np.ceil(np.log2(max_flow)))
                num_bisections = min(num_bisections, 4)  # safety cap
                

            # for safety, limit the number of bisections to 5 
            num_bisections = min(num_bisections, 4)

            if num_bisections == 0:
                return [image[0]], F_0_1, [(t0 + t1) / 2]

        left_images, left_of, left_timestamps = self._upsample_adaptive(I0, image, t0, (t0+t1)/2, num_bisections=num_bisections-1)
        right_images, right_of, right_timestamps = self._upsample_adaptive(image, I1, (t0+t1)/2, t1, num_bisections=num_bisections-1)

        timestamps = left_timestamps + [(t0+t1)/2] + right_timestamps
        images = left_images + [image[0]] + right_images

        of_return = []
        if was_first:
            of_return = F_0_1

        return images, of_return, timestamps

    def _upsample_optical_flow(self, datapoint):

        images = datapoint['images']
        fps = datapoint['fps']
        
        if self.output_name is not None:
            outputs = datapoint[self.output_name]

        all_images = []
        all_timestamps = []
        optical_flows = []

        if self.debug:
            pbar = tqdm.tqdm(range(len(images)-1))
        else:
            pbar = range(len(images)-1)

        dt = 1. / fps
        for i in pbar:
            current_image, next_image = images[i], images[i + 1]
            t0, t1  = i * dt, (i + 1) * dt

            current_image = to_nhwc3_batched(images[i])            # (1,H,W,3)
            next_image    = to_nhwc3_batched(images[i + 1])        # (1,H,W,3)

            total_frames, optical_flow, total_timestamps = self._upsample_adaptive(current_image,
                                                                                 next_image,
                                                                                 t0,
                                                                                 t1)
            total_frames = [current_image[0]] + total_frames
            timestamps = [t0] + total_timestamps

            sorted_indices = np.argsort(timestamps)
            total_frames = [total_frames[j] for j in sorted_indices]
            timestamps = [timestamps[i] for i in sorted_indices]

            if self.debug:
                print(f"Generated : {len(total_frames)}")
            all_images.append(total_frames)
            all_timestamps.append(timestamps)
            
            if self.return_optical_flows:
                optical_flows.append(optical_flow)

        images_per_patch = [len(a) for a in all_images]

        if self.return_optical_flows:
            optical_flows = np.concatenate(optical_flows)

        
        if self.output_name is None:
            return np.array(images), np.concatenate(all_images), np.array(images_per_patch), np.concatenate(all_timestamps), optical_flows
        else:
            return np.array(images), np.concatenate(all_images), np.array(images_per_patch), np.concatenate(all_timestamps), optical_flows, np.array(outputs)


    def _upsample_to_fixed_framerate(self, datapoint, new_framerate):
        images = datapoint['images']
        fps = datapoint['fps']

        if self.output_name is not None:
            outputs = datapoint[self.output_name]

        all_images = []
        all_timestamps = []
        optical_flows = []

        new_dt = 1.0 / new_framerate
        img_dt = 1.0 / fps

        T = len(images)
        video_time = img_dt * T

        if self.debug:
            pbar = tqdm.tqdm(np.arange(0, video_time, new_dt))
        else:
            pbar = np.arange(0, video_time, new_dt)

        i = 0
        image_deltas = []
        prev = 0

        for time in pbar:

            if time >= (i + 1) * img_dt:
                i += 1
                image_deltas.append(prev)
                prev = 0

            original_frame_time = i * img_dt
            next_original_frame_time = (i + 1) * img_dt

            dt_val = (time - original_frame_time) / (next_original_frame_time - original_frame_time)
            dt_rel = np.array([dt_val], dtype=np.float32)

            prev += 1

            # preprocess the images to fit the shape
            I0 = to_nhwc3_batched(images[i])                  # (1,H,W,3) float32
            I1 = to_nhwc3_batched(images[i + 1] if i + 1 < T else images[i])

            if time == original_frame_time:
                all_images.append(I0)
                all_timestamps.append(time)
                continue

            interpolated_image, F_0_1, F_1_0 = self.interpolation.interpolate(I0, I1, dt_rel)
            all_images.append(interpolated_image)
            all_timestamps.append(time)

        image_deltas.append(prev)

        if self.output_name is None:
            return np.array(images), np.concatenate(all_images, axis=0), np.array(image_deltas), np.array(all_timestamps), optical_flows
        return np.array(images), np.concatenate(all_images, axis=0), np.array(image_deltas), np.array(all_timestamps), optical_flows, outputs
        
    def __len__(self): 
        return len(self.videoDataset)

    def __getitem__(self, idx):
        datapoint = self.videoDataset[idx]
        print("Processing index :", idx, " path : ", datapoint.get("path", "N/A"))

        if self.upsample_to_events:
            return self._upsample_optical_flow(datapoint)
        else:
            return self._upsample_to_fixed_framerate(datapoint, self.new_framerate)

class VideoDatasetUpsamplingCacheToFolder(Dataset):
    def __init__(self, upsamplingDataset : VideoDatasetUpsampling,
                 output_path, save_mp4=True, bit_depth=8,
                 start_index = 0, override=False, save_optical_flow=True, folder_names=None):
        self.upsamplingDataset = upsamplingDataset
        # output folder of item i; default video_{i:05d}. A list keeps the paper's clip IDs when
        # caching a subset (reproducibility/regenerate.py)
        self.folder_names = folder_names
        self.output_path = output_path
        self.save_mp4 = save_mp4
        self.bit_depth = bit_depth
        self.start_index = start_index
        self.override = override
        self.save_optical_flow = save_optical_flow
        self.data = []
        self._cache()

    def _folder_name(self, i):
        return self.folder_names[i] if self.folder_names is not None else f"video_{i:05d}"

    def _write_mp4(self, frames_f32, out_path, fps):
        """Save float32 [0,1] RGB video to MP4, 8- or 10-bit."""
        h, w = frames_f32.shape[1:3]

        if self.bit_depth == 8:  # ---- H.264 8-bit ----
            frames_u8 = (frames_f32 * 255.0 + 0.5).clip(0, 255).astype(np.uint8)

            writer = imageio.get_writer(
                out_path,
                fps=fps,
                codec='libx264',
                pixelformat='yuv420p',  # 8-bit 4:2:0
                macro_block_size=None,  # disable automatic resizing
                # any ffmpeg option can go here, e.g.   ffmpeg_params=['-crf','18']
            )
            for f in frames_u8:
                writer.append_data(f)
            writer.close()

        elif self.bit_depth == 10:  # ---- HEVC Main10 ----
            frames_u16 = (frames_f32 * 1023.0 + 0.5).clip(0, 1023).astype(np.uint16)

            writer = imageio.get_writer(
                out_path,
                fps=fps,
                codec='libx265',
                pixelformat='yuv420p10le',  # 10-bit 4:2:0 (Main10)
                macro_block_size=None,
                input_params=['-pix_fmt', 'rgb30le']  # tell ffmpeg what we feed
            )
            for f in frames_u16:
                writer.append_data(f)
            writer.close()
        else:
            raise ValueError("bit_depth must be 8 or 10")

    def _cache(self):
        import json, os, gc, time
        import numpy as np
        from pathlib import Path

        print("Cacheing, start_index:", getattr(self, "start_index", 0))

        upsample_to_events = self.upsamplingDataset.upsample_to_events
        new_framerate = self.upsamplingDataset.new_framerate

        # ---- metadata ----
        metadata = {
            "number_of_videos": len(self.upsamplingDataset),
            "bit_depth": self.bit_depth,
            "model_path": self.upsamplingDataset.model_path,
            "optical_flow_saved": True,
            "optical_flow_name": "optical_flow.npz",
        }
        if upsample_to_events:
            metadata["upsampling_OF_saved"] = True
        else:
            metadata["upsampling_to_fixed_fps"] = True
            metadata["new_framerate"] = new_framerate

        os.makedirs(self.output_path, exist_ok=True)
        meta_path = os.path.join(self.output_path, "metadata.json")

        # merge existing metadata if present
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                old = json.load(f)
            for k, v in old.items():
                if k not in metadata:
                    metadata[k] = v

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # ---- progress tracker ----
        progress_path = os.path.join(self.output_path, "_progress.json")
        if os.path.exists(progress_path):
            try:
                progress = json.load(open(progress_path))
            except Exception:
                progress = {}
        else:
            progress = {}

        # helper: decide if sample i is already complete
        def is_done(i: int) -> bool:
            root = os.path.join(self.output_path, self._folder_name(i))
            mp4_name = "video_upsampled_using_optical_flow.mp4" if upsample_to_events \
                    else f"video_upsampled_fps_eq_{new_framerate}.mp4"
            return os.path.exists(os.path.join(root, "params.json")) and os.path.exists(os.path.join(root, mp4_name))

        # if override=False, auto-advance start_index to first not-done
        start_index = int(getattr(self, "start_index", 0) or 0)
        if not getattr(self, "override", False):
            for i in range(start_index, len(self.upsamplingDataset)):
                if not is_done(i):
                    start_index = i
                    break
            else:
                print("[info] Nothing to do, all items completed.")
                return

        # write initial progress
        progress.update({"start_index": start_index, "last_completed": progress.get("last_completed", start_index-1)})
        json.dump(progress, open(progress_path, "w"), indent=2)

        import tqdm, torch, tensorflow as tf

        # periodic cleanup knobs
        CLEAR_EVERY = 12

        pbar = tqdm.tqdm(range(start_index, len(self.upsamplingDataset)))
        for idx, i in enumerate(pbar, start=start_index):
            root_folder = os.path.join(self.output_path, self._folder_name(i))
            Path(root_folder).mkdir(parents=True, exist_ok=True)

            # skip if already done and not overriding
            if not getattr(self, "override", False) and is_done(i):
                progress["last_completed"] = i
                json.dump(progress, open(progress_path, "w"), indent=2)
                continue

            # build params, decide MP4 path
            if upsample_to_events:
                mp4_path = os.path.join(root_folder, "video_upsampled_using_optical_flow.mp4")
            else:
                mp4_path = os.path.join(root_folder, f"video_upsampled_fps_eq_{new_framerate}.mp4")

            params_path = os.path.join(root_folder, "params.json")

            # ----- compute one item -----
            print(f"processing : {idx}")
            
            ds_out = self.upsamplingDataset[i]
            if len(ds_out) == 5:
                raw_images, upsampled_images, images_per_patches, timestamps, optical_flows = ds_out
                outputs = raw_images
            else:
                raw_images, upsampled_images, images_per_patches, timestamps, optical_flows, outputs = ds_out

            params = {
                        "raw_images": "raw_images.npz",
                        "fps": self.upsamplingDataset.videoDataset.fps,
                        "optical_flow_path": "optical_flow.npz",
                    }
            
            if upsample_to_events:
                        params["new_images_per_pair_OF"] = images_per_patches.tolist()
                        params["timestamps_OF"] = timestamps.tolist()
            else:
                        params["new_images_per_pair_FF"] = images_per_patches.tolist()
                        params["timestamps_FF"] = timestamps.tolist()

            # merge any existing params
            if os.path.exists(params_path):
                existing = json.load(open(params_path))
                for k, v in existing.items():
                    if k not in params:
                        params[k] = v
            
            with open(params_path, "w") as f:
                json.dump(params, f, indent=4)

            # write outputs asap
            self._write_mp4(upsampled_images, mp4_path, fps=params["fps"])
            del upsampled_images

            np.savez_compressed(os.path.join(root_folder, params["raw_images"]), raw_images=raw_images)
            del raw_images
            
            if outputs is not None:
                np.savez_compressed(os.path.join(root_folder, "outputs.npz"), outputs=outputs)
                del outputs

            of_path = os.path.join(root_folder, params["optical_flow_path"])
            if self.save_optical_flow and optical_flows is not None and hasattr(optical_flows, "shape") and optical_flows.shape[0] != 0:
                if not os.path.exists(of_path):
                    np.savez_compressed(of_path, optical_flow=optical_flows)
            
            del optical_flows, images_per_patches, timestamps

            # mark success and persist progress
            progress["last_completed"] = i
            json.dump(progress, open(progress_path, "w"), indent=2)

            # light cleanup
            try: torch.cuda.empty_cache()
            except Exception: pass

            # periodic heavy cleanup
            if ((i - start_index + 1) % CLEAR_EVERY) == 0:
                gc.collect()
                try: tf.keras.backend.clear_session()
                except Exception: pass

        print("[info] Cacheing pass finished. last_completed =", progress.get("last_completed"), 
            "failed =", progress.get("failed", []))


if __name__ == "__main__":

    idx = 0
    debug = False
    video_ds = VideoDatasetFromPictures(os.path.join("src", "neural_network", "data", "ecoco_depthmaps_test"),
                                          split="train",
                                          global_fps=27,
                                          suffix="frames")

    upsampling_ds = VideoDatasetUpsampling(video_ds,
                                           model_path=os.path.join("models", "film_net", "Style", "saved_model"),
                                           upsample_to_events=True,
                                           debug = debug)

    original_images = video_ds[idx]
    x = upsampling_ds[idx]
    images, images_per_patches, timestamps, optical_flows = x
    os.makedirs(os.path.join("src", "prototyping"), exist_ok=True)
    np.savez_compressed(os.path.join("src", "prototyping", "opticalflow.npz"), optical_flows=optical_flows)

    raw_images = []

    prev = 0
    for delta in images_per_patches:
        raw_images.append(images[prev])
        prev = prev + delta

    raw_images.append(images[-1])

    print("shape of original_images ",len(original_images['images']))
    print(f"shape of images : {images.shape}")
    print(f"shape of timestamps : {timestamps.shape}")
    print(f"images_per_patches : {images_per_patches}")
    print(f"shape of images_per_patches : {images_per_patches.shape}")
    print("sum of images_per_patches : ",np.sum(images_per_patches))
    print("Shape of Optical Flow : ", optical_flows.shape)

    write_sequence_to_video(images, os.path.join("src", "prototyping","videoDatasetUpsamplingHighFPSOnly.mp4"))
    write_sequence_to_video(raw_images, os.path.join("src", "prototyping","videoDatasetUpsamplingOriginal.mp4"))

    write_sequence_and_original_to_video(upsampled=images, timestamps=timestamps,
                                         original=original_images["images"], original_fps=video_ds.fps,
                                         filename="videoDatasetUpsamplingHighFPSStiched.mp4")

    VideoDatasetUpsamplingCacheToFolder(upsamplingDataset=upsampling_ds,
                                        output_path=os.path.join("src", "neural_network", "data", "upsampled_train"))

    video_ds = VideoDatasetFromPictures(os.path.join("src", "neural_network", "data", "ecoco_depthmaps_test"),
                                          split="validation",
                                          global_fps=27,
                                          suffix="frames")

    upsampling_ds = VideoDatasetUpsampling(video_ds,
                                           model_path=os.path.join("models", "film_net", "Style", "saved_model"),
                                           upsample_to_events=True,
                                           debug = debug)

    VideoDatasetUpsamplingCacheToFolder(upsamplingDataset=upsampling_ds,
                                        output_path=os.path.join("src", "neural_network", "data", "upsampled_validation"))
