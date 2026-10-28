import imageio
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import json
import esim_torch
import matplotlib
from src.neural_network.utils.event_utils import *
import hashlib
from src.neural_network.dataloaders.utils import _mk_cache_paths_events, _memmap_load, _save_cached_npy

class EventDatasetFromUpsampledSaliency(Dataset):


    def __init__(self, directory,
                       preprocessing_params,
                       prediction_video_name, 
                       log_scale=True,
                       mode = "event",
                       B = 5,
                       contrast_threshold_neg = 0.2,
                       contrast_threshold_pos = 0.2,
                       force_recomputing=False,
                       device="cuda",
                       verbose=False,
                       save_xypt = False,
                       limit=-1,
                       grids_dtype=np.float16,
                       input_video="optical_flow"):

        self.preprocessing_params = preprocessing_params

        # "optical_flow": video_upsampled_using_optical_flow.mp4 + timestamps_OF (all paper DVS runs)
        # "fixed_200fps": video_upsampled_fps_eq_200.mp4 + timestamps_FF, the video the retina reads
        if input_video not in ("optical_flow", "fixed_200fps"):
            raise ValueError(f"input_video must be 'optical_flow' or 'fixed_200fps', got {input_video!r}")
        self.input_video = input_video

        self.directory = directory
        self.mode = mode
        self.B = preprocessing_params.get("num_bins", 5)
        self.log_scale = log_scale
        self.force_recomputing = force_recomputing
        self.device = device
        self.grids_dtype = grids_dtype
        self.verbose = verbose
        self.prediction_video_name = prediction_video_name
        self.save_xypt = save_xypt
        self._recomputed = set()  # folders regenerated in this process (force_recomputing applies once per clip)
        
        self.contrast_threshold_neg = preprocessing_params.get("contrast_threshold_neg", contrast_threshold_neg)
        self.contrast_threshold_pos = preprocessing_params.get("contrast_threshold_pos", contrast_threshold_pos)

        # load glboal metadata once
        with open(os.path.join(directory, "metadata.json"), "r") as f:
            self.read_metadata = json.load(f)

        # enumerate samples
        self.data_files = sorted([os.path.join(directory, el) for el in os.listdir(directory) if os.path.isdir(os.path.join(directory, el))])

        if limit is not None and limit > 0:
            self.data_files = self.data_files[:limit]

        # Pre-read per-sample params once (avoid JSON I/O in __getitem__)
        self.properties = []
        for folder in self.data_files:
            with open(os.path.join(folder, "params.json"), "r") as f:
                self.properties.append(json.load(f))

        # ESIM
        self.event_simulator = esim_torch.ESIM(
            contrast_threshold_neg=self.contrast_threshold_neg,
            contrast_threshold_pos=self.contrast_threshold_pos,
            refractory_period_ns=0,
        )

        # Cache filename builder bound with thresholds
        self._mk_paths = lambda folder: _mk_cache_paths_events(
            folder, self.B, self.contrast_threshold_neg, self.contrast_threshold_pos,
            version="v2" if input_video == "optical_flow" else "v2_ff200"
        )

    def __len__(self):
        return len(self.data_files)


    @staticmethod
    def _load_video_gray(path, fit_to_size=None):
        """
        Load an MP4 (or image stack) as grayscale uint8.
        If fit_to_size=(W,H) is given, resize each frame to exactly (H, W).

        Returns:
            np.ndarray with shape:
                - (N, H, W) uint8
        """
        import imageio.v3 as iio
        import numpy as np
        from PIL import Image

        vid = iio.imread(path)  # (N,H,W,C) or (N,H,W); dtype usually uint8

        # If RGB(A), convert to grayscale using luminance weights
        if vid.ndim == 4:
            # Keep only RGB if RGBA
            if vid.shape[-1] == 4:
                vid = vid[..., :3]
            # Convert to float for weighted sum, then back to uint8
            r = vid[..., 0].astype(np.float32)
            g = vid[..., 1].astype(np.float32)
            b = vid[..., 2].astype(np.float32)
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            vid = np.clip(gray, 0, 255).astype(np.uint8)
        elif vid.ndim == 3:
            # Already grayscale (N,H,W)
            if vid.dtype != np.uint8:
                # If it came in as float [0,1], scale up; otherwise just clip
                if np.issubdtype(vid.dtype, np.floating):
                    vid = np.clip(vid * 255.0, 0, 255).astype(np.uint8)
                else:
                    vid = np.clip(vid, 0, 255).astype(np.uint8)
        else:
            raise ValueError(f"Unexpected video array shape {vid.shape}")

        # Resize if requested
        if fit_to_size is None:
            return vid

        W, H = fit_to_size
        if W <= 0 or H <= 0:
            raise ValueError(f"fit_to_size must be positive (got {fit_to_size})")

        N = vid.shape[0]
        out = np.empty((N, H, W), dtype=np.uint8)

        # Fast path if already the right size
        if vid.shape[1] == H and vid.shape[2] == W:
            return vid

        for i in range(N):
            # PIL expects (W,H)
            im = Image.fromarray(vid[i], mode="L")
            im = im.resize((W, H), resample=Image.BILINEAR)
            out[i] = np.asarray(im, dtype=np.uint8)

        return out



    def _read_events_or_create(self, folder, properties):

        paths = self._mk_paths(folder)
        
        if self.save_xypt:
            have_xypt_cache = os.path.exists(paths['x']) and os.path.exists(paths['y']) and os.path.exists(paths['t']) and os.path.exists(paths['p'])
        else:
            have_xypt_cache = True


        have_cache = (os.path.exists(paths['grids'])
                      and os.path.exists(paths['outputs']) and have_xypt_cache)

        if have_cache and (not self.force_recomputing or folder in self._recomputed):
            # fast path: mmapped reads

            # (T, B, H, W), dtype=np.float16
            grids_np = _memmap_load(paths['grids'])
            # zero-copy views into tensors on CPU: later moved to CUDA
            event_grids = torch.as_tensor(grids_np)

            # (T+1, H, W) uint8
            frames_np = _memmap_load(paths['frames'])

            # read outputs 
            outputs_np = _memmap_load(paths['outputs'])

            if self.save_xypt:
                x_np = _memmap_load(paths["x"])  # uint16
                y_np = _memmap_load(paths["y"])  # uint16
                t_np = _memmap_load(paths["t"])  # int64
                p_np = _memmap_load(paths["p"])  # int8

                events = {
                    "x": torch.as_tensor(x_np),
                    "y": torch.as_tensor(y_np),
                    "t": torch.as_tensor(t_np),
                    "p": torch.as_tensor(p_np),
                }
            else:
                events = {"x" : None, 
                        "y" : None,
                        "t" : None,
                        "p" : None}


            result = {
                "frames": np.array(frames_np['raw_images']).astype(np.float32) / 255.,  # keep as numpy (cheap, contiguous)
                "outputs" : np.array(outputs_np),
                "events": events,
                "grids": event_grids,  # half precision on load
            }

            # optional: optical flow (unchanged from your logic)
            optical_flow_file = os.path.join(folder, "optical_flow.npz")
            if os.path.exists(optical_flow_file):
                optical_flow = np.load(optical_flow_file)["optical_flow"]
                
                if optical_flow.shape[0] != 0:
                    result["optical_flows"] = optical_flow.transpose((0, 3, 1, 2))

            return result


        # slow path ( first time read and data cacheing)
        if self.input_video == "optical_flow":
            frames = self._load_video_gray(os.path.join(folder, "video_upsampled_using_optical_flow.mp4"))  # (N, H, W), uint8
            timestamps = np.asarray(properties["timestamps_OF"], dtype=np.float64)
            images = properties["new_images_per_pair_OF"]  # list of counts per segment
        else:
            frames = self._load_video_gray(os.path.join(folder, "video_upsampled_fps_eq_200.mp4"))
            timestamps = np.asarray(properties["timestamps_FF"], dtype=np.float64)
            images = properties["new_images_per_pair_FF"]  # one segment per source frame, the last one extrapolated
        if len(frames) != len(timestamps):
            raise ValueError(f"{folder}: {len(frames)} frames but {len(timestamps)} timestamps ({self.input_video})")
        timestamps_ns = (timestamps * 1e9).astype("int64")

        # read the output frames and resize to the frame size
        if self.prediction_video_name:
            pred_video_path = os.path.join(folder, self.prediction_video_name)
            if os.path.exists(pred_video_path):
                pred_frames = self._load_video_gray(pred_video_path,
                                                   fit_to_size=(frames.shape[2], frames.shape[1]))  # (N,H,W) uint8
                outputs = pred_frames
        else:
            outputs = frames.copy()
    
        # Log intensities on GPU
        
        if self.log_scale:
            input_images = np.log(frames.astype("float32") / 255.0 + 1e-4)
        else:
            input_images = frames.astype("float32") / 255.0
            
        input_images = torch.from_numpy(input_images).to(self.device, non_blocking=True)
        timestamps_ns_t = torch.from_numpy(timestamps_ns).to(self.device, non_blocking=True)

        # reset + simulate events
        
        with torch.no_grad():
            self.event_simulator.reset()
            events = self.event_simulator.forward(input_images, timestamps_ns_t)

        x = events["x"]
        y = events["y"]
        t = events["t"]
        p = events["p"]

        # images: list of counts per segment (as in your original code)
        imgs_np = np.asarray(images, dtype=np.int64)

        # cumulative frame counts; cum_full has length len(images)
        cum_full = np.cumsum(imgs_np, dtype=np.int64)  # [im0, im0+im1, ..., sum(images)]
        # boundaries used by your original loop: [0, im0, im0+im1, ..., im0+...+im_{K-2}]
        starts_ix = np.concatenate(([0], cum_full[:-1]))    # length K
        ends_ix = cum_full[:-1]

        starts_ix_t = torch.from_numpy(starts_ix).to(timestamps_ns_t.device)
        t_starts = timestamps_ns_t.index_select(0, starts_ix_t).contiguous()
        
        # End times: first K-1 from timestamps at ends_ix, last one = last timestamp + 1 ns
        if len(ends_ix) > 0:
            ends_ix_t = torch.from_numpy(ends_ix).to(timestamps_ns_t.device)
            t_ends_first = timestamps_ns_t.index_select(0, ends_ix_t).contiguous()   # (K-1,)
            t_last = timestamps_ns_t[-1] + 1                                         # make half-open [t0, t1)
            t_ends = torch.cat([t_ends_first, t_last.view(1)], dim=0)                # (K,)
        else:
            # Edge case: only one segment -> starts=[0], ends=[last+1]
            t_ends = (timestamps_ns_t[-1] + 1).view(1)        


        N = timestamps_ns_t.numel()

        # Now cut events by these bounds
        lefts  = torch.searchsorted(t, t_starts, right=False)        # (K,)
        rights = torch.searchsorted(t, t_ends,   right=False)        # (K,)


        grids = []
        # load the raw_frames
        raw_frames = np.load(os.path.join(folder, "raw_images.npz"))['raw_images'].astype(np.float32) / 255.0

        assert cum_full[-1] == N, f"sum(images_per_pair_OF)={cum_full[-1]} != len(timestamps)={N}"

        for k in range(len(imgs_np)):
            s = int(lefts[k].item())
            e = int(rights[k].item())

            t0 = int(t_starts[k].item())
            t1 = int(t_ends[k].item())
            if e <= s or t1 <= t0:
                grids.append(torch.zeros((self.B, frames.shape[1], frames.shape[2]), device=t.device, dtype=torch.float32))
                continue

            xk, yk, tk, pk = x[s:e], y[s:e], t[s:e], p[s:e]

            # (optional) hard safety: clamp x/y once if your ESIM can emit border values
            # H, W = frames.shape[-2], frames.shape[-1]
            # xk = xk.clamp_(0, W-1); yk = yk.clamp_(0, H-1)

            g = events_to_voxels(xk, yk, tk, pk, t0, t1, H = frames.shape[1], W=frames.shape[2], B=self.B)
            grids.append(g)

        event_grids = torch.stack(grids, dim=0)
        if self.input_video == "fixed_200fps":
            # drop the segment after the last source frame, so there is one grid per pair of
            # source frames, as for the optical-flow video and the retina loader
            event_grids = event_grids[:-1]

        # normalize the outputs 
        outputs = outputs.astype(np.float32) / 255.0

        # write caches
        # cast & move to CPU arrays only once, right before saving
        _save_cached_npy(paths["grids"], event_grids.detach().to("cpu").to(torch.float16).numpy())
        _save_cached_npy(paths["outputs"], outputs)

        # Compact events to smaller dtypes on disk
        
        if self.save_xypt:
            _save_cached_npy(paths["x"], x.detach().to("cpu").to(torch.int32).numpy().astype(np.uint16))
            _save_cached_npy(paths["y"], y.detach().to("cpu").to(torch.int32).numpy().astype(np.uint16))
            _save_cached_npy(paths["t"], t.detach().to("cpu").to(torch.int64).numpy())
            _save_cached_npy(paths["p"], p.detach().to("cpu").to(torch.int8).numpy())

        del events, input_images, timestamps_ns_t
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            
        if self.save_xypt:
            events = {
                "x": x.to("cpu"),
                "y": y.to("cpu"),
                "t": t.to("cpu"),
                "p": p.to("cpu"),
            }
        else:
            events = {"x" : None, 
                      "y" : None,
                      "t" : None,
                      "p" : None}


        self._recomputed.add(folder)

        # Return in the same shapes/types you expect
        result = {
            "frames": raw_frames,                                 # np.uint8
            "outputs" : outputs,
            "events": events,
            "grids": event_grids.to("cpu"),                       # torch.float32 -> we saved float16
        }

        optical_flow_file = os.path.join(folder, "optical_flow.npz")
        if os.path.exists(optical_flow_file):
            optical_flow = np.load(optical_flow_file)["optical_flow"]
            # optical flow empty 
            if optical_flow.shape[0] != 0:
                result["optical_flows"] = optical_flow.transpose((0, 3, 1, 2))

        return result

    def __getitem__(self, idx):
        folder = self.data_files[idx]
        props = self.properties[idx]
        return self._read_events_or_create(folder, props)


if __name__ == "__main__":
    import tqdm

    ds = EventDatasetFromUpsampledSaliency(directory=os.path.join("src", "neural_network", "data", "saliency_dataset_upsampled"),
                                           prediction_video_name="maps.mp4",
                                           preprocessing_params = {"num_bins":5},
                                            mode="event",
                                            B=5,
                                            force_recomputing=True)

    # visualize and save a couple of datapoints
    no_datapoints = 20
    
    for i in tqdm.tqdm(range(0, no_datapoints)):
        datapoint = ds[i]

        imgs, events, grids = datapoint["frames"], datapoint["events"], datapoint["grids"]
        
        optical_flow_on = False
        if "optical_flows" in datapoint:
            optical_flow = datapoint["optical_flows"]
            optical_flow_on = True


        x,y, t, p = events['x'], events['y'], events['t'], events['p']

        # visualize a datapoint
        render_events_video(datapoint, output_path=os.path.join("visualizations", "events_dhf1k", f"sample_{i}.mp4"),
                            flows_on=optical_flow_on,
                            invert_rgb=True,
                            show_outputs=True,
                            overlay_outputs = True)
