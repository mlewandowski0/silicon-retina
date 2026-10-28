# Original records (extracts)

Short extracts of lab logs that show how the DHF1K selection and processed IDs came about. The
full logs are not published.

| File | Extracted from | Shows |
|---|---|---|
| `upsample_200fps_index_map.txt` | `upsample_200fps.txt` (fixed 200 FPS pass, 2025-10-24, 541 clips) | "Processing index : i  path : …/NNNN": the DHF1K clip behind each `video_{i:05d}` (535 of 541 lines; the rest were interleaved with progress bars) |
| `upsample_of_pass1_starts.txt` | `upsample_log.txt` (first optical-flow pass, 2025-10-23, counts of consecutive lines) | Total 542 clips (start 16 + 526). The worker restarts 77 times at index 40, which is DHF1K clip 058 in the 542-clip list |
| `upsample_of_pass2_starts.txt` | `upsampling_adaptive.txt` (optical-flow pass, 2025-10-24) | After 058 was removed: 541 clips (start 11 + 530), finishing with "All done" |

Other evidence used in `../audit.md`:
- The `copy_short_videos.py --max-frames 850` and `--max-frames 750` commands come from the author's shell history.
- `split.json` comes from `saliency_dataset_ready/`.
- The rest comes from W&B run metadata and git history.
