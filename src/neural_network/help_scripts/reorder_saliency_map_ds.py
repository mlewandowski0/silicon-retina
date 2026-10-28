import os
import argparse
import imageio as iio
import shutil
from tqdm import tqdm
import cv2
import json


def reorder_clip(video_path, annotation_path, out_folder):
    """Copy one DHF1K clip into out_folder as input.AVI + fixation.mp4 + maps.mp4 + params.json.

    video_path:      DHF1K video/NNN.AVI (copied byte for byte)
    annotation_path: DHF1K annotation/NNNN (fixation/*.png and maps/*.png are encoded to MP4)
    """
    os.makedirs(out_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


    fixation_path = os.path.join(annotation_path, "fixation")
    maps_path = os.path.join(annotation_path, "maps")

    shutil.copy(video_path, os.path.join(out_folder, f"input.AVI"))

    fixation_frames = 0
    with iio.get_writer(os.path.join(out_folder, "fixation.mp4"), fps=30,macro_block_size=1) as writer:
        fixation_list = sorted([os.path.join(fixation_path, x) for x in os.listdir(fixation_path) if x.endswith(".png") or x.endswith(".jpg")])
        for fixation in fixation_list:
            img = iio.imread(fixation)
            writer.append_data(img)
            fixation_frames += 1

    maps_frames = 0
    with iio.get_writer(os.path.join(out_folder, "maps.mp4"), fps=30, macro_block_size=1) as writer:
        fixation_list = sorted([os.path.join(maps_path, x) for x in os.listdir(maps_path) if x.endswith(".png") or x.endswith(".jpg")])
        for fixation in fixation_list:
            img = iio.imread(fixation)
            writer.append_data(img)
            maps_frames += 1


    params = {"fps" : fps, "input_frame_count": frame_count, "fixation_frame_count": fixation_frames, "maps_frame_count": maps_frames}
    with open(os.path.join(out_folder, "params.json"), "w") as f:
        json.dump(params, f)

    if frame_count != fixation_frames or frame_count != maps_frames:
        print(f"Warning: Frame count mismatch in video {os.path.basename(video_path)}: video frames = {frame_count}, fixation frames = {fixation_frames}, maps frames = {maps_frames}")
    return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to the folder containing the saliency map dataset to be reordered")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the folder where the reordered dataset will be saved")
    # the original download unpacked to video-002/video and annotation-003/annotation
    parser.add_argument("--video_subdir", type=str, default=os.path.join("video-002", "video"))
    parser.add_argument("--annotation_subdir", type=str, default=os.path.join("annotation-003", "annotation"))

    args = parser.parse_args()

    max_val = 701
    videos_path, annotations_path = os.path.join(args.input_path, args.video_subdir), os.path.join(args.input_path, args.annotation_subdir)

    os.makedirs(args.output_path, exist_ok=True)

    for i in tqdm(range(1, max_val)):
        video_name = str(i).zfill(3)
        folder_name = str(i).zfill(4)

        reorder_clip(os.path.join(videos_path, f"{video_name}.AVI"),
                     os.path.join(annotations_path, folder_name),
                     os.path.join(args.output_path, folder_name))
