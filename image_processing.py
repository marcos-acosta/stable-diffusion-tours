from pathlib import Path
from PIL import Image
from tqdm import tqdm

import argparse
import cv2
import numpy as np

STITCH = "stitch"
SPLIT = "split"
MODES = [STITCH, SPLIT]
SD_WIDTH = 512
SD_HEIGHT = 512


def match_sd_dimensions(frame: np.ndarray):
    shape = frame.shape
    height, width, _ = shape
    if height > width:
        height_start = (height - width) // 2
        height_end = height_start + width
        cropped = frame[height_start:height_end,:,:]
    else:
        width_start = (width - height) // 2
        width_end = width_start + height
        cropped = frame[:, width_start:width_end, :]
    return cv2.resize(cropped, (SD_WIDTH, SD_HEIGHT))


def split_frames(video_path: str, output_dir: str, target_fps: int = 15):
    video_path = Path(video_path)
    video_name = video_path.stem
    output_dir = Path(output_dir) / video_name
    output_dir.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(str(video_path))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_sample_rate = int(video_fps / target_fps)
    i = -1
    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))) as progress_bar:
        while cap.isOpened():
            i += 1
            if i % frame_sample_rate != 0:
                _ = cap.grab()
                progress_bar.update(1)
            else:
                ret, frame = cap.read()
                if not ret:
                    break
                match_sd_dimensionsped_frame = match_sd_dimensions(frame)
                Image.fromarray(match_sd_dimensionsped_frame).save(str(output_dir / f"frame_{i}.png"))
                progress_bar.update(1)


def stitch_frames_to_video(frame_dir: str, output_dir: str, fps: int = 15, original_dir: str = None):
    frame_dir = Path(frame_dir)
    output_path = Path(output_dir) / f"{frame_dir.name}.mp4"
    frame_paths = sorted(list(frame_dir.iterdir()),
                    key=lambda filename: int(filename.stem.split("_")[1]))
    height, width = cv2.imread(str(frame_paths[0])).shape[:2]
    dimensions = (width, height)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"avc1"), fps, dimensions)
    for frame_path in tqdm(frame_paths):
        frame = cv2.imread(str(frame_path))
        writer.write(frame)
    writer.release()
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help=f"Must be one of [{', '.join(MODES)}]; determines the action to run")
    parser.add_argument("--input_path", type=str, help="Path to the video to be split into frames (for SPLIT)")
    parser.add_argument("--output_dir", type=str, help="Directory to write result to")
    parser.add_argument("--input_dir", type=str, help="Path to the input frames to be stitched together (for STITCH)")
    parser.add_argument("--output_path", type=str, help="Filename to save output video as (for STITCH)")
    parser.add_argument("--fps", "-f", type=int, default=15, help="Target FPS of the output video")
    parser.add_argument("--side_by_side", "-s", action="store_true", help="If set, show side-by-side input and output frames")
    args = parser.parse_args()

    if args.mode == SPLIT:
        split_frames(args.input_path, args.output_dir, target_fps=args.fps)
    elif args.mode == STITCH:
        stitch_frames_to_video(args.input_dir, args.output_dir, fps=args.fps)
    else:
        raise ValueError(f"Mode must be one of [{','.join(MODES)}]")


if __name__ == "__main__":
    main()