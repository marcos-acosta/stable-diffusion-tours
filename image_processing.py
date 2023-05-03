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
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
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
                frame = match_sd_dimensions(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                Image.fromarray(frame).save(str(output_dir / f"frame_{i}.png"))
                progress_bar.update(1)


def stitch_frames_to_video(frame_dir: str, output_path: str, fps: int = 15, original_dir: str = None):
    frame_dir = Path(frame_dir)
    frame_paths = sorted(list(frame_dir.iterdir()),
                    key=lambda filename: int(filename.stem.split("_")[1]))
    height, width = cv2.imread(str(frame_paths[0])).shape[:2]
    if original_dir:
        width *= 2
    dimensions = (width, height)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"avc1"), fps, dimensions)
    for frame_path in tqdm(frame_paths):
        frame = cv2.imread(str(frame_path))
        if original_dir:
            original_frame = cv2.imread(str(Path(original_dir) / frame_path.name))
            frame = np.concatenate((original_frame, frame), axis=1)
        writer.write(frame)
    writer.release()
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=str, help=f"Must be one of [{', '.join(MODES)}]; determines the action to run")
    parser.add_argument("input", type=str, help="Input directory / path")
    parser.add_argument("output", type=str, help="Output directory / path")
    parser.add_argument("--fps", "-f", type=int, default=15, help="Target FPS of the output video")
    parser.add_argument("--original", type=str, help="If set during stitching, will show original frames next to diffusioned ones")
    args = parser.parse_args()

    if args.mode == SPLIT:
        split_frames(args.input, args.output, target_fps=args.fps)
    elif args.mode == STITCH:
        stitch_frames_to_video(args.input,
                               args.output,
                               fps=args.fps,
                               original_dir=args.original)
    else:
        raise ValueError(f"Mode must be one of [{','.join(MODES)}]")


if __name__ == "__main__":
    main()