from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


def decode_image(source: str | Path | bytes | bytearray | np.ndarray) -> np.ndarray:
    if isinstance(source, np.ndarray):
        image = source
    elif isinstance(source, (bytes, bytearray)):
        image = cv2.imdecode(np.frombuffer(source, dtype=np.uint8), cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)

    if image is None or image.size == 0:
        raise ValueError(f"Unable to decode image: {source!r}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected a three-channel BGR image.")
    return image


def iter_video_frames(
    video_path: str | Path,
    sample_every: int = 15,
    max_frames: int = 30,
) -> Iterator[tuple[int, float, np.ndarray]]:
    if sample_every < 1:
        raise ValueError("sample_every must be at least 1.")
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1.")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Unable to open video: {video_path}")

    frame_index = 0
    sampled = 0
    try:
        while sampled < max_frames:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % sample_every == 0:
                timestamp_seconds = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                yield frame_index, timestamp_seconds, frame
                sampled += 1
            frame_index += 1
    finally:
        capture.release()