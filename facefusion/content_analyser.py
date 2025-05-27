from functools import lru_cache

import cv2
import numpy
from tqdm import tqdm

from facefusion import inference_manager, state_manager, wording
from facefusion.filesystem import resolve_relative_path
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.typing import Fps, VisionFrame
from facefusion.vision import count_video_frame_total, detect_video_fps, get_video_frame, read_image

MODEL_SET = {}  # NSFW model set removed
PROBABILITY_LIMIT = 0.80
RATE_LIMIT = 10
STREAM_COUNTER = 0


def get_inference_pool():
    # NSFW model pool disabled
    return None


def clear_inference_pool() -> None:
    inference_manager.clear_inference_pool(__name__)


def get_model_options():
    return None  # NSFW model options disabled


def pre_check() -> bool:
    # No need to check/download NSFW models
    return True


def analyse_stream(vision_frame: VisionFrame, video_fps: Fps) -> bool:
    global STREAM_COUNTER

    STREAM_COUNTER += 1
    if STREAM_COUNTER % int(video_fps) == 0:
        return analyse_frame(vision_frame)
    return False


def analyse_frame(vision_frame: VisionFrame) -> bool:
    vision_frame = prepare_frame(vision_frame)
    return False  # Always returning False as NSFW detection is turned off


def forward(vision_frame: VisionFrame) -> float:
    # Bypass the model's inference for NSFW detection
    return 0.0  # Always return low probability


def prepare_frame(vision_frame: VisionFrame) -> VisionFrame:
    # Keeping frame preparation logic in case it's needed for non-NSFW purposes
    vision_frame = cv2.resize(vision_frame, (224, 224)).astype(numpy.float32)
    vision_frame -= numpy.array([104, 117, 123]).astype(numpy.float32)
    vision_frame = numpy.expand_dims(vision_frame, axis=0)
    return vision_frame


@lru_cache(maxsize=None)
def analyse_image(image_path: str) -> bool:
    frame = read_image(image_path)
    return analyse_frame(frame)


@lru_cache(maxsize=None)
def analyse_video(video_path: str, start_frame: int, end_frame: int) -> bool:
    video_frame_total = count_video_frame_total(video_path)
    video_fps = detect_video_fps(video_path)
    frame_range = range(start_frame or 0, end_frame or video_frame_total)
    rate = 0.0
    counter = 0

    with tqdm(total=len(frame_range), desc=wording.get('analysing'), unit='frame', ascii=' =', disable=state_manager.get_item('log_level') in ['warn', 'error']) as progress:
        for frame_number in frame_range:
            if frame_number % int(video_fps) == 0:
                frame = get_video_frame(video_path, frame_number)
                if analyse_frame(frame):
                    counter += 1
            rate = counter * int(video_fps) / len(frame_range) * 100
            progress.update()
            progress.set_postfix(rate=rate)
    return rate > RATE_LIMIT
