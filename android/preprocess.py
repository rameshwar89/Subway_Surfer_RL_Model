from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    width: int = 160
    height: int = 120
    crop_top: int = 0
    crop_bottom: int = 0
    grayscale: bool = True
    normalize: bool = True
    stack_size: int = 4


def crop_frame(frame: np.ndarray, crop_top: int = 0, crop_bottom: int = 0) -> np.ndarray:
    bottom = frame.shape[0] - crop_bottom if crop_bottom else frame.shape[0]
    return frame[crop_top:bottom, :]


def preprocess_frame(frame: np.ndarray, config: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    processed = crop_frame(frame, config.crop_top, config.crop_bottom)
    processed = cv2.resize(processed, (config.width, config.height), interpolation=cv2.INTER_AREA)
    if config.grayscale:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    if config.normalize:
        processed = processed.astype(np.float32) / 255.0
    return processed


class FrameStacker:
    def __init__(self, stack_size: int = 4) -> None:
        if stack_size < 1:
            raise ValueError("stack_size must be at least 1.")
        self._frames: deque[np.ndarray] = deque(maxlen=stack_size)

    def reset(self, frame: np.ndarray) -> np.ndarray:
        self._frames.clear()
        for _ in range(self._frames.maxlen or 1):
            self._frames.append(frame)
        return self.stack()

    def push(self, frame: np.ndarray) -> np.ndarray:
        self._frames.append(frame)
        if len(self._frames) < (self._frames.maxlen or 1):
            return self.reset(frame)
        return self.stack()

    def stack(self) -> np.ndarray:
        return np.stack(list(self._frames), axis=0)


def preprocess_batch(frames: Iterable[np.ndarray], config: PreprocessConfig = PreprocessConfig()) -> np.ndarray:
    return np.stack([preprocess_frame(frame, config) for frame in frames], axis=0)

