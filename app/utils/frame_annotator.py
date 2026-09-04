"""
Frame annotator untuk menggambar hasil deteksi YOLO ke frame
Memisahkan visualisasi dari logic deteksi (Single Responsibility)
"""
import logging
from typing import List, Tuple, Dict

import cv2
import numpy as np

from app.services.core_ml.detection.detector import Detection

logger = logging.getLogger(__name__)

# BGR colors per class
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "person": (255, 0, 255),      # Magenta
    "motorcycle": (0, 255, 255),  # Kuning
    "car": (0, 255, 0),           # Hijau
    "bus": (255, 0, 0),           # Biru
    "truck": (0, 165, 255),       # Oranye
}
DEFAULT_COLOR = (255, 255, 255)

# Font & line config
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
BOX_THICKNESS = 2
TEXT_PADDING = 4


def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    show_confidence: bool = True,
    show_track_id: bool = True
) -> np.ndarray:
    """
    Gambar bounding box + label hasil deteksi ke frame

    Args:
        frame: OpenCV frame (numpy array)
        detections: List of Detection objects hasil YOLO
        show_confidence: Tampilkan confidence di label
        show_track_id: Tampilkan track ID di label (jika ada)

    Returns:
        Frame baru dengan bounding box (frame asli tidak diubah)
    """
    annotated = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Build label text
        label_parts = [det.class_name]
        if show_confidence:
            label_parts.append(f"{det.confidence:.2f}")
        if show_track_id and det.track_id is not None:
            label_parts.append(f"#{det.track_id}")
        label = " ".join(label_parts)

        _draw_label(annotated, label, x1, y1, y2, color)

    return annotated


def _draw_label(
    frame: np.ndarray,
    label: str,
    x1: int,
    y1: int,
    y2: int,
    color: Tuple[int, int, int]
) -> None:
    """Gambar label dengan background di atas bounding box"""
    (text_w, text_h), baseline = cv2.getTextSize(
        label, FONT, FONT_SCALE, FONT_THICKNESS
    )

    # Label position: above the box, fallback di dalam box jika keluar frame
    label_y1 = y1 - text_h - TEXT_PADDING * 2
    if label_y1 < 0:
        label_y1 = y2

    # Background rectangle
    bg_top_left = (x1, label_y1)
    bg_bottom_right = (x1 + text_w + TEXT_PADDING, label_y1 + text_h + TEXT_PADDING)
    cv2.rectangle(frame, bg_top_left, bg_bottom_right, color, cv2.FILLED)

    # Text (warna hitam/putih kontras dengan background)
    text_pos = (x1 + TEXT_PADDING // 2, label_y1 + text_h + TEXT_PADDING // 2)
    cv2.putText(
        frame, label, text_pos, FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA
    )
