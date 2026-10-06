"""Evidence builder.

Produces, per clip:
- a DENSE, TIME-BADGED keyframe strip (the proven workhorse, strengthened):
  N motion-stratified natural frames, each carrying an ordinal badge and a
  temporal progress bar. Badges are metadata about *when*, never assertions
  about *what* — no extracted physics is drawn as content.
- POINTER-ONLY Set-of-Mark image: numbered candidate regions on the peak-
  interaction frame. Marks are hypotheses ("consider these regions"), not
  assertions; a wrong mark is an unused option. FastSAM when available,
  classical contour proposals otherwise.
- necessity metadata with verifiable answers for evidence-reading QA:
  pixel-arithmetic change localization (assumption-free) and a validity-gated
  nearest-mark-to-hand answer (only when hand track coverage passes the gate).

Hand tracking is used ONLY for frame selection and gated QA metadata — it is
never rendered. Contact/trajectory assertions are deliberately absent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .evidence_config import artifact_path, load_config, repo_path
from .evidence_utils import read_jsonl, write_json


# ---------------------------------------------------------------- video io ---

def read_video(path: Path, max_frames: int) -> tuple[list[np.ndarray], np.ndarray, float, int]:
    import cv2
    cap = cv2.VideoCapture(str(path))
    total = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    indices = np.unique(np.linspace(0, total - 1, min(max_frames, total)).round().astype(int))
    frames: list[np.ndarray] = []
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = cap.read()
        if ok:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, indices[:len(frames)], fps, total


def motion_energy(frames: list[np.ndarray]) -> np.ndarray:
    import cv2
    if not frames:
        return np.zeros(0, dtype=np.float32)
    gray = [cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (160, 90)) for frame in frames]
    scores = [0.0]
    for previous, current in zip(gray, gray[1:]):
        scores.append(float(np.mean(cv2.absdiff(previous, current))))
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) > 2:
        scores = np.convolve(scores, np.ones(3, dtype=np.float32) / 3, mode="same")
    return scores


def stratified_keyframes(scores: np.ndarray, count: int) -> list[int]:
    """One max-motion frame per temporal stratum -> temporal coverage guaranteed.
    Kept as an ablation baseline; the default is energy_keyframes."""
    if len(scores) == 0:
        return []
    count = min(count, len(scores))
    selected = []
    for phase in range(count):
        lo = int(round(phase * len(scores) / count))
        hi = max(lo + 1, int(round((phase + 1) * len(scores) / count)))
        selected.append(lo + int(np.argmax(scores[lo:hi])))
    return sorted(set(selected))


def energy_keyframes(scores: np.ndarray, count: int, floor: float = 0.15,
                     snap_window: int = 2) -> list[int]:
    """Energy-adaptive keyframe selection (motion-mass quantiles + transition snap).

    1. ALLOCATION: per-frame motion energy is treated as a probability mass and
       the frame budget is placed at equal quantiles of its cumulative curve —
       dense where the action happens, sparse during idle stretches. A uniform
       floor (`floor` of the total mass) preserves whole-clip coverage and makes
       near-static clips degrade gracefully to uniform sampling.
    2. SNAPPING: each pick moves to the strongest local energy CHANGE within
       +/- snap_window frames, so keyframes align with action-phase transitions
       (reach/grasp/insert/release boundaries) rather than arbitrary grid points.

    Deterministic pixel arithmetic only — no extracted physics is trusted.
    """
    n = len(scores)
    if n == 0:
        return []
    count = min(count, n)
    mass = np.maximum(scores.astype(np.float64), 0.0)
    total = float(mass.sum())
    uniform = np.full(n, 1.0 / n)
    weights = uniform if total <= 0 else (1.0 - floor) * mass / total + floor * uniform
    cumulative = np.cumsum(weights)
    cumulative /= cumulative[-1]
    targets = (np.arange(count) + 0.5) / count
    picks = np.clip(np.searchsorted(cumulative, targets), 0, n - 1)
    transition = np.abs(np.diff(scores, prepend=scores[:1]))
    snapped = []
    for pick in picks:
        lo, hi = max(0, int(pick) - snap_window), min(n, int(pick) + snap_window + 1)
        snapped.append(lo + int(np.argmax(transition[lo:hi])))
    selected = sorted(set(snapped))
    # snapping may merge neighbours; refill from the strongest unused transitions
    if len(selected) < count:
        for index in np.argsort(-transition):
            if int(index) not in selected:
                selected.append(int(index))
            if len(selected) == count:
                break
        selected = sorted(selected)
    return [int(i) for i in selected]


def maxmotion_keyframes(scores: np.ndarray, count: int) -> list[int]:
    """Global top-K motion frames, then sorted in time (no coverage guarantee)."""
    n = len(scores)
    if n == 0:
        return []
    count = min(count, n)
    chosen = np.argsort(-scores.astype(np.float64))[:count]
    return sorted(int(i) for i in chosen)


def ends_uniform_keyframes(scores: np.ndarray, count: int) -> list[int]:
    """First + last frame, remaining budget uniform in time."""
    n = len(scores)
    if n == 0:
        return []
    count = min(count, n)
    picks = np.linspace(0, n - 1, count).round().astype(int).tolist()
    picks[0], picks[-1] = 0, n - 1
    return sorted(set(picks))


def select_keyframes(scores: np.ndarray, count: int, method: str = "energy") -> list[int]:
    if method == "energy":
        return energy_keyframes(scores, count)
    if method == "stratified":
        return stratified_keyframes(scores, count)
    if method == "uniform":
        n = len(scores)
        return sorted(set(np.linspace(0, n - 1, min(count, n)).round().astype(int).tolist())) if n else []
    if method == "maxmotion":
        return maxmotion_keyframes(scores, count)
    if method in {"ends_uniform", "ends"}:
        return ends_uniform_keyframes(scores, count)
    raise ValueError(f"Unknown keyframe_method: {method}")


# -------------------------------------------------------------- hand track ---

def _detect_hands_tasks_api(frames: list[np.ndarray], cfg: dict[str, Any]) -> list[list[tuple[list[float], str]]] | None:
    """MediaPipe >=0.10.30 removed mp.solutions; use the tasks HandLandmarker.

    Returns per-frame lists of (normalized xyxy box, handedness label), or None
    when the tasks API / model file is unavailable.
    """
    model_path = repo_path(cfg, cfg["evidence"].get("hand_landmarker_model", "models/hand_landmarker.task"))
    if not model_path.exists():
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=float(cfg["evidence"]["min_detection_confidence"]),
        ))
    except Exception:
        return None
    per_frame: list[list[tuple[list[float], str]]] = []
    for frame in frames:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
        result = landmarker.detect(image)
        detections = []
        for detection_index, landmarks in enumerate(result.hand_landmarks or []):
            xy = np.asarray([(p.x, p.y) for p in landmarks], dtype=np.float32)
            x1, y1 = xy.min(0)
            x2, y2 = xy.max(0)
            label = "unknown"
            if result.handedness and detection_index < len(result.handedness) and result.handedness[detection_index]:
                label = result.handedness[detection_index][0].category_name.lower()
            detections.append(([float(x1), float(y1), float(x2), float(y2)], label))
        per_frame.append(detections)
    landmarker.close()
    return per_frame


def _detect_hands_legacy_api(frames: list[np.ndarray], cfg: dict[str, Any]) -> list[list[tuple[list[float], str]]] | None:
    try:
        import mediapipe as mp
        detector = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=2,
            min_detection_confidence=float(cfg["evidence"]["min_detection_confidence"]),
        )
    except Exception:
        return None
    per_frame: list[list[tuple[list[float], str]]] = []
    for frame in frames:
        result = detector.process(frame)
        detections = []
        handedness = result.multi_handedness or []
        for detection_index, landmarks in enumerate(result.multi_hand_landmarks or []):
            xy = np.asarray([(p.x, p.y) for p in landmarks.landmark], dtype=np.float32)
            x1, y1 = xy.min(0)
            x2, y2 = xy.max(0)
            label = handedness[detection_index].classification[0].label.lower() if detection_index < len(handedness) else "unknown"
            detections.append(([float(x1), float(y1), float(x2), float(y2)], label))
        per_frame.append(detections)
    detector.close()
    return per_frame


def detect_person_upper(frame: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Person segmentation mask (+ face keypoints) from the pose landmarker.

    Used only to DOWNWEIGHT SoM proposals sitting on the operator's body —
    never rendered. Returns None when the model is unavailable or no person
    is detected."""
    model_path = repo_path(cfg, cfg["evidence"].get("pose_landmarker_model", "models/pose_landmarker_lite.task"))
    if not model_path.exists():
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        landmarker = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE, num_poses=1,
        ))
    except Exception:
        return None
    try:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
        result = landmarker.detect(image)
    finally:
        landmarker.close()
    if not result.pose_landmarks:
        return None
    landmarks = result.pose_landmarks[0]
    height, width = frame.shape[:2]
    def _pt(i: int) -> np.ndarray:
        return np.asarray([landmarks[i].x * width, landmarks[i].y * height], dtype=np.float32)
    # indices: 0 nose, 7/8 ears, 11/12 shoulders, 23/24 hips
    shoulder_width = float(np.linalg.norm(_pt(11) - _pt(12)))
    # torso quad expanded ~20% laterally; hips may be off-frame, clamp handled by polygon test
    torso = np.asarray([_pt(11), _pt(12), _pt(24), _pt(23)], dtype=np.float32)
    torso_center = torso.mean(axis=0)
    torso = torso_center + 1.2 * (torso - torso_center)
    return {"face_points": [_pt(0), _pt(7), _pt(8)],
            "face_radius": max(float(np.linalg.norm(_pt(7) - _pt(8))), 0.15 * shoulder_width),
            "torso_polygon": torso}


def hands_per_frame(frames: list[np.ndarray], cfg: dict[str, Any]) -> tuple[list[list[list[float]]], str]:
    """ALL detected hand boxes per frame (hand-agnostic mode).

    Datasets that do not annotate an acting hand (Assembly101, IKEA ASM) must not
    have evidence anchored on one arbitrary hand: proposal ranking should consider
    proximity to EITHER hand. Returns ([[box, ...] per frame], tracker)."""
    per_frame = _detect_hands_tasks_api(frames, cfg)
    tracker = "mediapipe_tasks"
    if per_frame is None:
        per_frame = _detect_hands_legacy_api(frames, cfg)
        tracker = "mediapipe"
    if per_frame is None:
        return [[] for _ in frames], "unavailable"
    return [[box for box, _label in detections] for detections in per_frame], tracker


def hand_track(frames: list[np.ndarray], cfg: dict[str, Any], target_hand: str) -> tuple[list[list[float] | None], str]:
    if target_hand not in {"left", "right"}:
        raise ValueError(
            f"hand_track needs a single target hand, got {target_hand!r}; "
            "hand-agnostic clips must use hands_per_frame()")
    per_frame = _detect_hands_tasks_api(frames, cfg)
    tracker = "mediapipe_tasks"
    if per_frame is None:
        per_frame = _detect_hands_legacy_api(frames, cfg)
        tracker = "mediapipe"
    if per_frame is None:
        return [None] * len(frames), "unavailable"
    track: list[list[float] | None] = []
    previous_center: np.ndarray | None = None
    for detections in per_frame:
        candidates = []
        for box, label in detections:
            # MediaPipe labels assume a mirrored (selfie) view; our videos are not mirrored.
            if not cfg["evidence"].get("mediapipe_input_mirrored", False) and label in {"left", "right"}:
                label = "right" if label == "left" else "left"
            center = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            candidates.append((box, center, label))
        if not candidates:
            track.append(None)
        else:
            matching = [item for item in candidates if item[2] == target_hand]
            if matching:
                candidates = matching
            if previous_center is None:
                chosen = max(candidates, key=lambda item: (item[0][2] - item[0][0]) * (item[0][3] - item[0][1]))
            else:
                chosen = min(candidates, key=lambda item: np.linalg.norm(item[1] - previous_center))
            track.append(chosen[0])
            previous_center = chosen[1]
    return track, tracker


def track_coverage(track: list[list[float] | None]) -> float:
    return float(np.mean([box is not None for box in track])) if track else 0.0


# ---------------------------------------------------------------- proposers ---

def classical_proposals(frame: np.ndarray, max_regions: int, min_area_frac: float,
                        max_area_frac: float = 0.25) -> list[tuple[int, int, int, int]]:
    """Annotation-free contour proposals (fallback when FastSAM is unavailable)."""
    import cv2
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area_frac * width * height or area > max_area_frac * width * height:
            continue
        boxes.append((x, y, x + w, y + h, area))
    boxes.sort(key=lambda item: -item[4])
    return [tuple(int(v) for v in box[:4]) for box in boxes[:max_regions]]


def fastsam_proposals(frame: np.ndarray, model_path: Path, max_regions: int, min_area_frac: float,
                      device: str, max_area_frac: float = 0.25) -> list[tuple[int, int, int, int]] | None:
    """Candidate regions from FastSAM. Returns MORE than max_regions; the caller
    ranks by manipulation relevance (motion + hand proximity) and trims."""
    try:
        from ultralytics import FastSAM
    except ImportError:
        return None
    if not model_path.is_file():
        return None
    import cv2
    height, width = frame.shape[:2]
    result = FastSAM(str(model_path))(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), device=device,
                                      retina_masks=True, imgsz=640, conf=0.3, iou=0.9, verbose=False)
    if not result or result[0].boxes is None:
        return None
    boxes = []
    for box in result[0].boxes.xyxy.cpu().numpy():
        area = float(box[2] - box[0]) * float(box[3] - box[1])
        if area < min_area_frac * width * height or area > max_area_frac * width * height:
            continue
        boxes.append(tuple(float(v) for v in box[:4]))
    return [tuple(int(v) for v in box) for box in boxes[: max_regions * 6]]


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def rank_proposals(boxes: list[tuple[int, int, int, int]], motion_map: np.ndarray,
                   hand_centers_px: list[np.ndarray], max_regions: int,
                   hand_scale_px: float | None = None, min_regions: int = 3,
                   score_keep_frac: float = 0.45,
                   person: dict[str, Any] | None = None) -> list[tuple[int, int, int, int]]:
    """Rank candidate regions by manipulation relevance, not size.

    The person also moves, so raw motion energy alone favors head/torso
    segments. When the hand track is reliable, proximity to the manipulation
    zone (any tracked hand position, decay scaled by hand size) dominates;
    motion is log-damped and only breaks ties among nearby regions. Without a
    reliable track, motion is the only signal. Greedy IoU dedup, then an
    adaptive cutoff drops far/static leftovers instead of padding to
    max_regions with person/background segments.
    """
    if not boxes:
        return []
    height, width = motion_map.shape[:2]
    decay = hand_scale_px if hand_scale_px else 0.1 * float(np.hypot(width, height))
    scored = []
    for box in boxes:
        x1, y1, x2, y2 = (max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3]))
        if x2 <= x1 or y2 <= y1:
            continue
        motion = float(motion_map[y1:y2, x1:x2].mean())
        weight = 1.0
        if hand_centers_px:
            center = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
            distance = min(float(np.linalg.norm(center - hand)) for hand in hand_centers_px)
            weight = float(np.exp(-distance / max(decay, 1.0)))
        if hand_scale_px:
            # Graspable-scale prior: assembly components are roughly hand-sized;
            # torso/monitor/background segments are several times larger.
            box_diag = float(np.hypot(x2 - x1, y2 - y1))
            oversize = box_diag / max(hand_scale_px, 1.0)  # hand_scale_px ~ 2 hand diagonals
            if oversize > 1.0:
                weight *= float(np.exp(-(oversize - 1.0)))
        if person is not None:
            import cv2
            center = np.asarray([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
            face_distance = min(float(np.linalg.norm(center - p)) for p in person["face_points"])
            on_torso = cv2.pointPolygonTest(person["torso_polygon"].reshape(-1, 1, 2),
                                            (float(center[0]), float(center[1])), False) >= 0
            if face_distance < 3.0 * person["face_radius"] or on_torso:
                weight *= 0.05   # proposal sits on the operator's head/torso
        scored.append((weight * (1.0 + float(np.log1p(motion))), box))
    scored.sort(key=lambda item: -item[0])
    kept: list[tuple[float, tuple[int, int, int, int]]] = []
    for score, box in scored:
        if all(_box_iou(box, existing) < 0.5 for _, existing in kept):
            kept.append((score, box))
        if len(kept) >= max_regions:
            break
    if kept:
        threshold = score_keep_frac * kept[0][0]
        kept = [item for i, item in enumerate(kept) if i < min_regions or item[0] >= threshold]
    return [box for _, box in kept]


def motion_map_from_frames(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel mean absolute temporal difference at full resolution (float32)."""
    import cv2
    gray = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY).astype(np.float32) for f in frames]
    if len(gray) < 2:
        return np.zeros_like(gray[0])
    acc = np.zeros_like(gray[0])
    for a, b in zip(gray, gray[1:]):
        acc += np.abs(b - a)
    acc /= len(gray) - 1
    return cv2.GaussianBlur(acc, (9, 9), 0)


# ---------------------------------------------------------------- rendering ---

MARK_COLORS = [(255, 64, 64), (64, 200, 64), (64, 128, 255), (255, 200, 0),
               (200, 64, 255), (0, 200, 200), (255, 128, 0), (0, 160, 255)]


def _font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def badge_keyframe(frame: np.ndarray, ordinal: int, count: int, relative_time: float, size: int) -> np.ndarray:
    """Resize + burn a time badge (ordinal) and a temporal progress bar."""
    import cv2
    from PIL import Image, ImageDraw
    height, width = frame.shape[:2]
    scale = size / max(width, height)
    resized = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(resized)
    draw = ImageDraw.Draw(image)
    w, h = image.size
    badge = f"{ordinal}/{count}"
    font = _font(max(14, h // 14))
    box = draw.textbbox((0, 0), badge, font=font)
    pad = 4
    draw.rectangle([0, 0, box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), badge, fill=(255, 255, 0), font=font)
    bar_height = max(4, h // 60)
    draw.rectangle([0, h - bar_height, w - 1, h - 1], fill=(40, 40, 40))
    draw.rectangle([0, h - bar_height, int(relative_time * (w - 1)), h - 1], fill=(255, 200, 0))
    return np.asarray(image)


def draw_pointer_marks(frame: np.ndarray, boxes: list[tuple[int, int, int, int]], size: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    import cv2
    from PIL import Image, ImageDraw
    height, width = frame.shape[:2]
    scale = size / max(width, height)
    resized = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    image = Image.fromarray(resized)
    draw = ImageDraw.Draw(image)
    font = _font(max(14, image.size[1] // 16))
    marks = []
    for index, box in enumerate(boxes):
        color = MARK_COLORS[index % len(MARK_COLORS)]
        x1, y1, x2, y2 = [int(v * scale) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        radius = max(10, image.size[1] // 24)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)
        draw.text((cx - radius // 2 - 2, cy - radius // 2 - 4), str(index + 1), fill=(0, 0, 0), font=font)
        marks.append({"id": index + 1, "box": [int(v) for v in box],
                      "center": [float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)]})
    return np.asarray(image), marks


def save_rgb(path: Path, frame: np.ndarray) -> None:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def degrade_video(source: Path, destination: Path, cfg: dict[str, Any]) -> None:
    """Low-information counterfactual video for necessity training."""
    import cv2
    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    side = int(cfg["evidence"]["degrade_side"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    stride = max(1, int(cfg["evidence"]["degrade_frame_stride"]))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            height, width = frame.shape[:2]
            scale = side / max(width, height)
            small = cv2.resize(frame, (max(2, int(width * scale)), max(2, int(height * scale))), interpolation=cv2.INTER_AREA)
            blurred = cv2.GaussianBlur(small, (5, 5), 0)
            if writer is None:
                writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                                         max(2.0, fps / stride), (blurred.shape[1], blurred.shape[0]))
            writer.write(blurred)
        index += 1
    cap.release()
    if writer is not None:
        writer.release()


# ------------------------------------------------------------------ builder ---

def keyframe_change_pair(frames: list[np.ndarray], keyframe_indices: list[int]) -> int:
    """Pixel-arithmetic answer: between which consecutive keyframes is the change
    largest? Returns the 1-based ordinal of the earlier keyframe of the pair.
    Assumption-free (pure frame differencing), so safe as a necessity QA target."""
    import cv2
    if len(keyframe_indices) < 2:
        return 1
    gray = [cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY), (160, 90)) for i in keyframe_indices]
    diffs = [float(np.mean(cv2.absdiff(a, b))) for a, b in zip(gray, gray[1:])]
    return int(np.argmax(diffs)) + 1


def process_record(row: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    clip_id = row["clip_id"]
    # The cropped workspace view is the proven input (C1 finding: workspace
    # foveation helps). The full-scene original is person/background-dominated
    # and is kept in the manifest only for provenance.
    source_key = cfg["evidence"].get("source", "video")
    source = repo_path(cfg, row[source_key])
    frames, sampled_indices, fps, total = read_video(source, int(cfg["evidence"]["scan_frames"]))
    if not frames:
        raise RuntimeError(f"Unreadable video: {source}")
    if len(frames) == 1:
        # A handful of null clips are single-frame (0.07 s); duplicate the frame
        # so motion/keyframe logic degrades gracefully instead of failing.
        frames = frames * 2
        sampled_indices = np.concatenate([sampled_indices, sampled_indices])
    scores = motion_energy(frames)
    keyframes = select_keyframes(scores, int(cfg["evidence"]["num_keyframes"]),
                                 cfg["evidence"].get("keyframe_method", "energy"))
    peak = int(np.argmax(scores))
    hand_agnostic = row["target_hand"] == "both"
    if hand_agnostic:
        # no acting hand annotated -> keep BOTH hands, anchor evidence on neither
        multi_track, tracker = hands_per_frame(frames, cfg)
        raw_track = [boxes[0] if boxes else None for boxes in multi_track]
        coverage = float(np.mean([bool(b) for b in multi_track])) if multi_track else 0.0
    else:
        multi_track = None
        raw_track, tracker = hand_track(frames, cfg, row["target_hand"])
        coverage = track_coverage(raw_track)
    reliable = coverage >= float(cfg["evidence"]["min_track_coverage"])
    out = artifact_path(cfg, cfg["evidence"].get("output_dirname", "evidence"), clip_id)
    out.mkdir(parents=True, exist_ok=True)

    assets: list[dict[str, Any]] = []
    size = int(cfg["evidence"]["keyframe_size"])
    for ordinal, index in enumerate(keyframes, start=1):
        relative = float(sampled_indices[index]) / max(1, total - 1)
        badged = badge_keyframe(frames[index], ordinal, len(keyframes), relative, size)
        name = f"kf{ordinal:02d}.jpg"
        save_rgb(out / name, badged)
        assets.append({"type": "kf", "path": name, "ordinal": ordinal,
                       "sample_index": int(index), "source_frame": int(sampled_indices[index]),
                       "relative_time": relative})
    # plain (badge-free) copies for the no_badges ablation
    import cv2
    for ordinal, index in enumerate(keyframes, start=1):
        height, width = frames[index].shape[:2]
        scale = size / max(width, height)
        plain = cv2.resize(frames[index], (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
        save_rgb(out / f"plain_kf{ordinal:02d}.jpg", plain)

    som_backend = "none"
    marks: list[dict[str, Any]] = []
    if cfg["evidence"].get("include_som", True):
        peak_frame = frames[peak]
        max_marks = int(cfg["evidence"]["max_marks"])
        max_area = float(cfg["evidence"].get("max_mark_area_frac", 0.25))
        candidates = fastsam_proposals(peak_frame, repo_path(cfg, cfg["evidence"]["fastsam_path"]),
                                       max_marks, float(cfg["evidence"]["min_mark_area_frac"]),
                                       cfg["evidence"].get("som_device", "cpu"), max_area)
        if candidates is not None:
            som_backend = "fastsam"
        else:
            candidates = classical_proposals(peak_frame, max_marks * 6,
                                             float(cfg["evidence"]["min_mark_area_frac"]), max_area)
            som_backend = "classical"
        # Rank by manipulation relevance (motion around the peak + hand
        # proximity) so marks land on moving assembly parts, not on the
        # static person/background.
        lo, hi = max(0, peak - 3), min(len(frames), peak + 4)
        motion = motion_map_from_frames(frames[lo:hi])
        hand_centers_px: list[np.ndarray] = []
        hand_sizes: list[float] = []
        if reliable:
            height, width = peak_frame.shape[:2]
            # hand-agnostic: every detected hand contributes, so proposals are
            # ranked by proximity to EITHER hand rather than to one arbitrary track
            window = ([b for boxes in multi_track[lo:hi] for b in boxes] if hand_agnostic
                      else raw_track[lo:hi])
            for hand_box in window:
                if hand_box is not None:
                    hand_centers_px.append(np.asarray([(hand_box[0] + hand_box[2]) / 2 * width,
                                                       (hand_box[1] + hand_box[3]) / 2 * height], dtype=np.float32))
                    hand_sizes.append(float(np.hypot((hand_box[2] - hand_box[0]) * width,
                                                     (hand_box[3] - hand_box[1]) * height)))
        # decay ~ 1.5 hand-diagonals: marks concentrate on the graspable zone;
        # occlusion-flicker regions behind the operator sit several diagonals away
        hand_scale = 1.5 * float(np.median(hand_sizes)) if hand_sizes else None
        person = detect_person_upper(peak_frame, cfg)
        boxes = rank_proposals(candidates or [], motion, hand_centers_px, max_marks, hand_scale,
                               person=person)
        if boxes:
            som_image, marks = draw_pointer_marks(peak_frame, boxes, int(cfg["evidence"]["som_size"]))
            save_rgb(out / "som.jpg", som_image)
            assets.append({"type": "som", "path": "som.jpg", "sample_index": int(peak),
                           "source_frame": int(sampled_indices[peak])})

    # Necessity QA metadata (all answers verifiable; hand answer validity-gated).
    necessity: dict[str, Any] = {
        "keyframe_count": len(keyframes),
        "max_change_after_kf": keyframe_change_pair(frames, keyframes),
    }
    # "which mark is nearest the <target> hand" has no answer when no hand is the
    # subject, so hand-agnostic clips simply do not get this QA item (the prompt
    # builders key off its presence).
    if not hand_agnostic and reliable and marks and raw_track[peak] is not None:
        height, width = frames[peak].shape[:2]
        hand_box = raw_track[peak]
        hand_center = np.asarray([(hand_box[0] + hand_box[2]) / 2 * width,
                                  (hand_box[1] + hand_box[3]) / 2 * height])
        distances = [float(np.linalg.norm(hand_center - np.asarray(mark["center"]))) for mark in marks]
        necessity["mark_nearest_hand"] = int(marks[int(np.argmin(distances))]["id"])

    if cfg["evidence"].get("include_degraded_video", True):
        degraded = out / "degraded.mp4"
        if not degraded.exists():
            degrade_video(source, degraded, cfg)

    meta = {
        "clip_id": clip_id, "source": str(source), "tracker": tracker,
        "track_coverage": coverage, "reliable_localization": reliable,
        "som_backend": som_backend, "marks": marks,
        "keyframe_method": cfg["evidence"].get("keyframe_method", "energy"),
        "keyframe_indices": [int(sampled_indices[i]) for i in keyframes],
        "peak_source_frame": int(sampled_indices[peak]),
        "assets": assets, "necessity": necessity,
        "policy": ("pointer-only evidence: badges/bars encode time, marks are candidate "
                   "regions; no extracted trajectory or contact is rendered"),
    }
    write_json(out / "meta.json", meta)
    return meta


def records(cfg: dict[str, Any], splits: list[str] | None = None) -> list[dict[str, Any]]:
    root = artifact_path(cfg, "manifests")
    if splits is None:
        splits = ["official_train", "official_test"]
        if (root / "right_test.jsonl").exists():
            splits += ["right_test"]
    rows = [row for split in splits for row in read_jsonl(root / f"{split}.jsonl")]
    return list({row["clip_id"]: row for row in rows}.values())


def audit(cfg: dict[str, Any]) -> dict[str, Any]:
    rows = records(cfg)
    found = []
    failures = []
    reliable = 0
    mark_counts = []
    backends: dict[str, int] = {}
    for row in rows:
        path = artifact_path(cfg, cfg["evidence"].get("output_dirname", "evidence"), row["clip_id"], "meta.json")
        if not path.exists():
            failures.append(row["clip_id"])
            continue
        meta = json.loads(path.read_text())
        found.append(meta)
        reliable += int(meta["reliable_localization"])
        mark_counts.append(len(meta["marks"]))
        backends[meta["som_backend"]] = backends.get(meta["som_backend"], 0) + 1
    report = {"expected": len(rows), "complete": len(found), "missing": failures[:20],
              "missing_count": len(failures),
              "reliable": reliable, "reliable_rate": reliable / max(1, len(found)),
              "mean_marks": float(np.mean(mark_counts)) if mark_counts else 0.0,
              "som_backends": backends,
              "ok": len(found) == len(rows)}
    write_json(artifact_path(cfg, cfg["evidence"].get("output_dirname", "evidence"), "audit.json"), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("command", choices=["build", "audit"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--splits", default="")
    parser.add_argument("--override", action="append", default=[],
                        help="dotted config override, e.g. evidence.keyframe_method=stratified")
    args = parser.parse_args()
    cfg = load_config(args.config)
    for item in args.override:
        dotted, _, value = item.partition("=")
        node = cfg
        keys = dotted.split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value
    if args.command == "audit":
        print(json.dumps(audit(cfg), indent=2))
        return
    requested = [value for value in args.splits.split(",") if value]
    todo = records(cfg, requested or None)
    if args.limit:
        todo = todo[:args.limit]
    errors = []
    for index, row in enumerate(todo, 1):
        meta = artifact_path(cfg, cfg["evidence"].get("output_dirname", "evidence"), row["clip_id"], "meta.json")
        if args.resume and meta.exists():
            continue
        try:
            process_record(row, cfg)
        except Exception as exc:
            errors.append({"clip_id": row["clip_id"], "error": repr(exc)})
        if index % 100 == 0:
            print(f"evidence {index}/{len(todo)} errors={len(errors)}", flush=True)
    write_json(artifact_path(cfg, cfg["evidence"].get("output_dirname", "evidence"), "build_errors.json"), errors)
    if errors:
        raise RuntimeError(f"Evidence build failed for {len(errors)} clips; see build_errors.json")


if __name__ == "__main__":
    main()
