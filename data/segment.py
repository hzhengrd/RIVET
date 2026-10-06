"""Cut per-action clips out of continuous session videos into the
<root>/<label>/<clip_id>.mp4 layout data.common.list_clips()
expects. Used by the Assembly101/IKEA ASM prep scripts, which ship one
continuous video per session/scan plus a table of (start, end, label)
segments -- unlike HA-ViD, which ships pre-cut per-action clips.
"""
from __future__ import annotations
import os
import subprocess
from dataclasses import dataclass


@dataclass
class Segment:
    video_path: str     # source continuous video
    start_sec: float
    end_sec: float
    label: str
    clip_id: str
    crop: tuple | None = None      # (x1, y1, x2, y2) workspace ROI in source pixels


def cut_segment(seg: Segment, root: str, *, reencode: bool = True, fps: float | None = None) -> str:
    """Extracts [start_sec, end_sec) from seg.video_path into
    <root>/<label>/<clip_id>.mp4. reencode=True (default) re-encodes for a
    frame-accurate cut; stream-copy (-c copy) is faster but can only cut on
    keyframes, which is unsafe for short action segments.

    seg.crop restricts the frame to a workspace ROI. Full-scene datasets need
    this for parity with HA-ViD, whose proven input is already a cropped
    workspace view: on IKEA ASM's room-wide dev3 view MediaPipe finds 0 hands in
    48 full frames but 29-32 of 32 after the ROI crop. A crop forces reencode
    (a stream copy cannot change the frame size)."""
    out_dir = os.path.join(root, seg.label)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{seg.clip_id}.mp4")
    duration = seg.end_sec - seg.start_sec
    if duration <= 0:
        raise ValueError(f"non-positive duration for {seg.clip_id}: {seg.start_sec}..{seg.end_sec}")
    cmd = ["ffmpeg", "-y", "-ss", f"{seg.start_sec:.3f}", "-i", seg.video_path, "-t", f"{duration:.3f}"]
    if seg.crop:
        x1, y1, x2, y2 = (int(v) for v in seg.crop)
        cmd += ["-vf", f"crop={x2 - x1}:{y2 - y1}:{x1}:{y1}"]
        reencode = True
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
        if fps:
            cmd += ["-r", str(fps)]
    else:
        cmd += ["-c", "copy"]
    cmd += ["-an", out_path]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return out_path


def cut_segments(segments: list[Segment], root: str, **kwargs) -> list[str]:
    return [cut_segment(seg, root, **kwargs) for seg in segments]
