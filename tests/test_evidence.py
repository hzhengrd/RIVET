"""Offline tests for the released evidence builder: pure-python/numpy only.

Evidence-construction tests: keyframe selection and badge rendering.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:  # tests stay runnable via the plain-python fallback runner
    pytest = None

from data.evidence import (badge_keyframe, energy_keyframes, keyframe_change_pair,
                           select_keyframes, stratified_keyframes)
from data.evidence_utils import SLOTS, structured_answer

GRAMMAR = {
    "slot_vocab": {
        "status": ["active", "null"],
        "verb": ["insert", "screw", "null"],
        "manipulated_object": ["hex screw", "nut", "not applicable"],
        "target_object": ["hole c2", "stud", "not applicable"],
        "tool": ["hex screwdriver", "no tool"],
    },
    "tuples": [["screw", "hex screw", "hole c2", "hex screwdriver"]],
    "class_counts": {"verb": {"insert": 10, "screw": 1000, "null": 100}},
}

REWARD_CFG = {"grpo": {"reward": {"w_format": 0.5, "w_slot": 2.0, "w_valid": 0.5, "w_composite": 2.0,
                                  "macro_weighted": True, "validity_reward": True,
                                  "macro_alpha": 0.5, "max_class_weight": 10.0}}}

def test_stratified_keyframes_cover_time():
    scores = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6], dtype=np.float32)
    picks = stratified_keyframes(scores, 4)
    assert picks == sorted(picks)
    assert len(picks) == 4
    assert picks[0] < 2 and picks[-1] >= 6


def test_energy_keyframes_concentrate_on_action():
    scores = np.zeros(40, dtype=np.float32)
    scores[20:30] = 10.0                      # single action burst
    picks = energy_keyframes(scores, 6)
    assert len(picks) == 6 and picks == sorted(picks)
    in_burst = sum(1 for p in picks if 18 <= p <= 31)
    assert in_burst >= 4                      # budget concentrates on the burst
    assert min(picks) < 18                    # uniform floor keeps global coverage


def test_energy_keyframes_static_clip_falls_back_to_uniform():
    picks = energy_keyframes(np.zeros(30, dtype=np.float32), 5)
    assert len(picks) == 5
    gaps = np.diff(picks)
    assert gaps.max() - gaps.min() <= 3       # roughly even spacing


def test_select_keyframes_methods():
    scores = np.asarray([0.0, 1.0, 0.0, 5.0, 0.0, 1.0], dtype=np.float32)
    assert select_keyframes(scores, 3, "energy") == energy_keyframes(scores, 3)
    assert select_keyframes(scores, 3, "stratified") == stratified_keyframes(scores, 3)
    assert len(select_keyframes(scores, 3, "uniform")) == 3
    assert select_keyframes(scores, 2, "maxmotion") == [1, 3]
    ends = select_keyframes(scores, 3, "ends_uniform")
    assert ends[0] == 0 and ends[-1] == len(scores) - 1
    try:
        select_keyframes(scores, 3, "nope")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_keyframe_change_pair_pixel_arithmetic():
    frames = [np.full((60, 80, 3), value, dtype=np.uint8) for value in (10, 12, 200, 202)]
    assert keyframe_change_pair(frames, [0, 1, 2, 3]) == 2


def test_badge_render():
    frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3)).astype(np.uint8)
    badged = badge_keyframe(frame, 2, 6, 0.25, 224)
    assert badged.shape[1] == 224 and badged.ndim == 3
    assert not np.array_equal(badged[:20, :40], frame[:20, :40])
