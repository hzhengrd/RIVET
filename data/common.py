"""Shared helpers for clip preparation: paths, taxonomy, gold-slot
derivation, the structured answer format + tolerant parser, APT-token rendering,
and video frame reading. Pure-python (numpy/cv2 only) so it imports on the cluster
and in the offline sandbox."""
from __future__ import annotations
import json, os, re, glob
import numpy as np

# ---------------- paths (repo-root relative) ----------------
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LH_TRAIN_ROOT = "artifacts/havid_clips/lh_v0/videos_train"
LH_VAL_ROOT = "artifacts/havid_clips/lh_v0/videos_val"
GT_OFFICIAL = "artifacts/havid/ground_truth.jsonl"
MAPPING = "artifacts/havid_clips/mapping_semantics.txt"
COMP_SPLIT = "artifacts/splits/composition_split_v1.json"
PRIM_SPLIT = "artifacts/splits/primitive_split_v1.json"

OUT = "artifacts/havid"
C1_DIR = f"{OUT}/c1"
FEAT_DIR = f"{OUT}/feat"
CODES_DIR = f"{OUT}/codes"
SFT_DIR = f"{OUT}/sft"
TOKENIZER_PT = f"{OUT}/tokenizer.pt"
MANIFEST_DIR = f"{OUT}/manifests"

from data.taxonomy import build_taxonomy, ALL_SLOTS, ELEMENT_SLOTS, norm

# ---------------- taxonomy / gold slots ----------------

def load_tax(extra_gt=None):
    gts = [GT_OFFICIAL] + (extra_gt or [])
    return build_taxonomy(gts, MAPPING)


def gold_slots(tax, label):
    """label code -> canonical slot dict (status/verb/manip/target/tool). null clips
    map to the idle state."""
    if label == "null" or label not in tax.label_to_tuple:
        return {"status": "null", "verb": "null", "manipulated_object": "not applicable",
                "target_object": "not applicable", "tool": "no tool"}
    v, m, t, tool = tax.label_to_tuple[label]
    return {"status": "active", "verb": v, "manipulated_object": m,
            "target_object": t, "tool": tool}


def holistic_desc(tax, label):
    return tax.label_to_desc.get(label, "null")


# ---------------- structured answer (target) + tolerant parser ----------------
# Target string matches the legacy V10 structured format (key 'action_verb').
ANSWER_KEYS = ["status", "action_verb", "manipulated_object", "target_object", "tool"]


def structured_answer(slots):
    d = {"status": slots["status"],
         "action_verb": slots["verb"],
         "manipulated_object": slots["manipulated_object"],
         "target_object": slots["target_object"],
         "tool": slots["tool"]}
    return json.dumps(d)


_JSON_RE = re.compile(r"\{.*\}", re.S)
_KV_RE = re.compile(r'"?(status|action_verb|verb|manipulated_object|target_object|tool)"?\s*[:=]\s*"?([^",}\n]+)"?')


def parse_structured(text):
    """Tolerant recovery parser: try JSON, else key:value regex. Returns canonical
    slot dict {status,verb,manipulated_object,target_object,tool} (norm-ed)."""
    raw = {}
    m = _JSON_RE.search(text or "")
    if m:
        try:
            raw = json.loads(m.group(0))
        except Exception:
            raw = {}
    if not raw:
        for k, v in _KV_RE.findall(text or ""):
            raw[k] = v
    g = lambda *ks: next((raw[k] for k in ks if k in raw and raw[k] is not None), None)
    out = {
        "status": norm(g("status"), slot="status") or "active",
        "verb": norm(g("action_verb", "verb"), slot="verb"),
        "manipulated_object": norm(g("manipulated_object"), slot="manipulated_object"),
        "target_object": norm(g("target_object"), slot="target_object"),
        "tool": norm(g("tool"), slot="tool"),
    }
    if out["status"] == "null":
        out.update(verb="null", manipulated_object="not applicable",
                   target_object="not applicable", tool="no tool")
    return out


# ---------------- APT token rendering (Mode A in-context) ----------------
def apt_token(i):
    return f"<APT_{int(i)}>"


APT_ABSTAIN_TOKEN = "<APT_ABSTAIN>"


def render_apt_tokens(indices, abstain_index=None, max_len=24):
    """Discrete APT primitive indices -> a compact special-token string for the prompt.
    Consecutive repeats are collapsed (run-length), abstain -> <APT_ABSTAIN>."""
    if not indices:
        return APT_ABSTAIN_TOKEN
    toks, prev = [], None
    for i in indices[:max_len]:
        if i == prev:
            continue
        prev = i
        toks.append(APT_ABSTAIN_TOKEN if (abstain_index is not None and i == abstain_index)
                    else apt_token(i))
    return " ".join(toks)


def all_apt_special_tokens(codebook_size):
    return [apt_token(i) for i in range(codebook_size)] + [APT_ABSTAIN_TOKEN]


# ---------------- prompts ----------------
STRUCT_INSTR = ("Identify the assembly state and slots. Return only JSON with keys: "
                "status, action_verb, manipulated_object, target_object, tool. Use status "
                "active or null; if status is null, use action_verb null, tool no tool, and "
                "not applicable for the objects.")


def hand_focus(target="left"):
    """Hand-focus clause, with a trailing space when non-empty.

    Datasets that do not annotate which hand acts (Assembly101, IKEA ASM) use
    target_hand 'both' and get NO hand clause at all: the question is about the
    action taking place, not about any hand. Naming a hand there would ask the
    model to attend to something the label does not describe."""
    return "" if target == "both" else f"Focus only on the {target} hand. "


def question_noapt(target="left"):
    return (f"<video>\n{hand_focus(target)}{STRUCT_INSTR}")


def question_apt(apt_str, n_images, target="left"):
    imgs = "".join("<image>" for _ in range(n_images))
    return (f"<video>{imgs}\n{hand_focus(target)}The extra images are "
            f"foveated views (hand/object crops, motion, numbered candidates). "
            f"The following action-perception tokens encode the measured hand "
            f"kinematics and contact geometry: {apt_str}. {STRUCT_INSTR}")


# ---------------- clip enumeration ----------------
def clip_target_hand(root):
    return "right" if "rh" in os.path.basename(os.path.dirname(os.path.dirname(root))).lower() else "left"


def list_clips(root):
    """[(label, clip_id, path)] under root/<label>/*.mp4."""
    out = []
    for lab in sorted(os.listdir(root)):
        d = os.path.join(root, lab)
        if not os.path.isdir(d):
            continue
        for v in sorted(glob.glob(os.path.join(d, "*.mp4"))):
            out.append((lab, os.path.splitext(os.path.basename(v))[0], v))
    return out


def read_frames(path, max_frames=16):
    import cv2
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).astype(int)
        frames = [frames[i] for i in idx]
    return frames
