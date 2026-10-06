from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np

from .config import load_config, output_path, repo_path
from .experiment import variant_spec
from .utils import read_jsonl, stable_int, write_json

ELEMENTS = ("verb", "manipulated_object", "target_object", "tool")


def correctness(row: dict[str, Any], gold_key: str = "gold") -> dict[str, int]:
    gold, pred = row[gold_key], row["prediction"]
    result = {slot: int(pred.get(slot) == gold.get(slot)) for slot in ("status", *ELEMENTS)}
    result["exact"] = int(all(result[slot] for slot in ("status", *ELEMENTS)))
    result["active_composite"] = int(gold["status"] == "active" and all(result[slot] for slot in ELEMENTS))
    return result


def macro_accuracy(rows: list[dict[str, Any]], slot: str) -> float:
    per_class: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        per_class[row["gold"][slot]].append(int(row["prediction"].get(slot) == row["gold"][slot]))
    return float(np.mean([np.mean(values) for values in per_class.values()])) if per_class else float("nan")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    correct = [correctness(row) for row in rows]
    active = [i for i, row in enumerate(rows) if row["gold"]["status"] == "active"]
    summary: dict[str, Any] = {"n": len(rows), "n_active": len(active)}
    for slot in ("status", *ELEMENTS):
        summary[f"{slot}_micro"] = float(np.mean([item[slot] for item in correct]))
        summary[f"{slot}_macro"] = macro_accuracy(rows, slot)
    summary["active_composite_accuracy"] = float(np.mean([correct[i]["active_composite"] for i in active])) if active else float("nan")
    summary["exact_accuracy"] = float(np.mean([item["exact"] for item in correct]))
    null_rows = [row for row in rows if row["gold"]["status"] == "null"]
    summary["null_false_activation_rate"] = float(np.mean([row["prediction"].get("status") != "null" for row in null_rows])) if null_rows else float("nan")
    summary["strict_json_rate"] = float(np.mean([bool(row.get("strict_json")) for row in rows]))
    grammar_rows = [row for row in rows if row.get("grammar")]
    active_grammar = [row for row in grammar_rows
                      if row["gold"]["status"] == "active" and row["prediction"].get("status") == "active"]
    if active_grammar:
        # Novel-composition emission: active predictions outside the seen-tuple set.
        # On composition_test a nonzero correct-novel rate is the compositional win.
        novel = [row for row in active_grammar if not row["grammar"].get("in_train_tuple", True)]
        summary["novel_tuple_rate"] = len(novel) / len(active_grammar)
        summary["novel_tuple_correct_rate"] = (float(np.mean([correctness(row)["active_composite"] for row in novel]))
                                               if novel else 0.0)
    summary["latency_mean_s"] = float(np.mean([row.get("latency_s", 0) for row in rows]))
    if rows and rows[0].get("parameter_counts"):
        summary.update({f"params_{key}": value for key, value in rows[0]["parameter_counts"].items()})
    raw_rows = [row for row in rows if row.get("official_gold_raw")]
    if raw_rows:
        raw_correct = [correctness(row, "official_gold_raw") for row in raw_rows]
        raw_active = [index for index, row in enumerate(raw_rows) if row["official_gold_raw"]["status"] == "active"]
        summary["official_raw_exact_accuracy"] = float(np.mean([item["exact"] for item in raw_correct]))
        summary["official_raw_active_composite_accuracy"] = (
            float(np.mean([raw_correct[index]["active_composite"] for index in raw_active])) if raw_active else float("nan"))
    return summary


def _paired(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], metric: str):
    a = {row["clip_id"]: row for row in rows_a}
    b = {row["clip_id"]: row for row in rows_b}
    ids = sorted(set(a) & set(b))

    def score(row):
        value = correctness(row)[metric]
        if metric == "active_composite" and row["gold"]["status"] != "active":
            return np.nan
        return value

    av = np.asarray([score(a[key]) for key in ids], dtype=float)
    bv = np.asarray([score(b[key]) for key in ids], dtype=float)
    keep = ~(np.isnan(av) | np.isnan(bv))
    clusters = [a[key].get("session_id", key.split("_")[0]) for key, use in zip(ids, keep) if use]
    kept_ids = [key for key, use in zip(ids, keep) if use]
    return av[keep], bv[keep], clusters, kept_ids


def cluster_bootstrap(a: np.ndarray, b: np.ndarray, clusters: list[str], samples: int, seed: int) -> tuple[float, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, name in enumerate(clusters):
        groups[name].append(index)
    names = sorted(groups)
    rng = np.random.default_rng(seed)
    effects = []
    for _ in range(samples):
        chosen = rng.choice(names, len(names), replace=True)
        indices = [index for name in chosen for index in groups[name]]
        effects.append(float(np.mean(b[indices] - a[indices])))
    return tuple(np.quantile(effects, [0.025, 0.975]).tolist())


def permutation_p(a: np.ndarray, b: np.ndarray, clusters: list[str], samples: int, seed: int) -> float:
    difference = b - a
    observed = abs(float(difference.mean()))
    rng = np.random.default_rng(seed)
    names = sorted(set(clusters))
    membership = np.asarray([names.index(name) for name in clusters])
    null = [abs(float((difference * rng.choice([-1, 1], len(names))[membership]).mean())) for _ in range(samples)]
    return (1 + sum(value >= observed for value in null)) / (samples + 1)


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    from scipy.stats import binomtest
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    p = float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue) if n01 + n10 else 1.0
    return {"baseline_wrong_variant_right": n01, "baseline_right_variant_wrong": n10, "p": p}


def compare(base: list[dict[str, Any]], variant: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    samples = int(cfg["evaluation"]["bootstrap_samples"])
    permutations = int(cfg["evaluation"]["permutation_samples"])
    result = {}
    for metric in ("active_composite", "exact"):
        a, b, clusters, ids = _paired(base, variant, metric)
        seed = stable_int(metric + str(len(ids)))
        result[metric] = {"n": len(ids), "baseline": float(a.mean()), "variant": float(b.mean()),
                          "effect": float((b - a).mean()),
                          "ci95": cluster_bootstrap(a, b, clusters, samples, seed),
                          "permutation_p": permutation_p(a, b, clusters, permutations, seed),
                          "mcnemar": mcnemar(a, b)}
    return result


def holm(items: list[dict[str, Any]], alpha: float) -> None:
    indexed = sorted(enumerate(items), key=lambda pair: pair[1]["p_raw"])
    count = len(items)
    running = 0.0
    for rank, (index, item) in enumerate(indexed):
        adjusted = min(1.0, (count - rank) * item["p_raw"])
        running = max(running, adjusted)
        items[index]["p_holm"] = running
        items[index]["reject_holm"] = running < alpha


def _meta_from_path(path) -> dict[str, Any]:
    """Parse predictions/.../<train>/<variant>/seed_<n>/<eval>[__tag].jsonl.

    Tags after the eval split: `__dec_<decoding>`, `__bank_<mode>`, or an
    evidence-mode name (`wrong`, `absent`, ...). `__dec_*` is not evidence.
    """
    parts = path.parts
    try:
        pred_idx = parts.index("predictions")
        train_split = parts[pred_idx + 1]
        variant = parts[pred_idx + 2]
        seed = int(parts[pred_idx + 3].removeprefix("seed_"))
        tokens = path.stem.split("__")
        eval_split = tokens[0]
        evidence_mode = "correct"
        decoding = None
        bank_mode = None
        epoch = None
        for tag in tokens[1:]:
            if tag.startswith("dec_"):
                decoding = tag[len("dec_"):]
            elif tag.startswith("bank_"):
                bank_mode = tag[len("bank_"):]
            elif tag.startswith("epoch_"):
                try:
                    epoch = int(tag.split("_", 1)[1])
                except ValueError:
                    epoch = tag
            else:
                evidence_mode = tag
        return {
            "train_split": train_split,
            "variant": variant,
            "seed": seed,
            "eval_split": eval_split,
            "evidence_mode": evidence_mode,
            "decoding": decoding,
            "bank_mode": bank_mode,
            "epoch": epoch,
        }
    except (ValueError, IndexError):
        return {}


def aggregate(cfg: dict[str, Any]) -> None:
    root = output_path(cfg, "predictions")
    files = sorted(root.glob("**/*.jsonl"))
    summaries = []
    cache: dict[tuple, list[dict[str, Any]]] = {}
    for path in files:
        if "__pre_fix" in path.stem:
            continue
        rows = read_jsonl(path)
        if not rows:
            continue
        meta = _meta_from_path(path)
        train_split = rows[0].get("train_split") or meta.get("train_split")
        variant = rows[0].get("variant") or meta.get("variant")
        seed = int(rows[0].get("seed") or meta.get("seed") or 0)
        eval_split = rows[0].get("eval_split") or rows[0].get("split") or meta.get("eval_split")
        evidence_mode = rows[0].get("evidence_mode") or meta.get("evidence_mode") or "correct"
        decoding = rows[0].get("decoding") or meta.get("decoding") or ""
        bank_mode = rows[0].get("bank_mode") or meta.get("bank_mode") or ""
        epoch = rows[0].get("epoch") if rows[0].get("epoch") is not None else meta.get("epoch")
        if not train_split or not variant or not eval_split:
            print(f"skip unparseable prediction file: {path}")
            continue
        key = (train_split, variant, seed, eval_split, evidence_mode, decoding, bank_mode, epoch)
        cache[key] = rows
        summaries.append({"train_split": key[0], "variant": key[1], "seed": key[2],
                          "eval_split": key[3], "evidence_mode": key[4], "decoding": key[5],
                          "bank_mode": key[6], "epoch": key[7], **summarize(rows)})

    def spec_of(name: str) -> dict[str, Any]:
        try:
            return variant_spec(cfg, name)
        except KeyError:
            return {}

    comparisons = []
    tests = []
    for key, rows in cache.items():
        train, variant, seed, split, evidence_mode, decoding, bank_mode, epoch = key
        if evidence_mode != "correct":
            continue
        baseline_name = spec_of(variant).get("baseline")
        if not baseline_name:
            continue
        baseline_key = (train, baseline_name, seed, split, "correct", decoding, bank_mode, epoch)
        if baseline_key not in cache:
            baseline_key = (train, baseline_name, seed, split, "correct", decoding, bank_mode, None)
        if baseline_key not in cache:
            baseline_key = (train, baseline_name, seed, split, "correct", decoding, "", None)
        if baseline_key not in cache:
            continue
        result = compare(cache[baseline_key], rows, cfg)
        comparisons.append({"train_split": train, "variant": variant, "baseline": baseline_name,
                            "seed": seed, "eval_split": split, "decoding": decoding,
                            "bank_mode": bank_mode, **result})
        tests.append({"key": f"{train}/{variant}/{seed}/{split}/{decoding}/{bank_mode}/active",
                      "p_raw": result["active_composite"]["permutation_p"]})
    holm(tests, float(cfg["evaluation"]["holm_alpha"]))
    adjusted = {item["key"]: item for item in tests}
    for item in comparisons:
        key = (f"{item['train_split']}/{item['variant']}/{item['seed']}/"
               f"{item['eval_split']}/{item.get('decoding', '')}/{item.get('bank_mode', '')}/active")
        item["active_composite"].update({k: v for k, v in adjusted[key].items()
                                         if k.startswith("p_") or k.startswith("reject")})

    # Evidence-content controls: wrong/shuffled evidence vs the same checkpoint's
    # correct-evidence predictions. Negative effects = evidence content is read.
    controls = []
    control_tests = []
    for key, control_rows in cache.items():
        train, variant, seed, split, evidence_mode, decoding, bank_mode, epoch = key
        if evidence_mode == "correct":
            continue
        correct_key = (train, variant, seed, split, "correct", decoding, bank_mode, epoch)
        if correct_key not in cache:
            continue
        result = compare(cache[correct_key], control_rows, cfg)
        item = {"train_split": train, "variant": variant, "seed": seed, "eval_split": split,
                "control": evidence_mode, **result}
        controls.append(item)
        control_tests.append({"item": item, "p_raw": result["active_composite"]["permutation_p"]})
    holm(control_tests, float(cfg["evaluation"]["holm_alpha"]))
    for test in control_tests:
        test["item"]["active_composite"].update(p_holm=test["p_holm"], reject_holm=test["reject_holm"])

    # Harm curve: for each evidence variant, active composite vs video_only at each
    # corruption level. harm = max(0, video_only - rivet_at_level).
    harm_curves = []
    for (train, variant, seed, split, decoding, bank_mode, epoch), _ in {
        (s["train_split"], s["variant"], s["seed"], s["eval_split"], s.get("decoding", ""),
         s.get("bank_mode", ""), s.get("epoch")): s
        for s in summaries
    }.items():
        vo_key = (train, "video_only", seed, split, "correct", decoding, bank_mode, None)
        if vo_key not in cache:
            vo_key = (train, "video_only", seed, split, "correct", decoding, "", None)
        if vo_key not in cache:
            continue
        vo_active = summarize(cache[vo_key]).get("active_composite_accuracy", float("nan"))
        levels = {}
        for mode in cfg.get("evaluation", {}).get("harm_levels",
                                                   ["correct", "shuffled", "partial", "wrong", "absent"]):
            key = (train, variant, seed, split, mode, decoding, bank_mode, epoch)
            if key not in cache:
                continue
            active = summarize(cache[key]).get("active_composite_accuracy", float("nan"))
            levels[mode] = {
                "active_composite": active,
                "harm": float(max(0.0, vo_active - active)) if active == active else None,
                "delta_vs_video_only": float(active - vo_active) if active == active else None,
            }
        if levels:
            harm_curves.append({"train_split": train, "variant": variant, "seed": seed,
                                "eval_split": split, "video_only_active": vo_active, "levels": levels})

    report_dir = output_path(cfg, "reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "summary.json",
               {"summaries": summaries, "comparisons": comparisons, "evidence_controls": controls,
                "harm_curves": harm_curves,
                "missing_warning": "Results are valid only when every planned seed/split is present."})
    if summaries:
        with (report_dir / "summary.csv").open("w", newline="") as handle:
            fieldnames = sorted({key for row in summaries for key in row})
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)
    lines = ["# Results", "", f"Generated: {datetime.now().isoformat()}", "",
             "Primary endpoint: **exact** structured match (seed 17).", "",
             "## Runs", "",
             "| Train | Test | Variant | Epoch | Evidence | Decoding | Bank | Seed | Exact | Active | Status | Verb | Manip | Target | Tool | Strict JSON |",
             "|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        epoch = row.get("epoch")
        epoch_s = "" if epoch is None or epoch == "" else str(epoch)
        lines.append(
            f"| {row['train_split']} | {row['eval_split']} | {row['variant']} | {epoch_s} | {row['evidence_mode']} "
            f"| {row.get('decoding') or ''} | {row.get('bank_mode') or ''} | {row['seed']} | {row['exact_accuracy']:.4f} | {row['active_composite_accuracy']:.4f} "
            f"| {row['status_micro']:.4f} | {row['verb_micro']:.4f} | {row['manipulated_object_micro']:.4f} "
            f"| {row['target_object_micro']:.4f} | {row['tool_micro']:.4f} | {row['strict_json_rate']:.4f} |"
        )
    lines += ["", "## Paired comparisons (variant vs its declared baseline) — exact primary", "",
              "| Train/Test | Variant vs Baseline | Decoding | Seed | Exact effect | 95% cluster CI | p (exact) | Active effect |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        exact = row["exact"]
        active = row["active_composite"]
        lines.append(f"| {row['train_split']}/{row['eval_split']} | {row['variant']} vs {row['baseline']} "
                     f"| {row.get('decoding') or ''} | {row['seed']} | {exact['effect']:+.4f} | [{exact['ci95'][0]:+.4f}, {exact['ci95'][1]:+.4f}] "
                     f"| {exact.get('permutation_p', float('nan')):.4g} | {active['effect']:+.4f} |")
    # Residual on vs off within each parent (main matrix).
    by_key = {(s["train_split"], s["variant"], s["seed"], s["eval_split"], s["evidence_mode"],
               s.get("decoding", ""), s.get("bank_mode", ""), s.get("epoch")): s
              for s in summaries}
    residual_pairs = [("sft_e", "sft_e_res"), ("r_sft_e", "r_sft_e_res"), ("r_grpo_e", "r_grpo_e_res")]
    residual_lines = []
    for off, on in residual_pairs:
        for s in summaries:
            if s["variant"] != off or s["evidence_mode"] != "correct":
                continue
            key_on = (s["train_split"], on, s["seed"], s["eval_split"], "correct",
                      s.get("decoding", ""), s.get("bank_mode", ""), s.get("epoch"))
            if key_on not in by_key:
                continue
            other = by_key[key_on]
            residual_lines.append(
                f"| {s['train_split']}/{s['eval_split']} | {on} vs {off} | {s['seed']} "
                f"| {other['exact_accuracy'] - s['exact_accuracy']:+.4f} "
                f"| {other['exact_accuracy']:.4f} | {s['exact_accuracy']:.4f} |"
            )
    if residual_lines:
        lines += ["", "## Residual on vs off (exact)", "",
                  "| Train/Test | Pair | Seed | Δ exact (on−off) | Exact on | Exact off |",
                  "|---|---|---:|---:|---:|---:|", *residual_lines]
    if controls:
        lines += ["", "## Evidence-content controls (control minus correct; negative = evidence is read)", "",
                  "| Train/Test | Variant | Control | Seed | Exact effect | 95% cluster CI | Active effect |",
                  "|---|---|---|---:|---:|---:|---:|"]
        for row in controls:
            exact = row["exact"]
            active = row["active_composite"]
            lines.append(f"| {row['train_split']}/{row['eval_split']} | {row['variant']} | {row['control']} "
                         f"| {row['seed']} | {exact['effect']:+.4f} | [{exact['ci95'][0]:+.4f}, {exact['ci95'][1]:+.4f}] "
                         f"| {active['effect']:+.4f} |")
    if harm_curves:
        lines += ["", "## Harm curves (utilization without fragility)", "",
                  "| Train/Test | Variant | Seed | Level | Active | Δ vs video_only | harm |",
                  "|---|---|---:|---|---:|---:|---:|"]
        for curve in harm_curves:
            for level, stats in curve["levels"].items():
                lines.append(
                    f"| {curve['train_split']}/{curve['eval_split']} | {curve['variant']} "
                    f"| {curve['seed']} | {level} | {stats['active_composite']:.4f} "
                    f"| {stats['delta_vs_video_only']:+.4f} | {stats['harm']:.4f} |"
                )
    (report_dir / "results.md").write_text("\n".join(lines) + "\n")
    log_path = repo_path(cfg, "outputs/log.md")
    if log_path.exists():
        with log_path.open("a") as handle:
            handle.write(f"\n### Automated analysis {datetime.now().isoformat()}\n\n")
            handle.write(f"See `{(report_dir / 'results.md').relative_to(repo_path(cfg, '.'))}`. "
                         f"Parsed {len(files)} prediction files.\n")
    print(f"analyzed {len(files)} prediction files -> {report_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("command", choices=["aggregate"])
    args = parser.parse_args()
    aggregate(load_config(args.config))


if __name__ == "__main__":
    main()
