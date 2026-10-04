"""
P4-A: ラベル分布の点検(パイロット用)

Episodes/ の全 scene を読んで、Task ごとに goal・衝突などのラベルの偏りと、matched pair の手先・腕の差をまとめる。
build_metadata.py → validate_episode.py → check_task_pair.py のあとに実行する。

実行(プロジェクト直下で):
    python Assets/Python/summarize_labels.py

出力:
    Episodes/label_summary.json     全 episode のラベル一覧と集計
    Episodes/label_summary.md       人が読む用の表
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _cols(p, names):
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return np.array([[float(r[n]) for n in names] for r in rows])


def collect(root: Path):
    eps = []
    for sd in sorted(root.glob("scene_*")):
        if not (sd / "metadata.json").is_file():
            continue
        meta = _json(sd / "metadata.json")
        val = _json(sd / "validation.json") if (sd / "validation.json").is_file() else {}
        cands = {c["candidate_id"]: c for c in _json(sd / "candidates.json")["candidates"]}
        for e in meta["episodes"]:
            cid = e["candidate_id"]
            lab = e.get("labels") or {}
            c = cands.get(cid, {})
            eps.append({
                "scene_id": sd.name, "task_variant": e.get("task_variant"), "pair_id": e.get("pair_id"),
                "candidate_id": cid, "note": c.get("note"), "push_angle_deg": c.get("push_angle_deg"),
                "matched_pair_id": c.get("matched_pair_id"), "branch": c.get("branch"),
                "scene_validation_passed": val.get("passed"),
                "goal_reached": lab.get("goal_reached"), "goal_distance_mm": round(lab["goal_distance_m"] * 1000, 1) if "goal_distance_m" in lab else None,
                "target_contact": lab.get("target_contact"), "target_fell": lab.get("target_fell"),
                "secondary_collision": lab.get("secondary_collision"),
                "first_robot_secondary_step": lab.get("first_robot_secondary_step"),
                "first_target_secondary_step": lab.get("first_target_secondary_step"),
                "max_tcp_tracking_mm": round(e["tracking"]["max_tcp_err_m"] * 1000, 1) if e.get("tracking") else None,
                "dir": str(sd / "candidates" / cid),
            })
    return eps


def matched_pairs(eps):
    """同じ scene・同じ matched_pair_id の候補同士で、手先と腕の差(planned)を測る"""
    groups = defaultdict(list)
    for e in eps:
        if e["matched_pair_id"]:
            groups[(e["scene_id"], e["matched_pair_id"])].append(e)
    out = []
    for (sid, mp), g in sorted(groups.items()):
        if len(g) != 2:
            out.append({"scene_id": sid, "matched_pair_id": mp, "error": f"候補が {len(g)} 本(2 本のはず)"})
            continue
        a, b = g
        eef = ["tcp_x", "tcp_y", "tcp_z"]
        rot = ["rot_x", "rot_y", "rot_z", "rot_w"]
        pa, pb = _cols(Path(a["dir"]) / "actions" / "eef_numeric.csv", eef), _cols(Path(b["dir"]) / "actions" / "eef_numeric.csv", eef)
        ra, rb = _cols(Path(a["dir"]) / "actions" / "eef_numeric.csv", rot), _cols(Path(b["dir"]) / "actions" / "eef_numeric.csv", rot)
        ang = 2 * np.degrees(np.arccos(np.clip(np.abs((ra * rb).sum(1)), 0, 1)))
        links = {}
        for k in ["forearm_link", "wrist_1_link", "wrist_2_link"]:
            cols = [f"{k}_{x}" for x in "xyz"]
            d = np.linalg.norm(_cols(Path(a["dir"]) / "actions" / "fullbody_numeric.csv", cols)
                               - _cols(Path(b["dir"]) / "actions" / "fullbody_numeric.csv", cols), axis=1)
            links[k] = {"mean_cm": round(float(d.mean()) * 100, 1), "max_cm": round(float(d.max()) * 100, 1)}
        out.append({"scene_id": sid, "task_variant": a["task_variant"], "matched_pair_id": mp,
                    "candidates": {a["branch"] or a["candidate_id"]: a["candidate_id"], b["branch"] or b["candidate_id"]: b["candidate_id"]},
                    "tcp_diff_max_mm": round(float(np.linalg.norm(pa - pb, axis=1).max()) * 1000, 3),
                    "rot_diff_max_deg": round(float(ang.max()), 3), "link_diff": links,
                    "secondary_collision": {a["branch"] or a["candidate_id"]: a["secondary_collision"],
                                            b["branch"] or b["candidate_id"]: b["secondary_collision"]},
                    "only_one_collides": (a["secondary_collision"] is not None
                                          and bool(a["secondary_collision"]) != bool(b["secondary_collision"]))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    ap.add_argument("--episodes", default="Episodes")
    args = ap.parse_args()
    project = Path(args.project) if args.project else Path(__file__).resolve().parents[2]
    root = project / args.episodes
    eps = collect(root)
    if not eps:
        sys.exit(f"episode が見つからない: {root}")

    by_task = defaultdict(list)
    for e in eps:
        by_task[e["task_variant"]].append(e)
    summary = {}
    for t, es in sorted(by_task.items(), key=lambda kv: str(kv[0])):
        summary[str(t)] = {
            "episodes": len(es),
            "scenes": sorted({e["scene_id"] for e in es}),
            "validation_passed_scenes": sorted({e["scene_id"] for e in es if e["scene_validation_passed"]}),
            "goal_reached": dict(Counter(str(e["goal_reached"]) for e in es)),
            "secondary_collision": dict(Counter(str(e["secondary_collision"]) for e in es)),
            "target_contact": dict(Counter(str(e["target_contact"]) for e in es)),
            "target_fell": dict(Counter(str(e["target_fell"]) for e in es)),
            "goal_and_no_collision": sum(1 for e in es if e["goal_reached"] and not e["secondary_collision"]),
            "max_tcp_tracking_mm": max(e["max_tcp_tracking_mm"] or 0 for e in es),
        }
    mps = matched_pairs(eps)
    out = {"num_episodes": len(eps), "by_task": summary, "matched_pairs": mps,
           "episodes": [{k: v for k, v in e.items() if k != "dir"} for e in eps]}
    (root / "label_summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    L = ["# ラベル分布(パイロット)", "", f"episode 数: {len(eps)}", "", "## Task ごとの集計", "",
         "| Task | episodes | 検証 PASS の scene | goal True / False | 衝突 True / False / 対象なし | target 転倒 | goal かつ衝突なし | 追従誤差 最大 |",
         "|---|---|---|---|---|---|---|---|"]
    for t, s in summary.items():
        g, c = s["goal_reached"], s["secondary_collision"]
        L.append(f"| {t} | {s['episodes']} | {len(s['validation_passed_scenes'])}/{len(s['scenes'])} | "
                 f"{g.get('True', 0)} / {g.get('False', 0)} | {c.get('True', 0)} / {c.get('False', 0)} / {c.get('None', 0)} | "
                 f"{s['target_fell'].get('True', 0)} | {s['goal_and_no_collision']} | {s['max_tcp_tracking_mm']} mm |")
    L += ["", "## episode 一覧", "", "| scene | Task | 候補 | 押す方向 | matched pair | goal(距離) | 衝突 | 初回 robot→secondary | 初回 target→secondary |",
          "|---|---|---|---|---|---|---|---|---|"]
    for e in eps:
        mp = f"{e['matched_pair_id']} ({e['branch']})" if e["matched_pair_id"] else ""
        L.append(f"| {e['scene_id']} | {e['task_variant']} | {e['candidate_id']} | {e['push_angle_deg']}° | {mp} | "
                 f"{e['goal_reached']}({e['goal_distance_mm']} mm) | {e['secondary_collision']} | {e['first_robot_secondary_step']} | {e['first_target_secondary_step']} |")
    if mps:
        L += ["", "## matched pair", "", "| scene | Task | pair | TCP の差 最大 | 向きの差 最大 | 前腕の差 平均 | 手首 1 の差 平均 | 衝突 | 片方だけ衝突 |",
              "|---|---|---|---|---|---|---|---|---|"]
        for m in mps:
            if "error" in m:
                L.append(f"| {m['scene_id']} | | {m['matched_pair_id']} | {m['error']} | | | | | |")
                continue
            L.append(f"| {m['scene_id']} | {m['task_variant']} | {m['matched_pair_id']} | {m['tcp_diff_max_mm']} mm | {m['rot_diff_max_deg']}° | "
                     f"{m['link_diff']['forearm_link']['mean_cm']} cm | {m['link_diff']['wrist_1_link']['mean_cm']} cm | "
                     f"{m['secondary_collision']} | {m['only_one_collides']} |")
    (root / "label_summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
