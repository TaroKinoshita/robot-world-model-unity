"""
P7: 量産したデータの偏りを点検する(学習の前に、データの作り方を直すかどうか決めるため)。

見るもの:
  1. 候補の種類 × ラベル(種類だけでラベルが当たってしまわないか)
  2. matched pair(片方だけ衝突する組が何組あるか)
  3. 円柱の高さ・置き方の分布
  4. 追従誤差(planned と実際の TCP のずれ)
  5. 生成に失敗した seed の割合・外した候補の理由
  6. その他(倒れた・ロボット↔机・A/B の不安定・検証)

出力: <root>/inspection.md と inspection.json
使い方: python inspect_batch.py --episodes Episodes/_p7_20scenes
"""
import argparse
import json
import math
import re
import statistics as stx
from collections import Counter, defaultdict
from pathlib import Path

KINDS = ["matched_pair", "goal_dir", "random", "toward_secondary"]
KIND_JA = {"matched_pair": "matched pair", "goal_dir": "ゴール方向", "random": "ランダム", "toward_secondary": "円柱の方向"}


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def kind_of(c):
    note = (c.get("note") or "").lower()
    if c.get("matched_pair_id"):
        return "matched_pair"
    if "secondary" in note:
        return "toward_secondary"
    if "goal" in note:
        return "goal_dir"
    if "random" in note:
        return "random"
    i = c.get("candidate_index", -1)
    return "matched_pair" if i < 4 else "goal_dir" if i < 10 else "random" if i < 13 else "toward_secondary"


def pct(a, b):
    return f"{a}/{b}({100 * a / b:.0f}%)" if b else "—"


def q(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def collect(root):
    eps, scenes = [], []
    for sd in sorted(root.glob("scene_*")):
        if not (sd / "candidates.json").is_file():
            continue
        cj = _json(sd / "candidates.json")
        task = cj.get("task_variant")
        gen = _json(sd / "scene_generation.json") if (sd / "scene_generation.json").is_file() else {}
        val = _json(sd / "validation.json") if (sd / "validation.json").is_file() else {}
        scenes.append({"scene": sd.name, "task": task, "pair": cj.get("pair_id"), "gen": gen,
                       "val_passed": val.get("passed"), "val_failed": val.get("num_failed")})
        for c in cj["candidates"]:
            cd = sd / "candidates" / c["candidate_id"]
            fs = cd / "final" / "final_state.json"
            em = cd / "executed" / "executed_meta.json"
            if not fs.is_file():
                continue
            lab = _json(fs).get("labels", {})
            tr = (_json(em).get("tracking") or {}) if em.is_file() else {}
            eps.append({
                "scene": sd.name, "task": task, "pair": cj.get("pair_id"), "cid": c["candidate_id"],
                "kind": kind_of(c), "mp": c.get("matched_pair_id"), "branch": c.get("branch"),
                "angle": c.get("push_angle_deg"), "length": c.get("push_length_m"),
                "goal": lab.get("goal_reached"), "goal_dist": lab.get("goal_distance_m"),
                "coll": lab.get("secondary_collision"),
                "robot_sec": (lab.get("first_robot_secondary_step") or -1) >= 0 if task == "B" else None,
                "target_sec": (lab.get("first_target_secondary_step") or -1) >= 0 if task == "B" else None,
                "fell": lab.get("target_fell"), "contact": lab.get("target_contact"),
                "table": lab.get("robot_table_contact"),
                "tcp_err_max": tr.get("max_tcp_err_m"), "tcp_err_rms": tr.get("rms_tcp_err_m"),
                "joint_err_max": tr.get("max_joint_err_rad"),
                "mask_px": (_json(fs).get("target_mask") or {}).get("pixels"),
            })
    return eps, scenes


def type_predictability(rows, key):
    """種類ごとの多数派で当てたときの正解率 vs 全体の多数派で当てたときの正解率"""
    rows = [r for r in rows if r[key] is not None]
    if not rows:
        return None
    n = len(rows)
    overall = Counter(r[key] for r in rows).most_common(1)[0][1] / n
    by = defaultdict(list)
    for r in rows:
        by[r["kind"]].append(r[key])
    hit = sum(Counter(v).most_common(1)[0][1] for v in by.values())
    pure = [k for k, v in by.items() if len(set(v)) == 1]
    return {"n": n, "baseline_acc": overall, "by_type_acc": hit / n, "pure_types": pure}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    ap.add_argument("--episodes", default="Episodes")
    args = ap.parse_args()
    project = Path(args.project) if args.project else Path(__file__).resolve().parents[2]
    root = Path(args.episodes)
    if not root.is_absolute():
        root = project / root
    eps, scenes = collect(root)
    A = [e for e in eps if e["task"] == "A"]
    B = [e for e in eps if e["task"] == "B"]
    nB_scenes = sum(1 for s in scenes if s["task"] == "B")
    L, J = [], {"root": str(root), "num_episodes": len(eps), "num_scenes": len(scenes)}
    w = L.append

    w(f"# 量産データの点検（{root.name}）\n")
    w(f"- scene {len(scenes)}（ペア {nB_scenes}）、episode {len(eps)}（A {len(A)}・B {len(B)}）\n")

    # ---- 1. 種類 × ラベル ----
    w("## 1. 候補の種類 × ラベル\n")
    w("| 種類 | 本数(B) | goal True(A) | goal True(B) | 衝突(B) | ロボ→円柱 | ターゲット→円柱 | 倒れた(A+B) |")
    w("|---|---|---|---|---|---|---|---|")
    J["by_kind"] = {}
    for k in KINDS + ["all"]:
        a = A if k == "all" else [e for e in A if e["kind"] == k]
        b = B if k == "all" else [e for e in B if e["kind"] == k]
        if not b and not a:
            continue
        row = {
            "n_B": len(b), "goal_A": sum(bool(e["goal"]) for e in a), "n_A": len(a),
            "goal_B": sum(bool(e["goal"]) for e in b), "coll_B": sum(bool(e["coll"]) for e in b),
            "robot_sec": sum(bool(e["robot_sec"]) for e in b), "target_sec": sum(bool(e["target_sec"]) for e in b),
            "fell": sum(bool(e["fell"]) for e in a + b),
        }
        J["by_kind"][k] = row
        name = "**合計**" if k == "all" else KIND_JA[k]
        w(f"| {name} | {len(b)} | {pct(row['goal_A'], len(a))} | {pct(row['goal_B'], len(b))} | {pct(row['coll_B'], len(b))} | "
          f"{row['robot_sec']} | {row['target_sec']} | {pct(row['fell'], len(a) + len(b))} |")
    w("")
    w("種類だけでラベルを当てたら何 % 当たるか（各種類の多数派を答える。全体の多数派を答えるのが基準）：\n")
    w("| ラベル | 基準（全体の多数派） | 種類だけで当てる | 全部同じ答えの種類 |")
    w("|---|---|---|---|")
    J["type_predictability"] = {}
    for name, rows, key in [("goal(A)", A, "goal"), ("goal(B)", B, "goal"), ("衝突(B)", B, "coll")]:
        r = type_predictability(rows, key)
        if r is None:
            continue
        J["type_predictability"][name] = r
        w(f"| {name} | {100 * r['baseline_acc']:.0f}% | {100 * r['by_type_acc']:.0f}% | "
          f"{', '.join(KIND_JA[k] for k in r['pure_types']) or 'なし'} |")
    w("")
    # 種類 × (goal, 衝突) の組み合わせ(B)
    w("Task B の (goal, 衝突) の組み合わせ：\n")
    w("| 種類 | goal○衝突× | goal○衝突○ | goal×衝突○ | goal×衝突× |")
    w("|---|---|---|---|---|")
    for k in KINDS:
        b = [e for e in B if e["kind"] == k]
        if not b:
            continue
        c = Counter((bool(e["goal"]), bool(e["coll"])) for e in b)
        w(f"| {KIND_JA[k]} | {c[(True, False)]} | {c[(True, True)]} | {c[(False, True)]} | {c[(False, False)]} |")
    w("")
    # シーンごとのばらつき
    w("シーンごと（Task B）の goal / 衝突の本数（16 本中）：\n")
    per = defaultdict(lambda: [0, 0])
    for e in B:
        per[e["scene"]][0] += bool(e["goal"])
        per[e["scene"]][1] += bool(e["coll"])
    gs = [v[0] for v in per.values()]
    cs = [v[1] for v in per.values()]
    if gs:
        w(f"- goal：最小 {min(gs)}・中央 {stx.median(gs)}・最大 {max(gs)}、0 本のシーン {sum(g == 0 for g in gs)}")
        w(f"- 衝突：最小 {min(cs)}・中央 {stx.median(cs)}・最大 {max(cs)}、0 本のシーン {sum(c == 0 for c in cs)}\n")
    J["per_scene_B"] = {k: {"goal": v[0], "coll": v[1]} for k, v in per.items()}

    # ---- 2. matched pair ----
    w("## 2. matched pair（Task B）\n")
    mp = defaultdict(dict)
    for e in B:
        if e["mp"]:
            mp[(e["scene"], e["mp"])][e["branch"]] = bool(e["coll"])
    cnt = Counter()
    per_scene_one = defaultdict(int)
    for (s, m), d in mp.items():
        v = list(d.values())
        kind = "片方だけ" if len(v) == 2 and v[0] != v[1] else "両方" if all(v) else "どちらも当たらない"
        cnt[kind] += 1
        if kind == "片方だけ":
            per_scene_one[s] += 1
    scenes_B = sorted({e["scene"] for e in B})
    dist = Counter(per_scene_one.get(s, 0) for s in scenes_B)
    w(f"- 組の数 {len(mp)}：片方だけ衝突 {cnt['片方だけ']}・両方 {cnt['両方']}・どちらも当たらない {cnt['どちらも当たらない']}")
    w(f"- シーンあたりの「片方だけ衝突」の組：" + "、".join(f"{k} 組 = {dist[k]} シーン" for k in sorted(dist)))
    br = Counter(e["branch"] for e in B if e["mp"] and e["coll"])
    w(f"- 衝突した側の枝：" + "、".join(f"{k} {v}" for k, v in br.items()) + "\n")
    J["matched_pair"] = {"pairs": len(mp), **cnt, "per_scene_one_sided": dict(dist), "collided_branch": dict(br)}

    # ---- 3. 円柱 ----
    w("## 3. 円柱（secondary）の置き方\n")
    gens = [s["gen"] for s in scenes if s["task"] == "B" and s["gen"]]
    hs = [round(100 * (g.get("secondary") or {}).get("height_m", 0)) for g in gens]
    hc = Counter(hs)
    w("| 高さ (cm) | " + " | ".join(str(h) for h in range(24, 42, 2)) + " |")
    w("|---|" + "---|" * 9)
    w("| シーン数 | " + " | ".join(str(hc.get(h, 0)) for h in range(24, 42, 2)) + " |\n")
    ov = [1000 * (g.get("secondary") or {}).get("arm_overlap_m", 0) for g in gens]
    oh = [(g.get("secondary") or {}).get("predicted_other_robot_hits", 0) for g in gens]
    ob = Counter((g.get("secondary") or {}).get("overlapping_branch") for g in gens)
    opair = Counter((g.get("secondary") or {}).get("placed_for_pair") for g in gens)
    if ov:
        w(f"- 腕の重なり：{min(ov):.1f}〜{max(ov):.1f} mm（中央 {stx.median(ov):.1f}）")
        w(f"- ほかの候補への当たり予測：" + "、".join(f"{k} 本 = {v}" for k, v in sorted(Counter(oh).items())))
        w(f"- 当たる側の枝：" + "、".join(f"{k} {v}" for k, v in ob.items()) + "、対象の組：" + "、".join(f"{k} {v}" for k, v in opair.items()) + "\n")
    J["secondary"] = {"height_cm": dict(hc), "overlap_mm": ov, "other_hits": dict(Counter(oh)), "branch": dict(ob)}

    # ---- 4. 追従誤差 ----
    w("## 4. 追従誤差（TCP、planned と実際のずれの最大）\n")
    errs = [1000 * e["tcp_err_max"] for e in eps if e["tcp_err_max"] is not None]
    if errs:
        w(f"- 全 {len(errs)} episode：中央 {stx.median(errs):.2f} mm・95% {q(errs, .95):.2f} mm・最大 {max(errs):.2f} mm")
        w(f"- 3 mm 超 {sum(x > 3 for x in errs)}・5 mm 超 {sum(x > 5 for x in errs)}・10 mm 超 {sum(x > 10 for x in errs)}")
        worst = sorted((e for e in eps if e["tcp_err_max"] is not None), key=lambda e: -e["tcp_err_max"])[:5]
        w("- 大きい順：" + "、".join(f"{e['scene']}/{e['cid']}（{KIND_JA[e['kind']]}）{1000 * e['tcp_err_max']:.1f} mm" for e in worst))
        w("\n| 種類 | 中央 | 95% | 最大 |")
        w("|---|---|---|---|")
        for k in KINDS:
            x = [1000 * e["tcp_err_max"] for e in eps if e["kind"] == k and e["tcp_err_max"] is not None]
            if x:
                w(f"| {KIND_JA[k]} | {stx.median(x):.2f} | {q(x, .95):.2f} | {max(x):.2f} |")
        w("")
    J["tracking_mm"] = {"median": stx.median(errs) if errs else None, "p95": q(errs, .95) if errs else None,
                        "max": max(errs) if errs else None, "over3": sum(x > 3 for x in errs), "over5": sum(x > 5 for x in errs)}

    # ---- 5. 生成 ----
    w("## 5. 生成の失敗・外した候補\n")
    summ = sorted((root / "_batch").glob("batch_summary_*.json")) if (root / "_batch").is_dir() else []
    if summ:
        st = _json(summ[-1])
        recs = st.get("records", [])
        fails = [r for r in recs if r.get("status") != "ok"]
        w(f"- seed を試した数 {len(recs)}、成功 {len(recs) - len(fails)}、失敗 {len(fails)}（{100 * len(fails) / max(1, len(recs)):.0f}%）")
        for r in fails:
            w(f"  - seed {r['seed']}：{r['stage']}：{r['reason']}")
        pt = [r.get("playMin", 0) for r in recs if r.get("status") == "ok"]
        if pt:
            w(f"- Play の時間：中央 {stx.median(pt):.1f} 分・最大 {max(pt):.1f} 分")
        J["batch"] = {"tried": len(recs), "failed": len(fails), "fail_detail": fails}
    else:
        w("- バッチの記録なし（手で作ったデータ）")
    sa = Counter(g.get("attempts", {}).get("scene", 1) for g in gens)
    rj = Counter()
    for g in gens:
        rj.update(g.get("attempts", {}).get("reject_reasons", {}))
    w(f"- シーンの生成が何回目で成功したか：" + "、".join(f"{k} 回目 = {v}" for k, v in sorted(sa.items())))
    w(f"- 外した候補の理由（合計）：" + "、".join(f"{k} {v}" for k, v in rj.most_common()) + "\n")
    J["scene_attempts"] = dict(sa)
    J["reject_reasons"] = dict(rj)

    # ---- 5b. 円柱がターゲットを隠す(最終画像) ----
    w("## 5b. 円柱がターゲットを隠す（最終画像、Task B の target_mask の画素 ÷ 同じ候補の Task A）\n")
    amap = {(e["pair"], e["cid"]): e["mask_px"] for e in A}
    ratios = []
    for e in B:
        pa = amap.get((e["pair"], e["cid"]))
        if pa and e["mask_px"] is not None:
            ratios.append((e["mask_px"] / pa, e))
    if ratios:
        rs = [r for r, _ in ratios]
        w(f"- 見えている割合：中央 {100 * stx.median(rs):.0f}%、半分未満 {sum(r < .5 for r in rs)} 本、25% 未満 {sum(r < .25 for r in rs)} 本、0 {sum(r == 0 for r in rs)} 本（全 {len(rs)} 本）")
        low = sorted(ratios, key=lambda t: t[0])[:5]
        w("- 小さい順：" + "、".join(f"{e['scene']}/{e['cid']}（{KIND_JA[e['kind']]}）{100 * r:.0f}%" for r, e in low) + "\n")
        J["occlusion_final"] = {"median": stx.median(rs), "lt50": sum(r < .5 for r in rs), "lt25": sum(r < .25 for r in rs), "zero": sum(r == 0 for r in rs)}

    # ---- 6. その他 ----
    w("## 6. その他\n")
    unstable = []
    for pc in sorted(root.glob("pairs/*/pair_check.json")):
        d = _json(pc)
        unstable += [(d["pair_id"], u) for u in (d.get("sim_unstable_candidates") or [])]
    vfail = [s["scene"] for s in scenes if s["val_passed"] is not True]
    pfail = [pc.parent.name for pc in root.glob("pairs/*/pair_check.json") if _json(pc).get("passed") is not True]
    w(f"- 検証：scene FAIL {len(vfail)} {vfail if vfail else ''}、ペア FAIL {len(pfail)} {pfail if pfail else ''}")
    w(f"- A/B の不安定（触れていないのに 1 mm 超ずれた）：{len(unstable)} {unstable if unstable else ''}")
    w(f"- 倒れた：{sum(bool(e['fell']) for e in eps)}、ロボット↔机：{sum(bool(e['table']) for e in eps)}")
    w(f"- ターゲットに触れなかった：{sum(e['contact'] is False for e in eps)}")
    J.update({"validation_failed": vfail, "pair_failed": pfail, "sim_unstable": unstable,
              "fell": sum(bool(e['fell']) for e in eps), "robot_table": sum(bool(e['table']) for e in eps)})

    (root / "inspection.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (root / "inspection.json").write_text(json.dumps(J, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
