"""
P2: Task A / B ペアの比較(A = secondary 物体なし、B = あり)

チェック内容(pair ごと)
  1. 計画が同じ : planned_trajectory.csv と action(eef/fullbody numeric、raster 画像)がバイト単位で同一、
                  planned_trajectory.json と actions_meta.json は scene_id 以外が同一
  2. 条件が同じ : カメラ行列、ターゲットの初期 pose、ロボットの初期関節角、ゴール・判定パラメータが同一
  3. 初期画像   : robot-free 初期画像の A/B の差が secondary 物体のせいだけと追えるか
                  - semantic・depth の差 ⊆ B の secondary の画素
                  - secondary から shadow_px より外では、RGB の差が far_tol 階調以下(8bit)
                    (影・照り返しは近くで大きく変わる。遠くは HDRP の間接光などで ±1〜2 階調ゆれる)
  4. 結果の比較 : 候補ごとに 最終位置・goal・接触/衝突ラベル・初回接触 step を並べる(合否ではなく記録)
                  最終画像を A/B で横に並べた PNG を保存

実行(プロジェクト直下で):
    python Assets/Python/check_task_pair.py                  # Episodes/ の全 pair
    python Assets/Python/check_task_pair.py --pair pair_0000

出力: Episodes/pairs/<pair_id>/
    pair_check.json            チェック結果と比較表
    comparison.md              比較表(人が読む用)
    initial_robot_free.png     [A | B | RGB の差(赤) / secondary(黄)]
    <cid>_final.png            [A with_robot | B with_robot | A robot_free | B robot_free]
全部 PASS なら exit code 0。
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_episode import Report, _json, find_pair  # noqa: E402

SECONDARY_RGB = (255, 255, 0)   # semantic の secondary_object の色


def _drop(d, keys=("scene_id",)):
    if isinstance(d, dict):
        return {k: _drop(v, keys) for k, v in d.items() if k not in keys}
    if isinstance(d, list):
        return [_drop(v, keys) for v in d]
    return d


def same_bytes(a: Path, b: Path):
    return a.is_file() and b.is_file() and a.read_bytes() == b.read_bytes()


def check_plans(r, A, B, cids):
    for cid in cids:
        a, b = A / "candidates" / cid, B / "candidates" / cid
        r.check(f"{cid} 計画: planned_trajectory.csv がバイト単位で同一",
                same_bytes(a / "planned_trajectory.csv", b / "planned_trajectory.csv"))
        ja, jb = _json(a / "planned_trajectory.json"), _json(b / "planned_trajectory.json")
        r.check(f"{cid} 計画: planned_trajectory.json が scene_id 以外同一", _drop(ja) == _drop(jb))
        for f in ["eef_numeric.csv", "fullbody_numeric.csv"]:
            r.check(f"{cid} action: {f} がバイト単位で同一", same_bytes(a / "actions" / f, b / "actions" / f))
        r.check(f"{cid} action: actions_meta.json が scene_id 以外同一",
                _drop(_json(a / "actions" / "actions_meta.json")) == _drop(_json(b / "actions" / "actions_meta.json")))
        diff = []
        for sub in ["eef_raster", "fullbody_raster", "eef_depth", "fullbody_depth"]:
            if not (a / "actions" / sub).is_dir() and not (b / "actions" / sub).is_dir():
                continue   # 古いデータ(depth なし)
            na = sorted(p.name for p in (a / "actions" / sub).glob("*.png"))
            nb = sorted(p.name for p in (b / "actions" / sub).glob("*.png"))
            if na != nb:
                diff.append(f"{sub}: 枚数 {len(na)} vs {len(nb)}")
                continue
            diff += [f"{sub}/{n}" for n in na if not same_bytes(a / "actions" / sub / n, b / "actions" / sub / n)]
        r.check(f"{cid} action: raster 画像が全部バイト単位で同一", not diff, str(diff[:5]))


def check_conditions(r, A, B, sA, sB):
    ca, cb = _json(A / "camera.json"), _json(B / "camera.json")
    r.check("条件: カメラ行列(K・T_world_to_cam)が同一",
            ca.get("K") == cb.get("K") and ca.get("T_world_to_cam_cv") == cb.get("T_world_to_cam_cv"))
    tgt = _json(next((A / "candidates").iterdir()) / "planned_trajectory.json")["target"]["name"]
    oa = next(o for o in sA["objects"] if o["name"] == tgt)
    ob = next(o for o in sB["objects"] if o["name"] == tgt)
    r.check("条件: ターゲットの初期 pose が同一",
            oa["position_world"] == ob["position_world"] and oa["rotation_world_xyzw"] == ob["rotation_world_xyzw"],
            f"A={oa['position_world']}, B={ob['position_world']}")
    r.check("条件: ロボットの初期関節角が同一", sA["robot"]["joints"] == sB["robot"]["joints"])
    others_a = sorted(o["name"] for o in sA["objects"] if o["name"] != tgt)
    others_b = sorted(o["name"] for o in sB["objects"] if o["name"] != tgt)
    sec = sB["task"]["secondary_object"]
    r.check("条件: 物体の違いは secondary だけ(A にはターゲット以外なし、B は secondary だけ追加)",
            others_a == [] and others_b == [sec], f"A の他物体={others_a}, B の他物体={others_b}")
    fa = _json(next((A / "candidates").iterdir()) / "final" / "final_state.json")
    fb = _json(next((B / "candidates").iterdir()) / "final" / "final_state.json")
    r.check("条件: ゴール・判定パラメータが同一",
            fa["goal"] == fb["goal"] and fa["label_params"] == fb["label_params"])
    return tgt


def check_initial_images(r, A, B, out_dir, shadow_px, far_tol):
    ia, ib = A / "initial" / "robot_free", B / "initial" / "robot_free"
    rgb_a, rgb_b = cv2.imread(str(ia / "rgb.png"), cv2.IMREAD_COLOR), cv2.imread(str(ib / "rgb.png"), cv2.IMREAD_COLOR)
    sem_a, sem_b = cv2.imread(str(ia / "semantic.png"), cv2.IMREAD_COLOR), cv2.imread(str(ib / "semantic.png"), cv2.IMREAD_COLOR)
    dep_a, dep_b = cv2.imread(str(ia / "depth.exr"), cv2.IMREAD_UNCHANGED), cv2.imread(str(ib / "depth.exr"), cv2.IMREAD_UNCHANGED)

    sec = np.all(sem_b[:, :, ::-1] == SECONDARY_RGB, axis=2)
    r.check("初期画像: B の semantic に secondary が写っている / A には無い",
            sec.sum() > 0 and not np.all(sem_a[:, :, ::-1] == SECONDARY_RGB, axis=2).any(), f"B の secondary {int(sec.sum())} px")

    sem_diff = np.any(sem_a != sem_b, axis=2)
    r.check("初期画像: semantic の差 ⊆ secondary の画素", not (sem_diff & ~sec).any(),
            f"差 {int(sem_diff.sum())} px、うち secondary の外 {int((sem_diff & ~sec).sum())} px")
    da, db = dep_a[:, :, 2] if dep_a.ndim == 3 else dep_a, dep_b[:, :, 2] if dep_b.ndim == 3 else dep_b
    dep_diff = np.abs(da - db) > 1e-6
    r.check("初期画像: depth の差 ⊆ secondary の画素", not (dep_diff & ~sec).any(),
            f"差 {int(dep_diff.sum())} px、うち secondary の外 {int((dep_diff & ~sec).sum())} px")

    absd = np.abs(rgb_a.astype(int) - rgb_b.astype(int)).max(axis=2)   # 画素ごとの最大階調差
    rgb_diff = absd > 0
    dist = cv2.distanceTransform((~sec).astype(np.uint8), cv2.DIST_L2, 5)   # secondary からの距離 px
    outside = rgb_diff & ~sec
    near = ~sec & (dist <= shadow_px)
    far = dist > shadow_px
    far_max = int(absd[far].max()) if far.any() else 0
    near_max = int(absd[near].max()) if near.any() else 0
    r.check(f"初期画像: secondary から {shadow_px}px より外の RGB の差 ≤ {far_tol} 階調(近くの影・照り返しは OK)",
            far_max <= far_tol,
            f"差のある画素 {int(rgb_diff.sum())}(secondary 上 {int((rgb_diff & sec).sum())}、外 {int(outside.sum())})。"
            f"{shadow_px}px 以内: 最大 {near_max} 階調、>3 階調 {int((absd[near] > 3).sum())} px / "
            f"{shadow_px}px より外: 最大 {far_max} 階調、>1 階調 {int((absd[far] > 1).sum())} px")

    vis = rgb_b.copy()
    vis[outside & (absd > 3)] = (0, 0, 255)   # 3 階調より大きい差(secondary の外)= 赤
    vis[outside & (absd > 0) & (absd <= 3)] = (255, 160, 0)   # 1〜3 階調の差 = 水色
    vis[sec] = (0, 255, 255)            # secondary = 黄
    cv2.imwrite(str(out_dir / "initial_robot_free.png"), np.hstack([rgb_a, rgb_b, vis]))
    return {"secondary_px": int(sec.sum()), "rgb_diff_px": int(rgb_diff.sum()), "rgb_diff_outside_secondary_px": int(outside.sum()),
            "rgb_diff_gt3_outside_secondary_px": int((outside & (absd > 3)).sum()),
            "rgb_max_diff_within_shadow_px": near_max, "rgb_max_diff_beyond_shadow_px": far_max,
            "shadow_px": shadow_px, "far_tol": far_tol,
            "semantic_diff_px": int(sem_diff.sum()), "depth_diff_px": int(dep_diff.sum())}


UNSTABLE_MM = 1.0   # 円柱に触れていないのに A/B の最終位置がこれ以上違えば「不安定」の印を付ける(FAIL にはしない)


def compare_results(A, B, cids, tgt, out_dir):
    rows = []
    for cid in cids:
        row = {"candidate_id": cid}
        for v, S in (("A", A), ("B", B)):
            d = S / "candidates" / cid
            f = _json(d / "final" / "final_state.json")
            em = _json(d / "executed" / "executed_meta.json")
            o = next(x for x in f["objects"] if x["name"] == tgt)
            lab = f["labels"]
            row[v] = {
                "target_final_position": [round(x, 4) for x in o["final_position_world"]],
                "target_displacement_mm": round(o["displacement_m"] * 1000, 1),
                "goal_reached": lab["goal_reached"], "goal_distance_mm": round(lab["goal_distance_m"] * 1000, 1),
                "target_fell": lab["target_fell"], "target_contact": lab["target_contact"],
                "secondary_collision": lab["secondary_collision"],
                "first_robot_target_step": lab["first_robot_target_step"],
                "first_robot_secondary_step": lab["first_robot_secondary_step"],
                "first_target_secondary_step": lab["first_target_secondary_step"],
                "max_tcp_tracking_err_mm": round(em["tracking"]["max_tcp_err_m"] * 1000, 1),
                "final_mask_px": f["target_mask"]["pixels"],
            }
        pa = np.array(row["A"]["target_final_position"]); pb = np.array(row["B"]["target_final_position"])
        row["final_position_diff_A_B_mm"] = round(float(np.linalg.norm(pa - pb)) * 1000, 1)
        # P4-B: B で円柱に触れていないのに A と結果が違う候補 = シミュレーションが不安定(わずかな計算の違いが大きく育った)
        row["sim_unstable"] = (not row["B"]["secondary_collision"]) and float(np.linalg.norm(pa - pb)) * 1000 > UNSTABLE_MM
        rows.append(row)

        imgs = [cv2.imread(str(S / "candidates" / cid / "final" / k / "rgb.png"), cv2.IMREAD_COLOR)
                for k in ("with_robot", "robot_free") for S in (A, B)]
        cv2.imwrite(str(out_dir / f"{cid}_final.png"), np.hstack(imgs))
    return rows


def write_md(out_dir, pair_id, A, B, init, rows, r):
    L = [f"# {pair_id}: Task A ({A.name}) vs Task B ({B.name})", "",
         f"チェック: {'PASS' if r.passed else 'FAIL'}({sum(i['pass'] for i in r.items)}/{len(r.items)})", "",
         "## 初期画像(robot-free)の差", "",
         f"secondary {init['secondary_px']} px / RGB の差 {init['rgb_diff_px']} px(secondary の外 {init['rgb_diff_outside_secondary_px']} px、"
         f"うち 3 階調超 {init['rgb_diff_gt3_outside_secondary_px']} px)/ {init['shadow_px']}px 以内の最大差 {init['rgb_max_diff_within_shadow_px']} 階調、"
         f"それより外の最大差 {init['rgb_max_diff_beyond_shadow_px']} 階調 / semantic の差 {init['semantic_diff_px']} px / depth の差 {init['depth_diff_px']} px", "",
         "![initial](initial_robot_free.png)(左 A / 中 B / 右: 黄 = secondary、赤 = 3 階調超の差、水色 = 1〜3 階調の差)", "",
         "## 結果", ""]
    keys = ["goal_reached", "goal_distance_mm", "target_displacement_mm", "target_fell", "target_contact",
            "secondary_collision", "first_robot_target_step", "first_robot_secondary_step", "first_target_secondary_step",
            "max_tcp_tracking_err_mm", "final_mask_px"]
    for row in rows:
        L += [f"### {row['candidate_id']}(最終位置の A/B 差 {row['final_position_diff_A_B_mm']} mm)", "",
              "| 項目 | Task A | Task B |", "|---|---|---|"]
        L += [f"| {k} | {row['A'][k]} | {row['B'][k]} |" for k in keys]
        L += ["", f"![{row['candidate_id']}]({row['candidate_id']}_final.png)", ""]
    (out_dir / "comparison.md").write_text("\n".join(L), encoding="utf-8")


def check_pair(root: Path, pair_id: str, shadow_px: int, far_tol: int):
    r = Report()
    pair = find_pair(root, pair_id)
    if set(pair) != {"A", "B"}:
        r.check("pair: A と B が両方ある", False, str({k: v.name for k, v in pair.items()}))
        print(f"== {pair_id}: FAIL(A/B が揃っていない: {sorted(pair)})")
        return False
    A, B = pair["A"], pair["B"]
    out_dir = root / "pairs" / pair_id
    out_dir.mkdir(parents=True, exist_ok=True)
    sA, sB = _json(A / "scene_initial.json"), _json(B / "scene_initial.json")
    r.check("pair: A と B が互いを paired_scene_id で指している",
            sA["task"]["paired_scene_id"] == B.name and sB["task"]["paired_scene_id"] == A.name)
    ca = [c["candidate_id"] for c in _json(A / "candidates.json")["candidates"] if c["planned_ok"]]
    cb = [c["candidate_id"] for c in _json(B / "candidates.json")["candidates"] if c["planned_ok"]]
    r.check("pair: 候補の一覧が同じ", ca == cb, f"A={ca}, B={cb}")
    cids = [c for c in ca if c in cb]

    check_plans(r, A, B, cids)
    tgt = check_conditions(r, A, B, sA, sB)
    init = check_initial_images(r, A, B, out_dir, shadow_px, far_tol)
    rows = compare_results(A, B, cids, tgt, out_dir)

    out = {"pair_id": pair_id, "A": A.name, "B": B.name, "passed": r.passed,
           "num_checks": len(r.items), "num_failed": sum(not i["pass"] for i in r.items),
           "checks": r.items, "initial_image_diff": init,
           "sim_unstable_threshold_mm": UNSTABLE_MM,
           "sim_unstable_candidates": [row["candidate_id"] for row in rows if row["sim_unstable"]],
           "results": rows}
    (out_dir / "pair_check.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    write_md(out_dir, pair_id, A, B, init, rows, r)

    print(f"== {pair_id}({A.name} = A, {B.name} = B): {'PASS' if r.passed else 'FAIL'}({len(r.items) - out['num_failed']}/{len(r.items)})")
    for i in r.items:
        print(f"  [{'OK' if i['pass'] else 'NG'}] {i['check']}" + (f"  … {i['detail']}" if i["detail"] else ""))
    for row in rows:
        a, b = row["A"], row["B"]
        print(f"  {row['candidate_id']}: goal A={a['goal_reached']}({a['goal_distance_mm']} mm) B={b['goal_reached']}({b['goal_distance_mm']} mm) | "
              f"secondary A={a['secondary_collision']} B={b['secondary_collision']} | "
              f"初回 robot→secondary B={b['first_robot_secondary_step']}, target→secondary B={b['first_target_secondary_step']} | "
              f"最終位置の差 {row['final_position_diff_A_B_mm']} mm")
    print(f"  不安定(円柱に触れていないのに A/B が {UNSTABLE_MM} mm 超ずれた): {out['sim_unstable_candidates']}")
    print(f"  出力: {out_dir}")
    return r.passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    ap.add_argument("--episodes", default="Episodes")
    ap.add_argument("--pair", default=None, help="pair_XXXX(省略時は全部)")
    ap.add_argument("--shadow-px", type=int, default=40, help="secondary から何 px 以内の RGB の差を影・照り返しとして許すか")
    ap.add_argument("--far-tol", type=int, default=3, help="それより外で許す RGB の差(8bit 階調)")
    args = ap.parse_args()
    project = Path(args.project) if args.project else Path(__file__).resolve().parents[2]
    root = project / args.episodes
    if args.pair:
        pairs = [args.pair]
    else:
        pairs = sorted({(_json(d / "scene_initial.json").get("task") or {}).get("pair_id")
                        for d in root.glob("scene_*") if (d / "scene_initial.json").is_file()} - {None})
    if not pairs:
        sys.exit(f"pair が見つからない: {root}")
    ok = [check_pair(root, p, args.shadow_px, args.far_tol) for p in pairs]
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
