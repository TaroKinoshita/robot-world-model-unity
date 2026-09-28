"""
TODO5-8: episode フォルダの自動検証

チェック内容
  A. 読み込み   : load_episode() で全ファイルを numpy に読み込めるか(完了条件そのもの)
  B. ファイル   : metadata.json の全パスが実在するか / raster 画像が actions_meta の raster_steps と一致するか / 画像サイズ
  C. 投影       : 初期・最終の target mask と、target の 3D 箱を camera.json で投影した形が一致するか
                  EEF raster の点の中心と、EEF numeric を投影した位置が一致するか
  D. ラベル     : final_state の target_contact / secondary_collision が contacts.csv から計算し直した値と一致するか
  E. 時系列     : planned / actions / executed の step 数・時刻・planned_step 対応、contacts の step 範囲、
                  final pose と executed 最終行の一致

実行(プロジェクト直下で):
    pip install numpy opencv-python
    python Assets/Python/validate_episode.py                  # Episodes/ の全 scene
    python Assets/Python/validate_episode.py --scene scene_0000

出力: Episodes/scene_XXXX/validation.json と、コンソールの PASS / FAIL 一覧。全部 PASS なら exit code 0。

別のスクリプトから episode を使うときは:
    from validate_episode import load_episode
    ep = load_episode("Episodes/scene_0000/candidates/c000")
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"   # cv2 の import より前

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- 読み込み

def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _csv(p):
    """数値列は float の numpy 配列、文字列列は list で返す"""
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    cols = {}
    for j, name in enumerate(header):
        vals = [r[j] for r in body]
        try:
            cols[name] = np.array([float(v) for v in vals])
        except ValueError:
            cols[name] = vals
    return header, cols


def _img(p, flags=cv2.IMREAD_UNCHANGED):
    im = cv2.imread(str(p), flags)
    if im is None:
        raise IOError(f"画像を読めない: {p}")
    return im


def load_episode(ep_dir):
    """1 episode(= scene_XXXX/candidates/cXXX)を numpy にまとめて読む"""
    ep_dir = Path(ep_dir)
    scene_dir = ep_dir.parents[1]
    ep = {"dir": ep_dir, "scene_dir": scene_dir}
    ep["scene"] = _json(scene_dir / "scene_initial.json")
    ep["camera"] = _json(scene_dir / "camera.json")
    ep["episode"] = _json(ep_dir / "episode.json")

    ep["planned"] = _json(ep_dir / "planned_trajectory.json")
    ep["planned_csv"] = _csv(ep_dir / "planned_trajectory.csv")

    ep["actions_meta"] = _json(ep_dir / "actions" / "actions_meta.json")
    ep["eef_numeric"] = _csv(ep_dir / "actions" / "eef_numeric.csv")
    ep["fullbody_numeric"] = _csv(ep_dir / "actions" / "fullbody_numeric.csv")
    steps = ep["actions_meta"]["raster_steps"]
    ep["eef_raster"] = np.stack([_img(ep_dir / "actions" / "eef_raster" / f"step_{s:04d}.png", cv2.IMREAD_GRAYSCALE) for s in steps])
    ep["fullbody_raster"] = np.stack([_img(ep_dir / "actions" / "fullbody_raster" / f"step_{s:04d}.png", cv2.IMREAD_GRAYSCALE) for s in steps])

    ep["executed_meta"] = _json(ep_dir / "executed" / "executed_meta.json")
    ep["executed"] = _csv(ep_dir / "executed" / "executed_trajectory.csv")
    ep["contacts"] = _csv(ep_dir / "executed" / "contacts.csv")

    ep["final"] = _json(ep_dir / "final" / "final_state.json")
    ep["final_mask"] = _img(ep_dir / "final" / "target_mask.png", cv2.IMREAD_GRAYSCALE)
    ep["final_rgb"] = _img(ep_dir / "final" / "robot_free" / "rgb.png", cv2.IMREAD_COLOR)
    ep["final_depth"] = _img(ep_dir / "final" / "robot_free" / "depth.exr")
    ep["final_semantic"] = _img(ep_dir / "final" / "robot_free" / "semantic.png", cv2.IMREAD_COLOR)

    init = scene_dir / "initial" / "robot_free"
    ep["initial_rgb"] = _img(init / "rgb.png", cv2.IMREAD_COLOR)
    ep["initial_depth"] = _img(init / "depth.exr")
    ep["initial_semantic"] = _img(init / "semantic.png", cv2.IMREAD_COLOR)
    return ep

# ---------------------------------------------------------------- 幾何

def quat_rotate(q, v):
    """q = [x, y, z, w](Unity と同じ並び)で v を回す"""
    u, w = np.asarray(q[:3], float), float(q[3])
    v = np.asarray(v, float)
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def project(cam, pts):
    K, T = np.array(cam["K"], float), np.array(cam["T_world_to_cam_cv"], float)
    pts = np.atleast_2d(pts)
    Xc = (T @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
    uv = (K @ Xc.T).T
    return uv[:, :2] / uv[:, 2:3], Xc[:, 2]


def box_corners(pos, rot, scale):
    s = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)]) * np.asarray(scale)
    return np.array([np.asarray(pos) + quat_rotate(rot, c) for c in s])


def hull_mask(cam, corners, H, W):
    uv, z = project(cam, corners)
    m = np.zeros((H, W), np.uint8)
    if np.any(z <= 0):
        return m.astype(bool)
    hull = cv2.convexHull(uv.astype(np.float32)).reshape(-1, 2)
    cv2.fillConvexPoly(m, np.round(hull * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0

# ---------------------------------------------------------------- チェック

class Report:
    def __init__(self):
        self.items = []

    def check(self, name, ok, detail=""):
        self.items.append({"check": name, "pass": bool(ok), "detail": detail})
        return ok

    @property
    def passed(self):
        return all(i["pass"] for i in self.items)


def check_files(r, scene_dir, meta):
    r.check("files: metadata.missing_files が空", not meta.get("missing_files"), str(meta.get("missing_files")))

    missing = []
    def walk(d):
        for v in d.values():
            if isinstance(v, dict):
                walk(v)
            elif isinstance(v, str) and not (scene_dir / v).exists():
                missing.append(v)
    walk(meta["initial"])
    for e in meta["episodes"]:
        walk(e["files"])
    r.check("files: 全パスが実在(再確認)", not missing, str(missing))


def check_episode(r, ep, tol_px=1.5, iou_min=0.85):
    cid = ep["episode"]["candidate_id"]
    cam = ep["camera"]
    H, W = cam["height"], cam["width"]
    tgt_name = ep["planned"]["target"]["name"]

    # ---- B. raster ファイル・画像サイズ
    steps = ep["actions_meta"]["raster_steps"]
    files = sorted(p.name for p in (ep["dir"] / "actions" / "eef_raster").glob("*.png"))
    r.check(f"{cid} files: eef_raster の枚数 = raster_steps", len(files) == len(steps), f"{len(files)} vs {len(steps)}")
    shapes = {k: ep[k].shape[:2] for k in ["final_mask", "final_rgb", "final_depth", "final_semantic",
                                           "initial_rgb", "initial_depth", "initial_semantic"]}
    shapes["eef_raster"] = ep["eef_raster"].shape[1:3]
    shapes["fullbody_raster"] = ep["fullbody_raster"].shape[1:3]
    r.check(f"{cid} files: 全画像が {W}x{H}", all(s == (H, W) for s in shapes.values()), str(shapes))

    # ---- C. 投影: target の箱 vs mask(初期・最終)
    obj0 = next(o for o in ep["scene"]["objects"] if o["name"] == tgt_name)
    color = ep["final"]["target_mask"]["semantic_color_rgb"]
    init_mask = np.all(ep["initial_semantic"][:, :, ::-1] == color, axis=2)
    proj0 = hull_mask(cam, box_corners(obj0["position_world"], obj0["rotation_world_xyzw"], obj0["scale"]), H, W)
    v = iou(init_mask, proj0)
    r.check(f"{cid} 投影: 初期 target 箱 vs 初期 mask IoU ≥ {iou_min}", v >= iou_min, f"IoU={v:.3f}")

    fo = next(o for o in ep["final"]["objects"] if o["name"] == tgt_name)
    proj1 = hull_mask(cam, box_corners(fo["final_position_world"], fo["final_rotation_world_xyzw"], obj0["scale"]), H, W)
    fm = ep["final_mask"] > 127
    v = iou(fm, proj1)
    r.check(f"{cid} 投影: 最終 target 箱 vs target_mask IoU ≥ {iou_min}", v >= iou_min, f"IoU={v:.3f}")
    r.check(f"{cid} 投影: target_mask の画素数 = final_state 記録値", int(fm.sum()) == ep["final"]["target_mask"]["pixels"],
            f"{int(fm.sum())} vs {ep['final']['target_mask']['pixels']}")

    # ---- C. 投影: EEF raster の点 vs EEF numeric
    _, eef = ep["eef_numeric"]
    worst, n = 0.0, 0
    for k, s in enumerate(steps):
        uv, z = project(cam, [eef["tcp_x"][s], eef["tcp_y"][s], eef["tcp_z"][s]])
        u, vv = uv[0]
        if z[0] <= 0 or not (3 <= u < W - 3 and 3 <= vv < H - 3):
            continue
        ys, xs = np.nonzero(ep["eef_raster"][k] > 127)
        if len(xs) == 0:
            worst = float("inf"); break
        worst = max(worst, float(np.hypot(xs.mean() - u, ys.mean() - vv)))
        n += 1
    r.check(f"{cid} 投影: EEF raster の点の中心 vs EEF numeric ≤ {tol_px}px", n > 0 and worst <= tol_px, f"最大 {worst:.3f}px({n} 枚)")

    # ---- D. ラベル vs contacts
    _, c = ep["contacts"]
    rows = list(zip(*(c[k] if isinstance(c[k], list) else list(c[k]) for k in ["step", "event", "body", "body_kind", "other", "other_kind"]))) if c else []
    rows = [x for x in rows if x[1] != "exit"]
    robot_target = [int(x[0]) for x in rows if x[3] == "object" and x[5] == "robot_link" and x[2] == tgt_name]
    robot_other = [int(x[0]) for x in rows if x[3] == "object" and x[5] == "robot_link" and x[2] != tgt_name]
    tgt_obj = [int(x[0]) for x in rows if x[3] == "object" and x[5] == "object" and tgt_name in (x[2], x[4])]
    lab = ep["final"]["labels"]
    first = lambda l: min(l) if l else -1
    r.check(f"{cid} ラベル: target_contact = contacts から再計算", lab["target_contact"] == bool(robot_target),
            f"label={lab['target_contact']}, contacts={bool(robot_target)}")
    r.check(f"{cid} ラベル: secondary_collision = contacts から再計算",
            lab["secondary_collision"] == bool(robot_other or tgt_obj),
            f"label={lab['secondary_collision']}, robot-他物体={bool(robot_other)}, target-物体={bool(tgt_obj)}")
    r.check(f"{cid} ラベル: 初回接触 step が一致",
            (lab["first_robot_target_step"], lab["first_robot_secondary_step"], lab["first_target_secondary_step"])
            == (first(robot_target), first(robot_other), first(tgt_obj)),
            f"label=({lab['first_robot_target_step']}, {lab['first_robot_secondary_step']}, {lab['first_target_secondary_step']}) "
            f"contacts=({first(robot_target)}, {first(robot_other)}, {first(tgt_obj)})")
    goal = ep["final"]["goal"]["center_world"]
    d = float(np.hypot(fo["final_position_world"][0] - goal[0], fo["final_position_world"][2] - goal[2]))
    r.check(f"{cid} ラベル: goal_distance = 最終 pose から再計算", abs(d - lab["goal_distance_m"]) < 1e-4,
            f"{d:.5f} vs {lab['goal_distance_m']:.5f}")

    # ---- E. 時系列
    dt = ep["planned"]["dt"]
    n_plan = ep["planned"]["num_steps"]
    q = np.array(ep["planned"]["steps"]["q"])
    lens = {"planned.json num_steps": n_plan, "planned.json q": len(q),
            "planned.csv": len(ep["planned_csv"][1]["step"]),
            "eef_numeric": len(ep["eef_numeric"][1]["step"]), "fullbody_numeric": len(ep["fullbody_numeric"][1]["step"])}
    r.check(f"{cid} 時系列: planned / actions の step 数が一致", len(set(lens.values())) == 1, str(lens))
    t = np.array(ep["planned"]["steps"]["t"])
    r.check(f"{cid} 時系列: planned の t = step × dt", np.allclose(t, np.arange(n_plan) * dt, atol=1e-5))
    r.check(f"{cid} 時系列: raster_steps が範囲内で最後の step を含む",
            steps == sorted(set(steps)) and steps[0] == 0 and steps[-1] == n_plan - 1, f"{steps[:3]}...{steps[-2:]}")

    em = ep["executed_meta"]
    ex = ep["executed"][1]
    n_ex = len(ex["step"])
    r.check(f"{cid} 時系列: executed の行数 = planned + post_settle",
            n_ex == em["num_steps"] == em["num_planned_steps"] + em["post_settle_steps"] and em["num_planned_steps"] == n_plan,
            f"rows={n_ex}, meta={em['num_steps']}, planned={n_plan}+{em['post_settle_steps']}")
    r.check(f"{cid} 時系列: executed の t = (step + 1) × dt", np.allclose(ex["t"], (np.arange(n_ex) + 1) * dt, atol=1e-5))
    r.check(f"{cid} 時系列: executed の planned_step = min(step, N-1)",
            np.array_equal(ex["planned_step"], np.minimum(np.arange(n_ex), n_plan - 1)))
    names = ep["planned"]["joint_names"]
    cmd = np.stack([ex["cmd_" + n] for n in names], 1)
    r.check(f"{cid} 時系列: executed の指令値 = planned の q", np.allclose(cmd, q[np.minimum(np.arange(n_ex), n_plan - 1)], atol=1e-5))

    cs = np.array(c["step"]) if c and len(c["step"]) else np.array([])
    r.check(f"{cid} 時系列: contacts の step が範囲内・昇順",
            len(cs) == 0 or (cs.min() >= 0 and cs.max() < n_ex and np.all(np.diff(cs) >= 0)),
            f"{len(cs)} 行" + (f", {int(cs.min())}〜{int(cs.max())}" if len(cs) else ""))
    last = np.array([ex[f"{tgt_name}_{a}"][-1] for a in "xyz"])
    r.check(f"{cid} 時系列: final pose = executed 最終行", np.allclose(last, fo["final_position_world"], atol=1e-4),
            f"diff={np.abs(last - np.array(fo['final_position_world'])).max():.2e} m")


def validate_scene(scene_dir: Path):
    r = Report()
    meta = _json(scene_dir / "metadata.json")
    check_files(r, scene_dir, meta)
    for e in meta["episodes"]:
        ep_dir = scene_dir / "candidates" / e["candidate_id"]
        try:
            ep = load_episode(ep_dir)
            r.check(f"{e['candidate_id']} 読み込み: load_episode", True, "全ファイルを numpy に読み込めた")
        except Exception as ex:
            r.check(f"{e['candidate_id']} 読み込み: load_episode", False, repr(ex))
            continue
        try:
            check_episode(r, ep)
        except Exception as ex:
            r.check(f"{e['candidate_id']} 検証中の例外", False, repr(ex))

    out = {"scene_id": scene_dir.name, "passed": r.passed,
           "num_checks": len(r.items), "num_failed": sum(not i["pass"] for i in r.items), "checks": r.items}
    (scene_dir / "validation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"== {scene_dir.name}: {'PASS' if r.passed else 'FAIL'}({len(r.items) - out['num_failed']}/{len(r.items)})")
    for i in r.items:
        print(f"  [{'OK' if i['pass'] else 'NG'}] {i['check']}" + (f"  … {i['detail']}" if i["detail"] else ""))
    return r.passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None, help="Unity プロジェクトのルート(省略時はこのファイルの 2 つ上)")
    ap.add_argument("--episodes", default="Episodes")
    ap.add_argument("--scene", default=None)
    args = ap.parse_args()
    project = Path(args.project) if args.project else Path(__file__).resolve().parents[2]
    root = project / args.episodes
    scenes = [root / args.scene] if args.scene else sorted(p for p in root.glob("scene_*") if p.is_dir())
    if not scenes:
        sys.exit(f"scene が見つからない: {root}")
    ok = [validate_scene(s) for s in scenes]
    sys.exit(0 if all(ok) else 1)


if __name__ == "__main__":
    main()
