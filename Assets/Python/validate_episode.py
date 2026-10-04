"""
TODO5-8: episode フォルダの自動検証

チェック内容
  A. 読み込み   : load_episode() で全ファイルを numpy に読み込めるか(完了条件そのもの)
  B. ファイル   : metadata.json の全パスが実在するか / raster 画像が actions_meta の raster_steps と一致するか / 画像サイズ
  C. 投影       : 初期・最終の target mask と、target の 3D 箱を camera.json で投影した形が一致するか
                  EEF raster の点の中心と、EEF numeric を投影した位置が一致するか
  D. ラベル     : final_state の target_contact / secondary_collision が contacts.csv から計算し直した値と一致するか
  T. Task A/B   : scene_initial / candidates / final_state の task_variant が一致、Task A に secondary 物体が無い、
                  Task B にある(A/B ペア同士の比較は check_task_pair.py)
  E. 時系列     : planned / actions / executed の step 数・時刻・planned_step 対応、contacts の step 範囲、
                  final pose と executed 最終行の一致

実行(プロジェクト直下で):
    pip install numpy opencv-python
    python Assets/Python/validate_episode.py                  # Episodes/ の全 scene
    python Assets/Python/validate_episode.py --scene scene_0000

出力: Episodes/scene_XXXX/validation.json と、コンソールの PASS / FAIL 一覧。全部 PASS なら exit code 0。

別のスクリプトから episode を使うときは:
    from validate_episode import load_episode, find_pair, load_pair_episode
    ep = load_episode("Episodes/scene_0000/candidates/c000")
    pair = find_pair("Episodes", "pair_0000")              # {"A": Path(scene_0000), "B": Path(scene_0001)}
    epB = load_pair_episode("Episodes", "pair_0000", "B", "c000")
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


def _quat_rotate(q, v):
    """クォータニオン (x, y, z, w) でベクトルを回す"""
    x, y, z, w = q
    u = np.array([x, y, z]); v = np.asarray(v, dtype=float)
    return 2 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * v + 2 * w * np.cross(u, v)


def _tilt_deg(q0, q1):
    """初期姿勢から見た、上向きの軸の傾き(度)。Unity は左手系だが角度の大きさは同じ"""
    x0, y0, z0, w0 = q0
    inv0 = (-x0, -y0, -z0, w0)
    up = _quat_rotate(q1, _quat_rotate(inv0, [0.0, 1.0, 0.0]))
    return float(np.degrees(np.arccos(np.clip(up[1] / np.linalg.norm(up), -1.0, 1.0))))

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
    # P4-B(actions_meta v2): メッシュの本描画。raster = マスク、depth = 16bit(mm、0 = 何も無い)
    if (ep_dir / "actions" / "eef_depth").is_dir():
        ep["eef_depth"] = np.stack([_img(ep_dir / "actions" / "eef_depth" / f"step_{s:04d}.png") for s in steps])
        ep["fullbody_depth"] = np.stack([_img(ep_dir / "actions" / "fullbody_depth" / f"step_{s:04d}.png") for s in steps])

    ep["executed_meta"] = _json(ep_dir / "executed" / "executed_meta.json")
    ep["executed"] = _csv(ep_dir / "executed" / "executed_trajectory.csv")
    ep["contacts"] = _csv(ep_dir / "executed" / "contacts.csv")

    ep["final"] = _json(ep_dir / "final" / "final_state.json")
    ep["final_mask"] = _img(ep_dir / "final" / "target_mask.png", cv2.IMREAD_GRAYSCALE)
    ep["final_rgb"] = _img(ep_dir / "final" / "robot_free" / "rgb.png", cv2.IMREAD_COLOR)
    ep["final_depth"] = _img(ep_dir / "final" / "robot_free" / "depth.exr")
    ep["final_semantic"] = _img(ep_dir / "final" / "robot_free" / "semantic.png", cv2.IMREAD_COLOR)
    if (ep_dir / "final" / "contact_heatmaps.json").is_file():
        ep["contact_heatmaps"] = _json(ep_dir / "final" / "contact_heatmaps.json")
        ep["heatmap_target"] = _img(ep_dir / "final" / "contact_heatmap_target.png", cv2.IMREAD_GRAYSCALE)
        ep["heatmap_secondary"] = _img(ep_dir / "final" / "contact_heatmap_secondary.png", cv2.IMREAD_GRAYSCALE)

    task = ep["scene"].get("task") or {}
    ep["task_variant"] = task.get("task_variant")   # 古いデータは None
    ep["pair_id"] = task.get("pair_id")

    init = scene_dir / "initial" / "robot_free"
    ep["initial_rgb"] = _img(init / "rgb.png", cv2.IMREAD_COLOR)
    ep["initial_depth"] = _img(init / "depth.exr")
    ep["initial_semantic"] = _img(init / "semantic.png", cv2.IMREAD_COLOR)
    wr = scene_dir / "initial" / "with_robot"
    if (wr / "depth.exr").is_file():
        ep["initial_with_robot_depth"] = _img(wr / "depth.exr")
        ep["initial_with_robot_semantic"] = _img(wr / "semantic.png", cv2.IMREAD_COLOR)
    return ep

def find_pair(episodes_root, pair_id):
    """pair_id から {task_variant: scene_dir} を返す(scene_initial.json の task を見る)"""
    out = {}
    for d in sorted(Path(episodes_root).glob("scene_*")):
        f = d / "scene_initial.json"
        if not f.is_file():
            continue
        task = _json(f).get("task") or {}
        if task.get("pair_id") == pair_id:
            v = task.get("task_variant")
            if v in out:
                raise ValueError(f"{pair_id} の Task {v} が 2 つある: {out[v].name}, {d.name}")
            out[v] = d
    return out


def load_pair_episode(episodes_root, pair_id, task_variant, candidate_id):
    """scene × task(A/B) × candidate を pair_id から読む"""
    pair = find_pair(episodes_root, pair_id)
    if task_variant not in pair:
        raise KeyError(f"{pair_id} に Task {task_variant} が無い(見つかったのは {sorted(pair)})")
    return load_episode(pair[task_variant] / "candidates" / candidate_id)

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


def cam_z(cam, pts):
    """world 点のカメラ座標の z(前方向の距離、m)"""
    T = np.array(cam["T_world_to_cam_cv"], dtype=float)
    P = np.c_[np.asarray(pts, dtype=float), np.ones(len(pts))]
    return (P @ T.T)[:, 2]


def check_mesh_raster(r, ep, steps):
    """P4-B: メッシュの本描画(マスク + depth)の検証"""
    cid = ep["episode"]["candidate_id"]
    cam = ep["camera"]
    H, W = cam["height"], cam["width"]
    em, fm = ep["eef_raster"], ep["fullbody_raster"]
    ed, fd = ep["eef_depth"].astype(np.float64), ep["fullbody_depth"].astype(np.float64)
    r.check(f"{cid} 本描画: depth の枚数・大きさ = マスク", ed.shape == em.shape and fd.shape == fm.shape, f"{ed.shape} {em.shape}")
    r.check(f"{cid} 本描画: マスク = (depth > 0)", np.array_equal(em > 127, ed > 0) and np.array_equal(fm > 127, fd > 0))
    r.check(f"{cid} 本描画: どの画像にもロボットが写っている", bool((em > 127).any(axis=(1, 2)).all() and (fm > 127).any(axis=(1, 2)).all()))
    # 全身はグリッパーを含むので、グリッパーの画素は全身の画素でもあり、depth は全身 ≤ グリッパー(手前が勝つ)
    sub = (em > 127) & ~(fm > 127)
    both = (ed > 0) & (fd > 0)
    r.check(f"{cid} 本描画: グリッパーのマスク ⊂ 全身のマスク、全身の depth ≤ グリッパーの depth",
            not sub.any() and bool((fd[both] <= ed[both] + 0.5).all()), f"はみ出し {int(sub.sum())} px")
    # TCP: 投影した画素のまわり(5 px 以内)にグリッパーが写っていて、その depth が TCP のカメラ z と 3 cm 以内
    _, eef = ep["eef_numeric"]
    worst_px, worst_dz, n = 0.0, 0.0, 0
    for k, s in enumerate(steps):
        p = [eef["tcp_x"][s], eef["tcp_y"][s], eef["tcp_z"][s]]
        uv, z = project(cam, [p])
        u, v = uv[0]
        if z[0] <= 0 or not (5 <= u < W - 5 and 5 <= v < H - 5):
            continue
        ys, xs = np.nonzero(em[k] > 127)
        d = np.hypot(xs - u, ys - v)
        j = int(np.argmin(d))
        worst_px = max(worst_px, float(d[j]))
        win = ed[k, max(0, int(v) - 5):int(v) + 6, max(0, int(u) - 5):int(u) + 6]
        win = win[win > 0]
        if len(win):
            worst_dz = max(worst_dz, float(np.min(np.abs(win / 1000.0 - z[0]))))
        n += 1
    r.check(f"{cid} 本描画: TCP の投影から 5 px 以内にグリッパー、depth が TCP の z と 3 cm 以内",
            n > 0 and worst_px <= 5.0 and worst_dz <= 0.03, f"最大 {worst_px:.2f} px / {worst_dz * 1000:.1f} mm({n} 枚)")
    # カメラ幾何: 最初の step の全身の本描画 vs Perception の初期画像(ロボットあり)。開始姿勢が初期姿勢と同じ候補だけ
    if "initial_with_robot_depth" in ep and steps and steps[0] == 0:
        q0 = np.array(ep["planned"]["steps"]["q"][0], dtype=float)
        jpos = {j["name"]: j["position"] for j in (ep["scene"].get("robot") or {}).get("joints", [])}
        names = ep["planned"].get("joint_names", [])
        qi = np.array([jpos[n] for n in names], dtype=float) if names and all(n in jpos for n in names) else None
        same_start = qi is not None and len(qi) == len(q0) and np.allclose(q0, qi, atol=1e-3)
        if same_start:
            robot_px = np.any(ep["initial_with_robot_semantic"] != ep["initial_semantic"], axis=2)
            ren = fm[0] > 127
            v = iou(robot_px, ren)
            pd = ep["initial_with_robot_depth"]
            pd = pd[:, :, 2] if pd.ndim == 3 else pd   # Perception の depth は R チャンネル(cv2 は BGRA の順)
            m = robot_px & ren
            dz = np.abs(pd[m].astype(np.float64) - fd[0][m] / 1000.0)
            med = float(np.median(dz)) if m.any() else float("inf")
            r.check(f"{cid} 本描画: step 0 の全身 vs Perception の初期画像(ロボットあり): マスク IoU ≥ 0.9、depth の差の中央値 ≤ 5 mm",
                    v >= 0.9 and med <= 0.005, f"IoU={v:.3f}, 中央値 {med * 1000:.2f} mm, 90% 点 {float(np.percentile(dz, 90)) * 1000 if m.any() else -1:.2f} mm")


def _heat(points_uv, W, H, sigma):
    acc = np.zeros((H, W), dtype=np.float64)
    rad = int(np.ceil(3 * sigma))
    for u, v in points_uv:
        cx, cy = int(round(u)), int(round(v))
        for y in range(max(0, cy - rad), min(H - 1, cy + rad) + 1):
            for x in range(max(0, cx - rad), min(W - 1, cx + rad) + 1):
                acc[y, x] += np.exp(-((x - u) ** 2 + (y - v) ** 2) / (2 * sigma * sigma))
    mx = acc.max()
    return np.zeros((H, W), np.uint8) if mx <= 0 else np.round(255.0 * acc / mx).astype(np.uint8)


def check_heatmaps(r, ep):
    """P4-B: contact heatmap を contacts.csv と camera.json から作り直して照合"""
    cid = ep["episode"]["candidate_id"]
    cam = ep["camera"]
    H, W = cam["height"], cam["width"]
    meta = ep["contact_heatmaps"]
    tgt = ep["planned"]["target"]["name"]
    sec = (ep["scene"].get("task") or {}).get("secondary_object", "SecondObject")
    tol = meta.get("touch_tolerance_m", 0.001)
    _, c = ep["contacts"]
    n = len(c["step"]) if c else 0
    pt, ps = [], []
    for i in range(n):
        if c["event"][i] == "exit" or not (float(c["min_separation"][i]) <= tol):
            continue
        body, bk, other, ok = c["body"][i], c["body_kind"][i], c["other"][i], c["other_kind"][i]
        to_t = bk == "object" and body == tgt and ok == "robot_link"
        to_s = (bk == "object" and body == sec and ok == "robot_link") or \
               (bk == "object" and ok == "object" and {body, other} == {tgt, sec})
        if not (to_t or to_s):
            continue
        uv, z = project(cam, [[float(c["px"][i]), float(c["py"][i]), float(c["pz"][i])]])
        if z[0] <= 0:
            continue
        (pt if to_t else ps).append(uv[0])
    for name, pts, img in [("target", pt, ep["heatmap_target"]), ("secondary", ps, ep["heatmap_secondary"])]:
        ref = _heat(pts, W, H, meta["sigma_px"])
        d = int(np.abs(ref.astype(int) - img.astype(int)).max())
        r.check(f"{cid} heatmap: {name} を contacts から作り直して一致(差 ≤ 2)", d <= 2 and len(pts) == meta[name]["rows"],
                f"最大差 {d}, 行 {len(pts)} vs {meta[name]['rows']}")
    lab = ep["final"]["labels"]
    r.check(f"{cid} heatmap: target が空でない ⇔ target_contact", bool(ep["heatmap_target"].any()) == bool(lab["target_contact"]))
    if lab.get("secondary_applicable", True):
        r.check(f"{cid} heatmap: secondary が空でない ⇔ secondary_collision",
                bool(ep["heatmap_secondary"].any()) == bool(lab["secondary_collision"]))
    else:
        r.check(f"{cid} heatmap: Task A の secondary は空", not ep["heatmap_secondary"].any())


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

    # ---- C. 投影: EEF raster の点 vs EEF numeric(骨格線の古いデータだけ)
    _, eef = ep["eef_numeric"]
    worst, n = 0.0, 0
    for k, s in enumerate(steps if "eef_depth" not in ep else []):
        uv, z = project(cam, [eef["tcp_x"][s], eef["tcp_y"][s], eef["tcp_z"][s]])
        u, vv = uv[0]
        if z[0] <= 0 or not (3 <= u < W - 3 and 3 <= vv < H - 3):
            continue
        ys, xs = np.nonzero(ep["eef_raster"][k] > 127)
        if len(xs) == 0:
            worst = float("inf"); break
        worst = max(worst, float(np.hypot(xs.mean() - u, ys.mean() - vv)))
        n += 1
    if "eef_depth" in ep:
        check_mesh_raster(r, ep, steps)
    else:
        r.check(f"{cid} 投影: EEF raster の点の中心 vs EEF numeric ≤ {tol_px}px", n > 0 and worst <= tol_px, f"最大 {worst:.3f}px({n} 枚)")
    if "contact_heatmaps" in ep:
        check_heatmaps(r, ep)

    # ---- D. ラベル vs contacts
    _, c = ep["contacts"]
    rows = list(zip(*(c[k] if isinstance(c[k], list) else list(c[k]) for k in ["step", "event", "body", "body_kind", "other", "other_kind"]))) if c else []
    rows = [x for x in rows if x[1] != "exit"]
    # P4-B: min_separation の列があれば、触れていない(隙間 > touch_tolerance)行は数えない
    if c and "min_separation" in c:
        tol = ep["executed_meta"]["contact_summary"].get("touch_tolerance_m", 0.001)
        sep = list(c["min_separation"])
        keep = [i for i in range(len(sep)) if c["event"][i] != "exit" and float(sep[i]) <= tol]
        allrows = list(zip(*(c[k] if isinstance(c[k], list) else list(c[k]) for k in ["step", "event", "body", "body_kind", "other", "other_kind"])))
        rows = [allrows[i] for i in keep]
    robot_target = [int(x[0]) for x in rows if x[3] == "object" and x[5] == "robot_link" and x[2] == tgt_name]
    robot_other = [int(x[0]) for x in rows if x[3] == "object" and x[5] == "robot_link" and x[2] != tgt_name]
    tgt_obj = [int(x[0]) for x in rows if x[3] == "object" and x[5] == "object" and tgt_name in (x[2], x[4])]
    lab = ep["final"]["labels"]
    first = lambda l: min(l) if l else -1
    r.check(f"{cid} ラベル: target_contact = contacts から再計算", lab["target_contact"] == bool(robot_target),
            f"label={lab['target_contact']}, contacts={bool(robot_target)}")
    # secondary 物体があるか(古いデータは secondary_applicable が無い → 物体リストで判断)
    has_secondary = any(o["name"] != tgt_name for o in ep["scene"]["objects"])
    applicable = lab.get("secondary_applicable", has_secondary)
    r.check(f"{cid} ラベル: secondary_applicable = scene に secondary 物体がある", applicable == has_secondary,
            f"label={applicable}, scene objects={[o['name'] for o in ep['scene']['objects']]}")
    if applicable:
        r.check(f"{cid} ラベル: secondary_collision = contacts から再計算",
                lab["secondary_collision"] == bool(robot_other or tgt_obj),
                f"label={lab['secondary_collision']}, robot-他物体={bool(robot_other)}, target-物体={bool(tgt_obj)}")
        r.check(f"{cid} ラベル: 初回接触 step が一致",
                (lab["first_robot_target_step"], lab["first_robot_secondary_step"], lab["first_target_secondary_step"])
                == (first(robot_target), first(robot_other), first(tgt_obj)),
                f"label=({lab['first_robot_target_step']}, {lab['first_robot_secondary_step']}, {lab['first_target_secondary_step']}) "
                f"contacts=({first(robot_target)}, {first(robot_other)}, {first(tgt_obj)})")
    else:
        # Task A: 「衝突なし」ではなく「対象なし」= null。contacts にも secondary との接触が無いこと
        r.check(f"{cid} ラベル: Task A の secondary ラベルが null(対象なし)",
                lab["secondary_collision"] is None and lab["first_robot_secondary_step"] is None
                and lab["first_target_secondary_step"] is None,
                f"secondary_collision={lab['secondary_collision']}, first=({lab['first_robot_secondary_step']}, {lab['first_target_secondary_step']})")
        r.check(f"{cid} ラベル: Task A の contacts に secondary との接触が無い", not (robot_other or tgt_obj),
                f"robot-他物体={len(robot_other)} 行, target-物体={len(tgt_obj)} 行")
        r.check(f"{cid} ラベル: 初回接触 step(robot→target)が一致", lab["first_robot_target_step"] == first(robot_target),
                f"label={lab['first_robot_target_step']}, contacts={first(robot_target)}")
    goal = ep["final"]["goal"]["center_world"]
    d = float(np.hypot(fo["final_position_world"][0] - goal[0], fo["final_position_world"][2] - goal[2]))
    r.check(f"{cid} ラベル: goal_distance = 最終 pose から再計算", abs(d - lab["goal_distance_m"]) < 1e-4,
            f"{d:.5f} vs {lab['goal_distance_m']:.5f}")
    # P4-B: 倒れた判定(高さの低下 or 傾き)を最終 pose から再計算(target_tilt_deg が無い古いデータは高さだけ)
    lp = ep["final"].get("label_params", {})
    drop = fo["initial_position_world"][1] - fo["final_position_world"][1]
    fell = drop > lp.get("fall_drop_threshold_m", 0.02)
    if "target_tilt_deg" in lab:
        tilt = _tilt_deg(fo["initial_rotation_world_xyzw"], fo["final_rotation_world_xyzw"])
        r.check(f"{cid} ラベル: target_tilt_deg = 最終 pose から再計算", abs(tilt - lab["target_tilt_deg"]) < 0.05,
                f"{tilt:.2f} vs {lab['target_tilt_deg']:.2f}")
        fell = fell or tilt > lp.get("fall_tilt_deg", 30.0)
    r.check(f"{cid} ラベル: target_fell = 最終 pose から再計算", fell == lab["target_fell"], f"{fell} vs {lab['target_fell']}")
    still = True
    if "target_final_speed_mps" in lab:
        ex_ = ep["executed"][1]
        tname = fo["name"]
        v_last = float(np.linalg.norm([float(ex_[f"{tname}_v{a}"][-1]) for a in "xyz"]))
        r.check(f"{cid} ラベル: target_final_speed = executed の最後の行の速さ", abs(v_last - lab["target_final_speed_mps"]) < 1e-4,
                f"{v_last:.6f} vs {lab['target_final_speed_mps']:.6f}")
        still = lab["target_final_speed_mps"] < lp.get("goal_max_speed_mps", float("inf"))
    r.check(f"{cid} ラベル: goal_reached = 距離 ≤ 半径 かつ 倒れていない かつ 止まっている",
            lab["goal_reached"] == (lab["goal_distance_m"] <= ep["final"]["goal"]["radius_m"] + 1e-9 and not lab["target_fell"] and still),
            f"goal_reached={lab['goal_reached']}")

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


def check_task(r, scene_dir, meta):
    """T. Task A/B の整合(古いデータで task が無ければスキップ)"""
    scene = _json(scene_dir / "scene_initial.json")
    task = scene.get("task")
    if not task:
        r.check("task: (task 情報なし = P2 より前のデータ。スキップ)", True)
        return
    v = task.get("task_variant")
    cands = _json(scene_dir / "candidates.json")
    r.check("task: task_variant が A か B", v in ("A", "B"), str(v))
    r.check("task: metadata / candidates.json の task_variant・pair_id が scene_initial と一致",
            (meta["ids"].get("task_variant"), meta["ids"].get("pair_id")) == (v, task.get("pair_id"))
            and (cands.get("task_variant"), cands.get("pair_id")) == (v, task.get("pair_id")),
            f"scene=({v}, {task.get('pair_id')}), metadata=({meta['ids'].get('task_variant')}, {meta['ids'].get('pair_id')}), "
            f"candidates=({cands.get('task_variant')}, {cands.get('pair_id')})")
    names = [o["name"] for o in scene["objects"]]
    sec = task.get("secondary_object")
    want = (v == "B")
    r.check(f"task: secondary 物体 '{sec}' が {'ある' if want else '無い'}(Task {v})", (sec in names) == want, str(names))
    bad = []
    for e in meta["episodes"]:
        fs = scene_dir / "candidates" / e["candidate_id"] / "final" / "final_state.json"
        f = _json(fs) if fs.is_file() else {}
        if (e.get("task_variant"), e.get("pair_id")) != (v, task.get("pair_id")) or \
           (f.get("task_variant"), f.get("pair_id")) != (v, task.get("pair_id")):
            bad.append(e["candidate_id"])
    r.check("task: 全 episode.json・final_state.json の task_variant・pair_id が一致", not bad, str(bad))


def validate_scene(scene_dir: Path):
    r = Report()
    meta = _json(scene_dir / "metadata.json")
    check_files(r, scene_dir, meta)
    check_task(r, scene_dir, meta)
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
