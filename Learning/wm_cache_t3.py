"""
3 つ目のタスク（前腕で背の高い箱を押す、Assets/Task2Pilot/Task3Recorder.cs）の episode フォルダ → 学習用キャッシュ。
wm_cache_t2.py と同じ形（1 シーン = task 0 と task 1 の 2 行）。読み替え:
- goal  = 箱が倒れた（tipped、30° 以上傾いた）
- coll  = 腕（グリッパー以外）が箱に触れた
- final_pos = 箱の最後の中心位置、final_mask = 箱の最後の見え方（ロボットを除く）
- goal_xz = 箱の最初の位置（床面の中心）
- matched pair = 同じ dx の shoulder_a / shoulder_b（branch 0 / 1）
3 つ目のタスクだけの列: episodes/t3_tool_contact, t3_tipped (u8), t3_disp, t3_dx (f32)
  scenes/t3_tcp_x, t3_tcp_z, t3_box_lateral, t3_box_along, t3_box_w, t3_box_d, t3_box_h (f32)
使い方: python Learning/wm_cache_t3.py --episodes Episodes/_task3_v1
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wm_cache as WC                                                                    # noqa: E402
from wm_cache import load_action, load_initial, down_area, read_gray, KINDS, EEF_COLS   # noqa: E402


def canon_quat_tol(q, tol=1e-3):
    """wm_cache.canon_quat と同じ規則で、0 とみなす幅だけ広げる（1e-6 → 1e-3）。
    手先は真下向きで w ≈ 0 になり、数値の誤差（±1e-6 程度）だけで最初の符号の判定が
    matched pair の 2 本で逆になった（同じ向きなのに符号が全部反転し、差が最大 1.41）"""
    q = np.array(q, np.float64, copy=True)
    if len(q) == 0:
        return q.astype(np.float32)
    q0 = q[0]
    key = next((v for v in (q0[3], q0[0], q0[1], q0[2]) if abs(v) > tol), 1.0)
    if key < 0:
        q[0] = -q0
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]
    return q.astype(np.float32)


WC.canon_quat = canon_quat_tol   # load_action（canon_quat_cols）がこちらを使う

VERSION = "wm_cache_t3_v1"
BRANCH = {"shoulder_a": 0, "shoulder_b": 1}
T3_LAYOUT = ["tcp_x", "tcp_z", "box_lateral", "box_along", "box_w", "box_d", "box_h"]


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def nan3(v):
    return [np.nan if x is None else float(x) for x in v]


def build(roots, out_path, res, frames, log=print, share_eef=True):
    scenes = sorted((s for r in roots for s in Path(r).glob("scene_*") if (s / "metadata.json").is_file()), key=lambda s: s.name)
    if not scenes:
        sys.exit(f"シーンが無い: {roots}")
    log(f"[cache_t3] シーン {len(scenes)}")
    cam = _json(scenes[0] / "camera.json")
    s = res / cam["width"]
    K = np.asarray(cam["K"], np.float64).copy()
    K[0, 0] *= s; K[1, 1] *= s
    K[0, 2] = (K[0, 2] + 0.5) * s - 0.5; K[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    R, T = res, frames
    comp = dict(compression="lzf", shuffle=True)
    S, A, E = [], [], []
    share_diff = {"eef_num": 0.0, "eef_mask_px": 0, "eef_depth_mm": 0}
    full_cols_ref = None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.h5")
    with h5py.File(tmp, "w") as h:
        ga = h.create_group("actions")
        img_ds = {k: ga.create_dataset(k, (0, T, R, R), dtp, maxshape=(None, T, R, R), chunks=(1, T, R, R), **comp)
                  for k, dtp in [("eef_mask", np.uint8), ("full_mask", np.uint8), ("eef_depth_mm", np.uint16), ("full_depth_mm", np.uint16)]}
        pair_ids = []
        for pi, sd in enumerate(scenes):
            gen = _json(sd / "scene_generation.json"); ini = _json(sd / "scene_initial.json")
            lay = gen["layout"]
            rgb, dep = load_initial(sd, res)
            pair_ids.append(ini["task"]["pair_id"])
            for t in (0, 1):
                S.append({"scene_id": sd.name, "pair_idx": pi, "task": t, "rgb": rgb, "depth": dep, "seed": ini["seed"],
                          "goal_xz": [gen["goal"]["center_world"][0], gen["goal"]["center_world"][2]],
                          "target_pos": gen["target"]["position_world"], "target_yaw": 0.0,
                          "sec_pos": gen["secondary"]["position_world"] if t == 1 else [np.nan] * 3,
                          "sec_h": gen["secondary"]["height_m"] if t == 1 else np.nan,
                          "t2": [lay[k] for k in T3_LAYOUT]})
            cands = _json(sd / "candidates.json")["candidates"]
            mp_members = {}
            for ci, c in enumerate(cands):
                cd = sd / "candidates" / c["candidate_id"]
                a = load_action(cd, res, frames)
                if full_cols_ref is None:
                    full_cols_ref = a["full_cols"]
                ai = len(A)
                for k, d in img_ds.items():
                    d.resize(ai + 1, axis=0); d[ai] = a[k]
                A.append({"eef_num": a["eef_num"], "full_num": a["full_num"], "raster_steps": a["raster_steps"], "eef_num_raw": a["eef_num_raw"], "shared_from": -1})
                mp = int(c["matched_pair_id"].replace("mp", "")); br = BRANCH[c["branch"]]
                mem = mp_members.setdefault(mp, []); mem.append((ai, br))
                if len(mem) == 2 and share_eef:
                    src = next(x for x, b in mem if b == 0); dst = next(x for x, b in mem if b == 1)
                    if len(A[src]["eef_num"]) != len(A[dst]["eef_num"]):
                        raise ValueError(f"{sd.name} mp{mp}: matched pair の長さが違う")
                    share_diff["eef_num"] = max(share_diff["eef_num"], float(np.abs(A[src]["eef_num"] - A[dst]["eef_num"]).max()))
                    share_diff["eef_mask_px"] = max(share_diff["eef_mask_px"], int((img_ds["eef_mask"][src] != img_ds["eef_mask"][dst]).sum()))
                    share_diff["eef_depth_mm"] = max(share_diff["eef_depth_mm"], int(np.abs(img_ds["eef_depth_mm"][src].astype(int) - img_ds["eef_depth_mm"][dst].astype(int)).max()))
                    A[dst]["shared_from"] = src
                    A[dst]["eef_num"] = A[src]["eef_num"].copy(); A[dst]["raster_steps"] = A[src]["raster_steps"].copy()
                    for k in ["eef_mask", "eef_depth_mm"]:
                        img_ds[k][dst] = img_ds[k][src]
                fs = _json(cd / "final" / "final_state.json"); lab = fs["labels"]
                tgt = next(o for o in fs["objects"] if o["name"] == "Target")
                fm = np.round(down_area(read_gray(cd / "final" / "target_mask.png").astype(np.float32), res)).astype(np.uint8)
                cp = nan3(fs["contacts"]["first_point_world"])
                base = {"action_idx": ai, "pair_idx": pi, "cand_idx": ci, "kind": KINDS.index("matched_pair"), "mp": mp, "branch": br,
                        "goal": bool(lab["goal_reached"]), "fell": bool(lab["tipped"]), "goal_dist": float(lab["goal_distance_m"]),
                        "final_pos": tgt["final_position_world"], "final_yaw": 0.0, "final_mask": fm,
                        "heat_target": np.zeros((R, R), np.uint8), "heat_secondary": np.zeros((R, R), np.uint8),
                        "contact_point": cp, "contact_valid": not np.isnan(cp[0]),
                        "t2": [int(lab["tool_contact"]), int(lab["tipped"]), float(lab["displacement_m"]), float(c["approach_dx"])]}
                for t in (0, 1):
                    E.append({**base, "scene_idx": 2 * pi + t, "task": t, "coll": -1 if t == 0 else int(bool(lab["secondary_collision"]))})
            if (pi + 1) % 20 == 0 or pi == len(scenes) - 1:
                log(f"[cache_t3] {pi + 1}/{len(scenes)}（action {len(A)}、episode {len(E)}）")

        N = len(A); Lmax = max(len(a["eef_num"]) for a in A)
        en = np.zeros((N, Lmax, 7), np.float32); fn = np.zeros((N, Lmax, len(full_cols_ref)), np.float32); er = np.zeros_like(en)
        ln = np.zeros(N, np.int32)
        for i, a in enumerate(A):
            L = len(a["eef_num"]); ln[i] = L
            en[i, :L] = a["eef_num"]; fn[i, :L] = a["full_num"]; er[i, :len(a["eef_num_raw"])] = a["eef_num_raw"]
        ga.create_dataset("eef_num", data=en, chunks=(1,) + en.shape[1:], **comp)
        ga.create_dataset("full_num", data=fn, chunks=(1,) + fn.shape[1:], **comp)
        ga.create_dataset("eef_num_raw", data=er, chunks=(1,) + er.shape[1:], **comp)
        ga.create_dataset("num_len", data=ln)
        ga.create_dataset("eef_shared_from", data=np.array([a["shared_from"] for a in A], np.int32))
        ga.create_dataset("raster_steps", data=np.stack([a["raster_steps"] for a in A]))
        ga.attrs["eef_cols"] = json.dumps(EEF_COLS); ga.attrs["full_cols"] = json.dumps(full_cols_ref)

        gs = h.create_group("scenes")
        gs.create_dataset("initial_rgb", data=np.stack([x["rgb"] for x in S]), chunks=(1, 3, R, R), **comp)
        gs.create_dataset("initial_depth_mm", data=np.stack([x["depth"] for x in S]), chunks=(1, R, R), **comp)
        gs.create_dataset("scene_id", data=np.array([x["scene_id"] for x in S], dtype="S16"))
        gs.create_dataset("pair_idx", data=np.array([x["pair_idx"] for x in S], np.int32))
        gs.create_dataset("task", data=np.array([x["task"] for x in S], np.int8))
        gs.create_dataset("goal_xz", data=np.array([x["goal_xz"] for x in S], np.float32))
        gs.create_dataset("target_init_pos", data=np.array([x["target_pos"] for x in S], np.float32))
        gs.create_dataset("target_init_yaw_deg", data=np.array([x["target_yaw"] for x in S], np.float32))
        gs.create_dataset("secondary_pos", data=np.array([x["sec_pos"] for x in S], np.float32))
        gs.create_dataset("secondary_height", data=np.array([x["sec_h"] for x in S], np.float32))
        t2s = np.array([x["t2"] for x in S], np.float32)
        for j, k in enumerate(["t3_" + x for x in T3_LAYOUT]):
            gs.create_dataset(k, data=t2s[:, j])

        ge = h.create_group("episodes")
        for k, dtp in [("scene_idx", np.int32), ("action_idx", np.int32), ("pair_idx", np.int32), ("task", np.int8),
                       ("cand_idx", np.int8), ("kind", np.int8), ("mp", np.int8), ("branch", np.int8),
                       ("goal", np.uint8), ("coll", np.int8), ("fell", np.uint8), ("goal_dist", np.float32),
                       ("final_yaw", np.float32), ("contact_valid", np.uint8)]:
            ge.create_dataset(k, data=np.array([e[k] for e in E], dtp))
        ge.create_dataset("final_pos", data=np.array([e["final_pos"] for e in E], np.float32))
        ge.create_dataset("contact_point", data=np.array([e["contact_point"] for e in E], np.float32))
        for k in ["final_mask", "heat_target", "heat_secondary"]:
            ge.create_dataset(k, data=np.stack([e[k] for e in E]), chunks=(1, R, R), **comp)
        t2e = np.array([e["t2"] for e in E], np.float32)
        for j, k in enumerate(["t3_tool_contact", "t3_tipped"]):
            ge.create_dataset(k, data=t2e[:, j].astype(np.uint8))
        ge.create_dataset("t3_disp", data=t2e[:, 2])
        ge.create_dataset("t3_dx", data=t2e[:, 3])
        ge.attrs["kinds"] = json.dumps(KINDS)

        gp = h.create_group("pairs")
        gp.create_dataset("pair_id", data=np.array(pair_ids, dtype="S16"))
        gp.create_dataset("seed", data=np.array([S[2 * i]["seed"] for i in range(len(scenes))], np.int64))
        h.attrs.update({"version": VERSION, "res": res, "frames": frames, "source_root": ";".join(str(r) for r in roots),
                        "created": dt.datetime.now().isoformat(timespec="seconds"), "K": K,
                        "T_world_to_cam_cv": np.asarray(cam["T_world_to_cam_cv"], np.float64), "git_commits": "[]",
                        "contact_touch_tolerance_m": 0.001, "depth_unit": "mm (uint16), 0 = nothing", "image_layout": "row 0 = top (OpenCV)",
                        "eef_shared_in_matched_pairs": bool(share_eef), "eef_share_maxdiff": json.dumps(share_diff),
                        "task": "task3: forearm pushes a tall box; goal = tipped, coll = arm contact; task 0 rows duplicate task 1 with coll = -1"})
    os.replace(tmp, out_path)
    log(f"[cache_t3] 書いた: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB) scenes {len(S)}, actions {len(A)}, episodes {len(E)}; EEF 共有前の最大の差 {share_diff}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", default=["Episodes/_task3_v1"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--frames", type=int, default=25)
    args = ap.parse_args()
    project = Path(__file__).resolve().parents[1]
    roots = [Path(e) if Path(e).is_absolute() else project / e for e in args.episodes]
    out = Path(args.out) if args.out else roots[0] / "_cache" / f"wm_{args.res}x{args.frames}.h5"
    if not out.is_absolute():
        out = project / out
    build(roots, out, args.res, args.frames)


if __name__ == "__main__":
    main()
