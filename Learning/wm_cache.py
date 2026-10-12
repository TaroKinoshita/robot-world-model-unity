"""
P10: episode フォルダ → 学習用キャッシュ(HDF5 1 ファイル)

- 画像は 256 px → 128 px(既定)。action 画像は 25 枚のまま
    mask・RGB・heatmap: 面積平均(cv2.INTER_AREA)
    depth: 2x2 の中で 0 でない画素の平均(背景 0 を混ぜない)、mm の uint16
- action(planned から作った 4 表現)は Task A/B で同じなので、ペア × 候補ごとに 1 回だけ保存する
  (保存する前に A と B がバイト単位で同じことを確かめる)
- 数値の action は全 step(50 Hz)のまま、最長に合わせて 0 で埋めて長さを別に持つ
- quaternion は q と -q が同じ向きなので、符号をそろえる(canon_quat、v2)。
  そろえないと matched pair の 2 本で同じ向きなのに符号だけ違い、EEF 数値から枝が分かってしまった(P8 の 20 シーンで 40/40 組)
- matched pair の 2 本は「同じ手先の経路」が定義なので、EEF の action(数値・画像)は wrist_down 側のものを両方に使う(v3、--no-share-eef で切れる)。
  2 本の EEF は IK→FK の数値誤差で 0.2 mm 以下しか違わないが、その差の符号が枝でそろっていて(全 40 組で同じ向き)、
  ほとんど変わらない成分(rot_y・rot_w)を標準化すると拡大され、小さいモデルでも枝を当てられた(P14)。共有前の差は attrs の eef_share_maxdiff に残す
- 元の値も残す(論文で処理と元の差を説明するため):
    actions/eef_num_raw   (N,Lmax,7)  CSV のままの EEF 数値(符号をそろえる前・共有する前)
    actions/eef_shared_from (N,)      共有したなら、元にした action の番号(wrist_down 側)。していなければ -1
  画像の元(共有前の wrist_up 側の EEF 画像)と全身の数値の元は、episode フォルダの actions/ にそのまま残っている(キャッシュは書き換えない)

使い方(Windows の Python 3.11、cv2・h5py が要る):
  python Learning/wm_cache.py --episodes Episodes/_p8_20scenes [--out <file>] [--res 128]

中身(/ はグループ):
  attrs        : version, res, frames, source_root, created, K(res 用), T_world_to_cam_cv, git_commits
  scenes/      : scene 単位(A/B 別)。initial_rgb (S,3,R,R) u8, initial_depth_mm (S,R,R) u16,
                 scene_id, pair_idx, task(0=A,1=B), goal_xz (S,2), target_init_pos (S,3), target_init_yaw_deg (S,),
                 secondary_pos (S,3)・secondary_height (S,)(A は NaN)
  actions/     : ペア × 候補。eef_mask・full_mask (N,T,R,R) u8, eef_depth_mm・full_depth_mm (N,T,R,R) u16,
                 eef_num (N,Lmax,7) f32, full_num (N,Lmax,69) f32, num_len (N,), raster_steps (N,T)
  episodes/    : episode 単位。scene_idx・action_idx・pair_idx・task・cand_idx・kind・mp(-1/0/1)・branch(-1/0=down/1=up)、
                 goal (u8)・coll (i8, A は -1)・fell・goal_dist・final_pos (E,3)・final_yaw_deg・
                 contact_point (E,3)・contact_valid、final_mask・heat_target・heat_secondary (E,R,R) u8
  pairs/       : pair_id, seed
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"   # cv2 の import より前
import argparse
import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

VERSION = "wm_cache_v3"   # v2: quaternion の符号をそろえた(canon_quat)。v3: matched pair の EEF action を共有(share_eef)
KINDS = ["matched_pair", "goal_dir", "goal_edge", "random", "toward_secondary"]
EEF_COLS = ["tcp_x", "tcp_y", "tcp_z", "rot_x", "rot_y", "rot_z", "rot_w"]


def _json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def kind_of(c):
    note = (c.get("note") or "").lower()
    if c.get("matched_pair_id"):
        return "matched_pair"
    if "no toward-secondary" in note:
        return "random"
    if "secondary" in note:
        return "toward_secondary"
    if "goal edge" in note:
        return "goal_edge"
    if "goal" in note:
        return "goal_dir"
    return "random"


def yaw_deg(q):
    x, y, z, w = q
    # Unity: Y 軸まわりの回転(左手系)。箱は立っている前提
    return math.degrees(math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + x * x)))


# ---------- 縮小 ----------

def down_area(img, res):
    return cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)


def down_depth(d, res):
    """0 = 何もない。f×f の中で 0 でない画素の平均。全部 0 なら 0"""
    h = d.shape[0]
    f = h // res
    d = d.astype(np.float64).reshape(res, f, res, f)
    nz = (d > 0)
    s = d.sum(axis=(1, 3))
    n = nz.sum(axis=(1, 3))
    out = np.zeros((res, res), np.float64)
    m = n > 0
    out[m] = s[m] / n[m]
    return out


def read_gray(p):
    a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(p)
    if a.ndim == 3:
        a = a[:, :, 0]
    return a


def read_numeric(p, cols):
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = [[float(row[c]) for c in cols] for row in r]
    return np.asarray(rows, np.float32)


def canon_quat(q):
    """(L, 4) xyzw の列の符号をそろえる。最初の step は w > 0(w ≈ 0 なら最初の 0 でない成分 > 0)、
    あとは 1 つ前と同じ半球(内積 ≥ 0)。同じ向きの列なら、元の符号にかかわらず同じ結果になる"""
    q = np.array(q, np.float64, copy=True)
    if len(q) == 0:
        return q.astype(np.float32)
    q0 = q[0]
    key = q0[3] if abs(q0[3]) > 1e-6 else next((v for v in (q0[0], q0[1], q0[2]) if abs(v) > 1e-6), 1.0)
    if key < 0:
        q[0] = -q0
    for t in range(1, len(q)):
        if np.dot(q[t], q[t - 1]) < 0:
            q[t] = -q[t]
    return q.astype(np.float32)


def canon_quat_cols(arr, cols):
    """列名で quaternion(rot_x..rot_w、<link>_rx..<link>_rw)を探して符号をそろえる"""
    arr = arr.copy()
    for i, c in enumerate(cols):
        if c.endswith("rot_x") or c.endswith("_rx"):
            stem = c[:-1]
            names = [stem + a for a in ("x", "y", "z", "w")]
            if all(n in cols for n in names):
                idx = [cols.index(n) for n in names]
                if idx == list(range(i, i + 4)):
                    arr[:, i:i + 4] = canon_quat(arr[:, i:i + 4])
    return arr


def full_cols(p):
    with open(p, newline="", encoding="utf-8") as f:
        head = next(csv.reader(f))
    return [c for c in head if c not in ("step", "t", "phase")]


def first_contact_point(contacts_csv, tol):
    """ロボット → ターゲットの最初の「触れた」接触点(world)。無ければ None"""
    with open(contacts_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["body"] == "Target" and row["other_kind"] == "robot_link":
                try:
                    if float(row["min_separation"]) <= tol:
                        return [float(row["px"]), float(row["py"]), float(row["pz"])]
                except ValueError:
                    continue
    return None


# ---------- 読み込み ----------

def load_action(cd, res, frames):
    am = _json(cd / "actions" / "actions_meta.json")
    steps = am.get("raster_steps")
    out = {}
    for rep, mdir, ddir in [("eef", "eef_raster", "eef_depth"), ("full", "fullbody_raster", "fullbody_depth")]:
        mf = sorted((cd / "actions" / mdir).glob("*.png"))
        df = sorted((cd / "actions" / ddir).glob("*.png"))
        if len(mf) != frames or len(df) != frames:
            raise ValueError(f"{cd}: {rep} の枚数が {frames} じゃない({len(mf)}/{len(df)})")
        out[f"{rep}_mask"] = np.stack([np.round(down_area(read_gray(p).astype(np.float32), res)).astype(np.uint8) for p in mf])
        out[f"{rep}_depth_mm"] = np.stack([np.round(down_depth(read_gray(p), res)).astype(np.uint16) for p in df])
    out["eef_num_raw"] = read_numeric(cd / "actions" / "eef_numeric.csv", EEF_COLS)
    out["eef_num"] = canon_quat_cols(out["eef_num_raw"], EEF_COLS)
    fc = full_cols(cd / "actions" / "fullbody_numeric.csv")
    out["full_num"] = canon_quat_cols(read_numeric(cd / "actions" / "fullbody_numeric.csv", fc), fc)
    out["full_cols"] = fc
    out["raster_steps"] = np.asarray(steps if steps else np.linspace(0, len(out["eef_num"]) - 1, frames).round(), np.int32)
    return out


def load_initial(sd, res):
    rgb = cv2.imread(str(sd / "initial" / "robot_free" / "rgb.png"), cv2.IMREAD_COLOR)[:, :, ::-1]   # RGB
    rgb = down_area(rgb, res).transpose(2, 0, 1).copy()
    d = cv2.imread(str(sd / "initial" / "robot_free" / "depth.exr"), cv2.IMREAD_UNCHANGED)
    if d.ndim == 3:
        d = d[:, :, 2]   # BGRA の R
    d = np.nan_to_num(d.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
    return rgb.astype(np.uint8), np.round(down_depth(d, res)).astype(np.uint16)


def load_final(cd, res, tol):
    fs = _json(cd / "final" / "final_state.json")
    lab = fs["labels"]
    tgt = next(o for o in fs["objects"] if o["name"] == "Target")
    out = {
        "goal": bool(lab["goal_reached"]),
        "coll": -1 if lab.get("secondary_collision") is None else int(bool(lab["secondary_collision"])),
        "fell": bool(lab["target_fell"]),
        "goal_dist": float(lab["goal_distance_m"]),
        "final_pos": tgt["final_position_world"],
        "final_yaw": yaw_deg(tgt["final_rotation_world_xyzw"]),
        "final_mask": np.round(down_area(read_gray(cd / "final" / "target_mask.png").astype(np.float32), res)).astype(np.uint8),
        "heat_target": np.round(down_area(read_gray(cd / "final" / "contact_heatmap_target.png").astype(np.float32), res)).astype(np.uint8),
        "heat_secondary": np.round(down_area(read_gray(cd / "final" / "contact_heatmap_secondary.png").astype(np.float32), res)).astype(np.uint8),
    }
    cp = first_contact_point(cd / "executed" / "contacts.csv", tol)
    out["contact_point"] = cp if cp else [np.nan] * 3
    out["contact_valid"] = cp is not None
    return out


def git_commit(sd):
    try:
        g = (_json(sd / "metadata.json").get("provenance") or {}).get("git") or {}
    except Exception:
        g = {}
    if not g:
        try:
            txt = (sd / "metadata.json").read_text(encoding="utf-8")
            i = txt.find('"commit"')
            if i >= 0:
                return txt[i:].split('"')[3]
        except Exception:
            pass
    return g.get("commit", "")


# ---------- 本体 ----------

def build(root: Path, out_path: Path, res: int, frames: int, tol: float, log=print, share_eef=True):
    scenes = {}
    for sd in sorted(root.glob("scene_*")):
        if not (sd / "candidates.json").is_file() or not (sd / "metadata.json").is_file():
            continue
        task = (_json(sd / "scene_initial.json").get("task") or {})
        scenes[sd.name] = {"dir": sd, "task": task.get("task_variant"), "pair": task.get("pair_id")}
    pairs = sorted({s["pair"] for s in scenes.values() if s["pair"]})
    pair_scene = {p: {s["task"]: n for n, s in scenes.items() if s["pair"] == p} for p in pairs}
    pairs = [p for p in pairs if set(pair_scene[p]) == {"A", "B"}]
    if not pairs:
        sys.exit(f"ペアが見つからない: {root}")
    log(f"[cache] {root}: ペア {len(pairs)}")

    first = scenes[pair_scene[pairs[0]]["A"]]["dir"]
    cam = _json(first / "camera.json")
    s = res / cam["width"]
    K = np.asarray(cam["K"], np.float64).copy()
    K[0, 0] *= s; K[1, 1] *= s
    K[0, 2] = (K[0, 2] + 0.5) * s - 0.5; K[1, 2] = (K[1, 2] + 0.5) * s - 0.5

    S = []          # scenes 行
    A = []          # actions 行(後で書く)
    E = []          # episodes 行
    act_index = {}
    mp_members = {}     # (pair_idx, mp) → [(action_idx, branch)]
    share_diff = {"eef_num": 0.0, "eef_mask_px": 0, "eef_depth_mm": 0}
    full_cols_ref = None
    commits = set()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.h5")
    R, T = res, frames
    comp = dict(compression="lzf", shuffle=True)
    with h5py.File(tmp, "w") as h:
        ga = h.create_group("actions")
        # 画像は 1 件ずつ書く(全部メモリに載せない。1 万 episode で 10 GB 超になるため)
        img_ds = {}
        for k, dtp in [("eef_mask", np.uint8), ("full_mask", np.uint8), ("eef_depth_mm", np.uint16), ("full_depth_mm", np.uint16)]:
            img_ds[k] = ga.create_dataset(k, (0, T, R, R), dtp, maxshape=(None, T, R, R), chunks=(1, T, R, R), **comp)
        n_cand_total = 0
        for pi, p in enumerate(pairs):
            for t in ["A", "B"]:
                sd = scenes[pair_scene[p][t]]["dir"]
                commits.add(git_commit(sd))
                gen = _json(sd / "scene_generation.json") if (sd / "scene_generation.json").is_file() else {}
                rgb, dep = load_initial(sd, res)
                sec = gen.get("secondary") or {}
                S.append({"scene_id": sd.name, "pair_idx": pi, "task": 0 if t == "A" else 1, "rgb": rgb, "depth": dep,
                          "goal_xz": [gen["goal"]["center_world"][0], gen["goal"]["center_world"][2]],
                          "target_pos": gen["target"]["position_world"], "target_yaw": gen["target"]["yaw_deg"],
                          "sec_pos": sec.get("position_world", [np.nan] * 3) if t == "B" else [np.nan] * 3,
                          "sec_h": sec.get("height_m", np.nan) if t == "B" else np.nan,
                          "seed": _json(sd / "scene_initial.json").get("seed")})
            sA = scenes[pair_scene[p]["A"]]["dir"]; sB = scenes[pair_scene[p]["B"]]["dir"]
            cands = _json(sA / "candidates.json")["candidates"]
            for ci, c in enumerate(cands):
                cid = c["candidate_id"]
                a = load_action(sA / "candidates" / cid, res, frames)
                # A/B で action が同じか(planned を共有しているはず)
                for f in ["eef_numeric.csv", "fullbody_numeric.csv"]:
                    if (sA / "candidates" / cid / "actions" / f).read_bytes() != (sB / "candidates" / cid / "actions" / f).read_bytes():
                        raise ValueError(f"{p} {cid}: A/B で {f} が違う")
                if full_cols_ref is None:
                    full_cols_ref = a["full_cols"]
                ai = len(A)
                for k, d in img_ds.items():
                    d.resize(ai + 1, axis=0)
                    d[ai] = a[k]
                A.append({**{k: a[k] for k in ["eef_num", "full_num", "raster_steps"]}, "eef_num_raw": a["eef_num_raw"], "shared_from": -1})
                act_index[(pi, cid)] = ai
                mp = -1
                if c.get("matched_pair_id"):
                    mp = int(c["matched_pair_id"].replace("mp", ""))
                br = {"wrist_down": 0, "wrist_up": 1}.get(c.get("branch"), -1)
                if mp >= 0:
                    mp_members.setdefault((pi, mp), []).append((ai, br))
                    mem = mp_members[(pi, mp)]
                    if len(mem) == 2 and share_eef:
                        src = next(x for x, b in mem if b == 0); dst = next(x for x, b in mem if b == 1)
                        if len(A[src]["eef_num"]) != len(A[dst]["eef_num"]):
                            raise ValueError(f"{p} mp{mp}: matched pair の長さが違う")
                        share_diff["eef_num"] = max(share_diff["eef_num"], float(np.abs(A[src]["eef_num"] - A[dst]["eef_num"]).max()))
                        share_diff["eef_mask_px"] = max(share_diff["eef_mask_px"], int((img_ds["eef_mask"][src] != img_ds["eef_mask"][dst]).sum()))
                        share_diff["eef_depth_mm"] = max(share_diff["eef_depth_mm"], int(np.abs(img_ds["eef_depth_mm"][src].astype(int) - img_ds["eef_depth_mm"][dst].astype(int)).max()))
                        A[dst]["shared_from"] = src
                        A[dst]["eef_num"] = A[src]["eef_num"].copy()
                        A[dst]["raster_steps"] = A[src]["raster_steps"].copy()
                        for k in ["eef_mask", "eef_depth_mm"]:
                            img_ds[k][dst] = img_ds[k][src]
                for t, sd in [("A", sA), ("B", sB)]:
                    fin = load_final(sd / "candidates" / cid, res, tol)
                    E.append({"scene_idx": 2 * pi + (0 if t == "A" else 1), "action_idx": ai, "pair_idx": pi,
                              "task": 0 if t == "A" else 1, "cand_idx": ci, "kind": KINDS.index(kind_of(c)),
                              "mp": mp, "branch": br, **fin})
                n_cand_total += 1
            log(f"[cache] {p}: 候補 {len(cands)}(合計 action {len(A)}、episode {len(E)})")

        # ---- actions ----
        N = len(A)
        Lmax = max(len(a["eef_num"]) for a in A)
        en = np.zeros((N, Lmax, 7), np.float32); fn = np.zeros((N, Lmax, len(full_cols_ref)), np.float32)
        ln = np.zeros(N, np.int32)
        for i, a in enumerate(A):
            L = len(a["eef_num"]); ln[i] = L
            en[i, :L] = a["eef_num"]; fn[i, :L] = a["full_num"]
        # 1 件ずつ読むので、chunk も 1 件ずつにする(まとめて圧縮すると 1 件読むたびに複数件を展開してしまう)
        ga.create_dataset("eef_num", data=en, chunks=(1,) + en.shape[1:], **comp)
        ga.create_dataset("full_num", data=fn, chunks=(1,) + fn.shape[1:], **comp)
        ga.create_dataset("num_len", data=ln)
        er = np.zeros((N, Lmax, 7), np.float32)
        for i, a in enumerate(A):
            er[i, :len(a["eef_num_raw"])] = a["eef_num_raw"]
        ga.create_dataset("eef_num_raw", data=er, chunks=(1,) + er.shape[1:], **comp)
        ga.create_dataset("eef_shared_from", data=np.array([a["shared_from"] for a in A], np.int32))
        ga.create_dataset("raster_steps", data=np.stack([a["raster_steps"] for a in A]))
        ga.attrs["eef_cols"] = json.dumps(EEF_COLS)
        ga.attrs["full_cols"] = json.dumps(full_cols_ref)

        # ---- scenes ----
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

        # ---- episodes ----
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
        ge.attrs["kinds"] = json.dumps(KINDS)

        # ---- pairs ----
        gp = h.create_group("pairs")
        gp.create_dataset("pair_id", data=np.array(pairs, dtype="S16"))
        gp.create_dataset("seed", data=np.array([S[2 * i]["seed"] for i in range(len(pairs))], np.int64))

        h.attrs.update({"version": VERSION, "res": res, "frames": frames, "source_root": str(root),
                        "created": dt.datetime.now().isoformat(timespec="seconds"),
                        "K": K, "T_world_to_cam_cv": np.asarray(cam["T_world_to_cam_cv"], np.float64),
                        "git_commits": json.dumps(sorted(commits)), "contact_touch_tolerance_m": tol,
                        "depth_unit": "mm (uint16), 0 = nothing", "image_layout": "row 0 = top (OpenCV)",
                        "eef_shared_in_matched_pairs": bool(share_eef), "eef_share_maxdiff": json.dumps(share_diff)})
    if share_eef:
        log(f"[cache] matched pair の EEF action を共有(wrist_down 側)。共有前の最大の差: {share_diff}")
    os.replace(tmp, out_path)
    log(f"[cache] 書いた: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)  scenes {len(S)}, actions {len(A)}, episodes {len(E)}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="Episodes/_p8_20scenes", help="プロジェクトからの相対 or 絶対パス")
    ap.add_argument("--out", default=None, help="省略時は <episodes>/_cache/wm_<res>x<frames>.h5")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--frames", type=int, default=25)
    ap.add_argument("--touch-tol", type=float, default=0.001)
    ap.add_argument("--no-share-eef", action="store_true", help="matched pair の EEF action を共有しない(v2 と同じ)")
    args = ap.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = Path(args.episodes)
    if not root.is_absolute():
        root = project / root
    out = Path(args.out) if args.out else root / "_cache" / f"wm_{args.res}x{args.frames}.h5"
    t0 = dt.datetime.now()
    build(root, out, args.res, args.frames, args.touch_tol, share_eef=not args.no_share_eef)
    print(f"[cache] {(dt.datetime.now() - t0).total_seconds():.0f} s")


if __name__ == "__main__":
    main()
