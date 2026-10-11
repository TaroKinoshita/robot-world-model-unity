"""
P12: 中間報告(2026-10-30)用の図と数字

  python Docs/interim_20261030/make_figures.py
出力: このフォルダの fig*.png と numbers.json(materials.md の表の元)
使うデータ(消していない元データから毎回計算する):
  Episodes/_p8_20scenes              今の仕様の 20 シーン(seed 4〜23)
  Episodes/_p7_20scenes              P8 の修正前(同じ seed)
  Episodes/_archive/p4a_oldpush_*    元の押し方(開いた指 1 本)
  Episodes/_archive/p4b_pgs_3seeds_* PGS ソルバーの 3 シーン
  Episodes/                          3 シーンのパイロット(seed 1〜3、P5)
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import csv
import glob
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[1]
EP = PROJ / "Episodes"
P8 = EP / "_p8_20scenes"
P7 = EP / "_p7_20scenes"
OLD = sorted(EP.glob("_archive/p4a_oldpush_*"))[0]
PGS = sorted(EP.glob("_archive/p4b_pgs_3seeds_*"))[0]

# 色(dataviz の参照パレット、明るい背景)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, INK, INK2 = "#8a8986", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#b8b7b2", "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight"})
NUM = {}


def J(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def rgb(p):
    return cv2.imread(str(p), cv2.IMREAD_COLOR)[:, :, ::-1]


def tilt_deg(q):
    x, y, z, w = q
    # 回した上向きの軸 (0,1,0) の y 成分
    uy = 1 - 2 * (x * x + z * z)
    return math.degrees(math.acos(max(-1.0, min(1.0, uy))))


def scenes(root, task=None):
    out = []
    for sd in sorted(root.glob("scene_*")):
        if not (sd / "candidates.json").is_file():
            continue
        t = (J(sd / "scene_initial.json").get("task") or {})
        if task and t.get("task_variant") != task:
            continue
        out.append((sd, t))
    return out


def read_num(p):
    with open(p, newline="", encoding="utf-8") as f:
        r = list(csv.DictReader(f))
    return r


# ====================== Fig 1: matched pair(手首の反転) ======================
def fig1():
    # Task B で mp0 が片方だけ衝突するシーンを、番号の小さい順に探す
    for sd, t in scenes(P8, "B"):
        c = J(sd / "candidates.json")["candidates"]
        lab = [J(sd / "candidates" / x["candidate_id"] / "final" / "final_state.json")["labels"] for x in c[:2]]
        if lab[0]["secondary_collision"] != lab[1]["secondary_collision"]:
            break
    cd0, cd1 = sd / "candidates" / "c000", sd / "candidates" / "c001"
    e0 = read_num(cd0 / "actions" / "eef_numeric.csv"); e1 = read_num(cd1 / "actions" / "eef_numeric.csv")
    f0 = read_num(cd0 / "actions" / "fullbody_numeric.csv"); f1 = read_num(cd1 / "actions" / "fullbody_numeric.csv")
    n = min(len(e0), len(e1))
    t = np.array([float(e0[i]["t"]) for i in range(n)])
    P = lambda r, k: np.array([float(r[k + "_x"]), float(r[k + "_y"]), float(r[k + "_z"])])
    tcp = np.array([1000 * np.linalg.norm(P(e0[i], "tcp") - P(e1[i], "tcp")) for i in range(n)])
    fa = np.array([100 * np.linalg.norm(P(f0[i], "forearm_link") - P(f1[i], "forearm_link")) for i in range(n)])
    w1 = np.array([100 * np.linalg.norm(P(f0[i], "wrist_1_link") - P(f1[i], "wrist_1_link")) for i in range(n)])
    phase = [e0[i]["phase"] for i in range(n)]
    coll = {x: J(sd / "candidates" / x / "final" / "final_state.json")["labels"] for x in ["c000", "c001"]}
    hit = "c000" if coll["c000"]["secondary_collision"] else "c001"
    hit_step = coll[hit]["first_robot_secondary_step"]

    # 画像: robot-free の初期画像に、2 本の全身 mask(同じ時刻)を重ねる
    am = J(cd0 / "actions" / "actions_meta.json")
    steps = am.get("raster_steps") or list(range(25))
    k = int(np.argmin([abs(s - hit_step) for s in steps]))
    base = rgb(sd / "initial" / "robot_free" / "rgb.png").astype(np.float32) / 255
    m0 = cv2.imread(str(sorted((cd0 / "actions" / "fullbody_raster").glob("*.png"))[k]), 0) > 127
    m1 = cv2.imread(str(sorted((cd1 / "actions" / "fullbody_raster").glob("*.png"))[k]), 0) > 127
    img = base.copy()
    for m, col in [(m0, C2), (m1, C1)]:
        cc = np.array(matplotlib.colors.to_rgb(col))
        img[m] = 0.45 * img[m] + 0.55 * cc
    both = m0 & m1
    img[both] = 0.45 * base[both] + 0.55 * np.array(matplotlib.colors.to_rgb("#6250d6"))

    fig = plt.figure(figsize=(12, 4.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.2, 1.2], wspace=0.32)
    ax = fig.add_subplot(gs[0]); ax.imshow(img); ax.axis("off")
    ax.set_title(f"(a) same TCP path, two IK branches\n{sd.name} (Task B), step {steps[k]}", fontsize=9, color=INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C2, label="wrist-down"), Patch(color=C1, label="wrist-up"), Patch(color="#6250d6", label="both")],
              loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, fontsize=8)
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, tcp, color=INK2, lw=2)
    ax2.set_ylim(0, max(0.2, tcp.max() * 1.3)); ax2.set_xlabel("time (s)"); ax2.set_ylabel("TCP difference (mm)")
    ax2.set_title(f"(b) end-effector (TCP):\nmax {tcp.max():.3f} mm", fontsize=9, color=INK)
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(t, fa, color=C3, lw=2, label="forearm"); ax3.plot(t, w1, color=C4, lw=2, label="wrist 1")
    ax3.axvline(t[min(hit_step, n - 1)], color=GRAY, lw=1, ls="--")
    ax3.text(t[min(hit_step, n - 1)] - 0.15, 3, f"{hit.replace('c000', 'wrist-down').replace('c001', 'wrist-up')}\ntouches\ncylinder", fontsize=8, color=INK2, ha="right")
    ax3.set_ylim(0, 26)
    ax3.set_xlabel("time (s)"); ax3.set_ylabel("link difference (cm)"); ax3.legend(frameon=False, fontsize=8, loc="center right")
    ax3.set_title(f"(c) arm: forearm mean {fa.mean():.1f} cm,\nwrist 1 mean {w1.mean():.1f} cm", fontsize=9, color=INK)
    fig.savefig(HERE / "fig1_matched_pair.png"); plt.close(fig)
    NUM["fig1"] = {"scene": sd.name, "tcp_max_mm": float(tcp.max()), "forearm_mean_cm": float(fa.mean()),
                   "wrist1_mean_cm": float(w1.mean()), "collides": hit, "first_contact_step": hit_step}


# ====================== Fig 2: 押し方と「倒れた」の判定 ======================
def fig2():
    def collect(root):
        rows = []
        for sd, t in scenes(root):
            for c in J(sd / "candidates.json")["candidates"]:
                fs = J(sd / "candidates" / c["candidate_id"] / "final" / "final_state.json")
                o = next(x for x in fs["objects"] if x["name"] == "Target")
                rows.append({"sd": sd, "cid": c["candidate_id"], "tilt": tilt_deg(o["final_rotation_world_xyzw"]),
                             "drop_mm": 1000 * (o["initial_position_world"][1] - o["final_position_world"][1]),
                             "goal": fs["labels"]["goal_reached"], "task": t.get("task_variant"), "pos": o["final_position_world"]})
        return rows
    old, new = collect(OLD), collect(P8)
    # 例の画像: 元の押し方で「goal に入った」のに横倒し(Task A)/ 今の押し方で goal(Task A)
    ex_old = max((r for r in old if r["task"] == "A"), key=lambda r: (r["goal"], r["tilt"]))
    ex_new = next(r for r in new if r["task"] == "A" and r["goal"] and r["tilt"] < 5)

    def crop(r):
        im = rgb(r["sd"] / "candidates" / r["cid"] / "final" / "robot_free" / "rgb.png")
        cam = J(r["sd"] / "camera.json")
        T = np.array(cam["T_world_to_cam_cv"]); K = np.array(cam["K"])
        p = T @ np.r_[r["pos"], 1.0]; u = K[0, 0] * p[0] / p[2] + K[0, 2]; v = K[1, 1] * p[1] / p[2] + K[1, 2]
        h = 40; u0 = int(np.clip(u - h, 0, 256 - 2 * h)); v0 = int(np.clip(v - h, 0, 256 - 2 * h))
        return cv2.resize(np.ascontiguousarray(im[v0:v0 + 2 * h, u0:u0 + 2 * h]), (240, 240), interpolation=cv2.INTER_NEAREST)

    # 立方体は横倒しでも見た目がほぼ同じなので、画像ではなく数字で見せる(押し方の動画は Assets/Capture/push_methods_comparison.mp4)
    fig, c = plt.subplots(figsize=(6.4, 3.6))
    c.scatter([r["tilt"] for r in new], [r["drop_mm"] for r in new], s=14, color=C1, label=f"new push (P8, n={len(new)})", alpha=0.7, edgecolors="none")
    c.scatter([r["tilt"] for r in old], [r["drop_mm"] for r in old], s=26, color=C2, label=f"old push (P4-A, n={len(old)})", edgecolors="white", linewidths=0.6)
    c.axvline(30, color=INK2, lw=1, ls="--"); c.axhline(20, color=GRAY, lw=1, ls=":")
    c.text(31, 15, "new rule: fell if tilt > 30°\n(or drop ≥ 20 mm)", fontsize=8, color=INK2)
    c.text(31, 21, "old rule: fell only if drop ≥ 20 mm", fontsize=8, color=GRAY)
    no = sum(r["tilt"] > 30 for r in old); nn = sum(r["tilt"] <= 30 for r in new)
    c.annotate(f"old push: {no}/{len(old)} episodes here\n(tilt ≈ 90°, drop ≈ 0)\n{sum(bool(r['goal']) for r in old)} of them labelled goal = True",
               xy=(90, 0), xytext=(52, 7), fontsize=8, color=INK2, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    c.annotate(f"new push: {nn}/{len(new)} here\n(max tilt {max(r['tilt'] for r in new):.1f}°)", xy=(0.5, 0), xytext=(4, 9), fontsize=8, color=INK2,
               arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    c.set_xlim(-3, 100); c.set_ylim(-3, 28)
    c.set_xlabel("final tilt of the box (°)"); c.set_ylabel("height drop of the centre (mm)")
    c.legend(frameon=False, fontsize=8, loc="upper right")
    c.set_title("A cube on its side keeps its centre height → tilt is needed", fontsize=9, color=INK)
    fig.savefig(HERE / "fig2_push_and_fall.png"); plt.close(fig)
    NUM["fig2"] = {"old_n": len(old), "old_tilted_gt30": sum(r["tilt"] > 30 for r in old), "old_drop_ge20": sum(r["drop_mm"] >= 20 for r in old),
                   "old_goal_true": sum(bool(r["goal"]) for r in old),
                   "old_goal_true_and_tilted": sum(bool(r["goal"]) and r["tilt"] > 30 for r in old),
                   "new_n": len(new), "new_tilted_gt30": sum(r["tilt"] > 30 for r in new), "new_max_tilt": max(r["tilt"] for r in new)}


# ====================== Fig 3: 衝突 = 「触れた」接触だけ ======================
def fig3():
    mins, reported_only = [], 0
    n_b = 0
    for sd, t in scenes(P8, "B"):
        for c in J(sd / "candidates.json")["candidates"]:
            n_b += 1
            best = None
            with open(sd / "candidates" / c["candidate_id"] / "executed" / "contacts.csv", newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    pair = {r["body"], r["other"]}
                    if "SecondObject" in pair and ({"Target"} & pair or r["other_kind"] == "robot_link" or r["body_kind"] == "robot_link"):
                        try:
                            v = float(r["min_separation"])
                        except ValueError:
                            continue
                        best = v if best is None else min(best, v)
            if best is not None:
                mins.append(1000 * best)
                reported_only += best > 0.001
    mins = np.array(mins)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bins = np.linspace(min(-4, mins.min()), max(4, mins.max()) + 0.25, 40)
    ax.hist(mins[mins <= 1], bins=bins, color=C1, label=f"touching (gap ≤ 1 mm): {int((mins <= 1).sum())}", edgecolor="white", linewidth=0.8)
    ax.hist(mins[mins > 1], bins=bins, color=C4, label=f"reported but not touching: {int((mins > 1).sum())}", edgecolor="white", linewidth=0.8)
    ax.axvline(1, color=INK2, lw=1, ls="--")
    ax.set_xlabel("smallest gap to the cylinder in an episode (mm; < 0 = overlap)")
    ax.set_ylabel("Task B episodes")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"PhysX reports contacts within the contact offset (P8, {n_b} Task B episodes)", fontsize=9, color=INK)
    fig.savefig(HERE / "fig3_touch_only.png"); plt.close(fig)
    NUM["fig3"] = {"task_b_episodes": n_b, "with_reported_contact": int(len(mins)), "touching": int((mins <= 1).sum()),
                   "reported_not_touching": int(reported_only), "max_gap_mm": float(mins.max())}


# ====================== Fig 4: A/B のずれ(PGS → TGS + 候補ごとの物理シーン) ======================
def fig4():
    def diffs(root):
        out = []
        for pc in sorted(root.glob("pairs/*/pair_check.json")):
            for r in J(pc)["results"]:
                if r["B"].get("secondary_collision") is False:     # 円柱に触れていない候補だけ
                    out.append(float(r["final_position_diff_A_B_mm"]))
        return np.array(out)
    pgs, tgs = diffs(PGS), diffs(P8)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    rng = np.random.default_rng(0)
    floor = 1e-3
    for i, (d, col, name) in enumerate([(pgs, C2, "PGS solver\n(P4-B, 3 scenes)"), (tgs, C1, "TGS + fresh physics\nscene (P8, 20 scenes)")]):
        y = np.maximum(d, floor)
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(y)), y, s=16, color=col, edgecolors="white", linewidths=0.5)
        ax.text(i, 25, f"n = {len(d)}\nmax {d.max():.1f} mm\n> 1 mm: {(d > 1).sum()}\n= 0: {(d == 0).sum()}", ha="center", fontsize=8, color=INK2)
    ax.set_yscale("log"); ax.set_ylim(5e-4, 2000)
    ax.axhline(3, color=INK2, lw=1, ls="--"); ax.text(1.42, 3.3, "3 mm (H1/H3 margin)", fontsize=8, color=INK2, ha="right")
    ax.axhline(1, color=GRAY, lw=1, ls=":"); ax.text(1.42, 0.62, "1 mm (unstable flag)", fontsize=8, color=GRAY, ha="right")
    ax.text(-0.45, 1.3e-3, "0 mm drawn at 0.001", fontsize=7, color=GRAY)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["PGS solver\n(P4-B, 3 scenes)", "TGS + fresh physics\nscene (P8, 20 scenes)"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("|final position A − B| (mm, log)")
    ax.set_title("Task A vs B for candidates that never touch the cylinder", fontsize=9, color=INK, pad=12)
    fig.savefig(HERE / "fig4_ab_stability.png"); plt.close(fig)
    NUM["fig4"] = {"pgs_n": int(len(pgs)), "pgs_max_mm": float(pgs.max()), "pgs_gt1": int((pgs > 1).sum()), "pgs_zero": int((pgs == 0).sum()),
                   "tgs_n": int(len(tgs)), "tgs_max_mm": float(tgs.max()), "tgs_gt1": int((tgs > 1).sum()), "tgs_zero": int((tgs == 0).sum()),
                   "pgs_sorted_top": sorted(pgs.tolist())[-3:]}


# ====================== Fig 5: 候補の内訳と結果 ======================
KJA = ["matched pair", "toward goal", "goal edge", "random", "toward cylinder"]


def kind_of(c):
    note = (c.get("note") or "").lower()
    if c.get("matched_pair_id"):
        return 0
    if "no toward-secondary" in note:
        return 3
    if "secondary" in note:
        return 4
    if "goal edge" in note:
        return 2
    if "goal" in note:
        return 1
    return 3


def outcomes(root):
    cnt = {k: Counter() for k in range(5)}
    for sd, t in scenes(root, "B"):
        for c in J(sd / "candidates.json")["candidates"]:
            l = J(sd / "candidates" / c["candidate_id"] / "final" / "final_state.json")["labels"]
            cnt[kind_of(c)][(bool(l["goal_reached"]), bool(l["secondary_collision"]))] += 1
    return cnt


def fig5():
    cats = [((True, False), "goal, no collision", C3), ((True, True), "goal + collision", C4),
            ((False, True), "collision, no goal", C2), ((False, False), "neither", "#c9c8c2")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharex=True)
    out = {}
    for ax, (root, name) in zip(axes, [(P7, "P7: before (goal-directed 6)"), (P8, "P8: now (3 goal + 3 goal-edge)")]):
        cnt = outcomes(root)
        ks = [k for k in range(5) if sum(cnt[k].values()) > 0][::-1]
        for yi, k in enumerate(ks):
            tot = sum(cnt[k].values()); left = 0
            for key, lab, col in cats:
                v = cnt[k][key] / tot
                if v > 0:
                    ax.barh(yi, v, left=left, color=col, edgecolor="white", linewidth=2, height=0.62)
                    if v >= 0.08:
                        ax.text(left + v / 2, yi, f"{100 * v:.0f}%", ha="center", va="center", fontsize=7.5, color=INK)
                left += v
            ax.text(1.01, yi, f"n={tot}", va="center", fontsize=7.5, color=INK2)
        ax.set_yticks(range(len(ks))); ax.set_yticklabels([KJA[k] for k in ks])
        ax.set_xlim(0, 1); ax.set_xlabel("share of Task B episodes"); ax.grid(axis="y", visible=False)
        ax.set_title(name, fontsize=9, color=INK)
        out[name] = {KJA[k]: {f"{a}/{b}": v for (a, b), v in cnt[k].items()} for k in range(5)}
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=col, label=lab) for _, lab, col in cats], loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.1))
    fig.subplots_adjust(wspace=0.45)
    fig.savefig(HERE / "fig5_candidate_mix.png"); plt.close(fig)
    NUM["fig5"] = out
    # 種類だけで当てたときの正解率(inspection.json から)
    NUM["shortcut"] = {}
    for root, key in [(P7, "P7"), (P8, "P8")]:
        ins = J(root / "inspection.json")
        NUM["shortcut"][key] = {k: {"baseline": v["baseline_acc"], "by_type": v["by_type_acc"]} for k, v in ins["type_predictability"].items()}


# ====================== Fig 6: 円柱の隠れと高さ(P8 の修正) ======================
def fig6():
    # P7 で初期画像のターゲットが一番隠れたシーン(validation の初期 IoU が最小)
    worst, wv = None, 1.0
    for sd, t in scenes(P7, "B"):
        v = J(sd / "validation.json")
        for c in v["checks"]:
            if "初期 target 箱" in c["check"] and not c["pass"]:
                iou = float(c["detail"].split("=")[1])
                if iou < wv:
                    worst, wv = sd, iou
    same = P8 / worst.name
    fig = plt.figure(figsize=(11, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.4], wspace=0.3)
    for i, (sd, title) in enumerate([(worst, f"(a) P7 {worst.name}: target hidden\n(initial mask IoU {wv:.2f})"),
                                     (same, f"(b) P8, same seed: cylinder placed\nout of the camera's line of sight")]):
        ax = fig.add_subplot(gs[i]); ax.imshow(rgb(sd / "initial" / "robot_free" / "rgb.png")); ax.axis("off")
        ax.set_title(title, fontsize=9, color=INK)
    ax = fig.add_subplot(gs[2])
    hs = list(range(24, 42, 2)); w = 0.38
    for j, (root, name, col) in enumerate([(P7, "P7 (lowest height first)", C2), (P8, "P8 (random order, 32–40 cm)", C1)]):
        hc = Counter(round(100 * J(sd / "scene_generation.json")["secondary"]["height_m"]) for sd, t in scenes(root, "B"))
        ax.bar(np.arange(len(hs)) + (j - 0.5) * w, [hc.get(h, 0) for h in hs], width=w, color=col, label=name, edgecolor="white", linewidth=1)
    ax.set_xticks(range(len(hs))); ax.set_xticklabels([str(h) for h in hs]); ax.set_xlabel("cylinder height (cm)"); ax.set_ylabel("scenes")
    ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.grid(axis="x", visible=False); ax.set_ylim(0, 23)
    ax.set_title("(c) cylinder height per scene (20 scenes each)", fontsize=9, color=INK)
    fig.savefig(HERE / "fig6_occlusion_height.png"); plt.close(fig)
    NUM["fig6"] = {"p7_worst_scene": worst.name, "p7_worst_initial_iou": wv}
    for root, key in [(P7, "P7"), (P8, "P8")]:
        ins = J(root / "inspection.json")
        NUM["fig6"][key] = {"height_cm": ins["secondary"]["height_cm"], "occlusion_final": ins.get("occlusion_final"),
                            "validation_failed": ins["validation_failed"]}


# ====================== パイロットの数字 ======================
def pilot_numbers():
    out = {}
    for root, key in [(EP, "pilot3_P5"), (P8, "p8_20")]:
        ins = J(root / "inspection.json") if (root / "inspection.json").is_file() else None
        if ins is None:
            continue
        bk = ins["by_kind"]["all"]
        out[key] = {
            "scenes": ins["num_scenes"], "episodes": ins["num_episodes"],
            "goal_A": [bk["goal_A"], bk["n_A"]], "goal_B": [bk["goal_B"], bk["n_B"]], "coll_B": [bk["coll_B"], bk["n_B"]],
            "robot_to_cyl": bk["robot_sec"], "target_to_cyl": bk["target_sec"], "fell": ins["fell"], "robot_table": ins["robot_table"],
            "tracking_mm": ins["tracking_mm"], "matched_pair": ins["matched_pair"], "sim_unstable": ins["sim_unstable"],
            "validation_failed": ins["validation_failed"], "pair_failed": ins["pair_failed"],
            "height_cm": ins["secondary"]["height_cm"], "batch": ins.get("batch", {}).get("tried"),
            "batch_failed": ins.get("batch", {}).get("failed"),
        }
    # 検証の項目数(A・B・ペア)
    for root, key in [(EP, "pilot3_P5"), (P8, "p8_20")]:
        tot = {"A": [0, 0], "B": [0, 0], "pair": [0, 0]}
        for sd, t in scenes(root):
            v = J(sd / "validation.json")
            k = t.get("task_variant")
            tot[k][0] += v["num_checks"] - v["num_failed"]; tot[k][1] += v["num_checks"]
        for pc in root.glob("pairs/*/pair_check.json"):
            v = J(pc); tot["pair"][0] += v["num_checks"] - v["num_failed"]; tot["pair"][1] += v["num_checks"]
        out.setdefault(key, {})["checks"] = tot
    # バッチの時間
    sm = sorted((P8 / "_batch").glob("batch_summary_*.json"))
    if sm:
        recs = J(sm[-1])["records"]
        out["p8_20"]["play_min"] = [float(np.median([r["playMin"] for r in recs])), float(max(r["playMin"] for r in recs))]
    # キャッシュ
    cc = P8 / "_cache" / "wm_128x25.h5"
    if cc.is_file():
        out["p8_20"]["cache_mb"] = cc.stat().st_size / 1e6
    NUM["pilot"] = out


if __name__ == "__main__":
    # 点検の数字は毎回作り直す(inspect_batch.py の数え方を直したときに古い数字が残らないように)
    import subprocess, sys
    for root in [EP, P7, P8]:
        subprocess.run([sys.executable, str(PROJ / "Assets" / "Python" / "inspect_batch.py"), "--episodes", str(root)], check=True,
                       stdout=subprocess.DEVNULL)
    for f in [fig1, fig2, fig3, fig4, fig5, fig6, pilot_numbers]:
        f(); print("done", f.__name__)
    (HERE / "numbers.json").write_text(json.dumps(NUM, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(NUM, indent=1, ensure_ascii=False, default=str)[:4000])
