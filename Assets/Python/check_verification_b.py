"""検証 B: ロボットの姿勢を変えても robot-free 画像が完全に同じかを確かめる。

RobotFreeCapture (CaptureController) で撮った SOLO 出力と、そのときの撮影ログ CSV を使う。
- ロボットなしの全キャプチャを 1 枚目と比較し、RGB / semantic / instance / depth の差分画素数を数える(全部 0 で PASS)
- ロボットありのキャプチャで、ロボット(arm/hand)が実際に写っていたかも数える(写っていないとテストにならない)

使い方:
    python Assets/Python/check_verification_b.py                 # 最新の solo_* と最新の robot_free_log_*.csv
    python Assets/Python/check_verification_b.py --solo <dir> --log <csv>
"""
import argparse
import csv
import glob
import json
import os
import sys

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

ARM = (0, 255, 0)
HAND = (0, 0, 255)


def latest(pattern):
    c = glob.glob(pattern)
    return max(c, key=os.path.getmtime) if c else None


def load(seq, step):
    p = lambda s: os.path.join(seq, f"step{step}.camera{s}")
    rgb = cv2.imread(p(".png"), cv2.IMREAD_UNCHANGED)
    sem = cv2.imread(p(".semantic segmentation.png"), cv2.IMREAD_UNCHANGED)
    ins = cv2.imread(p(".instance segmentation.png"), cv2.IMREAD_UNCHANGED)
    dep = cv2.imread(p(".Depth.exr"), cv2.IMREAD_UNCHANGED)
    if dep is not None and dep.ndim == 3:
        dep = dep[:, :, 2]  # BGRA の R
    return rgb, sem, ins, dep


def count_color(img_bgra, rgb):
    b, g, r = rgb[2], rgb[1], rgb[0]
    return int(np.sum((img_bgra[:, :, 0] == b) & (img_bgra[:, :, 1] == g) & (img_bgra[:, :, 2] == r)))


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo", default=None)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    solo = a.solo or latest(os.path.join(os.path.expanduser("~"), "AppData", "LocalLow", "DefaultCompany", "PerceptionTestHDRP", "solo*"))
    log = a.log or latest(os.path.join(root, "CaptureLogs", "robot_free_log_*.csv"))
    seq = os.path.join(solo, "sequence.0")
    print(f"solo: {solo}\nlog : {log}")
    with open(log, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    free = [int(r["capture_index"]) for r in rows if r["robot_visible"].lower() in ("false", "0")]
    vis = [int(r["capture_index"]) for r in rows if r["robot_visible"].lower() in ("true", "1")]
    print(f"captures: {len(rows)} (robot-free {len(free)}, with-robot {len(vis)})")

    ref = load(seq, free[0])
    names = ["rgb", "semantic", "instance", "depth"]
    worst = {n: 0 for n in names}
    ok = True
    for s in free[1:]:
        cur = load(seq, s)
        diffs = []
        for n, x, y in zip(names, ref, cur):
            if x is None or y is None:
                diffs.append(f"{n}=MISSING")
                ok = False
                continue
            d = np.any(x != y, axis=-1) if x.ndim == 3 else (x != y)
            nd = int(np.sum(d))
            worst[n] = max(worst[n], nd)
            diffs.append(f"{n}={nd}")
            if nd:
                ok = False
        print(f"  step{s} vs step{free[0]}: " + ", ".join(diffs))

    robot_px = []
    for s in vis:
        sem = load(seq, s)[1]
        robot_px.append(count_color(sem, ARM) + count_color(sem, HAND))
    free_robot_px = [count_color(load(seq, s)[1], ARM) + count_color(load(seq, s)[1], HAND) for s in free]

    total = ref[0].shape[0] * ref[0].shape[1]
    print(f"\nworst differing pixels (of {total}): " + json.dumps(worst))
    print(f"robot pixels in with-robot captures: min={min(robot_px)}, max={max(robot_px)}")
    print(f"robot pixels in robot-free captures: max={max(free_robot_px)}")
    if min(robot_px) == 0:
        print("WARNING: some with-robot capture shows no robot (test is weaker)")
    if max(free_robot_px) > 0:
        ok = False
    print("\n検証 B: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
