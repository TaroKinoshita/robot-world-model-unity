"""
検証A: ロボットがターゲットを隠す姿勢でも、robot-free の RGB / semantic / depth が欠けないかを判定する。

使い方:
    python check_verification_a.py "<solo_Nのパス>/sequence.0"

前提: 偶数step = ロボットあり、奇数step = robot-free(RobotFreeCapture.cs の出力)
必要: pip install opencv-python numpy pillow
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # cv2 の import より前に必要

import sys
import glob
import re
import numpy as np
import cv2
from PIL import Image

TARGET = (255, 0, 0)                     # target_object の semantic 色
ROBOT = [(0, 0, 255), (0, 255, 0)]       # hand, arm


def load_rgb(d, s):
    return np.array(Image.open(os.path.join(d, f"step{s}.camera.png")))[..., :3].astype(int)


def load_sem(d, s):
    return np.array(Image.open(os.path.join(d, f"step{s}.camera.semantic segmentation.png")))[..., :3]


def load_depth(d, s):
    x = cv2.imread(os.path.join(d, f"step{s}.camera.Depth.exr"), cv2.IMREAD_UNCHANGED)
    return x[..., 2] if x.ndim == 3 else x   # OpenCV は BGRA 順なので R = index 2


def count(sem, color):
    return int((sem == color).all(-1).sum())


def main(d):
    steps = sorted(int(re.search(r"step(\d+)\.camera\.png$", f).group(1))
                   for f in glob.glob(os.path.join(d, "step*.camera.png")))
    free_steps = [s for s in steps if s % 2 == 1]
    ref = free_steps[0]
    ref_rgb, ref_sem, ref_dep = load_rgb(d, ref), load_sem(d, ref), load_depth(d, ref)
    ref_target = count(ref_sem, TARGET)

    print(f"steps: {len(steps)}  / 基準 robot-free: step{ref} (target {ref_target}px)\n")
    print("pair | あり:target見え | あり:robot px | なし:target | なし:depth穴 | なし:NaN | RGB差 | sem差 | depth差")
    print("-" * 100)

    all_ok = True
    occluded = 0
    for s in free_steps:
        p = (s - 1) // 2
        sem_with = load_sem(d, s - 1)
        vis = count(sem_with, TARGET)
        robot_px = sum(count(sem_with, c) for c in ROBOT)

        rgb, sem, dep = load_rgb(d, s), load_sem(d, s), load_depth(d, s)
        tgt = count(sem, TARGET)
        sky = (sem == 0).all(-1)
        holes = int(((dep <= 0) & ~sky).sum())      # 物体があるのに深度0
        nans = int(np.isnan(dep).sum())
        d_rgb = int(np.abs(rgb - ref_rgb).max())
        d_sem = int((sem != ref_sem).any(-1).sum())
        d_dep = float(np.nanmax(np.abs(dep - ref_dep)))

        ok = (tgt == ref_target and holes == 0 and nans == 0
              and d_rgb == 0 and d_sem == 0 and d_dep == 0.0)
        all_ok &= ok
        if vis < ref_target:
            occluded += 1
        mark = "OK" if ok else "NG"
        print(f"{p:4d} | {vis:5d}/{ref_target:<5d}    | {robot_px:12d} | {tgt:10d} | {holes:10d} | {nans:7d}"
              f" | {d_rgb:5d} | {d_sem:5d} | {d_dep:.6f}  {mark}")

    print("-" * 100)
    print(f"ターゲットが隠れた姿勢: {occluded} / {len(free_steps)}")
    if occluded == 0:
        print("判定: 保留(ターゲットを隠す姿勢が1つもない → 姿勢を変えて再撮影)")
    elif all_ok:
        print("判定: 検証A クリア(隠れた姿勢でも robot-free は完全一致・欠けなし)")
    else:
        print("判定: NG あり(NG の行を確認)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
