"""P1.5-3: 1 サンプル分のデータ読み込み時間を測る(実ファイル)

本番の Full-Image サンプル = action mask PNG x T + action depth EXR x T + 初期 RGB/depth + 正解(最終 mask など)。
action depth EXR はまだ無いので、同じ解像度の initial depth EXR を T 回読んで代用する。

  python Assets/Python/bench_data_loading.py [--episode Episodes/scene_0000/candidates/c000] [--reps 5]
"""
import argparse
import glob
import os
import statistics
import sys
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from validate_episode import load_episode  # noqa: E402


def timed(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts), min(ts), max(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="Episodes/scene_0000/candidates/c000")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    ep_dir = a.episode
    scene_dir = os.path.dirname(os.path.dirname(ep_dir))
    masks = sorted(glob.glob(os.path.join(ep_dir, "actions", "fullbody_raster", "*.png")))
    T = len(masks)
    depth_exr = os.path.join(scene_dir, "initial", "robot_free", "depth.exr")
    rgb_png = os.path.join(scene_dir, "initial", "robot_free", "rgb.png")
    final_mask = os.path.join(ep_dir, "final", "target_mask.png")

    def load_masks():
        return np.stack([cv2.imread(p, cv2.IMREAD_UNCHANGED) for p in masks])

    def load_depths():
        return np.stack([cv2.imread(depth_exr, cv2.IMREAD_UNCHANGED)[..., 2] for _ in range(T)])

    def load_sample():
        m = load_masks(); d = load_depths()
        rgb = cv2.imread(rgb_png, cv2.IMREAD_UNCHANGED); dep = cv2.imread(depth_exr, cv2.IMREAD_UNCHANGED)
        fm = cv2.imread(final_mask, cv2.IMREAD_UNCHANGED)
        return m, d, rgb, dep, fm

    def load_npz_like():   # 参考: 1 サンプルを 1 つの圧縮なし配列ファイルにまとめた場合
        return np.load(npy_path)

    print(f"episode: {ep_dir}  (T={T} frames)")
    r_m = timed(load_masks, a.reps); print(f"mask PNG x{T}:            median {r_m[0]*1000:.1f} ms")
    r_d = timed(load_depths, a.reps); print(f"depth EXR x{T} (proxy):   median {r_d[0]*1000:.1f} ms")
    r_s = timed(load_sample, a.reps); print(f"full sample (files):      median {r_s[0]*1000:.1f} ms")
    r_e = timed(lambda: load_episode(ep_dir), a.reps); print(f"load_episode() as-is:     median {r_e[0]*1000:.1f} ms")

    m, d, rgb, dep, fm = load_sample()
    arr = np.concatenate([m[:, None].astype(np.float16) / 255.0, d[:, None].astype(np.float16)], axis=1)
    npy_path = os.path.join(os.environ.get("TEMP", "."), "bench_sample.npy")
    np.save(npy_path, arr)
    r_n = timed(load_npz_like, a.reps)
    size_mb = os.path.getsize(npy_path) / 2**20
    print(f"packed .npy (fp16 {arr.shape}, {size_mb:.1f} MB): median {r_n[0]*1000:.1f} ms")
    os.remove(npy_path)


if __name__ == "__main__":
    main()
