'''
matched pair の EEF action を共有する前の差を、位置（mm）と回転（度）に分けて出す。

wm_cache.py のログの「共有前の最大の差 eef_num」は、位置と quaternion の成分をまとめた最大値なので、
どちらから来た差か分からない。キャッシュに残っている eef_num_raw（共有する前の値）から分けて出す。

使い方（プロジェクトのフォルダで）:
    py -3.11 Learning/check_eef_share_diff.py
    py -3.11 Learning/check_eef_share_diff.py --cache Episodes/_main_v1/_cache/wm_128x25.h5
'''
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def q(v):
    return f'中央 {np.median(v):.4f} / 95% {np.percentile(v, 95):.4f} / 最大 {v.max():.4f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=str(ROOT / 'Episodes/_main_v1/_cache/wm_128x25.h5'))
    args = ap.parse_args()
    with h5py.File(args.cache, 'r') as h:
        g = h['actions']
        cols = json.loads(g.attrs['eef_cols'])
        src = g['eef_shared_from'][:]
        nlen = g['num_len'][:]
        raw = g['eef_num_raw']
        dsts = np.nonzero(src >= 0)[0]
        print(f'cache: {args.cache}')
        print(f'列: {cols}（位置は Unity の world 座標、単位 m として mm に直す）')
        print(f'共有した matched pair: {len(dsts)} 組')
        pos_mm, ang_deg, pos_comp, quat_comp = [], [], [], []
        for d in dsts:
            s = int(src[d]); n = int(nlen[d])
            a = raw[int(d)][:n].astype(np.float64); b = raw[s][:n].astype(np.float64)
            pos_mm.append(np.linalg.norm(a[:, :3] - b[:, :3], axis=1).max() * 1000)
            pos_comp.append(np.abs(a[:, :3] - b[:, :3]).max())
            qa = a[:, 3:7] / np.linalg.norm(a[:, 3:7], axis=1, keepdims=True)
            qb = b[:, 3:7] / np.linalg.norm(b[:, 3:7], axis=1, keepdims=True)
            dot = np.sum(qa * qb, axis=1)
            sign = np.where(dot < 0, -1.0, 1.0)[:, None]          # q と -q は同じ回転なので符号をそろえる
            quat_comp.append(np.abs(qa - sign * qb).max())
            ang_deg.append(np.degrees(2 * np.arccos(np.clip(np.abs(dot), 0, 1))).max())
    pos_mm = np.array(pos_mm); ang_deg = np.array(ang_deg)
    print()
    print(f'位置の差（mm、1 組の中の最大）: {q(pos_mm)}')
    print(f'回転の差（度、1 組の中の最大）: {q(ang_deg)}')
    print(f'成分の最大の差: 位置 {max(pos_comp):.6f}（m）、quaternion {max(quat_comp):.6f}')
    print(f'位置の差が 0.2 mm 以上の組: {(pos_mm >= 0.2).sum()} / {len(pos_mm)}')
    which = '位置' if max(pos_comp) >= max(quat_comp) else 'quaternion'
    print(f'→ wm_cache.py のログの最大の差（0.0027 など）は、{which} の成分から来ている')


if __name__ == '__main__':
    main()
