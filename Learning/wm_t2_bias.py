'''
探索的（判定に使わない）: H5 で画像の腕全体モデルが「肩 B 側が止まる」に寄る理由を見る。
matched pair ごとに d = c_B − c_A（予測した衝突確率の差）を出し、
 (1) 片方だけ止まる組で、d が正しい向きに動いているか（AUROC）
 (2) 両方同じ結果の組で d の平均（ずれ）
 (3) しきい値 0 の代わりに、val の「両方同じ組」の d の中央値（test のときだけ）を引いたときの balanced
を、数値・画像の腕全体モデルで出す。
'''
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wm_eval, subprocess
from wm_dataset import make_splits
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'Episodes/_task2_v1/_cache/wm_128x25.h5'; RUNS = ROOT / 'Docs/encoder_fix/results_t2'
t = wm_eval.load_truth(CACHE); ids = t['pair_ids']; sp = make_splits(CACHE)
B1 = t['task'] == 1

def pair_d(pred, pairs):
    m = dict(zip(pred['ep'].tolist(), pred['coll'].tolist())); rows = []
    for p in pairs:
        e = np.nonzero(B1 & (t['pair_idx'] == p))[0]
        for mp in np.unique(t['mp'][e]):
            ee = e[t['mp'][e] == mp]
            if len(ee) != 2: continue
            a = ee[t['branch'][ee] == 0]; b = ee[t['branch'][ee] == 1]
            if len(a) != 1 or len(b) != 1 or a[0] not in m: continue
            rows.append((m[b[0]] - m[a[0]], int(t['coll'][a[0]]), int(t['coll'][b[0]])))
    return np.array(rows)

out = ['# 画像モデルの H5 の偏り（探索的）\n', 'd = c_B − c_A。片方だけ止まる組で、B だけ止まるなら d > 0 が正解。\n',
       '| モデル | 評価 | AUROC（d で A だけ / B だけを分ける） | 両方同じ組の d の平均 | balanced（しきい値 0） | balanced（しきい値 = val の両方同じ組の d 中央値、test のみ） |', '|---|---|---|---|---|---|']
for rep in ['full_num', 'full_img']:
    for split in ['val', 'test']:
        P = [wm_eval.load_predictions(f) for f in sorted((RUNS / f'preds_{split}').glob(f'{rep}__*t2.npz'))]
        Ptr = [wm_eval.load_predictions(f) for f in sorted((RUNS / 'preds_val').glob(f'{rep}__*t2.npz'))]
        pr = [ids.index(p) for p in sp[split]]; ptr = [ids.index(p) for p in sp['val']]
        au, bias, b0, bt = [], [], [], []
        for p, q in zip(P, Ptr):
            r = pair_d(p, pr); rt = pair_d(q, ptr) if split == 'test' else None
            dis = r[r[:, 1] != r[:, 2]]; same = r[r[:, 1] == r[:, 2]]
            pos = dis[dis[:, 2] == 1, 0]; neg = dis[dis[:, 1] == 1, 0]
            au.append(np.mean([(x > y) + 0.5 * (x == y) for x in pos for y in neg])); bias.append(same[:, 0].mean())
            thr = np.median(rt[rt[:, 1] == rt[:, 2], 0]) if rt is not None and len(rt) else 0.0
            b0.append(0.5 * ((pos > 0).mean() + (neg < 0).mean())); bt.append(0.5 * ((pos > thr).mean() + (neg < thr).mean()))
        out.append(f'| {rep} | {split} | {np.mean(au):.3f} | {np.mean(bias):+.3f} | {np.mean(b0):.3f} | ' + (f'{np.mean(bt):.3f}' if split == 'test' else '—') + ' |')
md = '\n'.join(out) + '\n'; (RUNS / 'h5_bias.md').write_text(md, encoding='utf-8'); print(md)
