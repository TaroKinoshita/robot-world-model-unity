'''
protocol 追記 12 の探索的分析：区画の余裕（幅 / 2 − |ずれ|）でシーンペアを 3 等分し、
衝突 AUPRC の 腕全体 − 手先 を出す（数値・画像、5 seed 平均、シーンペア単位 bootstrap）。判定には使わない。
使い方: python Learning/wm_t2_tertiles.py   （wm_predict.py の val / test 予測の後）
'''
import sys, glob
from pathlib import Path
import numpy as np, h5py
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wm_eval
from wm_dataset import make_splits

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / 'Episodes/_task2_v1/_cache/wm_128x25.h5'
RUNS = ROOT / 'Docs/encoder_fix/results_t2'
truth = wm_eval.load_truth(CACHE); ids = truth['pair_ids']; sp = make_splits(CACHE)
with h5py.File(CACHE, 'r') as h:
    sp_idx = h['scenes/pair_idx'][:]; w = h['scenes/t2_width'][:]; off = h['scenes/t2_offset'][:]
clear = {}
for p, ww, oo in zip(sp_idx, w, off):
    clear[int(p)] = float(ww / 2 - abs(oo))


def load(split, rep):
    return [wm_eval.load_predictions(f) for f in sorted((RUNS / f'preds_{split}').glob(f'{rep}__*t2.npz'))]


def auprc_set(preds, eps):
    vals = []
    for p in preds:
        m = dict(zip(p['ep'].tolist(), p['coll'].tolist()))
        vals.append(wm_eval.average_precision(truth['coll'][eps].astype(float), np.array([m[e] for e in eps])))
    return float(np.mean(vals))


out = ['# 区画の余裕ごとの衝突 AUPRC（探索的、protocol 追記 12。判定に使わない）\n',
       '余裕 = 区画の幅 / 2 − |区画のずれ|。評価するシーンペアの中で 3 等分。値は 5 seed 平均、[ ] はシーンペア単位 bootstrap 95% CI（2000 回）。\n']
for split_name, splits in [('test', ['test']), ('val + test', ['val', 'test'])]:
    pairs = [ids.index(p) for s in splits for p in sp[s]]
    preds = {r: sum((load(s, r) for s in splits), []) for r in ['eef_num', 'full_num', 'eef_img', 'full_img']}
    # val と test の予測は別ファイルなので seed ごとにまとめる
    if len(splits) == 2:
        for r in preds:
            a, b = load('val', r), load('test', r)
            preds[r] = [{k: np.concatenate([x[k], y[k]]) for k in ('ep', 'coll')} for x, y in zip(a, b)]
    c = np.array([clear[p] for p in pairs]); q = np.quantile(c, [1 / 3, 2 / 3])
    groups = [('狭い', c <= q[0]), ('中', (c > q[0]) & (c <= q[1])), ('広い', c > q[1])]
    out.append(f'\n## {split_name}（{len(pairs)} シーンペア、境目 {q[0]*100:.1f} cm / {q[1]*100:.1f} cm）\n')
    out.append('| 余裕 | ペア | 衝突率 | 手先 数値 | 腕全体 数値 | **差 数値** | 手先 画像 | 腕全体 画像 | **差 画像** |\n|---|---|---|---|---|---|---|---|---|')
    rng = np.random.default_rng(0)
    for name, msk in groups:
        gp = np.array(pairs)[msk]
        eps_of = {int(p): np.nonzero((truth['pair_idx'] == p) & (truth['task'] == 1))[0] for p in gp}
        allep = np.concatenate([eps_of[int(p)] for p in gp])
        row = [name, str(len(gp)), f"{truth['coll'][allep].mean():.2f}"]
        for f in ['num', 'img']:
            e, a = auprc_set(preds[f'eef_{f}'], allep), auprc_set(preds[f'full_{f}'], allep)
            bs = []
            for _ in range(2000):
                bp = gp[rng.integers(0, len(gp), len(gp))]
                be = np.concatenate([eps_of[int(p)] for p in bp])
                if truth['coll'][be].min() == truth['coll'][be].max():
                    continue
                bs.append(auprc_set(preds[f'full_{f}'], be) - auprc_set(preds[f'eef_{f}'], be))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            row += [f'{e:.3f}', f'{a:.3f}', f'**{a - e:+.3f}** [{lo:+.3f}, {hi:+.3f}]']
        out.append('| ' + ' | '.join(row) + ' |')
md = '\n'.join(out) + '\n'
(RUNS / 'tertiles.md').write_text(md, encoding='utf-8'); print(md)
