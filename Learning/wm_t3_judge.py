'''
3 つ目のタスクの H1′（protocol 追記 13）：箱の最後の位置の誤差（Task B の行）で、手先 − 腕全体 の 95% CI 下限 > 0。数値・画像で Holm。
探索的：片方だけ腕が触れる組（matched pair）に限った同じ差。
H5・H2 は wm_test_judge.py（WMJ_RUNS / WMJ_CACHE / WMJ_TAG=t3）で出す。
使い方: python Learning/wm_t3_judge.py [--split test|val]   （wm_predict.py --split <split> --name <split> の後）
'''
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wm_eval
from wm_dataset import make_splits

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser(); ap.add_argument('--split', default='test')
ap.add_argument('--runs', default='Docs/encoder_fix/results_t3'); ap.add_argument('--cache', default='Episodes/_task3_v1/_cache/wm_128x25.h5')
args = ap.parse_args()
RUNS = ROOT / args.runs; CACHE = ROOT / args.cache
truth = wm_eval.load_truth(CACHE); ids = truth['pair_ids']; sp = make_splits(CACHE)
pairs = [ids.index(p) for p in sp[args.split]]
models = {r: [wm_eval.load_predictions(f) for f in sorted((RUNS / f'preds_{args.split}').glob(f'{r}__*t3.npz'))] for r in ['eef_num', 'full_num', 'eef_img', 'full_img']}
res = wm_eval.evaluate(truth, models, pairs=pairs, n_boot=2000, tasks=('B',))
out = [f'# 3 つ目のタスク：H1′（{args.split}、{len(pairs)} シーンペア、seed {len(models["full_num"])} 個、protocol 追記 13）\n',
       '箱の最後の位置の誤差（mm、Task B の行）。差 = 手先 − 腕全体（正なら腕全体のほうが小さい）。CI はシーンペア単位 bootstrap 95%。\n',
       '| 形式 | 手先 | 腕全体 | 差 | p（差 ≤ 0） | Holm | 判定 |', '|---|---|---|---|---|---|---|']
pv, rows = {}, {}
for f in ['num', 'img']:
    c = wm_eval.diff_ci(res, 'B', f'eef_{f}', f'full_{f}', 'pos_err_mm')
    pv[f] = c['p_le0']; rows[f] = c
hp = wm_eval.holm(pv)
for f in ['num', 'img']:
    c = rows[f]; e = res['point']['B'][f'eef_{f}']['pos_err_mm']; a = res['point']['B'][f'full_{f}']['pos_err_mm']
    v = 'supported' if c['lo'] > 0 and hp[f][1] else ('refuted' if c['hi'] < 0 else 'not supported')
    out.append(f"| {f} | {e:.1f} | {a:.1f} | **{c['diff']:+.1f}** [{c['lo']:+.1f}, {c['hi']:+.1f}] | {c['p_le0']:.3f} | {hp[f][0]:.3f} | **{v}** |")

# 探索的：片方だけ触れる組の episode に限った位置の誤差
B = truth['task'] == 1
dis = []
for p in pairs:
    e = np.nonzero(B & (truth['pair_idx'] == p))[0]
    for mp in np.unique(truth['mp'][e]):
        ee = e[truth['mp'][e] == mp]
        if len(ee) == 2 and truth['coll'][ee[0]] != truth['coll'][ee[1]]:
            dis += list(ee)
dis = np.array(dis)
out.append(f'\n## 探索的：片方だけ腕が触れる組（{len(dis) // 2} 組）の位置の誤差（mm、5 seed 平均）\n')
out.append('| モデル | 誤差 |\n|---|---|')
for r, ps in models.items():
    errs = []
    for p in ps:
        m = {int(k): i for i, k in enumerate(p['ep'])}
        errs.append(np.mean([np.linalg.norm(p['final_pos'][m[int(x)]] - truth['final_pos'][x]) * 1000 for x in dis]))
    out.append(f'| {r} | {np.mean(errs):.1f} |')
md = '\n'.join(out) + '\n'
(RUNS / f't3_h1_{args.split}.md').write_text(md, encoding='utf-8'); print(md)
