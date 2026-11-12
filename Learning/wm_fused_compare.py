'''
protocol 追記 11（探索的、test は使わない）: early fusion（fused_ss5）と今の画像モデル（late fusion、main）の Full-Img を、
H5 セットと val で比べる。balanced（主）と matched pair 内の AUROC（副）、seed 5 つの平均と、seed をそろえた差。
CI はシーンペア単位のブートストラップ（2 つのモデルで同じ引き直しを使う＝対応あり）。

    py -3.11 Learning/wm_fused_compare.py
'''
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'Learning'))
import wm_eval                                       # noqa: E402
from wm_dataset import make_splits, load_excluded    # noqa: E402
from wm_h5_deep import items                         # noqa: E402

BASE = ROOT / 'Docs/encoder_fix/results_main_v1v2_ckpt'
FUSED = ROOT / 'Docs/encoder_fix/results_fused_ss5'
NB = 2000


def load(d, rep, suffix=''):
    fs = sorted(d.glob(f'{rep}__*{suffix}.npz'))
    return [dict(np.load(f)) for f in fs]


def stats(its, w):
    '''its: [(pair, hit, d)]、w: pair → 重み。balanced と AUROC（片方だけ当たる組、d>0 なら up 側を当たりと予測）'''
    acc = {}
    for br in (0, 1):
        a = [(w.get(p, 0.0), (d > 0) == (br == 1)) for p, h, d in its if h == br]
        tot = sum(x for x, _ in a)
        acc[br] = sum(x * y for x, y in a) / tot if tot else np.nan
    bal = 0.5 * (acc[0] + acc[1])
    # AUROC: 当たった側の c − 当たらなかった側の c > 0 の割合（重み付き）
    s = [(w.get(p, 0.0), d if h == 1 else -d) for p, h, d in its if h >= 0]
    tot = sum(x for x, _ in s)
    auc = sum(x * ((v > 0) + 0.5 * (v == 0)) for x, v in s) / tot if tot else np.nan
    return bal, auc


def compare(name, truth, pairs, base, fused):
    vp = np.array(sorted(pairs)); rng = np.random.default_rng(0)
    draws = rng.integers(0, len(vp), size=(NB, len(vp)))
    ib = [items(truth, p, vp) for p in base]; iff = [items(truth, p, vp) for p in fused]

    def mean(ilist, w):
        r = np.array([stats(i, w) for i in ilist]); return r.mean(0)
    w1 = {int(p): 1.0 for p in vp}
    pb, pf = mean(ib, w1), mean(iff, w1)
    bt = []
    for b in range(NB):
        w = {}
        for p in vp[draws[b]]:
            w[int(p)] = w.get(int(p), 0.0) + 1.0
        bt.append(np.concatenate([mean(ib, w), mean(iff, w)]))
    bt = np.array(bt)
    ci = lambda x: f'[{np.nanpercentile(x, 2.5):.3f}, {np.nanpercentile(x, 97.5):.3f}]'
    per_b = [stats(i, w1) for i in ib]; per_f = [stats(i, w1) for i in iff]
    out = [f'### {name}（{len(vp)} シーンペア）', '',
           '| | late（今の画像） | early fusion | 差（early − late） |', '|---|---|---|---|',
           f'| balanced（主） | {pb[0]:.3f} {ci(bt[:, 0])} | {pf[0]:.3f} {ci(bt[:, 2])} | {pf[0] - pb[0]:+.3f} {ci(bt[:, 2] - bt[:, 0])} |',
           f'| AUROC（副） | {pb[1]:.3f} {ci(bt[:, 1])} | {pf[1]:.3f} {ci(bt[:, 3])} | {pf[1] - pb[1]:+.3f} {ci(bt[:, 3] - bt[:, 1])} |',
           '', '| seed | ' + ' | '.join(str(s) for s in range(len(per_b))) + ' |', '|---|' + '---|' * len(per_b),
           '| late balanced | ' + ' | '.join(f'{x[0]:.3f}' for x in per_b) + ' |',
           '| early balanced | ' + ' | '.join(f'{x[0]:.3f}' for x in per_f) + ' |', '']
    return '\n'.join(out)


def main():
    md = ['# early fusion（fused_ss5）と late fusion の Full-Img（protocol 追記 11、探索的、test は使っていない）', '']
    h5c = ROOT / 'Episodes/_h5_v1/_cache/wm_128x25.h5'
    t = wm_eval.load_truth(h5c); ex, _ = load_excluded(h5c)
    pairs = [i for i, p in enumerate(t['pair_ids']) if p not in ex]
    md.append(compare('H5 評価専用セット', t, pairs, load(BASE / 'preds_h5_v1', 'full_img', '__main'),
                      load(FUSED / 'preds_h5_v1', 'full_img', '__fused')))
    mc = ROOT / 'Episodes/_main_v1v2/_cache/wm_128x25.h5'
    t = wm_eval.load_truth(mc)
    pairs = [t['pair_ids'].index(p) for p in make_splits(mc)['val']]
    md.append(compare('val', t, pairs, load(BASE / 'preds', 'full_img', '__main__val'),
                      load(FUSED / 'preds_val', 'full_img', '__fused')))
    s = '\n'.join(md)
    (FUSED / 'compare_late_vs_early.md').write_text(s, encoding='utf-8'); print(s)


if __name__ == '__main__':
    main()
