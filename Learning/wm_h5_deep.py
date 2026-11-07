import sys, glob, numpy as np
from pathlib import Path
ROOT = Path(r'C:\UnityProjects\PerceptionTestHDRP')
sys.path.insert(0, str(ROOT / 'Learning'))
import wm_eval
from wm_dataset import make_splits, load_excluded

RUNS = ROOT / 'Docs/encoder_fix/results_main_v1v2_ckpt'
rng = np.random.default_rng(0); NB = 2000


def items(truth, pred, pairs):
    """matched pair ごと（片方だけ当たる組も、両方同じ組も）: pair, 当たった枝(-1=どちらも同じ), c_up - c_down"""
    c = dict(zip(pred['ep'].tolist(), pred['coll'].tolist()))
    eps = np.nonzero((truth['task'] == 1) & (truth['mp'] >= 0) & np.isin(truth['pair_idx'], pairs))[0]
    g = {}
    for e in eps:
        g.setdefault((int(truth['pair_idx'][e]), int(truth['mp'][e])), []).append(int(e))
    out = []
    for (p, mp), m in sorted(g.items()):
        if len(m) != 2 or not all(e in c for e in m):
            continue
        up = [e for e in m if truth['branch'][e] == 1]; dn = [e for e in m if truth['branch'][e] == 0]
        if len(up) != 1 or len(dn) != 1:
            continue
        yu, yd = truth['coll'][up[0]] == 1, truth['coll'][dn[0]] == 1
        hit = -1 if yu == yd else (1 if yu else 0)
        out.append((p, hit, c[up[0]] - c[dn[0]]))
    return out


def auroc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    pos = np.asarray(pos)[:, None]; neg = np.asarray(neg)[None, :]
    return float(((pos > neg).mean() + 0.5 * (pos == neg).mean()))


def stats(it, w=None):
    """w: pair -> 重み（bootstrap）"""
    w = w or {}
    W = lambda p: w.get(p, 1.0) if w else 1.0
    d = {0: [], 1: []}; ww = {0: [], 1: []}
    for p, h, x in it:
        if h >= 0 and W(p) > 0:
            d[h].append(x); ww[h].append(W(p))
    acc = {}
    for h in (0, 1):
        x = np.array(d[h]); v = np.array(ww[h])
        corr = (x > 0) if h == 1 else (x < 0)
        corr = corr + 0.5 * (x == 0)
        acc[h] = float((corr * v).sum() / v.sum())
    # AUROC: 「up が当たった組」の差 vs 「down が当たった組」の差（重みは複製で近似）
    pos = np.repeat(d[1], np.array(ww[1]).astype(int)); neg = np.repeat(d[0], np.array(ww[0]).astype(int))
    return {'down': acc[0], 'up': acc[1], 'bal': 0.5 * (acc[0] + acc[1]), 'auroc': auroc(pos, neg),
            'bias_same': float(np.mean([x for p, h, x in it if h == -1])) if any(h == -1 for _, h, _ in it) else np.nan}


def boot(per_seed, pairs):
    vp = np.array(sorted(set(pairs)))
    pt = {k: np.nanmean([stats(it)[k] for it in per_seed]) for k in ['down', 'up', 'bal', 'auroc']}
    bs = {k: [] for k in pt}
    for b in range(NB):
        dr = vp[rng.integers(0, len(vp), len(vp))]
        w = {}
        for p in dr:
            w[int(p)] = w.get(int(p), 0) + 1
        w = {p: w.get(p, 0) for p in vp.tolist()}
        s = [stats(it, w) for it in per_seed]
        for k in bs:
            bs[k].append(np.nanmean([x[k] for x in s]))
    ci = {k: (np.nanpercentile(v, 2.5), np.nanpercentile(v, 97.5)) for k, v in bs.items()}
    return pt, ci


def run(name, cache, pred_dir, pairs_kind):
    truth = wm_eval.load_truth(cache); ids = truth['pair_ids']
    if pairs_kind == 'val':
        pairs = [ids.index(p) for p in make_splits(cache)['val']]
        suffix = '__val.npz'
    else:
        ex, _ = load_excluded(cache); pairs = [i for i, p in enumerate(ids) if p not in ex]
        suffix = '.npz'
    print(f'\n### {name}（{len(pairs)} シーンペア）')
    for rep in ['full_num', 'full_img']:
        fs = sorted(glob.glob(str(pred_dir / f'{rep}__*main{suffix}')))
        per = []
        for f in fs:
            d = np.load(f); per.append(items(truth, {'ep': d['ep'], 'coll': d['coll']}, pairs))
        nd = sum(1 for _, h, _ in per[0] if h == 0); nu = sum(1 for _, h, _ in per[0] if h == 1); ns = sum(1 for _, h, _ in per[0] if h == -1)
        print(f'{rep}: seeds {len(fs)}、組 down {nd} / up {nu} / 両方同じ {ns}')
        for i, it in enumerate(per):
            s = stats(it)
            print(f'  seed{i}: down {s["down"]:.3f} up {s["up"]:.3f} bal {s["bal"]:.3f} AUROC {s["auroc"]:.3f}  (両方同じ組の c_up−c_down 平均 {s["bias_same"]:+.3f})')
        # seed 平均の確率で 1 つにまとめた版（参考）
        ens = [(p, h, np.mean([per[k][j][2] for k in range(len(per))])) for j, (p, h, _) in enumerate(per[0])]
        s = stats(ens)
        print(f'  ensemble(参考): down {s["down"]:.3f} up {s["up"]:.3f} bal {s["bal"]:.3f} AUROC {s["auroc"]:.3f}')
        pt, ci = boot(per, [p for p, _, _ in per[0]])
        print('  seed 平均 [95% CI、シーンペア単位 bootstrap]: ' + '  '.join(f'{k} {pt[k]:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]' for k in pt))
        dn = np.array([x for _, h, x in ens if h == 0]); up = np.array([x for _, h, x in ens if h == 1])
        print(f'  c_up−c_down 中央値: down 側が当たった組 {np.median(dn):+.3f}、up 側 {np.median(up):+.3f}')


run('val（v1v2）', ROOT / 'Episodes/_main_v1v2/_cache/wm_128x25.h5', RUNS / 'preds', 'val')
run('H5 評価専用セット', ROOT / 'Episodes/_h5_v1/_cache/wm_128x25.h5', RUNS / 'preds_h5_v1', 'all')
