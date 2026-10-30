'''
H5（matched pair）の中身を分けて見る（学習なし。保存した val の予測を使う）。

片方だけ円柱に当たる matched pair を、「当たったのが wrist_up 側か wrist_down 側か」で分けて、
衝突する方に高い衝突確率を付けた割合（h5_acc）を出す。
「手首の向きだけ」で答える基準は、train で当たりやすい方（wrist_up）をいつも選ぶので、
wrist_down 側が当たった組では必ず外れる。ここで当てられれば、手首の向き以上のことを見ている証拠になる。

使い方:
    py -3.11 Learning/wm_h5_breakdown.py
'''
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_dataset import make_splits   # noqa: E402
import wm_eval                        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPS = ['eef_num', 'full_num', 'eef_img', 'full_img']


def h5_items(truth, pred, va_set):
    '''片方だけ当たる組ごとに (pair, 当たった側の枝 0=down/1=up, 正解 1/0.5/0, 確率の差 当たった側 − 外れた側)'''
    c = {int(e): float(v) for e, v in zip(pred['ep'], pred['coll'])}
    out = []
    eps = np.nonzero((truth['task'] == 1) & (truth['mp'] >= 0) & np.isin(truth['pair_idx'], list(va_set)))[0]
    groups = {}
    for e in eps:
        groups.setdefault((int(truth['pair_idx'][e]), int(truth['mp'][e])), []).append(int(e))
    for (p, mp), m in sorted(groups.items()):
        if len(m) != 2 or not all(e in c for e in m):
            continue
        y = [truth['coll'][e] == 1 for e in m]
        if y[0] == y[1]:
            continue
        hit, miss = (m[0], m[1]) if y[0] else (m[1], m[0])
        acc = 1.0 if c[hit] > c[miss] else 0.5 if c[hit] == c[miss] else 0.0
        out.append((p, int(truth['branch'][hit]), acc, c[hit] - c[miss]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=str(ROOT / 'Episodes/_main_v1/_cache/wm_128x25.h5'))
    ap.add_argument('--runs', default=str(ROOT / 'Docs/encoder_fix/results_main_ss'))
    ap.add_argument('--tag', default='main')
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    runs = Path(args.runs)
    truth = wm_eval.load_truth(args.cache)
    sp = make_splits(args.cache)
    ids = truth['pair_ids']
    tr = [ids.index(p) for p in sp['train']]
    va = [ids.index(p) for p in sp['val']]
    # train で、片方だけ当たる組のうち wrist_up 側が当たった割合
    tr_items = h5_items(truth, {'ep': np.arange(len(truth['task'])), 'coll': (truth['coll'] == 1).astype(float)}, set(tr))
    up_rate = np.mean([b for _, b, _, _ in tr_items])
    print(f'train: 片方だけ当たる組 {len(tr_items)}、そのうち wrist_up 側が当たった割合 {up_rate:.3f}')
    eval_eps = np.nonzero(np.isin(truth['pair_idx'], va))[0]
    models = {}
    for r in REPS:
        fs = sorted((runs / 'preds').glob(f'{r}__*__{args.tag}__val.npz'))
        models[r] = [wm_eval.load_predictions(f) for f in fs]
    models['base_kind_branch'] = [wm_eval.label_baseline(truth, tr, eval_eps, by=('kind', 'branch'))]
    rng = np.random.default_rng(0)
    vpairs = np.array(sorted(va))
    draws = rng.integers(0, len(vpairs), size=(args.n_boot, len(vpairs)))
    lines = ['| モデル | 全部（組の数） | wrist_up 側が当たった組 | **wrist_down 側が当たった組** | 確率の差の平均（down 側が当たった組） |',
             '|---|---|---|---|---|']
    for name, preds in models.items():
        per_seed = [h5_items(truth, p, set(va)) for p in preds]
        n_all = len(per_seed[0]); n_up = sum(1 for it in per_seed[0] if it[1] == 1); n_dn = n_all - n_up

        def stat(w, branch):
            vals = []
            for items in per_seed:
                a = [(w[p], acc) for p, b, acc, _ in items if branch is None or b == branch]
                tot = sum(x for x, _ in a)
                vals.append(sum(x * y for x, y in a) / tot if tot > 0 else np.nan)
            return float(np.nanmean(vals))

        w1 = {int(p): 1.0 for p in vpairs}
        cells = []
        for br in [None, 1, 0]:
            pt = stat(w1, br)
            bt = []
            for b in range(args.n_boot):
                w = {}
                for p in vpairs[draws[b]]:
                    w[int(p)] = w.get(int(p), 0) + 1.0
                for p in vpairs:
                    w.setdefault(int(p), 0.0)
                bt.append(stat(w, br))
            bt = np.array(bt); bt = bt[np.isfinite(bt)]
            cells.append(f'{pt:.3f} [{np.percentile(bt, 2.5):.3f}, {np.percentile(bt, 97.5):.3f}]')
        dd = np.mean([np.mean([d for _, b, _, d in items if b == 0]) for items in per_seed])
        lines.append(f'| {name} | {cells[0]}（{n_all}） | {cells[1]}（{n_up}） | {cells[2]}（{n_dn}） | {dd:+.3f} |')
    md = '# H5 の中身（val、当たった側の枝で分けた h5_acc、95% CI はシーン単位の bootstrap）\n\n'
    md += f'- train の片方だけ当たる組のうち wrist_up 側が当たった割合: {up_rate:.3f}（手首の向きだけの基準は wrist_up をいつも選ぶ）\n'
    md += '- h5_acc = 当たる側に高い衝突確率を付けた割合（同点 0.5）。手先モデルは matched pair の入力が同じなので 0.5\n\n'
    md += '\n'.join(lines) + '\n'
    (runs / f'h5_breakdown_{args.tag}.md').write_text(md, encoding='utf-8')
    print(md)


if __name__ == '__main__':
    main()
