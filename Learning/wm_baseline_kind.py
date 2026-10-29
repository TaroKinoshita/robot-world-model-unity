'''
候補の種類だけで当てる基準（action を見ない）の成績を、量産データの test で出す。

学習したモデルは、少なくともこれを超えないと「action から学べた」とは言えない。
中身は wm_eval.label_baseline（train のペアだけで、種類ごとの割合・平均の移動を取る）。
  kind        : 候補の種類だけ
  kind+branch : 種類 + matched pair の枝（wrist-down / up）。H5 の比較の相手

分け方は wm_dataset.make_splits（excluded_pairs.json のペアは外す）。
初めて実行すると <cache と同じ場所>/wm_128x25_splits.json ができる（以後の学習もこれを使う）。

使い方（プロジェクトのフォルダで。GPU は使わない）:
    py -3.11 Learning/wm_baseline_kind.py
    py -3.11 Learning/wm_baseline_kind.py --split val
'''
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_dataset import make_splits   # noqa: E402
import wm_eval                        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def metrics(truth, pred, task):
    ep = pred['ep']
    m = truth['task'][ep] == task
    e = ep[m]
    out = {'episodes': int(len(e))}
    d = pred['final_pos'][m][:, [0, 2]] - truth['final_pos'][e][:, [0, 2]]
    out['pos_err_mm'] = float(np.linalg.norm(d, axis=1).mean() * 1000)
    y = truth['goal'][e] == 1
    out['goal_auprc'] = wm_eval.average_precision(y, pred['goal'][m])
    out['goal_acc'] = float(((pred['goal'][m] >= 0.5) == y).mean())
    out['goal_rate'] = float(y.mean())
    if task == 1:
        yc = truth['coll'][e] == 1
        out['coll_auprc'] = wm_eval.average_precision(yc, pred['coll'][m])
        out['coll_acc'] = float(((pred['coll'][m] >= 0.5) == yc).mean())
        out['coll_rate'] = float(yc.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', default=str(ROOT / 'Episodes/_main_v1/_cache/wm_128x25.h5'))
    ap.add_argument('--split', default='test', choices=['val', 'test'])
    args = ap.parse_args()
    truth = wm_eval.load_truth(args.cache)
    sp = make_splits(args.cache)
    ids = truth['pair_ids']
    tr = [ids.index(p) for p in sp['train']]
    ev = [ids.index(p) for p in sp[args.split]]
    eval_eps = np.nonzero(np.isin(truth['pair_idx'], ev))[0]
    n_ex = len(sp['_info'].get('excluded_pairs', []))
    print(f'cache: {args.cache}')
    print(f'ペア: train {len(tr)} / {args.split} {len(ev)}（除外 {n_ex} ペア）、評価する episode {len(eval_eps)}')
    print()
    print('| 基準 | Task | episode | 位置誤差 mm | goal AUPRC | goal 正解率 | goal の割合 | 衝突 AUPRC | 衝突 正解率 | 衝突の割合 |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for name, by in [('kind', ('kind',)), ('kind+branch', ('kind', 'branch'))]:
        pred = wm_eval.label_baseline(truth, tr, eval_eps, by=by)
        for task, tname in [(0, 'A'), (1, 'B')]:
            r = metrics(truth, pred, task)
            c = ' | '.join(f'{r[k]:.3f}' if k in r else '—' for k in ['coll_auprc', 'coll_acc', 'coll_rate'])
            print(f"| {name} | {tname} | {r['episodes']} | {r['pos_err_mm']:.1f} | {r['goal_auprc']:.3f} | "
                  f"{r['goal_acc']:.3f} | {r['goal_rate']:.3f} | {c} |")


if __name__ == '__main__':
    main()
