'''
保存した重み（wm_encoder_study.py が <runs>/models/*.pt に残す）で、別のデータを予測して評価する。学習はしない。

使いみち:
  - H5 セット（Episodes/_h5_v1、評価専用。円柱を当てる腕を wrist_down / wrist_up で半々に指定した）
  - 最後の test（--split test）

H5 は「当たった側の枝」ごとの正解率と、その平均（balanced）を出す。手首の向きだけで答える基準は balanced = 0.5 になる。
goal は予測した最終位置から（protocol 追記 6、τ は学習時に train で決めたもの）。

使い方（プロジェクトのフォルダで）:
    py -3.11 Learning/wm_predict.py --runs Docs/encoder_fix/results_main_v1v2_ckpt --cache Episodes/_h5_v1/_cache/wm_128x25.h5 --name h5_v1
    py -3.11 Learning/wm_predict.py --runs Docs/encoder_fix/results_main_v1v2_ckpt --cache Episodes/_main_v1v2/_cache/wm_128x25.h5 --split test --name test
'''
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_dataset import WMDataset, make_splits, load_excluded   # noqa: E402
import wm_eval                                                  # noqa: E402
import wm_encoder_study as S                                    # noqa: E402
from wm_h5_breakdown import h5_items                            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPS = ['eef_num', 'full_num', 'eef_img', 'full_img']


def abspath(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_truth(cache):
    t = wm_eval.load_truth(cache)
    with h5py.File(cache, 'r') as h:
        t['goal_xz'] = h['scenes/goal_xz'][:][h['episodes/scene_idx'][:]]
    return t


def predict(ckpt_path, cache, truth, pairs, dev):
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    c = ck['config']; nd = ck['num_dim']
    z = np.zeros(nd if nd else 1, np.float32); o = np.ones(nd if nd else 1, np.float32)
    model = S.StudyWM(c['rep'], c['enc'], ck['frames'], nd, z, o, (0.0, 0.0, 0.0),
                      cube=bool(c.get('cube')), spatial=bool(c.get('spatial')))
    model.load_state_dict(ck['state_dict'])
    model.to(dev)
    ds = WMDataset(cache, c['rep'], task='both', num_steps=c['steps'])
    ds.idx = ds.idx[np.isin(ds.meta['pair_idx'][ds.idx], pairs)]
    P, _ = S.evaluate_split(model, ds, dev)
    tau = (ck.get('goal_tau_mm') or 3.0) / 1000.0
    gp = 1 / (1 + np.exp(-(S.GOAL_RADIUS - S.goal_dist(P, truth)) / tau))
    return {'ep': P['ep'].astype(np.int64), 'final_pos': P['final_pos'].astype(np.float32), 'goal': gp.astype(np.float32),
            'goal_head': P['goal'].astype(np.float32), 'coll': P['coll'].astype(np.float32)}


def group_baseline(train_truth, train_pairs, truth, eval_eps, by):
    '''学習データ（別のキャッシュでもよい）の train ペアで、(task, *by) ごとの割合・平均の移動を取り、評価データに当てはめる'''
    tr = np.nonzero(np.isin(train_truth['pair_idx'], train_pairs))[0]
    key = lambda T, i: (int(T['task'][i]),) + tuple(int(T[b][i]) for b in by)
    disp = train_truth['final_pos'] - train_truth['init_pos']
    groups = {}
    for i in tr:
        groups.setdefault(key(train_truth, i), []).append(i)
    fall = {t: tr[train_truth['task'][tr] == t] for t in (0, 1)}
    goal = np.zeros(len(eval_eps), np.float32); coll = np.zeros(len(eval_eps), np.float32); pos = np.zeros((len(eval_eps), 3), np.float32)
    for n, i in enumerate(eval_eps):
        g = np.asarray(groups.get(key(truth, i), fall[int(truth['task'][i])]))
        goal[n] = train_truth['goal'][g].mean(); coll[n] = (train_truth['coll'][g] == 1).mean()
        pos[n] = truth['init_pos'][i] + disp[g].mean(0)
    return {'ep': np.asarray(eval_eps, np.int64), 'goal': goal, 'coll': coll, 'final_pos': pos}


def h5_table(truth, models, pairs, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    vp = np.array(sorted(pairs))
    draws = rng.integers(0, len(vp), size=(n_boot, len(vp)))
    rows = ['| モデル | wrist_down 側が当たった組 | wrist_up 側が当たった組 | **balanced（2 つの平均）** | 組の数（down / up） |', '|---|---|---|---|---|']
    out = {}
    for name, preds in models.items():
        per = [h5_items(truth, p, set(int(x) for x in vp)) for p in preds]

        def stat(w):
            r = {0: [], 1: [], 'bal': []}
            for items in per:
                acc = {}
                for br in (0, 1):
                    a = [(w.get(p, 0.0), y) for p, b, y, _ in items if b == br]
                    tot = sum(x for x, _ in a)
                    acc[br] = sum(x * y for x, y in a) / tot if tot > 0 else np.nan
                r[0].append(acc[0]); r[1].append(acc[1]); r['bal'].append(0.5 * (acc[0] + acc[1]))
            return {k: float(np.nanmean(v)) for k, v in r.items()}

        pt = stat({int(p): 1.0 for p in vp})
        bt = {0: [], 1: [], 'bal': []}
        for b in range(n_boot):
            w = {}
            for p in vp[draws[b]]:
                w[int(p)] = w.get(int(p), 0.0) + 1.0
            s = stat(w)
            for k in bt:
                bt[k].append(s[k])
        ci = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in bt.items()}
        nd = sum(1 for it in per[0] if it[1] == 0); nu = sum(1 for it in per[0] if it[1] == 1)
        out[name] = {'point': pt, 'ci': ci, 'n_down': nd, 'n_up': nu}
        f = lambda k: f'{pt[k]:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]'
        rows.append(f'| {name} | {f(0)} | {f(1)} | {f("bal")} | {nd} / {nu} |')
    return '\n'.join(rows) + '\n', out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True, help='wm_encoder_study.py の --out（models/ がある所）')
    ap.add_argument('--tag', default='main')
    ap.add_argument('--cache', required=True, help='予測するデータのキャッシュ')
    ap.add_argument('--split', default=None, choices=[None, 'train', 'val', 'test'], help='省略 = 除外以外の全ペア（H5 セット用）')
    ap.add_argument('--train-cache', default='Episodes/_main_v1v2/_cache/wm_128x25.h5', help='基準（種類・枝）を作る学習データ')
    ap.add_argument('--name', required=True)
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    runs = abspath(args.runs); cache = abspath(args.cache); tcache = abspath(args.train_cache)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    truth = load_truth(cache)
    ids = truth['pair_ids']
    if args.split:
        pairs = [ids.index(p) for p in make_splits(cache)[args.split]]
    else:
        ex, _ = load_excluded(cache)
        pairs = [i for i, p in enumerate(ids) if p not in ex]
    print(f'予測するデータ: {cache}（{args.split or "除外以外の全ペア"}、{len(pairs)} ペア）、device {dev}')
    pd = runs / f'preds_{args.name}'; pd.mkdir(exist_ok=True)
    models = {}
    for r in REPS:
        fs = sorted((runs / 'models').glob(f'{r}__*__{args.tag}.pt'))
        if not fs:
            print(f'{r}: 重みが無い（{runs / "models"}）'); continue
        models[r] = []
        for f in fs:
            p = predict(f, cache, truth, pairs, dev)
            np.savez_compressed(pd / f'{f.stem}.npz', **p)
            models[r].append(p)
        print(f'{r}: seed {len(fs)} 個を予測')
    ttruth = wm_eval.load_truth(tcache)
    tids = ttruth['pair_ids']
    tpairs = [tids.index(p) for p in make_splits(tcache)['train']]
    eval_eps = np.nonzero(np.isin(truth['pair_idx'], pairs))[0]
    models['base_kind'] = [group_baseline(ttruth, tpairs, truth, eval_eps, ('kind',))]
    models['base_kind_branch'] = [group_baseline(ttruth, tpairs, truth, eval_eps, ('kind', 'branch'))]
    res = wm_eval.evaluate(truth, models, pairs=pairs, n_boot=args.n_boot)
    md = wm_eval.report_markdown(res, None, title=f'{args.name} での評価（{runs.name}、tag {args.tag}）', truth=truth,
                                 models={k: v for k, v in models.items() if k in REPS})
    h5md, h5res = h5_table(truth, models, pairs, args.n_boot)
    md += '\n## H5（当たった側の枝ごとの h5_acc と、その平均。手首の向きだけの基準は balanced = 0.5）\n\n' + h5md
    (runs / f'eval_{args.name}.md').write_text(md, encoding='utf-8')
    js = wm_eval.to_jsonable(res); js['h5_by_branch'] = h5res
    (runs / f'eval_{args.name}.json').write_text(json.dumps(js, indent=1, ensure_ascii=False), encoding='utf-8')
    print(md)


if __name__ == '__main__':
    main()
