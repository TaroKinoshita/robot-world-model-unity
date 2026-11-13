'''
test（1 回だけ）の仮説判定。protocol 追記 8 のルール。
- H1・H2・H4（と参考の旧 H5 = h5_acc 腕全体 − 手先）: wm_eval_val.judge と同じ基準
- H5（主）: 腕全体モデルの balanced（5 seed 平均、シーンペア単位 bootstrap）。CI 下限 > 0.5 で supported。数値・画像の 2 つで Holm
- AUROC（副、判定には使わない）と seed ごとの値も出す
使い方: python Learning/wm_test_judge.py   （wm_predict.py --split test --name test の後）
'''
import json, sys, os
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import wm_eval, wm_eval_val
from wm_dataset import make_splits
src = (Path(__file__).resolve().parent / 'wm_h5_deep.py').read_text(encoding='utf-8')
exec(src[:src.index("\nrun('val")])   # items / stats / boot を使う（末尾の実行部分は除く）

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / os.environ.get('WMJ_RUNS', 'Docs/encoder_fix/results_main_v1v2_ckpt')   # 追記 12: 2 つ目のタスクは環境変数で差し替える
CACHE = ROOT / os.environ.get('WMJ_CACHE', 'Episodes/_main_v1v2/_cache/wm_128x25.h5')
TAG = os.environ.get('WMJ_TAG', 'main')
truth = wm_eval.load_truth(CACHE); ids = truth['pair_ids']; sp = make_splits(CACHE)
te = [ids.index(p) for p in sp['test']]; tr = [ids.index(p) for p in sp['train']]
models = {r: [wm_eval.load_predictions(f) for f in sorted((RUNS / 'preds_test').glob(f'{r}__*{TAG}.npz'))] for r in wm_eval_val.REPS}
eval_eps = np.nonzero(np.isin(truth['pair_idx'], te))[0]
models['base_kind'] = [wm_eval.label_baseline(truth, tr, eval_eps, by=('kind',))]
models['base_kind_branch'] = [wm_eval.label_baseline(truth, tr, eval_eps, by=('kind', 'branch'))]
res = wm_eval.evaluate(truth, models, pairs=te, n_boot=2000)
hyp = wm_eval_val.judge(res)
out = [f'# test の仮説判定（{len(te)} シーンペア、seed {len(models["full_num"])} 個、protocol 追記 8）\n']
for f in ['num', 'img']:
    out.append(f"- H1（{f}）：**{hyp['H1'][f]['verdict']}** — " + '、'.join(f"{m} 低下 {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]（マージン {v['margin']}）" for m, v in hyp['H1'][f]['degradation'].items()))
for f in ['num', 'img']:
    h = hyp['H2'][f]; out.append(f"- H2（{f}、coll_auprc 腕全体 − 手先）：**{h['verdict']}** — {h['diff']:+.3f} [{h['lo']:+.3f}, {h['hi']:+.3f}]、Holm p = {h['holm_p']:.3f}")
for f in ['num', 'img']:
    h = hyp['H4'][f]; c, r = h['coll_rate_drop'], h['regret_drop']
    out.append(f"- H4（{f}）：**{h['verdict']}** — 衝突率の低下 {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]、regret の低下 {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")
# H5（主）: balanced
out.append('\n## H5（主 = balanced、副 = AUROC）\n')
out.append('| モデル | seed | wrist_down 側 | wrist_up 側 | balanced | AUROC | 両方同じ組の c_up − c_down |\n|---|---|---|---|---|---|---|')
pvals, summ = {}, {}
for f in ['num', 'img']:
    per = [items(truth, {'ep': p['ep'], 'coll': p['coll']}, te) for p in models[f'full_{f}']]
    for i, it in enumerate(per):
        s = stats(it); out.append(f"| full_{f} | {i} | {s['down']:.3f} | {s['up']:.3f} | {s['bal']:.3f} | {s['auroc']:.3f} | {s['bias_same']:+.3f} |")
    # bootstrap（balanced の分布も取って p を出す）
    vp = np.array(sorted({p for p, _, _ in per[0]})); bs = {k: [] for k in ['down', 'up', 'bal', 'auroc']}
    r2 = np.random.default_rng(0)
    for b in range(2000):
        dr = vp[r2.integers(0, len(vp), len(vp))]; w = {int(p): 0 for p in vp}
        for p in dr: w[int(p)] += 1
        ss = [stats(it, w) for it in per]
        for k in bs: bs[k].append(np.nanmean([x[k] for x in ss]))
    pt = {k: float(np.nanmean([stats(it)[k] for it in per])) for k in bs}
    ci = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in bs.items()}
    pvals[f] = float(np.mean(np.array(bs['bal']) <= 0.5)); summ[f] = (pt, ci, len([1 for _, h, _ in per[0] if h == 0]), len([1 for _, h, _ in per[0] if h == 1]))
hp = wm_eval.holm(pvals)
out.append('\n| モデル | wrist_down 側 | wrist_up 側 | **balanced** | p（≤ 0.5）/ Holm | 判定 | AUROC（副） | 組 down / up |\n|---|---|---|---|---|---|---|---|')
for f in ['num', 'img']:
    pt, ci, nd, nu = summ[f]; lo, hi = ci['bal']
    v = 'supported' if lo > 0.5 and hp[f][1] else ('refuted' if hi < 0.5 else 'inconclusive')
    out.append(f"| full_{f} | {pt['down']:.3f} [{ci['down'][0]:.3f}, {ci['down'][1]:.3f}] | {pt['up']:.3f} [{ci['up'][0]:.3f}, {ci['up'][1]:.3f}] | **{pt['bal']:.3f} [{lo:.3f}, {hi:.3f}]** | {pvals[f]:.3f} / {hp[f][0]:.3f} | **{v}** | {pt['auroc']:.3f} [{ci['auroc'][0]:.3f}, {ci['auroc'][1]:.3f}] | {nd} / {nu} |")
md = '\n'.join(out) + '\n'
(RUNS / 'test_judgement.md').write_text(md, encoding='utf-8'); print(md)
