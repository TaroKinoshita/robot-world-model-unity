'''
保存した val の予測で、意思決定の評価・matched pair（H5）・仮説の判定を、シーン単位の bootstrap で出す（学習なし）。

- 予測: <runs>/preds/*__<tag>__val.npz（wm_encoder_study.py が保存。goal は予測位置から = protocol 追記 6）
- 基準: 候補の種類だけ（base_kind）、種類 + matched pair の枝（base_kind_branch）。train のペアだけで作る
- H3 は mask の予測が要るので判定しない（mask IoU は run の JSON にある）
- **これは val での確認。仮説の最終判定は test で 1 回だけ行う**

使い方（プロジェクトのフォルダで。GPU は使わない）:
    py -3.11 Learning/wm_eval_val.py
    py -3.11 Learning/wm_eval_val.py --runs Docs/encoder_fix/results_main_ss --tag main --n-boot 2000
'''
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_dataset import make_splits   # noqa: E402
import wm_eval                        # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPS = ['eef_num', 'full_num', 'eef_img', 'full_img']


def judge(res):
    '''wm_eval.judge_hypotheses から H3（mask が要る）を除いたもの'''
    out = {'H3': {}}
    h1 = {}
    for f in ['num', 'img']:
        e, fl = f'eef_{f}', f'full_{f}'
        parts = {}
        for m, margin in wm_eval.H1_MARGINS.items():
            c = wm_eval.diff_ci(res, 'A', e, fl, m) if m in wm_eval.LOWER_IS_BETTER else wm_eval.diff_ci(res, 'A', fl, e, m)
            parts[m] = {**c, 'margin': margin}
        if all(v['hi'] <= v['margin'] for v in parts.values()):
            v = 'supported'
        elif any(v['lo'] > v['margin'] for v in parts.values()):
            v = 'rejected'
        else:
            v = 'inconclusive'
        h1[f] = {'verdict': v, 'degradation': parts}
    out['H1'] = h1
    for H, met in [('H2', 'coll_auprc'), ('H5', 'h5_acc')]:
        h = {f: wm_eval.diff_ci(res, 'B', f'full_{f}', f'eef_{f}', met) for f in ['num', 'img']}
        for f in h:
            h[f]['verdict'] = 'supported' if h[f]['lo'] > 0 else 'not supported'
        hp = wm_eval.holm({f: h[f]['p_le0'] for f in h})
        for f in h:
            h[f]['holm_p'], h[f]['holm_sig'] = hp[f]
        out[H] = h
    h4 = {}
    for f in ['num', 'img']:
        cr = wm_eval.diff_ci(res, 'B', f'eef_{f}', f'full_{f}', 'coll_rate')
        rg = wm_eval.diff_ci(res, 'B', f'eef_{f}', f'full_{f}', 'regret')
        v = 'supported' if cr['lo'] > 0 and rg['lo'] > 0 and out['H2'][f]['verdict'] == 'supported' else 'not supported'
        h4[f] = {'verdict': v, 'coll_rate_drop': cr, 'regret_drop': rg}
    out['H4'] = h4
    # 数値 vs 画像（同じ範囲）の差も参考に出す（画像 − 数値）
    out['num_vs_img'] = {f'{t}/{sc}/{m}': wm_eval.diff_ci(res, t, f'{sc}_img', f'{sc}_num', m)
                         for t, ms in [('A', ['pos_err_mm', 'goal_auprc', 'success_rate']), ('B', ['coll_auprc', 'success_rate', 'h5_acc'])]
                         for sc in ['eef', 'full'] for m in ms}
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
    models = {}
    for r in REPS:
        fs = sorted((runs / 'preds').glob(f'{r}__*__{args.tag}__val.npz'))
        if fs:
            models[r] = [wm_eval.load_predictions(f) for f in fs]
            print(f'{r}: seed {len(fs)} 個')
    eval_eps = np.nonzero(np.isin(truth['pair_idx'], va))[0]
    models['base_kind'] = [wm_eval.label_baseline(truth, tr, eval_eps, by=('kind',))]
    models['base_kind_branch'] = [wm_eval.label_baseline(truth, tr, eval_eps, by=('kind', 'branch'))]
    print(f'評価するペア（val）: {len(va)}、bootstrap {args.n_boot} 回')
    res = wm_eval.evaluate(truth, models, pairs=va, n_boot=args.n_boot)
    hyp = judge(res)
    md = wm_eval.report_markdown(res, hyp, title=f'val での評価（{runs.name}、tag {args.tag}）— 最終判定ではない（test は未使用）',
                                 truth=truth, models={k: v for k, v in models.items() if k in REPS})
    md += '\n## 数値 vs 画像（画像 − 数値、95% CI。位置誤差は小さい方が良い）\n\n'
    for k, c in hyp['num_vs_img'].items():
        md += f"- {k}: {c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]\n"
    (runs / f'eval_val_{args.tag}.md').write_text(md, encoding='utf-8')
    (runs / f'eval_val_{args.tag}.json').write_text(json.dumps(wm_eval.to_jsonable(res, hyp), indent=1, ensure_ascii=False), encoding='utf-8')
    print(md)


if __name__ == '__main__':
    main()
