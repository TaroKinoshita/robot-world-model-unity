"""
P11: 評価(予測精度・意思決定の質・H1〜H5)と、シーン単位の paired bootstrap

予測ファイル(モデル 1 つ × 学習 seed 1 つ)= npz:
    ep          (N,)      int    キャッシュの episode 番号(WMDataset の x["ep"])
    final_pos   (N, 3)    float  最終物体位置(world, m)
    goal        (N,)      float  goal に届く確率
    coll        (N,)      float  衝突する確率(Task A は何でもよい。使わない)
    final_mask  (N, R, R) float  最終物体 mask の確率(任意)
    contact_point (N, 3)  float  最初の接触点(任意)
  save_predictions(path, ...) で書ける。

指標(Task A / B は別々に出す。衝突系は B だけ):
    pos_err_mm      最終位置誤差(XZ、mm)の平均
    goal_auprc      goal の AUPRC
    coll_auprc      衝突の AUPRC
    mask_iou        最終 mask の IoU(予測は 0.5 で 2 値化、正解は 128 で 2 値化)の平均。正解・予測とも空なら数えない
    contact_err_mm  接触点の誤差(正解の接触がある episode だけ)
    --- 意思決定(シーンごとに 16 候補から 1 本選ぶ。score = P(goal)、B は P(goal)×(1−P(coll)))---
    success_rate    選んだ候補が「goal かつ 衝突なし」の割合
    coll_rate       選んだ候補が衝突した割合(B)
    regret          最良候補の成功(0/1)− 選んだ候補の成功
    dist_regret_mm  選んだ候補のゴール距離 − 衝突しない候補の中の最小ゴール距離(参考)
    ranking_acc     シーン内の候補の組のうち、本当の成功が違う組で、score の順が正しい割合(同点は 0.5)
    h5_acc          片方だけ衝突する matched pair で、衝突する方に高い衝突確率を付けた割合(同点は 0.5、B)
    random_success / oracle_success   下限(ランダムに選ぶ)と上限(正解を知っている)

bootstrap: 評価に使うシーン(ペア)を復元抽出し、全モデル・全 seed・両 Task に同じ抽出を使う。
各抽出で、seed ごとに指標を計算して seed の平均を取る。95% CI はパーセンタイル。
"""
import json
from pathlib import Path

import h5py
import numpy as np

MODELS = ["eef_num", "full_num", "eef_img", "full_img"]
LOWER_IS_BETTER = {"pos_err_mm", "contact_err_mm", "coll_rate", "regret", "dist_regret_mm"}
TABLE_TOP_Y = 0.773


# ================= 正解 =================

def load_truth(cache_path):
    with h5py.File(cache_path, "r") as h:
        e = h["episodes"]
        t = {k: e[k][:] for k in ["pair_idx", "task", "cand_idx", "kind", "mp", "branch", "goal", "coll",
                                  "goal_dist", "final_pos", "contact_point", "contact_valid"]}
        t["final_mask"] = e["final_mask"][:] >= 128
        t["init_pos"] = h["scenes/target_init_pos"][:][e["scene_idx"][:]]
        t["pair_ids"] = [p.decode() for p in h["pairs/pair_id"][:]]
        t["res"] = int(h.attrs["res"])
    t["n_pairs"] = len(t["pair_ids"])
    return t


def save_predictions(path, ep, final_pos, goal, coll, final_mask=None, contact_point=None):
    d = {"ep": np.asarray(ep, np.int64), "final_pos": np.asarray(final_pos, np.float32),
         "goal": np.asarray(goal, np.float32), "coll": np.asarray(coll, np.float32)}
    if final_mask is not None:
        d["final_mask"] = np.asarray(final_mask, np.float16)
    if contact_point is not None:
        d["contact_point"] = np.asarray(contact_point, np.float32)
    np.savez_compressed(path, **d)


def label_baseline(truth, train_pairs, eval_eps, by=("kind",)):
    """
    action を見ない単純な基準(予測と同じ形の dict を返す)。train のペアだけで、(task, *by) ごとの平均を取る
      by=("kind",)          : 候補の種類だけ(「種類だけで当てる」)
      by=("kind", "branch") : 種類 + matched pair の枝(wrist-down / up)。H5 の比較の相手
    goal・coll = そのグループの train での割合、final_pos = 初期位置 + そのグループの平均の移動
    """
    train_pairs = set(int(p) for p in train_pairs)
    tr = np.isin(truth["pair_idx"], list(train_pairs))
    disp = truth["final_pos"] - truth["init_pos"]
    def key(i):
        return (int(truth["task"][i]),) + tuple(int(truth[b][i]) for b in by)
    groups = {}
    for i in np.nonzero(tr)[0]:
        groups.setdefault(key(i), []).append(i)
    fall = {}
    for t in (0, 1):
        m = np.nonzero(tr & (truth["task"] == t))[0]
        fall[t] = m
    eval_eps = np.asarray(eval_eps)
    goal = np.zeros(len(eval_eps)); coll = np.zeros(len(eval_eps)); pos = np.zeros((len(eval_eps), 3), np.float32)
    for n, i in enumerate(eval_eps):
        g = np.asarray(groups.get(key(i), fall[int(truth["task"][i])]))
        goal[n] = truth["goal"][g].mean()
        coll[n] = (truth["coll"][g] == 1).mean()
        pos[n] = truth["init_pos"][i] + disp[g].mean(0)
    return {"ep": eval_eps.astype(np.int64), "goal": goal.astype(np.float32), "coll": coll.astype(np.float32), "final_pos": pos}


def load_predictions(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


# ================= 基本の指標 =================

def average_precision(y, s, w=None):
    """AUPRC(sklearn の average_precision_score と同じ定義: Σ (R_n − R_{n−1}) P_n、同じ score はまとめる)。w は重み(bootstrap 用)"""
    y = np.asarray(y, bool); s = np.asarray(s, np.float64)
    w = np.ones(len(y)) if w is None else np.asarray(w, np.float64)
    m = w > 0
    y, s, w = y[m], s[m], w[m]
    pos = (w * y).sum()
    if pos <= 0:
        return np.nan
    o = np.argsort(-s, kind="mergesort")
    y, s, w = y[o], s[o], w[o]
    tp = np.cumsum(w * y); fp = np.cumsum(w * ~y)
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]      # 同じ score のかたまりの最後
    tp, fp = tp[last], fp[last]
    prec = tp / (tp + fp); rec = tp / pos
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def mask_iou(pred_prob, truth_bool, thr=0.5):
    p = pred_prob >= thr
    inter = (p & truth_bool).sum(axis=(1, 2)); union = (p | truth_bool).sum(axis=(1, 2))
    out = np.full(len(p), np.nan)
    ok = union > 0
    out[ok] = inter[ok] / union[ok]
    return out


# ================= 1 モデル × 1 seed の、episode とシーンごとの値 =================

def episode_scene_stats(truth, pred, task):
    """task: "A" or "B"。返り値の配列は、episode ごと(ep_*)とペアごと(sc_*、長さ = 全ペア数)"""
    tk = 0 if task == "A" else 1
    order = np.argsort(pred["ep"])
    ep = pred["ep"][order]
    sel = truth["task"][ep] == tk
    ep = ep[sel]
    P = {k: v[order][sel] for k, v in pred.items() if k != "ep"}
    pair = truth["pair_idx"][ep]
    y_goal = truth["goal"][ep].astype(bool)
    y_coll = (truth["coll"][ep] == 1) if tk == 1 else np.zeros(len(ep), bool)
    st = {"ep": ep, "ep_pair": pair, "y_goal": y_goal, "s_goal": P["goal"].astype(np.float64)}
    if tk == 1:
        st["y_coll"] = y_coll; st["s_coll"] = P["coll"].astype(np.float64)
    d = (P["final_pos"] - truth["final_pos"][ep])[:, [0, 2]]
    st["ep_pos_err_mm"] = 1000 * np.linalg.norm(d, axis=1)
    if "final_mask" in P:
        st["ep_mask_iou"] = mask_iou(P["final_mask"].astype(np.float32), truth["final_mask"][ep])
    if "contact_point" in P:
        cv = truth["contact_valid"][ep].astype(bool)
        ce = np.full(len(ep), np.nan)
        ce[cv] = 1000 * np.linalg.norm(P["contact_point"][cv] - truth["contact_point"][ep][cv], axis=1)
        st["ep_contact_err_mm"] = ce

    # ---- シーン(ペア)ごと ----
    n = truth["n_pairs"]
    score = st["s_goal"] * (1 - st["s_coll"]) if tk == 1 else st["s_goal"]
    succ_b = y_goal & ~y_coll
    succ = succ_b.astype(np.float64)
    gd = truth["goal_dist"][ep]
    keys = ["sc_has", "sc_success", "sc_coll", "sc_regret", "sc_dist_regret_mm", "sc_rank_correct", "sc_rank_n",
            "sc_random_success", "sc_oracle_success", "sc_h5_correct", "sc_h5_n"]
    sc = {k: np.zeros(n) for k in keys}
    cand = truth["cand_idx"][ep]
    for p in np.unique(pair):
        m = np.nonzero(pair == p)[0]
        sc["sc_has"][p] = 1
        # 同点はキャッシュの候補番号が小さい方(決まった順)
        j = m[np.lexsort((cand[m], -score[m]))[0]]
        sc["sc_success"][p] = succ[j]
        sc["sc_coll"][p] = y_coll[j]
        sc["sc_oracle_success"][p] = succ[m].max()
        sc["sc_random_success"][p] = succ[m].mean()
        sc["sc_regret"][p] = succ[m].max() - succ[j]
        safe = m[~y_coll[m]]
        sc["sc_dist_regret_mm"][p] = 1000 * (gd[j] - gd[safe].min()) if len(safe) else np.nan
        # ranking: 本当の成功が違う組
        a, b = np.meshgrid(m, m, indexing="ij")
        diff = succ_b[a] & ~succ_b[b]                   # a が成功、b が失敗
        if diff.any():
            sa, sb = score[a][diff], score[b][diff]
            sc["sc_rank_correct"][p] = (sa > sb).sum() + 0.5 * (sa == sb).sum()
            sc["sc_rank_n"][p] = diff.sum()
        # H5
        if tk == 1:
            for mp in (0, 1):
                mm = m[truth["mp"][ep[m]] == mp]
                if len(mm) == 2 and y_coll[mm[0]] != y_coll[mm[1]]:
                    hit, miss = (mm[0], mm[1]) if y_coll[mm[0]] else (mm[1], mm[0])
                    c = st["s_coll"]
                    sc["sc_h5_correct"][p] += 1.0 if c[hit] > c[miss] else 0.5 if c[hit] == c[miss] else 0.0
                    sc["sc_h5_n"][p] += 1
    st.update(sc)
    return st


def metrics_from_stats(st, w_pair, task):
    """w_pair: ペアごとの重み(長さ = 全ペア数。評価に使わないペアは 0)"""
    w_ep = w_pair[st["ep_pair"]]
    out = {}

    def wmean(v, w):
        ok = np.isfinite(v) & (w > 0)
        return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan

    out["pos_err_mm"] = wmean(st["ep_pos_err_mm"], w_ep)
    out["goal_auprc"] = average_precision(st["y_goal"], st["s_goal"], w_ep)
    if task == "B":
        out["coll_auprc"] = average_precision(st["y_coll"], st["s_coll"], w_ep)
    if "ep_mask_iou" in st:
        out["mask_iou"] = wmean(st["ep_mask_iou"], w_ep)
    if "ep_contact_err_mm" in st:
        out["contact_err_mm"] = wmean(st["ep_contact_err_mm"], w_ep)
    ws = w_pair * st["sc_has"]
    for k, name in [("sc_success", "success_rate"), ("sc_regret", "regret"), ("sc_dist_regret_mm", "dist_regret_mm"),
                    ("sc_random_success", "random_success"), ("sc_oracle_success", "oracle_success")]:
        out[name] = wmean(st[k], ws)
    if task == "B":
        out["coll_rate"] = wmean(st["sc_coll"], ws)
    rn = (ws * st["sc_rank_n"]).sum()
    out["ranking_acc"] = float((ws * st["sc_rank_correct"]).sum() / rn) if rn > 0 else np.nan
    if task == "B":
        hn = (ws * st["sc_h5_n"]).sum()
        out["h5_acc"] = float((ws * st["sc_h5_correct"]).sum() / hn) if hn > 0 else np.nan
        out["h5_n"] = float(hn)
    return out


# ================= bootstrap =================

def evaluate(truth, models, pairs=None, n_boot=2000, seed=0, tasks=("A", "B")):
    """
    models: {"full_img": [pred_seed0, pred_seed1, ...], ...}(pred は load_predictions の dict)
    pairs : 評価に使うペア番号(None なら予測に出てくる全ペア)
    返り値: {"point": {task: {model: {metric: 値}}}, "boot": {task: {model: {metric: (n_boot,) 配列}}}, "pairs": [...]}
    """
    if pairs is None:
        eps = np.concatenate([p["ep"] for ps in models.values() for p in ps])
        pairs = np.unique(truth["pair_idx"][eps])
    pairs = np.asarray(sorted(pairs))
    n = truth["n_pairs"]
    w0 = np.zeros(n); w0[pairs] = 1
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(pairs), size=(n_boot, len(pairs)))      # 全モデル・全 seed・両 Task で共通
    W = np.zeros((n_boot, n))
    for b in range(n_boot):
        np.add.at(W[b], pairs[draws[b]], 1)
    res = {"point": {}, "boot": {}, "pairs": [truth["pair_ids"][p] for p in pairs], "n_boot": n_boot}
    for task in tasks:
        res["point"][task] = {}; res["boot"][task] = {}
        for name, preds in models.items():
            stats = [episode_scene_stats(truth, p, task) for p in preds]
            pts = [metrics_from_stats(s, w0, task) for s in stats]
            keys = list(pts[0].keys())
            res["point"][task][name] = {k: float(np.nanmean([p[k] for p in pts])) for k in keys}
            res["point"][task][name]["n_seeds"] = len(preds)
            bt = {k: np.empty(n_boot) for k in keys}
            for b in range(n_boot):
                ms = [metrics_from_stats(s, W[b], task) for s in stats]
                for k in keys:
                    bt[k][b] = np.nanmean([m[k] for m in ms])
            res["boot"][task][name] = bt
    return res


def diff_ci(res, task, a, b, metric, alpha=0.05):
    """a − b の点推定と (1−alpha) CI、片側 p 値(a − b ≤ 0 の割合)"""
    pa = res["point"][task][a][metric]; pb = res["point"][task][b][metric]
    d = res["boot"][task][a][metric] - res["boot"][task][b][metric]
    d = d[np.isfinite(d)]
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = (np.sum(d <= 0) + 1) / (len(d) + 1)
    return {"diff": pa - pb, "lo": float(lo), "hi": float(hi), "p_le0": float(p)}


def holm(pvals, alpha=0.05):
    """Holm 補正。pvals: {名前: p}。返り値 {名前: (補正後 p, 有意か)}"""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items); out = {}; running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p)); running = adj
        out[k] = (adj, adj < alpha)
    return out


# ================= 仮説 =================

H1_MARGINS = {"pos_err_mm": 3.0, "goal_auprc": 0.05, "ranking_acc": 0.05}


def judge_hypotheses(res):
    """proposal §2.6 の基準。どちらの形式(num / img)でも別々に判定し、両方を主張するときの Holm も付ける"""
    out = {}
    fmts = ["num", "img"]
    # H1(Task A): 手先のみの性能低下がマージン以内(3 指標とも)
    h1 = {}
    for f in fmts:
        e, fl = f"eef_{f}", f"full_{f}"
        parts = {}
        for m, margin in H1_MARGINS.items():
            # 低下 = 手先が悪い方向を正にする
            c = diff_ci(res, "A", e, fl, m) if m in LOWER_IS_BETTER else diff_ci(res, "A", fl, e, m)
            parts[m] = {**c, "margin": margin}
        if all(v["hi"] <= v["margin"] for v in parts.values()):
            v = "supported"
        elif any(v["lo"] > v["margin"] for v in parts.values()):
            v = "rejected"
        else:
            v = "inconclusive"
        h1[f] = {"verdict": v, "degradation": parts}
    out["H1"] = h1
    # H2(Task B): 腕全体の衝突 AUPRC が高い
    h2 = {f: diff_ci(res, "B", f"full_{f}", f"eef_{f}", "coll_auprc") for f in fmts}
    for f in fmts:
        h2[f]["verdict"] = "supported" if h2[f]["lo"] > 0 else "not supported"
    hp = holm({f: h2[f]["p_le0"] for f in fmts})
    for f in fmts:
        h2[f]["holm_p"], h2[f]["holm_sig"] = hp[f]
    out["H2"] = h2
    # H3(Task A・B 別): 画像は IoU が高く、数値は位置誤差が 3 mm 以上悪化しない
    h3 = {}
    for task in ["A", "B"]:
        for sc in ["eef", "full"]:
            iou = diff_ci(res, task, f"{sc}_img", f"{sc}_num", "mask_iou")
            pos = diff_ci(res, task, f"{sc}_num", f"{sc}_img", "pos_err_mm")
            v = "supported" if iou["lo"] > 0 and pos["hi"] <= 3.0 else "not supported"
            h3[f"{task}/{sc}"] = {"verdict": v, "iou_gain": iou, "num_pos_excess_mm": pos}
    out["H3"] = h3
    # H4(Task B): 腕全体の方が、選んだ候補の衝突率と regret が低い、かつ同じ形式で H2
    h4 = {}
    for f in fmts:
        cr = diff_ci(res, "B", f"eef_{f}", f"full_{f}", "coll_rate")
        rg = diff_ci(res, "B", f"eef_{f}", f"full_{f}", "regret")
        v = "supported" if cr["lo"] > 0 and rg["lo"] > 0 and h2[f]["verdict"] == "supported" else "not supported"
        h4[f] = {"verdict": v, "coll_rate_drop": cr, "regret_drop": rg}
    out["H4"] = h4
    # H5(Task B): matched pair の正解率
    h5 = {f: diff_ci(res, "B", f"full_{f}", f"eef_{f}", "h5_acc") for f in fmts}
    for f in fmts:
        h5[f]["verdict"] = "supported" if h5[f]["lo"] > 0 else "not supported"
    hp = holm({f: h5[f]["p_le0"] for f in fmts})
    for f in fmts:
        h5[f]["holm_p"], h5[f]["holm_sig"] = hp[f]
    out["H5"] = h5
    return out


# ================= 書き出し =================

def _fmt(v, metric=""):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:.1f}" if metric.endswith("_mm") else f"{v:.3f}"


KINDS_JA = ["matched pair", "ゴール方向", "ゴールの縁", "ランダム", "円柱の方向"]   # wm_cache の KINDS の順


def per_kind(truth, models, task):
    """候補の種類ごとの goal・衝突の AUPRC(点推定、seed の平均)。種類の中でラベルが全部同じなら None"""
    tk = 0 if task == "A" else 1
    out = {}
    for name, preds in models.items():
        out[name] = {}
        for k in range(len(KINDS_JA)):
            row = {}
            for lab, key in [("goal", "goal"), ("coll", "coll")] if tk == 1 else [("goal", "goal")]:
                vals = []
                for p in preds:
                    ep = p["ep"]; sel = (truth["task"][ep] == tk) & (truth["kind"][ep] == k)
                    y = truth["goal"][ep[sel]].astype(bool) if lab == "goal" else (truth["coll"][ep[sel]] == 1)
                    if sel.sum() and 0 < y.sum() < len(y):
                        vals.append(average_precision(y, p[key][sel].astype(np.float64)))
                row[lab] = float(np.mean(vals)) if vals else None
            out[name][KINDS_JA[k]] = row
    return out


def report_markdown(res, hyp=None, title="評価", truth=None, models=None):
    pred_models = models
    L = [f"# {title}", "", f"- 評価したペア {len(res['pairs'])}、bootstrap {res['n_boot']} 回（シーン単位、全モデル・両 Task で同じ抽出）", ""]
    for task in res["point"]:
        models = list(res["point"][task])
        metrics = []
        for name in models:   # モデルによって無い指標もある(mask や接触点を出さないモデル)
            metrics += [k for k in res["point"][task][name] if k not in ("n_seeds", "h5_n") and k not in metrics]
        L += [f"## Task {task}", "", "| 指標 | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for m in metrics:
            row = []
            for name in models:
                if m not in res["boot"][task][name]:
                    row.append("—"); continue
                bt = res["boot"][task][name][m]; bt = bt[np.isfinite(bt)]
                ci = f" [{_fmt(np.percentile(bt, 2.5), m)}, {_fmt(np.percentile(bt, 97.5), m)}]" if len(bt) else ""
                row.append(_fmt(res["point"][task][name][m], m) + ci)
            L.append(f"| {m} | " + " | ".join(row) + " |")
        L.append("")
    # ---- action を見ない基準との比較(P14) ----
    bases = [b for b in ("base_kind", "base_kind_branch") if b in res["point"].get("B", res["point"][next(iter(res["point"]))])]
    reps_here = [m for m in MODELS if m in res["point"][next(iter(res["point"]))]]
    if bases and reps_here:
        L += ["## 基準との比較（モデル − 基準、95% CI）", "",
              "- base_kind = 候補の種類だけ（train の割合）、base_kind_branch = 種類 + matched pair の枝。H5 は 0.5 ではなく base_kind_branch と比べる", ""]
        for task, mets in [("A", [("goal_auprc", "base_kind"), ("ranking_acc", "base_kind")]),
                           ("B", [("goal_auprc", "base_kind"), ("coll_auprc", "base_kind"), ("ranking_acc", "base_kind"), ("h5_acc", "base_kind_branch")])]:
            if task not in res["point"]:
                continue
            L += [f"| Task {task} | " + " | ".join(reps_here) + " |", "|---|" + "---|" * len(reps_here)]
            for m, b in mets:
                if b not in res["point"][task]:
                    continue
                row = []
                for r in reps_here:
                    c = diff_ci(res, task, r, b, m)
                    row.append(f"{c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]")
                L.append(f"| {m} − {b} | " + " | ".join(row) + " |")
            L.append("")
    if truth is not None and pred_models is not None:
        L += ["## 候補の種類ごと（AUPRC の点推定。種類の中でラベルが全部同じなら —）", ""]
        for task in res["point"]:
            pk = per_kind(truth, pred_models, task)
            names = list(pk)
            labs = ["goal", "coll"] if task == "B" else ["goal"]
            L += [f"| Task {task} | " + " | ".join(f"{n} {l}" for n in names for l in labs) + " |", "|---|" + "---|" * (len(names) * len(labs))]
            for kj in KINDS_JA:
                L.append(f"| {kj} | " + " | ".join(_fmt(pk[n][kj].get(l)) for n in names for l in labs) + " |")
            L.append("")
    if hyp:
        L += ["## 仮説", ""]
        for f in ["num", "img"]:
            h = hyp["H1"][f]
            parts = "、".join(f"{m} 低下 {_fmt(v['diff'], m)} [{_fmt(v['lo'], m)}, {_fmt(v['hi'], m)}]（マージン {v['margin']}）" for m, v in h["degradation"].items())
            L.append(f"- H1（{f}）：**{h['verdict']}** — {parts}")
        for H, met in [("H2", "coll_auprc"), ("H5", "h5_acc")]:
            for f in ["num", "img"]:
                h = hyp[H][f]
                L.append(f"- {H}（{f}、{met} 腕全体 − 手先）：**{h['verdict']}** — {h['diff']:.3f} [{h['lo']:.3f}, {h['hi']:.3f}]、Holm p = {h['holm_p']:.3f}")
        for k, h in hyp["H3"].items():
            L.append(f"- H3（{k}）：**{h['verdict']}** — IoU 改善 {h['iou_gain']['diff']:.3f} [{h['iou_gain']['lo']:.3f}, {h['iou_gain']['hi']:.3f}]、"
                     f"数値の位置誤差の超過 {h['num_pos_excess_mm']['diff']:.1f} mm [{h['num_pos_excess_mm']['lo']:.1f}, {h['num_pos_excess_mm']['hi']:.1f}]")
        for f in ["num", "img"]:
            h = hyp["H4"][f]
            L.append(f"- H4（{f}）：**{h['verdict']}** — 衝突率の低下 {h['coll_rate_drop']['diff']:.3f} [{h['coll_rate_drop']['lo']:.3f}, {h['coll_rate_drop']['hi']:.3f}]、"
                     f"regret の低下 {h['regret_drop']['diff']:.3f} [{h['regret_drop']['lo']:.3f}, {h['regret_drop']['hi']:.3f}]")
    return "\n".join(L) + "\n"


def to_jsonable(res, hyp=None):
    def conv(o):
        if isinstance(o, dict):
            return {k: conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return None   # bootstrap の配列は書かない
        if isinstance(o, (np.floating, np.integer, np.bool_)):
            return o.item()
        return o
    return {"point": conv(res["point"]), "pairs": res["pairs"], "n_boot": res["n_boot"], "hypotheses": conv(hyp) if hyp else None}
