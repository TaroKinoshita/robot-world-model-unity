"""
P11: 評価コードの動作確認(仮の予測値を入れる)

  python Learning/test_wm_eval.py --episodes Episodes/_p8_20scenes [--boot 1000]

1. 指標の単体テスト(sklearn と一致、小さな手作りの例)
2. 仮の予測:正解にノイズを足した「仮モデル」を 4 表現 × 3 seed 作る
   - 腕全体(full_*)は衝突の予測を良くしてある → H2・H5 が supported になるはず
   - 画像(*_img)は mask を良くしてあり、位置誤差は数値と同じ → H3 が supported になるはず
   - goal の予測は全モデル同じ質 → H1 は「低下が小さい」(supported か inconclusive)
   - 同じモデルどうしの差は 0、CI も 0
   - 完璧な予測は AUPRC 1・IoU 1・regret 0、ランダムな予測はほぼ基準どおり
3. 結果を <root>/_eval_dummy/report.md・report.json に書く
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wm_eval as E   # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": str(detail)})
    print(f"  [{'OK' if ok else 'NG'}] {name}  {detail}")


def fake_preds(truth, rng, goal_q, coll_q, pos_sigma_mm, mask_shift_px, mask_noise):
    """正解にノイズを足す。q はロジットの離れ具合(大きいほど良い)"""
    n = len(truth["task"])
    ep = np.arange(n)
    sig = lambda z: 1 / (1 + np.exp(-z))
    g = sig(goal_q * (2 * truth["goal"].astype(float) - 1) + rng.normal(0, 1.5, n))
    yc = (truth["coll"] == 1).astype(float)
    c = sig(coll_q * (2 * yc - 1) + rng.normal(0, 1.5, n))
    pos = truth["final_pos"] + np.c_[rng.normal(0, pos_sigma_mm / 1000, n), np.zeros(n), rng.normal(0, pos_sigma_mm / 1000, n)]
    m = truth["final_mask"].astype(np.float32)
    sh = rng.integers(-mask_shift_px, mask_shift_px + 1, size=(n, 2))
    mk = np.stack([np.roll(np.roll(m[i], sh[i, 0], 0), sh[i, 1], 1) for i in range(n)])
    mk = np.clip(mk + rng.normal(0, mask_noise, mk.shape), 0, 1)
    cp = truth["contact_point"] + rng.normal(0, 0.005, (n, 3))
    return {"ep": ep, "final_pos": pos.astype(np.float32), "goal": g.astype(np.float32), "coll": c.astype(np.float32),
            "final_mask": mk.astype(np.float16), "contact_point": np.nan_to_num(cp).astype(np.float32)}


def unit_tests():
    print("== 1. 単体テスト")
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(1)
    worst = 0
    for _ in range(200):
        n = int(rng.integers(5, 200))
        y = rng.random(n) < rng.uniform(0.05, 0.6)
        if not y.any():
            y[0] = True
        s = np.round(rng.random(n), int(rng.integers(1, 4)))   # 同点も混ぜる
        worst = max(worst, abs(E.average_precision(y, s) - average_precision_score(y, s)))
    check("AUPRC: sklearn と一致(同点を含む 200 通り)", worst < 1e-12, f"最大の差 {worst:.2e}")
    y = rng.random(50) < 0.3; s = rng.random(50); w = rng.integers(0, 4, 50)
    rep = np.repeat(np.arange(50), w)
    check("AUPRC: 重み = 同じ episode を重み回並べたもの", abs(E.average_precision(y, s, w) - average_precision_score(y[rep], s[rep])) < 1e-12)
    t = np.zeros((3, 4, 4), bool); t[0, :2, :2] = True; t[1, 0, 0] = True
    p = np.zeros((3, 4, 4)); p[0, :2, :2] = 1; p[1, 0, :2] = 1
    iou = E.mask_iou(p, t)
    check("IoU: 一致 1、半分 0.5、両方空は数えない(NaN)", iou[0] == 1 and iou[1] == 0.5 and np.isnan(iou[2]), iou)
    hp = E.holm({"a": 0.01, "b": 0.04})
    check("Holm: p 0.01・0.04 → 0.02・0.04", abs(hp["a"][0] - 0.02) < 1e-12 and abs(hp["b"][0] - 0.04) < 1e-12)


def toy_scene_test():
    """1 ペア・4 候補の手作りの例で、選択・regret・ranking・H5 を確かめる"""
    print("== 2. 手作りのシーン")
    T = {"pair_idx": np.zeros(8, int), "task": np.r_[np.zeros(4, int), np.ones(4, int)], "cand_idx": np.r_[0:4, 0:4],
         "kind": np.zeros(8, int), "mp": np.r_[0, 0, -1, -1, 0, 0, -1, -1], "branch": np.r_[0, 1, -1, -1, 0, 1, -1, -1],
         "goal": np.r_[0, 0, 1, 1, 0, 0, 1, 1], "coll": np.r_[-1, -1, -1, -1, 1, 0, 1, 0],
         "goal_dist": np.r_[.1, .1, .02, .01, .1, .1, .02, .01], "final_pos": np.zeros((8, 3)),
         "contact_point": np.zeros((8, 3)), "contact_valid": np.zeros(8), "final_mask": np.zeros((8, 2, 2), bool),
         "pair_ids": ["pair_0000"], "n_pairs": 1}
    # B: c002 は goal だが衝突 → 成功は c003 だけ。モデルは c002 を一番高くする(衝突を見落とす)
    pred = {"ep": np.arange(8), "final_pos": np.zeros((8, 3)), "goal": np.r_[.1, .1, .9, .8, .1, .1, .9, .8],
            "coll": np.r_[0, 0, 0, 0, .9, .2, .1, .1]}
    st = E.episode_scene_stats(T, pred, "B")
    m = E.metrics_from_stats(st, np.ones(1), "B")
    # score = g(1-c): c000 .01, c001 .08, c002 .81, c003 .72 → c002 を選ぶ → 成功 0、衝突 1、regret 1
    check("選択(B): score 最大の c002 → 成功 0・衝突 1・regret 1", m["success_rate"] == 0 and m["coll_rate"] == 1 and m["regret"] == 1, m)
    # ranking: 成功 c003 vs 失敗 3 本 → c003(.72) > c000(.01)、c001(.08) は正しい、c002(.81) は逆 → 2/3
    check("ranking: 2/3", abs(m["ranking_acc"] - 2 / 3) < 1e-12, m["ranking_acc"])
    # H5: mp0 = c000(衝突) と c001(なし)、衝突確率 .9 > .2 → 正解 1/1
    check("H5: 衝突する方に高い確率 → 1/1", m["h5_acc"] == 1 and m["h5_n"] == 1, (m["h5_acc"], m["h5_n"]))
    check("dist_regret: 選んだ c002(20 mm)− 衝突しない中の最小 c003(10 mm)= 10 mm", abs(m["dist_regret_mm"] - 10) < 1e-6, m["dist_regret_mm"])
    mA = E.metrics_from_stats(E.episode_scene_stats(T, pred, "A"), np.ones(1), "A")
    check("選択(A): c002 → 成功 1・regret 0、oracle 1・random 0.5", mA["success_rate"] == 1 and mA["regret"] == 0 and mA["oracle_success"] == 1 and mA["random_success"] == 0.5, mA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="Episodes/_p8_20scenes")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = Path(args.episodes)
    if not root.is_absolute():
        root = project / root
    cache = root / "_cache" / "wm_128x25.h5"

    unit_tests()
    toy_scene_test()

    print("== 3. 仮モデル")
    truth = E.load_truth(cache)
    rng = np.random.default_rng(0)
    # (goal_q, coll_q, 位置 σ mm, mask のずれ px, mask のノイズ)
    spec = {"eef_num": (2.0, 0.6, 8.0, 3, 0.3), "full_num": (2.0, 2.0, 8.0, 3, 0.3),
            "eef_img": (2.0, 0.6, 8.0, 1, 0.1), "full_img": (2.0, 2.0, 8.0, 1, 0.1)}
    models = {k: [fake_preds(truth, rng, *v) for _ in range(3)] for k, v in spec.items()}
    out_dir = root / "_eval_dummy"; out_dir.mkdir(exist_ok=True)
    E.save_predictions(out_dir / "preds_full_img_seed0.npz", **models["full_img"][0])
    rt = E.load_predictions(out_dir / "preds_full_img_seed0.npz")
    check("予測ファイル: 書いて読み直しても同じ", np.array_equal(rt["goal"], models["full_img"][0]["goal"]) and rt["final_mask"].shape == models["full_img"][0]["final_mask"].shape)

    # 完璧な予測・ランダムな予測・同じモデルのコピー
    perfect = {"ep": np.arange(len(truth["task"])), "final_pos": truth["final_pos"], "goal": truth["goal"].astype(float),
               "coll": (truth["coll"] == 1).astype(float), "final_mask": truth["final_mask"].astype(np.float32)}
    rand = {**perfect, "goal": rng.random(len(truth["task"])), "coll": rng.random(len(truth["task"]))}
    models["perfect"] = [perfect]
    models["random_scores"] = [rand]            # goal・衝突の確率だけランダム(位置・mask は正解のまま)
    models["copy_full_img"] = models["full_img"]

    t0 = time.time()
    res = E.evaluate(truth, models, n_boot=args.boot, seed=0)
    print(f"  bootstrap {args.boot} 回: {time.time() - t0:.0f} s")
    pB, pA = res["point"]["B"], res["point"]["A"]
    check("完璧: AUPRC 1・IoU 1・位置誤差 0・regret 0", pB["perfect"]["goal_auprc"] == 1 and pB["perfect"]["coll_auprc"] == 1 and
          abs(pB["perfect"]["mask_iou"] - 1) < 1e-9 and pB["perfect"]["pos_err_mm"] == 0 and pB["perfect"]["regret"] == 0 and pA["perfect"]["regret"] == 0)
    check("完璧: 成功率 = oracle、ranking 1・H5 1", pB["perfect"]["success_rate"] == pB["perfect"]["oracle_success"] and pB["perfect"]["ranking_acc"] == 1 and pB["perfect"]["h5_acc"] == 1)
    base_g = truth["goal"][truth["task"] == 1].mean()
    check("ランダム: goal AUPRC ≒ 正例の割合(±0.1)", abs(pB["random_scores"]["goal_auprc"] - base_g) < 0.1, f"{pB['random_scores']['goal_auprc']:.3f} vs {base_g:.3f}")
    check("ランダム: 成功率 ≒ random_success(±0.25、20 シーンなので粗い)", abs(pB["random_scores"]["success_rate"] - pB["random_scores"]["random_success"]) < 0.25,
          f"{pB['random_scores']['success_rate']:.2f} vs {pB['random_scores']['random_success']:.2f}")
    d = E.diff_ci(res, "B", "full_img", "copy_full_img", "coll_auprc")
    check("同じモデルの差: 0、CI も [0, 0]", d["diff"] == 0 and d["lo"] == 0 and d["hi"] == 0, d)
    check("順序: perfect ≥ full_img ≥ eef_img ≥ random(衝突 AUPRC)",
          pB["perfect"]["coll_auprc"] >= pB["full_img"]["coll_auprc"] >= pB["eef_img"]["coll_auprc"] >= pB["random_scores"]["coll_auprc"],
          [round(pB[k]["coll_auprc"], 3) for k in ["perfect", "full_img", "eef_img", "random_scores"]])

    hyp = E.judge_hypotheses(res)
    check("H2: 作り込んだ差(衝突の予測)を検出 → 両形式 supported", all(hyp["H2"][f]["verdict"] == "supported" for f in ["num", "img"]),
          {f: (round(hyp["H2"][f]["lo"], 3), round(hyp["H2"][f]["hi"], 3)) for f in ["num", "img"]})
    check("H5: 両形式 supported", all(hyp["H5"][f]["verdict"] == "supported" for f in ["num", "img"]),
          {f: (round(hyp["H5"][f]["diff"], 3), round(hyp["H5"][f]["lo"], 3)) for f in ["num", "img"]})
    check("H3: mask を良くした画像 → supported(A・B × eef・full)", all(v["verdict"] == "supported" for v in hyp["H3"].values()),
          {k: v["verdict"] for k, v in hyp["H3"].items()})
    check("H1: goal・位置は同じ質 → rejected にはならない", all(hyp["H1"][f]["verdict"] != "rejected" for f in ["num", "img"]),
          {f: hyp["H1"][f]["verdict"] for f in ["num", "img"]})
    check("H4: 判定が出る(supported / not supported)", all(hyp["H4"][f]["verdict"] in ("supported", "not supported") for f in ["num", "img"]),
          {f: hyp["H4"][f]["verdict"] for f in ["num", "img"]})

    (out_dir / "report.md").write_text(E.report_markdown(res, hyp, "評価(仮の予測値で動作確認)"), encoding="utf-8")
    (out_dir / "report.json").write_text(json.dumps(E.to_jsonable(res, hyp), indent=2, ensure_ascii=False), encoding="utf-8")
    ok = all(r["pass"] for r in RESULTS)
    (out_dir / "test_result.json").write_text(json.dumps({"passed": ok, "checks": RESULTS}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"== {'PASS' if ok else 'FAIL'}({sum(r['pass'] for r in RESULTS)}/{len(RESULTS)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
