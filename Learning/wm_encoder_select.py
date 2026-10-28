"""
P15: Docs/encoder_fix/protocol.md §3 のルールどおりにエンコーダを選び、レポートを書く
使い方: python Learning/wm_encoder_select.py [--res Docs/encoder_fix/results] → Docs/encoder_fix/report.md
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ORDER = ["base", "attn", "gru", "tf"]
FIT = {"pos_err_mm": 20.0, "goal_auprc": 0.95, "coll_auprc": 0.95}
FMT = {"num": ["eef_num", "full_num"], "img": ["eef_img", "full_img"]}


def load(res):
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted((res / "runs").glob("*.json"))]


def agg(runs, rep, enc, lr, steps="raster", cube=False):
    rs = [r for r in runs if r["config"]["rep"] == rep and r["config"]["enc"] == enc and r["config"]["steps"] == steps
          and abs(r["config"]["lr"] - lr) < 1e-12 and bool(r["config"].get("cube")) == cube]
    if not rs:
        return None
    m = lambda f: float(np.mean([f(r) for r in rs]))
    return {"n": len(rs), "params": rs[0]["params"],
            "A_pos": m(lambda r: r["train"]["A"]["pos_err_mm"]), "A_goal": m(lambda r: r["train"]["A"]["goal_auprc"]),
            "B_pos": m(lambda r: r["train"]["B"]["pos_err_mm"]), "B_goal": m(lambda r: r["train"]["B"]["goal_auprc"]),
            "B_coll": m(lambda r: r["train"]["B"].get("coll_auprc_nonmp", r["train"]["B"]["coll_auprc"])),   # 追記 1
            "B_coll_all": m(lambda r: r["train"]["B"]["coll_auprc"]),
            "A_iou": m(lambda r: r["train"]["A"]["mask_iou"]),
            "val_loss": m(lambda r: r["val_loss"]["total"]),
            "vA_pos": m(lambda r: r["val"]["A"]["pos_err_mm"]), "vA_goal": m(lambda r: r["val"]["A"]["goal_auprc"]),
            "vB_coll": m(lambda r: r["val"]["B"]["coll_auprc"]),
            "vA_iou": m(lambda r: r["val"]["A"]["mask_iou"]),
            "sec": m(lambda r: r["train_sec"])}


def fits(a):
    return (a["A_pos"] <= FIT["pos_err_mm"] and a["B_pos"] <= FIT["pos_err_mm"] and a["A_goal"] >= FIT["goal_auprc"]
            and a["B_goal"] >= FIT["goal_auprc"] and a["B_coll"] >= FIT["coll_auprc"])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", default=str(ROOT / "Docs/encoder_fix/results"))
    args = ap.parse_args(); res = Path(args.res)
    runs = load(res)
    L = ["# P15 エンコーダ比較の結果（自動生成：Learning/wm_encoder_select.py）", "",
         f"- 読んだ run：{len(runs)}。ルール：`Docs/encoder_fix/protocol.md`", "- 数値はすべて **train** の値（seed の平均）。val loss だけ val。test は使っていない", ""]

    # 診断
    L += ["## 診断（選ぶのには使わない）", "", "| run | 時間 | 箱の位置を渡す | train A 位置 mm | train A goal | train B 位置 mm | train B 衝突 | val loss |", "|---|---|---|---|---|---|---|---|"]
    for rep in ["eef_num", "full_num", "eef_img", "full_img"]:
        for steps, cube in [("all", False), ("raster", False), ("all", True)]:
            a = agg(runs, rep, "base", 1e-3, steps, cube)
            if a is None or (rep.endswith("img") and steps == "all"):
                continue
            L.append(f"| {rep} base | {'全 step' if steps == 'all' else '25 時刻'} | {'○' if cube else ''} | {a['A_pos']:.1f} | {a['A_goal']:.3f} | "
                     f"{a['B_pos']:.1f} | {a['B_coll']:.3f} | {a['val_loss']:.3f} |")
    q2 = res / "q2.json"
    if q2.exists():
        q = json.loads(q2.read_text())
        L += ["", "Q2（手で取り出した特徴 → 勾配ブースティング、Task A）:", "", "| 入力 | train 位置 mm | train goal | val 位置 mm | val goal |", "|---|---|---|---|---|"]
        for k in ["action_only", "scene_only", "action+scene"]:
            v = q[k]; L.append(f"| {k} | {v['train']['pos_err_mm']:.1f} | {v['train']['goal_auprc']:.3f} | {v['val']['pos_err_mm']:.1f} | {v['val']['goal_auprc']:.3f} |")
        L.append(f"| 箱が動かないと仮定 | {q['no_motion_baseline']['train']['pos_err_mm']:.1f} | — | {q['no_motion_baseline']['val']['pos_err_mm']:.1f} | — |")

    # 本番の比較
    L += ["", "## 候補の比較（25 時刻、100 epoch、seed 3 つの平均）", "",
          f"フィットの基準：位置 ≤ {FIT['pos_err_mm']:.0f} mm、goal AUPRC ≥ {FIT['goal_auprc']}、衝突 AUPRC ≥ {FIT['coll_auprc']}（Task A・B。衝突は matched pair 以外 = protocol 追記 1）", "",
          "| 形式 | 候補 | lr | rep | seed 数 | パラメータ | A 位置 | A goal | B 位置 | B goal | B 衝突（pair 以外） | B 衝突（全部） | A IoU | フィット | val loss |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    choice = {}
    for fmt, reps in FMT.items():
        cands = []
        for enc in ORDER:
            for lr in [1e-3, 3e-4]:
                ag = {rep: agg(runs, rep, enc, lr) for rep in reps}
                if any(v is None for v in ag.values()):
                    continue
                ok_all = all(fits(v) for v in ag.values())
                vl = float(np.mean([v["val_loss"] for v in ag.values()]))
                n = min(v["n"] for v in ag.values())
                for rep, a in ag.items():
                    L.append(f"| {fmt} | {enc} | {lr:g} | {rep} | {a['n']} | {a['params']:,} | {a['A_pos']:.1f} | {a['A_goal']:.3f} | {a['B_pos']:.1f} | "
                             f"{a['B_goal']:.3f} | {a['B_coll']:.3f} | {a['B_coll_all']:.3f} | {a['A_iou']:.3f} | {'✅' if fits(a) else '❌'} | {a['val_loss']:.3f} |")
                cands.append({"enc": enc, "lr": lr, "fit": ok_all, "val_loss": vl, "n": n,
                              "params": max(v["params"] for v in ag.values())})
        passing = [c for c in cands if c["fit"] and c["n"] >= 3]
        if passing:
            best = min(passing, key=lambda c: c["val_loss"])
            near = [c for c in passing if c["val_loss"] <= best["val_loss"] * 1.02]
            pick = sorted(near, key=lambda c: (c["params"], ORDER.index(c["enc"])))[0]
            choice[fmt] = {**pick, "best_val": best["val_loss"], "n_passing": len(passing), "n_cands": len(cands)}
        else:
            choice[fmt] = {"enc": None, "n_passing": 0, "n_cands": len(cands), "incomplete": any(c["n"] < 3 for c in cands)}
    L += ["", "## 選ばれたエンコーダ（protocol §3）", ""]
    for fmt, c in choice.items():
        if c["enc"]:
            L.append(f"- **{fmt}**：`{c['enc']}`、lr {c['lr']:g}（基準を満たした候補 {c['n_passing']}/{c['n_cands']}、val loss {c['val_loss']:.3f}、パラメータ最大 {c['params']:,}）")
        else:
            if c.get("incomplete"):
                L.append(f"- **{fmt}**：まだ選べない（seed 3 つが揃っていない候補がある。途中経過）")
            else:
                L.append(f"- **{fmt}**：基準を満たす候補なし（{c['n_cands']} 候補を評価済み）")
    # 参考: val での成績（選ぶのには使わない。val は 3 シーンだけなので、ばらつきが大きい）
    L += ["", "## 参考：val（学習に使っていない 3 シーン）での成績（選ぶのには使わない）", "",
          "| 形式 | 候補 | lr | rep | val A 位置 mm | val A goal | val B 衝突（全部） | val A IoU | val loss |", "|---|---|---|---|---|---|---|---|---|"]
    for fmt, reps in FMT.items():
        c = choice.get(fmt, {})
        rows = [("base", 1e-3, "P14 と同じ構造（lr 1e-3）")]
        if c.get("enc"):
            rows.append((c["enc"], c["lr"], "選ばれた候補"))
        for enc, lr, note in rows:
            for rep in reps:
                a = agg(runs, rep, enc, lr)
                if a:
                    L.append(f"| {fmt} | {enc}（{note}） | {lr:g} | {rep} | {a['vA_pos']:.1f} | {a['vA_goal']:.3f} | {a['vB_coll']:.3f} | {a.get('vA_iou', float('nan')):.3f} | {a['val_loss']:.3f} |")
    (ROOT / "Docs/encoder_fix/report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (res / "selection.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
    print("\n".join(L[-4:]))


if __name__ == "__main__":
    main()
