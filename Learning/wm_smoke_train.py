"""
P14: 4 表現で「学習 → 保存 → 読み込み → 予測 → 評価」が最後まで通るかの確認(精度は見ない)

- モデルは 4 表現で action のエンコーダだけが違う、小さいもの(SmokeWM)
    共通: 初期画像(RGB+depth)の CNN、ゴール(xz・画面上の位置)の MLP → ヘッド
    eef_num / full_num: 1D conv + 長さでマスクした平均   eef_img / full_img: 時刻 × 2ch をチャネルに並べた 2D CNN
    ヘッド: 最終位置(初期位置からの移動ではなく world)、goal、衝突、最終 mask(64 → 128 px)、最初の接触点
- 確かめること(結果は <out>/smoke_report.md・smoke.json)
    1. 最初の step で、全部のパラメータの塊に 0 でない勾配が流れる
    2. train の loss が下がる
    3. 保存 → 別のモデルに読み込み → 同じ入力で同じ出力
    4. 予測を npz に書いて、wm_eval で評価・H1〜H5 の判定まで出る(基準:種類だけ・種類+枝)

使い方(Windows の Python 3.11):
  python Learning/wm_smoke_train.py --cache Episodes/_p8_20scenes/_cache/wm_128x25.h5 --out Episodes/_p8_20scenes/_smoke
     [--epochs 15] [--num-steps all|raster] [--eval-split test|train|val] [--seeds 0]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_dataset import WMDataset, make_splits, prepare_batch, REPS   # noqa: E402
import wm_eval                                                         # noqa: E402


def conv(ci, co, s=2):
    return nn.Sequential(nn.Conv2d(ci, co, 3, s, 1), nn.GroupNorm(8, co), nn.SiLU())


class ImgEnc(nn.Module):
    def __init__(self, cin, d=128):
        super().__init__()
        self.net = nn.Sequential(conv(cin, 32), conv(32, 64), conv(64, 96), conv(96, 128), conv(128, d), nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        return self.net(x)


class NumEnc(nn.Module):
    def __init__(self, cin, mean, std, d=128):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(nn.Conv1d(cin, 64, 5, 1, 2), nn.SiLU(), nn.Conv1d(64, 128, 5, 2, 2), nn.SiLU(),
                                 nn.Conv1d(128, d, 5, 2, 2), nn.SiLU())

    def forward(self, a, length):
        a = (a - self.mean) / self.std
        L = a.shape[1]
        valid = (torch.arange(L, device=a.device)[None] < length[:, None]).float()
        a = a * valid[..., None]
        h = self.net(a.transpose(1, 2))                                   # (B, d, L/4)
        m = F.interpolate(valid[:, None], size=h.shape[-1], mode="nearest")  # 長さのマスクを縮める
        return (h * m).sum(-1) / m.sum(-1).clamp(min=1)


class SmokeWM(nn.Module):
    def __init__(self, rep, frames, num_dim=None, num_mean=None, num_std=None, pos_mean=(0, 0, 0), d=128):
        super().__init__()
        self.rep = rep
        self.register_buffer("pos_mean", torch.as_tensor(pos_mean, dtype=torch.float32))   # 位置の出力の中心(train の平均)
        self.obs = ImgEnc(4, d)
        self.goal = nn.Sequential(nn.Linear(4, 64), nn.SiLU(), nn.Linear(64, d))
        self.act = NumEnc(num_dim, num_mean, num_std, d) if rep.endswith("_num") else ImgEnc(2 * frames, d)
        self.trunk = nn.Sequential(nn.Linear(3 * d, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU())
        self.head_pos = nn.Linear(256, 3)
        self.head_cls = nn.Linear(256, 2)        # goal, coll の logit
        self.head_cp = nn.Linear(256, 3)
        self.dec_fc = nn.Linear(256, 64 * 8 * 8)
        self.dec = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, 64, 3, 1, 1), nn.SiLU(),
                                 nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, 1, 1), nn.SiLU(),
                                 nn.Upsample(scale_factor=2), nn.Conv2d(32, 16, 3, 1, 1), nn.SiLU(),
                                 nn.Conv2d(16, 1, 3, 1, 1))     # 64 px → 128 に拡大して出す

    def forward(self, b):
        zo = self.obs(b["obs"])
        zg = self.goal(torch.cat([b["goal_xz"], b["goal_uv"]], 1))
        if self.rep.endswith("_num"):
            za = self.act(b["action"], b["action_len"])
        else:
            a = b["action"]                                  # (B, T, 2, R, R)
            za = self.act(a.flatten(1, 2))
        h = self.trunk(torch.cat([zo, zg, za], 1))
        m = self.dec(self.dec_fc(h).view(-1, 64, 8, 8))
        m = F.interpolate(m, scale_factor=2, mode="bilinear", align_corners=False)
        cls = self.head_cls(h)
        return {"final_pos": self.head_pos(h) + self.pos_mean, "goal_logit": cls[:, 0], "coll_logit": cls[:, 1],
                "mask_logit": m, "contact_point": self.head_cp(h) + self.pos_mean}


def losses(out, b):
    L = {}
    L["pos"] = F.smooth_l1_loss(out["final_pos"], b["final_pos"], beta=0.01) * 10
    L["goal"] = F.binary_cross_entropy_with_logits(out["goal_logit"], b["goal"])
    cv = b["coll_valid"]
    L["coll"] = F.binary_cross_entropy_with_logits(out["coll_logit"][cv], b["coll"][cv]) if cv.any() else out["coll_logit"].sum() * 0
    L["mask"] = F.binary_cross_entropy_with_logits(out["mask_logit"], b["final_mask"])
    pv = b["contact_valid"]
    L["contact"] = F.smooth_l1_loss(out["contact_point"][pv], b["contact_point"][pv], beta=0.01) * 10 if pv.any() else out["contact_point"].sum() * 0
    L["total"] = sum(L.values())
    return L


def num_stats(ds):
    """train の数値 action の平均・標準偏差(長さの中だけ)"""
    xs = []
    for j in range(len(ds)):
        x = ds[j]
        xs.append(x["action"][: x["action_len"]].numpy())
    a = np.concatenate(xs)
    return a.mean(0), np.maximum(a.std(0), 1e-3)


def batches(ds, bs, shuffle, g=None):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, generator=g, drop_last=False)


@torch.no_grad()
def predict(model, ds, dev, bs=32):
    model.eval()
    P = {"ep": [], "final_pos": [], "goal": [], "coll": [], "final_mask": [], "contact_point": []}
    for b in batches(ds, bs, False):
        b = prepare_batch(b, dev)
        o = model(b)
        P["ep"].append(b["ep"].cpu().numpy())
        P["final_pos"].append(o["final_pos"].cpu().numpy())
        P["goal"].append(torch.sigmoid(o["goal_logit"]).cpu().numpy())
        P["coll"].append(torch.sigmoid(o["coll_logit"]).cpu().numpy())
        P["final_mask"].append(torch.sigmoid(o["mask_logit"])[:, 0].cpu().numpy())
        P["contact_point"].append(o["contact_point"].cpu().numpy())
    return {k: np.concatenate(v) for k, v in P.items()}


def run_one(rep, seed, args, splits, dev, log):
    torch.manual_seed(seed); np.random.seed(seed)
    kw = dict(task="both", splits=splits, in_memory=True, num_steps=args.num_steps)
    tr = WMDataset(args.cache, rep, split="train", **kw)
    ev = WMDataset(args.cache, rep, split=args.eval_split, **kw)
    nd = mean = std = None
    if rep.endswith("_num"):
        mean, std = num_stats(tr); nd = len(mean)
    model = SmokeWM(rep, tr.frames, nd, mean, std, args.pos_mean).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    g = torch.Generator(); g.manual_seed(seed)
    info = {"rep": rep, "seed": seed, "n_train": len(tr), "n_eval": len(ev), "params": nparam, "epochs": []}

    # 1. 最初の step の勾配
    b0 = prepare_batch(next(iter(batches(tr, args.bs, True, g))), dev)
    model.train(); opt.zero_grad()
    L0 = losses(model(b0), b0); L0["total"].backward()
    grads = {}
    for name, mod in [("obs", model.obs), ("goal", model.goal), ("act", model.act), ("trunk", model.trunk),
                      ("head_pos", model.head_pos), ("head_cls", model.head_cls), ("head_cp", model.head_cp), ("dec", model.dec)]:
        gn = [p.grad.norm().item() for p in mod.parameters() if p.grad is not None]
        grads[name] = {"n_with_grad": len(gn), "n_params": len(list(mod.parameters())), "norm": float(np.sqrt(np.sum(np.square(gn)))) if gn else 0.0}
    info["grad_ok"] = all(v["n_with_grad"] == v["n_params"] and v["norm"] > 0 for v in grads.values())
    info["grads"] = grads
    opt.zero_grad()

    # 2. 学習
    t0 = time.time()
    for ep in range(args.epochs):
        model.train(); acc = {}; n = 0
        for b in batches(tr, args.bs, True, g):
            b = prepare_batch(b, dev)
            L = losses(model(b), b)
            opt.zero_grad(); L["total"].backward(); opt.step()
            k = len(b["ep"]); n += k
            for kk, v in L.items():
                acc[kk] = acc.get(kk, 0.0) + v.item() * k
        e = {kk: v / n for kk, v in acc.items()}
        info["epochs"].append(e)
        log(f"  [{rep} s{seed}] epoch {ep + 1:2d}/{args.epochs}  " + "  ".join(f"{kk} {v:.4f}" for kk, v in e.items()))
    info["train_sec"] = time.time() - t0
    first, last = info["epochs"][0]["total"], info["epochs"][-1]["total"]
    info["loss_first"], info["loss_last"] = first, last
    info["loss_down"] = last < 0.7 * first

    # 3. 保存 → 読み込み → 同じ出力
    ck = Path(args.out) / f"ckpt_{rep}_s{seed}.pt"
    torch.save({"state": model.state_dict(), "rep": rep, "frames": tr.frames, "num_dim": nd,
                "num_mean": mean, "num_std": std, "pos_mean": args.pos_mean, "num_steps": args.num_steps, "seed": seed}, ck)
    c = torch.load(ck, map_location=dev, weights_only=False)
    m2 = SmokeWM(c["rep"], c["frames"], c["num_dim"], c["num_mean"], c["num_std"], c["pos_mean"]).to(dev)
    m2.load_state_dict(c["state"])
    p1 = predict(model, ev, dev); p2 = predict(m2, ev, dev)
    info["reload_maxdiff"] = float(max(np.abs(p1[k] - p2[k]).max() for k in p1 if k != "ep"))
    info["reload_ok"] = info["reload_maxdiff"] < 1e-5

    # 4. 予測を保存
    pf = Path(args.out) / f"pred_{rep}_s{seed}_{args.eval_split}.npz"
    wm_eval.save_predictions(pf, p2["ep"], p2["final_pos"], p2["goal"], p2["coll"], p2["final_mask"], p2["contact_point"])
    info["pred_file"] = str(pf)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--reps", nargs="+", default=REPS)
    ap.add_argument("--num-steps", default="all", choices=["all", "raster"])
    ap.add_argument("--eval-split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    logf = open(out / "smoke_log.txt", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    splits = make_splits(args.cache)
    truth = wm_eval.load_truth(args.cache)
    tr_ds = WMDataset(args.cache, "eef_num", split="train", splits=splits)
    args.pos_mean = truth["final_pos"][tr_ds.idx].mean(0).tolist()
    log(f"===== smoke {time.strftime('%Y-%m-%d %H:%M:%S')}  device {dev}  torch {torch.__version__}  cache {tr_ds.cache_version}  "
        f"num_steps {args.num_steps}  epochs {args.epochs}  eval {args.eval_split}")

    infos = []
    models = {}
    for rep in args.reps:
        for seed in args.seeds:
            inf = run_one(rep, seed, args, splits, dev, log)
            infos.append(inf)
            models.setdefault(rep, []).append(wm_eval.load_predictions(inf["pred_file"]))
            log(f"  [{rep} s{seed}] grad_ok {inf['grad_ok']}  loss {inf['loss_first']:.3f} → {inf['loss_last']:.3f}  "
                f"reload_maxdiff {inf['reload_maxdiff']:.2e}  {inf['train_sec']:.0f} s  params {inf['params']:,}")

    # 基準(action を見ない)
    ev_ds = WMDataset(args.cache, "eef_num", split=args.eval_split, splits=splits)
    pair_ids = truth["pair_ids"]
    train_pairs = [pair_ids.index(p) for p in splits["train"]]
    models["base_kind"] = [wm_eval.label_baseline(truth, train_pairs, ev_ds.idx, by=("kind",))]
    models["base_kind_branch"] = [wm_eval.label_baseline(truth, train_pairs, ev_ds.idx, by=("kind", "branch"))]

    res = wm_eval.evaluate(truth, models, n_boot=args.n_boot)
    hyp = wm_eval.judge_hypotheses(res) if all(r in models for r in REPS) else None
    md = wm_eval.report_markdown(res, hyp, title=f"P14 smoke(精度は意味なし。通るかの確認)eval={args.eval_split}・num_steps={args.num_steps}",
                                 truth=truth, models=models)
    checks = [{"rep": i["rep"], "seed": i["seed"], "grad_ok": i["grad_ok"], "loss_down": i["loss_down"], "reload_ok": i["reload_ok"],
               "loss_first": round(i["loss_first"], 4), "loss_last": round(i["loss_last"], 4), "train_sec": round(i["train_sec"], 1),
               "params": i["params"]} for i in infos]
    ok = all(c["grad_ok"] and c["loss_down"] and c["reload_ok"] for c in checks) and (hyp is not None or set(args.reps) != set(REPS))
    head = ["# P14 smoke", "", f"- 結果: **{'PASS' if ok else 'FAIL'}**（勾配・loss 低下・保存と読み込み・評価と判定まで）", "",
            "| rep | seed | 勾配 | loss 低下 | 読み込み | loss 最初 → 最後 | 学習 s | パラメータ |", "|---|---|---|---|---|---|---|---|"]
    for c in checks:
        head.append(f"| {c['rep']} | {c['seed']} | {'✅' if c['grad_ok'] else '❌'} | {'✅' if c['loss_down'] else '❌'} | "
                    f"{'✅' if c['reload_ok'] else '❌'} | {c['loss_first']} → {c['loss_last']} | {c['train_sec']} | {c['params']:,} |")
    (out / f"smoke_report_{args.eval_split}_{args.num_steps}.md").write_text("\n".join(head) + "\n\n" + md, encoding="utf-8")
    (out / f"smoke_{args.eval_split}_{args.num_steps}.json").write_text(json.dumps(
        {"ok": ok, "checks": checks, "infos": infos, "eval": wm_eval.to_jsonable(res, hyp)}, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    log(f"===== smoke {'PASS' if ok else 'FAIL'} → {out}")


if __name__ == "__main__":
    main()
