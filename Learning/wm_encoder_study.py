"""
P15: action エンコーダの公平な比較（ルールは Docs/encoder_fix/protocol.md。実験の前に固定した）

- エンコーダ以外は P14 の SmokeWM と同じ（初期画像 CNN・ゴール MLP・trunk・ヘッド・mask デコーダ・損失）
- 候補（数値・画像に同じ 4 種類）: base / attn / gru / tf
- 1 run = 1 rep × 1 enc × 1 lr × 1 seed。結果は <out>/runs/<name>.json（あれば飛ばす = 途中から再開できる）

使い方（Windows の Python 3.11）:
  py -3.11 Learning/wm_encoder_study.py sweep --plan diag  --out Docs/encoder_fix/results
  py -3.11 Learning/wm_encoder_study.py sweep --plan main  --out Docs/encoder_fix/results
  py -3.11 Learning/wm_encoder_study.py summary --out Docs/encoder_fix/results
"""
import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_dataset import WMDataset, make_splits, prepare_batch, REPS   # noqa: E402
from wm_smoke_train import ImgEnc, NumEnc, conv, losses, num_stats, batches   # noqa: E402
import wm_eval                                                                # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "Episodes/_p8_20scenes/_cache/wm_128x25.h5"
D = 128


# ================= エンコーダ =================

class NumTokens(nn.Module):
    """数値: 1 時刻ずつ小さい MLP（正規化してから）"""
    def __init__(self, cin, mean, std, d=D):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(cin, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, a):                       # (B, T, cin) → (B, T, d)
        return self.net((a - self.mean) / self.std)


class ImgTokens(nn.Module):
    """画像: 1 枚ずつ（mask + depth の 2ch）同じ CNN"""
    def __init__(self, d=D):
        super().__init__()
        self.net = nn.Sequential(conv(2, 32), conv(32, 64), conv(64, 96), conv(96, 128), conv(128, d),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, a):                       # (B, T, 2, R, R) → (B, T, d)
        B, T = a.shape[:2]
        return self.net(a.flatten(0, 1)).view(B, T, -1)


def time_embed(T, d, device):
    """正弦波の時刻埋め込み（長さが変わっても使える）"""
    pos = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    ang = pos / (10000 ** (i / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2] = torch.sin(ang); pe[:, 1::2] = torch.cos(ang)
    return pe


class SeqEnc(nn.Module):
    """トークン → 時刻埋め込み → attn / gru / tf でまとめる → (B, d)"""
    def __init__(self, tok, kind, d=D):
        super().__init__()
        self.tok, self.kind = tok, kind
        if kind == "attn":
            self.q = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            self.mha = nn.MultiheadAttention(d, 4, batch_first=True)
            self.out = nn.Sequential(nn.Linear(d, d), nn.SiLU())
        elif kind == "gru":
            self.gru = nn.GRU(d, d, num_layers=2, batch_first=True)
            self.out = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU())
        elif kind == "tf":
            self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, 2)
            self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU())
        else:
            raise ValueError(kind)

    def forward(self, a, length=None):
        x = self.tok(a)                                         # (B, T, d)
        B, T, d = x.shape
        x = x + time_embed(T, d, x.device)[None]
        pad = None
        if length is not None:
            pad = torch.arange(T, device=x.device)[None] >= length[:, None]   # True = 0 埋め
        if self.kind == "attn":
            h, _ = self.mha(self.q.expand(B, -1, -1), x, x, key_padding_mask=pad)
            return self.out(h[:, 0])
        if self.kind == "gru":
            if pad is not None:
                L = length.clamp(min=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(x, L, batch_first=True, enforce_sorted=False)
                o, hN = self.gru(packed)
                o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=T)
                m = (~pad).float()[..., None]
                mean = (o * m).sum(1) / m.sum(1).clamp(min=1)
            else:
                o, hN = self.gru(x); mean = o.mean(1)
            return self.out(torch.cat([hN[-1], mean], 1))
        # tf
        x = torch.cat([self.cls.expand(B, -1, -1), x], 1)
        if pad is not None:
            pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), pad], 1)
        h = self.tf(x, src_key_padding_mask=pad)
        return self.out(h[:, 0])


class StudyWM(nn.Module):
    """P14 の SmokeWM と同じ構造。action エンコーダだけ差し替える。cube=True は診断 Q4 専用（箱の初期位置・向きを渡す）"""
    def __init__(self, rep, enc, frames, num_dim=None, num_mean=None, num_std=None, pos_mean=(0, 0, 0), cube=False, d=D):
        super().__init__()
        self.rep, self.enc, self.cube = rep, enc, cube
        self.register_buffer("pos_mean", torch.as_tensor(pos_mean, dtype=torch.float32))
        self.obs = ImgEnc(4, d)
        self.goal = nn.Sequential(nn.Linear(4 + (5 if cube else 0), 64), nn.SiLU(), nn.Linear(64, d))
        num = rep.endswith("_num")
        if enc == "base":
            self.act = NumEnc(num_dim, num_mean, num_std, d) if num else ImgEnc(2 * frames, d)
        else:
            tok = NumTokens(num_dim, num_mean, num_std, d) if num else ImgTokens(d)
            self.act = SeqEnc(tok, enc, d)
        self.trunk = nn.Sequential(nn.Linear(3 * d, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU())
        self.head_pos = nn.Linear(256, 3)
        self.head_cls = nn.Linear(256, 2)
        self.head_cp = nn.Linear(256, 3)
        self.dec_fc = nn.Linear(256, 64 * 8 * 8)
        self.dec = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, 64, 3, 1, 1), nn.SiLU(),
                                 nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, 1, 1), nn.SiLU(),
                                 nn.Upsample(scale_factor=2), nn.Conv2d(32, 16, 3, 1, 1), nn.SiLU(),
                                 nn.Conv2d(16, 1, 3, 1, 1))

    def forward(self, b):
        zo = self.obs(b["obs"])
        g = [b["goal_xz"], b["goal_uv"]]
        if self.cube:
            g.append(b["cube"])
        zg = self.goal(torch.cat(g, 1))
        num = self.rep.endswith("_num")
        if self.enc == "base":
            za = self.act(b["action"], b["action_len"]) if num else self.act(b["action"].flatten(1, 2))
        else:
            za = self.act(b["action"], b["action_len"] if num else None)
        h = self.trunk(torch.cat([zo, zg, za], 1))
        m = self.dec(self.dec_fc(h).view(-1, 64, 8, 8))
        m = F.interpolate(m, scale_factor=2, mode="bilinear", align_corners=False)
        cls = self.head_cls(h)
        return {"final_pos": self.head_pos(h) + self.pos_mean, "goal_logit": cls[:, 0], "coll_logit": cls[:, 1],
                "mask_logit": m, "contact_point": self.head_cp(h) + self.pos_mean}


# ================= 学習と評価 =================

def add_cube(b, cube_tab):
    """診断 Q4: 箱の初期位置(x,y,z)と向き(sin, cos)"""
    b["cube"] = cube_tab[b["scene"].long()]
    return b


@torch.no_grad()
def evaluate_split(model, ds, dev, cube_tab=None):
    model.eval()
    P = {"ep": [], "final_pos": [], "goal": [], "coll": [], "final_mask": []}
    tot, n = {}, 0
    for b in batches(ds, 32, False):
        b = prepare_batch(b, dev)
        if cube_tab is not None:
            b = add_cube(b, cube_tab)
        o = model(b)
        L = losses(o, b); k = len(b["ep"]); n += k
        for kk, v in L.items():
            tot[kk] = tot.get(kk, 0.0) + v.item() * k
        P["ep"].append(b["ep"].cpu().numpy()); P["final_pos"].append(o["final_pos"].cpu().numpy())
        P["goal"].append(torch.sigmoid(o["goal_logit"]).cpu().numpy()); P["coll"].append(torch.sigmoid(o["coll_logit"]).cpu().numpy())
        P["final_mask"].append(torch.sigmoid(o["mask_logit"])[:, 0].cpu().numpy())
    P = {k: np.concatenate(v) for k, v in P.items()}
    return P, {k: v / n for k, v in tot.items()}


def point_metrics(P, truth):
    """Task A / B ごとの点推定（bootstrap なし）"""
    out = {}
    ep = P["ep"]
    for tk, name in [(0, "A"), (1, "B")]:
        s = truth["task"][ep] == tk
        if not s.any():
            continue
        e = ep[s]
        d = (P["final_pos"][s] - truth["final_pos"][e])[:, [0, 2]]
        m = {"pos_err_mm": float(1000 * np.linalg.norm(d, axis=1).mean()),
             "goal_auprc": float(wm_eval.average_precision(truth["goal"][e].astype(bool), P["goal"][s])),
             "mask_iou": float(np.nanmean(wm_eval.mask_iou(P["final_mask"][s], truth["final_mask"][e])))}
        if tk == 1:
            m["coll_auprc"] = float(wm_eval.average_precision(truth["coll"][e] == 1, P["coll"][s]))
            # protocol 追記 1: フィットの基準は matched pair 以外で見る（手先モデルは matched pair を構造上区別できないため）
            nm = truth["mp"][e] < 0
            m["coll_auprc_nonmp"] = float(wm_eval.average_precision(truth["coll"][e][nm] == 1, P["coll"][s][nm]))
        out[name] = m
    return out


def run_name(c):
    s = f"{c['rep']}__{c['enc']}__{c['steps']}__lr{c['lr']:g}__s{c['seed']}"
    return s + ("__cube" if c.get("cube") else "")


def run_one(c, out, dev, truth, splits, pos_mean, log):
    name = run_name(c)
    f = out / "runs" / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    torch.manual_seed(c["seed"]); np.random.seed(c["seed"])
    kw = dict(task="both", splits=splits, in_memory=True, num_steps=c["steps"])
    tr = WMDataset(CACHE, c["rep"], split="train", **kw)
    va = WMDataset(CACHE, c["rep"], split="val", **kw)
    nd = mean = std = None
    if c["rep"].endswith("_num"):
        mean, std = num_stats(tr); nd = len(mean)
    cube_tab = None
    if c.get("cube"):
        import h5py
        with h5py.File(CACHE, "r") as h:
            p = h["scenes/target_init_pos"][:]; yw = np.deg2rad(h["scenes/target_init_yaw_deg"][:])
        tab = np.c_[p, np.sin(4 * yw), np.cos(4 * yw)].astype(np.float32)   # 立方体なので 90° ごとに同じ → 4 倍角
        cube_tab = torch.from_numpy(tab).to(dev)
    model = StudyWM(c["rep"], c["enc"], tr.frames, nd, mean, std, pos_mean, cube=bool(c.get("cube"))).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=1e-4)
    g = torch.Generator(); g.manual_seed(c["seed"])
    steps_per_ep = math.ceil(len(tr) / 16); total = steps_per_ep * c["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    curve = []
    t0 = time.time()
    for ep in range(c["epochs"]):
        model.train(); acc = {}; n = 0
        for b in batches(tr, 16, True, g):
            b = prepare_batch(b, dev)
            if cube_tab is not None:
                b = add_cube(b, cube_tab)
            L = losses(model(b), b)
            opt.zero_grad(); L["total"].backward(); opt.step(); sched.step()
            k = len(b["ep"]); n += k
            for kk, v in L.items():
                acc[kk] = acc.get(kk, 0.0) + v.item() * k
        e = {kk: round(v / n, 5) for kk, v in acc.items()}
        if (ep + 1) % 10 == 0 or ep == 0:
            _, vl = evaluate_split(model, va, dev, cube_tab)
            e["val_total"] = round(vl["total"], 5)
            log(f"  [{name}] epoch {ep + 1:3d}  train {e['total']:.4f}  val {e['val_total']:.4f}")
        curve.append(e)
    sec = time.time() - t0
    Ptr, Ltr = evaluate_split(model, tr, dev, cube_tab)
    Pva, Lva = evaluate_split(model, va, dev, cube_tab)
    res = {"config": c, "name": name, "params": nparam, "train_sec": round(sec, 1),
           "train_loss": {k: round(v, 5) for k, v in Ltr.items()}, "val_loss": {k: round(v, 5) for k, v in Lva.items()},
           "train": point_metrics(Ptr, truth), "val": point_metrics(Pva, truth), "curve": curve}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(res, indent=1), encoding="utf-8")
    tm = res["train"]
    log(f"  [{name}] done {sec:.0f}s params {nparam:,}  TRAIN A pos {tm['A']['pos_err_mm']:.1f} goal {tm['A']['goal_auprc']:.3f} | "
        f"B pos {tm['B']['pos_err_mm']:.1f} goal {tm['B']['goal_auprc']:.3f} coll {tm['B']['coll_auprc']:.3f} | val {Lva['total']:.4f}")
    return res


def plan(which, epochs):
    P = []
    if which == "diag":       # Q1・Q3・Q4（数値だけ、seed 0、lr 1e-3）＋ 比較用に画像 base
        for rep in ["eef_num", "full_num"]:
            P.append(dict(rep=rep, enc="base", steps="all", lr=1e-3, seed=0, epochs=epochs))              # Q1
            P.append(dict(rep=rep, enc="base", steps="raster", lr=1e-3, seed=0, epochs=epochs))           # Q3
            P.append(dict(rep=rep, enc="base", steps="all", lr=1e-3, seed=0, epochs=epochs, cube=True))   # Q4
        for rep in ["eef_img", "full_img"]:
            P.append(dict(rep=rep, enc="base", steps="raster", lr=1e-3, seed=0, epochs=epochs))
    elif which == "main":     # protocol §2・§3: 4 候補 × 2 lr × 4 rep × 3 seed、全部 25 時刻
        for seed in [0, 1, 2]:
            for lr in [1e-3, 3e-4]:
                for enc in ["base", "attn", "gru", "tf"]:
                    for rep in REPS:
                        P.append(dict(rep=rep, enc=enc, steps="raster", lr=lr, seed=seed, epochs=epochs))
    elif which == "params":   # パラメータ数だけ数える
        pass
    return P


def count_params():
    tr = {r: WMDataset(CACHE, r, split="train", num_steps="raster") for r in REPS}
    rows = []
    for enc in ["base", "attn", "gru", "tf"]:
        for rep in REPS:
            nd = 7 if rep == "eef_num" else 69 if rep == "full_num" else None
            m = StudyWM(rep, enc, tr[rep].frames, nd, np.zeros(nd or 1), np.ones(nd or 1))
            rows.append((enc, rep, sum(p.numel() for p in m.parameters())))
    return rows


def summary(out):
    runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((out / "runs").glob("*.json"))]
    (out / "all_runs.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "curve"} for r in runs], indent=1), encoding="utf-8")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sweep", "params", "summary"])
    ap.add_argument("--plan", default="diag")
    ap.add_argument("--out", default=str(ROOT / "Docs/encoder_fix/results"))
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "log.txt", "a", encoding="utf-8")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    if args.cmd == "params":
        for enc, rep, n in count_params():
            log(f"params {enc:5s} {rep:9s} {n:,}")
        return
    if args.cmd == "summary":
        summary(out); return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    splits = make_splits(CACHE)
    truth = wm_eval.load_truth(CACHE)
    tr0 = WMDataset(CACHE, "eef_num", split="train", splits=splits)
    pos_mean = truth["final_pos"][tr0.idx].mean(0).tolist()
    P = plan(args.plan, args.epochs)
    log(f"===== P15 sweep {args.plan} {time.strftime('%Y-%m-%d %H:%M:%S')}  device {dev}  torch {torch.__version__}  runs {len(P)}")
    for i, c in enumerate(P):
        try:
            log(f"--- [{i + 1}/{len(P)}] {run_name(c)}")
            run_one(c, out, dev, truth, splits, pos_mean, log)
        except Exception:
            log("ERROR " + run_name(c) + "\n" + traceback.format_exc())
    summary(out)
    log(f"===== P15 sweep {args.plan} finished {time.strftime('%Y-%m-%d %H:%M:%S')}")
    (out / f"DONE_{args.plan}").write_text(time.strftime('%Y-%m-%d %H:%M:%S'), encoding="utf-8")


if __name__ == "__main__":
    main()
