"""
P15: action エンコーダの公平な比較（ルールは Docs/encoder_fix/protocol.md。実験の前に固定した）

- エンコーダ以外は P14 の SmokeWM と同じ（初期画像 CNN・ゴール MLP・trunk・ヘッド・mask デコーダ・損失）
- 候補（数値・画像に同じ 4 種類）: base / attn / gru / tf
- 1 run = 1 rep × 1 enc × 1 lr × 1 seed。結果は <out>/runs/<name>.json（あれば飛ばす = 途中から再開できる）

使い方（Windows の Python 3.11）:
  py -3.11 Learning/wm_encoder_study.py sweep --plan diag  --out Docs/encoder_fix/results
  py -3.11 Learning/wm_encoder_study.py sweep --plan main  --out Docs/encoder_fix/results
  py -3.11 Learning/wm_encoder_study.py summary --out Docs/encoder_fix/results

量産データ（protocol 追記 3・§5）: E0 base だけ、学習ペアを 25 / 50 / 100% にした学習曲線 × seed 3 つ
  py -3.11 Learning/wm_encoder_study.py sweep --plan main_v1 --cache Episodes/_main_v1/_cache/wm_128x25.h5 --out Docs/encoder_fix/results_main_v1
  （seed 0 の 12 run が先に終わるように並べてある。途中で止めても、もう一度同じコマンドで続きから回る）

early stopping（毎 epoch val を測って、val loss 最小の epoch の重みを使う）:
  py -3.11 Learning/wm_encoder_study.py sweep --plan es_v1 --cache Episodes/_main_v1/_cache/wm_128x25.h5 --out Docs/encoder_fix/results_es_v1 --epochs 40

診断: 予測した最終位置からゴールまでの距離で goal を判定（goal ヘッドが原因かを確かめる。SS と同じ 4 run）:
  py -3.11 Learning/wm_encoder_study.py sweep --plan ss_gp --cache Episodes/_main_v1/_cache/wm_128x25.h5 --out Docs/encoder_fix/results_es_v1 --epochs 40

SS（座標チャネル + spatial softmax、protocol 追記 5）。まず動作確認、それから本番:
  py -3.11 Learning/wm_encoder_study.py sweep --plan ss_smoke --cache Episodes/_main_v1/_cache/wm_128x25.h5 --out Temp/study_ss_smoke --epochs 1
  py -3.11 Learning/wm_encoder_study.py sweep --plan ss_v1 --cache Episodes/_main_v1/_cache/wm_128x25.h5 --out Docs/encoder_fix/results_es_v1 --epochs 40

F（1 時刻ずつ CNN / MLP → 1D CNN、protocol 追記 4）。まず動作確認、それから本番:
  py -3.11 Learning/wm_encoder_study.py sweep --plan f_smoke --cache Episodes/_main_v1/_cache/wm_128x25.h5 --out Temp/study_f_smoke --epochs 1
  py -3.11 Learning/wm_encoder_study.py sweep --plan f_v1 --cache Episodes/_main_v1/_cache/wm_128x25.h5 --out Docs/encoder_fix/results_es_v1 --epochs 40

案 A を early stopping 付きで（画像 2 rep、全学習ペア、seed 0）:
  py -3.11 Learning/wm_encoder_study.py sweep --plan fused_es --cache Episodes/_main_v1/_cache/wm_128x25.h5 --out Docs/encoder_fix/results_es_v1 --epochs 40

案 A（画像の action を初期画像・ゴールと同じ CNN に重ねる = enc fused）の小さい試し（100 epoch では効果なし。results_fused_pilot）:
  py -3.11 Learning/wm_encoder_study.py sweep --plan fused_pilot --cache Episodes/_main_v1/_cache/wm_128x25.h5 --out Docs/encoder_fix/results_fused_pilot
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
        elif kind == "conv":      # F（protocol 追記 4）: 数値の E0 と同じ形の 1D CNN で時間方向にまとめる
            self.net = nn.Sequential(nn.Conv1d(d, 64, 5, 1, 2), nn.SiLU(), nn.Conv1d(64, 128, 5, 2, 2), nn.SiLU(),
                                     nn.Conv1d(128, d, 5, 2, 2), nn.SiLU())
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
        if self.kind == "conv":                               # 時刻の埋め込みなし（1D CNN が順番を扱う）
            valid = torch.ones(B, T, device=x.device) if length is None else \
                (torch.arange(T, device=x.device)[None] < length[:, None]).float()
            h = self.net((x * valid[..., None]).transpose(1, 2))             # (B, d, T/4)
            m = F.interpolate(valid[:, None], size=h.shape[-1], mode="nearest")
            return (h * m).sum(-1) / m.sum(-1).clamp(min=1)
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


def goal_heatmap(uv, R, sigma=3.0):
    """ゴールの中心（画面上 0〜1）を、R×R の画像に描く（ガウス、σ = 3 px）。fused 用"""
    ax = torch.arange(R, device=uv.device, dtype=torch.float32)
    u = uv[:, 0] * R - 0.5; v = uv[:, 1] * R - 0.5                 # wm_dataset.project の逆（画素の中心）
    gx = torch.exp(-((ax[None] - u[:, None]) ** 2) / (2 * sigma ** 2))
    gy = torch.exp(-((ax[None] - v[:, None]) ** 2) / (2 * sigma ** 2))
    return (gy[:, :, None] * gx[:, None, :])[:, None]               # (B, 1, R, R)


class SSImgEnc(nn.Module):
    """SS（protocol 追記 5）: 座標チャネル（CoordConv, Liu et al. 2018）+ spatial softmax（Levine et al. 2016）
    平均プーリングで消える「画面のどこにあるか」を残す。出力 = 各特徴マップの期待位置 (x, y) と平均（k × 3）→ 線形 → d"""
    def __init__(self, cin, d=D, k=64):
        super().__init__()
        self.net = nn.Sequential(conv(cin + 2, 32), conv(32, 64), conv(64, 96, 1), conv(96, 128, 1),
                                 nn.Conv2d(128, k, 3, 1, 1))          # 128 → 64 → 32 → 32 → 32（px）
        self.log_temp = nn.Parameter(torch.zeros(1))                   # softmax の温度（学習する）
        self.out = nn.Sequential(nn.Linear(3 * k, d), nn.SiLU())

    def forward(self, x):
        B, _, H, W = x.shape
        gy, gx = torch.meshgrid(torch.linspace(-1, 1, H, device=x.device), torch.linspace(-1, 1, W, device=x.device), indexing="ij")
        x = torch.cat([x, gx.expand(B, 1, H, W), gy.expand(B, 1, H, W)], 1)
        f = self.net(x)                                                 # (B, k, h, w)
        b, k, h, w = f.shape
        p = torch.softmax(f.flatten(2) / self.log_temp.exp(), -1).view(b, k, h, w)
        ex = (p.sum(2) * torch.linspace(-1, 1, w, device=x.device)).sum(-1)    # 期待位置 x (b, k)
        ey = (p.sum(3) * torch.linspace(-1, 1, h, device=x.device)).sum(-1)    # 期待位置 y (b, k)
        return self.out(torch.cat([ex, ey, f.mean((2, 3))], 1))


class StudyWM(nn.Module):
    """P14 の SmokeWM と同じ構造。action エンコーダだけ差し替える。cube=True は診断 Q4 専用（箱の初期位置・向きを渡す）"""
    def __init__(self, rep, enc, frames, num_dim=None, num_mean=None, num_std=None, pos_mean=(0, 0, 0), cube=False, d=D, spatial=False):
        super().__init__()
        self.rep, self.enc, self.cube = rep, enc, cube
        self.register_buffer("pos_mean", torch.as_tensor(pos_mean, dtype=torch.float32))
        Img = SSImgEnc if spatial else ImgEnc           # spatial = protocol 追記 5（2D の画像エンコーダを全部 SS に）
        self.obs = Img(4, d)
        self.goal = nn.Sequential(nn.Linear(4 + (5 if cube else 0), 64), nn.SiLU(), nn.Linear(64, d))
        num = rep.endswith("_num")
        if enc == "base":
            self.act = NumEnc(num_dim, num_mean, num_std, d) if num else Img(2 * frames, d)
        elif enc == "fused":     # 画像だけ: 初期画像 4ch + ゴール 1ch + action 50ch を最初から重ねて 1 つの CNN（Action Image・Transporter の考え方）
            assert not num, "fused は画像の action 専用"
            self.act = Img(4 + 1 + 2 * frames, d)
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
        elif self.enc == "fused":
            R = b["obs"].shape[-1]
            za = self.act(torch.cat([b["obs"], goal_heatmap(b["goal_uv"], R), b["action"].flatten(1, 2)], 1))
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
             # 診断: goal ヘッドを使わず、予測した最終位置とゴール中心の距離（近いほど goal らしい）で判定した AUPRC
             # （goal のラベル = 最終位置が中心から 30 mm 以内・倒れていない・止まっている。EpisodeRecorder.cs）
             **({"goal_from_pos_auprc": float(wm_eval.average_precision(truth["goal"][e].astype(bool),
                 -np.linalg.norm(P["final_pos"][s][:, [0, 2]] - truth["goal_xz"][e], axis=1)))} if "goal_xz" in truth else {}),
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
    if c.get("frac", 1.0) < 1.0:
        s += f"__f{int(round(c['frac'] * 100)):03d}"
    if c.get("es"):
        s += f"__es{c['epochs']}"
    if c.get("spatial"):
        s += "__ss"
    if c.get("tag"):
        s += "__" + c["tag"]
    return s + ("__cube" if c.get("cube") else "")


def subset_train(splits, frac, seed=0):
    """学習ペアを frac だけ使う。順番は seed 0 で固定なので 25% ⊂ 50% ⊂ 100%（学習曲線用）。val・test は変えない"""
    if frac >= 1.0:
        return splits
    tr = list(splits["train"])
    order = np.random.default_rng(seed).permutation(len(tr))
    k = max(1, int(round(frac * len(tr))))
    sp = dict(splits); sp["train"] = sorted(tr[i] for i in order[:k])
    return sp


def run_one(c, out, dev, truth, splits, pos_mean, log):
    name = run_name(c)
    f = out / "runs" / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    torch.manual_seed(c["seed"]); np.random.seed(c["seed"])
    splits = subset_train(splits, c.get("frac", 1.0))
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
    model = StudyWM(c["rep"], c["enc"], tr.frames, nd, mean, std, pos_mean, cube=bool(c.get("cube")), spatial=bool(c.get("spatial"))).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=1e-4)
    g = torch.Generator(); g.manual_seed(c["seed"])
    steps_per_ep = math.ceil(len(tr) / 16); total = steps_per_ep * c["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    curve = []
    best = (float("inf"), 0, None)     # early stopping 用（val loss、epoch、重み）
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
        if c.get("es") or (ep + 1) % 10 == 0 or ep == 0:      # es のときは毎 epoch val を測る
            _, vl = evaluate_split(model, va, dev, cube_tab)
            e["val_total"] = round(vl["total"], 5)
            if c.get("es") and vl["total"] < best[0]:
                best = (vl["total"], ep + 1, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            if (ep + 1) % 10 == 0 or ep == 0:
                log(f"  [{name}] epoch {ep + 1:3d}  train {e['total']:.4f}  val {e['val_total']:.4f}")
        curve.append(e)
        if c.get("patience") and ep + 1 - best[1] >= c["patience"]:   # 改善がない epoch が続いたら打ち切り
            log(f"  [{name}] epoch {ep + 1}: {c['patience']} epoch 改善なしで打ち切り")
            break
    sec = time.time() - t0
    if c.get("es") and best[2] is not None:             # val loss がいちばん低かった epoch の重みに戻す
        model.load_state_dict(best[2])
        log(f"  [{name}] early stopping: best epoch {best[1]}（val {best[0]:.4f}）")
    Ptr, Ltr = evaluate_split(model, tr, dev, cube_tab)
    Pva, Lva = evaluate_split(model, va, dev, cube_tab)
    res = {"config": c, "name": name, "params": nparam, "train_sec": round(sec, 1), "cache": str(CACHE),
           "n_train_pairs": len(splits["train"]), "n_val_pairs": len(splits["val"]), "best_epoch": best[1] if c.get("es") else None,
           "train_loss": {k: round(v, 5) for k, v in Ltr.items()}, "val_loss": {k: round(v, 5) for k, v in Lva.items()},
           "train": point_metrics(Ptr, truth), "val": point_metrics(Pva, truth), "curve": curve}
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(res, indent=1), encoding="utf-8")
    pd = out / "preds"; pd.mkdir(exist_ok=True)                      # val の予測（mask 以外）を残す。あとで分析できるように
    wm_eval.save_predictions(pd / f"{name}__val.npz", Pva["ep"], Pva["final_pos"], Pva["goal"], Pva["coll"])
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
    elif which == "main_v1":  # protocol 追記 3・§5: E0 base だけ、学習ペア 25 / 50 / 100%、seed 0 → 1 → 2 の順
        for seed in [0, 1, 2]:
            for frac in [0.25, 0.5, 1.0]:
                for rep in REPS:
                    lr = 1e-3 if rep.endswith("_num") else 3e-4
                    P.append(dict(rep=rep, enc="base", steps="raster", lr=lr, seed=seed, epochs=epochs, frac=frac))
    elif which == "main_v1_smoke":  # 動くかだけ確かめる（学習ペア 25%・seed 0・4 rep。--epochs 2 などで）
        for rep in REPS:
            lr = 1e-3 if rep.endswith("_num") else 3e-4
            P.append(dict(rep=rep, enc="base", steps="raster", lr=lr, seed=0, epochs=epochs, frac=0.25))
    elif which == "es_v1":   # 過学習の対策: 毎 epoch val を測り、val loss 最小の epoch を使う（4 rep、全学習ペア、seed 0。--epochs 40 で）
        for rep in REPS:
            lr = 1e-3 if rep.endswith("_num") else 3e-4
            P.append(dict(rep=rep, enc="base", steps="raster", lr=lr, seed=0, epochs=epochs, es=True))
    elif which in ("ss_v1", "ss_smoke"):   # protocol 追記 5: 2D の画像エンコーダを全部 SS（座標チャネル + spatial softmax）に。画像を先に
        smoke = which == "ss_smoke"
        for rep in ["eef_img", "full_img", "eef_num", "full_num"]:
            lr = 1e-3 if rep.endswith("_num") else 3e-4
            c = dict(rep=rep, enc="base", steps="raster", lr=lr, seed=0, epochs=epochs, es=True, patience=8, spatial=True)
            if smoke:
                c["frac"] = 0.25
            P.append(c)
    elif which == "ss_gp":   # 診断: SS と同じ 4 run をもう一度回し、予測位置から計算した goal（goal_from_pos）も出す
        for rep in ["eef_img", "full_img", "eef_num", "full_num"]:
            lr = 1e-3 if rep.endswith("_num") else 3e-4
            P.append(dict(rep=rep, enc="base", steps="raster", lr=lr, seed=0, epochs=epochs, es=True, patience=8, spatial=True, tag="gp"))
    elif which in ("f_v1", "f_smoke"):   # protocol 追記 4: F（conv）を数値・画像の両方に。画像を先に（遅い・エラーを早く見るため）
        smoke = which == "f_smoke"
        for rep in ["eef_img", "full_img", "eef_num", "full_num"]:
            lr = 1e-3 if rep.endswith("_num") else 3e-4
            c = dict(rep=rep, enc="conv", steps="raster", lr=lr, seed=0, epochs=epochs, es=True, patience=8)
            if smoke:
                c["frac"] = 0.25
            P.append(c)
    elif which == "fused_es":   # 案 A を early stopping 付きでやり直す（fused_pilot は 100 epoch の過学習で比べてしまったため）
        for rep in ["eef_img", "full_img"]:
            P.append(dict(rep=rep, enc="fused", steps="raster", lr=3e-4, seed=0, epochs=epochs, es=True))
    elif which == "fused_pilot":   # 案 A の小さい試し: 画像 2 rep × 学習ペア 25 / 50%、seed 0（base の同じ run と比べる）
        for frac in [0.25, 0.5]:
            for rep in ["eef_img", "full_img"]:
                P.append(dict(rep=rep, enc="fused", steps="raster", lr=3e-4, seed=0, epochs=epochs, frac=frac))
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
    ap.add_argument("--cache", default=None, help="省略時は P15 と同じ _p8_20scenes のキャッシュ")
    args = ap.parse_args()
    global CACHE
    if args.cache:
        CACHE = Path(args.cache) if Path(args.cache).is_absolute() else ROOT / args.cache
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
    import h5py
    with h5py.File(CACHE, "r") as h:                                  # episode ごとのゴール中心（x, z）。goal_from_pos 用
        truth["goal_xz"] = h["scenes/goal_xz"][:][h["episodes/scene_idx"][:]]
    tr0 = WMDataset(CACHE, "eef_num", split="train", splits=splits)
    pos_mean = truth["final_pos"][tr0.idx].mean(0).tolist()
    P = plan(args.plan, args.epochs)
    log(f"===== P15 sweep {args.plan} {time.strftime('%Y-%m-%d %H:%M:%S')}  device {dev}  torch {torch.__version__}  runs {len(P)}")
    log(f"cache {CACHE}  pairs train {len(splits['train'])} / val {len(splits['val'])} / test {len(splits['test'])}  excluded {len(splits.get('_info', {}).get('excluded_pairs', []))}")
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
