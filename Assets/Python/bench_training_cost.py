"""P1.5: 学習コストの概算(使い捨ての仮モデル。本番モデルではない)

1 条件 = 1 プロセスで計測して JSON 1 行を出力する。--sweep で全条件を子プロセスとして順に回す。

  python Assets/Python/bench_training_cost.py --sweep --out CaptureLogs/bench_<日時>.jsonl
  python Assets/Python/bench_training_cost.py --rep full_image --res 256 --frames 99 --batch 16 --amp

計測値:
  alloc_gb   = torch.cuda.max_memory_allocated()  (テンソルが実際に使った量)
  reserved_gb= torch.cuda.max_memory_reserved()   (PyTorch が確保した量。OOM に近いのはこちら)
  smi_gb     = 計測中の nvidia-smi 使用量 - 開始前の使用量 (CUDA コンテキストを含む実使用量)
  step_s     = ウォームアップ後の 1 step(forward+backward+optimizer)の平均秒数
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def smi_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return float(out[0])
    except Exception:
        return float("nan")


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class FrameEncoder(nn.Module):
    """1 枚ずつ特徴にする小さい CNN(全フレーム共通の重み)"""
    def __init__(self, cin, dim=256):
        super().__init__()
        self.net = nn.Sequential(conv_block(cin, 32), conv_block(32, 64), conv_block(64, 128), conv_block(128, dim))

    def forward(self, x):            # (N, C, H, W) -> (N, dim, H/16, W/16)
        return self.net(x)


class ImageActionModel(nn.Module):
    def __init__(self, res, act_ch=2, dim=256):
        super().__init__()
        self.scene_enc = FrameEncoder(4, dim)          # robot-free RGB + depth
        self.act_enc = FrameEncoder(act_ch, dim)       # action raster (mask + depth)
        self.temporal = nn.GRU(dim, dim, batch_first=True)
        self.film = nn.Linear(dim, 2 * dim)
        self.dec = nn.Sequential(                       # H/16 -> H, 2 outputs: final mask, contact heatmap
            nn.ConvTranspose2d(dim, 128, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 2, 4, 2, 1))
        self.head = nn.Linear(2 * dim, 3 + 2)           # final pos (3) + goal / collision logits

    def forward(self, scene, act):                      # scene (B,4,H,W), act (B,T,C,H,W)
        B, T = act.shape[:2]
        s = self.scene_enc(scene)                       # (B,dim,h,w)
        a = self.act_enc(act.flatten(0, 1))             # (B*T,dim,h,w)
        a = a.mean(dim=(2, 3)).view(B, T, -1)           # (B,T,dim)
        _, hT = self.temporal(a)
        z = hT[-1]                                      # (B,dim)
        g, b = self.film(z).chunk(2, dim=-1)
        sf = s * (1 + g[:, :, None, None]) + b[:, :, None, None]
        maps = self.dec(sf)                             # (B,2,H,W)
        vec = self.head(torch.cat([s.mean(dim=(2, 3)), z], dim=-1))
        return vec, maps


class NumericActionModel(nn.Module):
    def __init__(self, act_dim, dim=256):
        super().__init__()
        self.scene_enc = FrameEncoder(4, dim)
        self.inp = nn.Linear(act_dim, dim)
        self.temporal = nn.GRU(dim, dim, batch_first=True)
        self.film = nn.Linear(dim, 2 * dim)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(dim, 128, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 2, 4, 2, 1))
        self.head = nn.Linear(2 * dim, 5)

    def forward(self, scene, act):                      # act (B,T,D)
        s = self.scene_enc(scene)
        _, hT = self.temporal(self.inp(act))
        z = hT[-1]
        g, b = self.film(z).chunk(2, dim=-1)
        maps = self.dec(s * (1 + g[:, :, None, None]) + b[:, :, None, None])
        return self.head(torch.cat([s.mean(dim=(2, 3)), z], dim=-1)), maps


def run_one(a):
    dev = "cuda"
    torch.backends.cudnn.benchmark = True
    smi0 = smi_used_mb()
    # Windows ドライバは VRAM 不足時に RAM へ溢れて(sysmem fallback)極端に遅くなるので、上限を明示して OOM にする
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    limit_gb = a.mem_limit_gb if a.mem_limit_gb > 0 else max(0.5, total_gb - smi0 / 1024 - 0.3)
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit_gb / total_gb), 0)
    a._limit_gb = round(limit_gb, 2)
    B, H = a.batch, a.res
    if a.rep in ("full_image", "eef_image"):
        model = ImageActionModel(H).to(dev)
        act = torch.rand(B, a.frames, 2, H, H, device=dev)
    else:
        model = NumericActionModel(a.act_dim).to(dev)
        act = torch.randn(B, a.steps_numeric, a.act_dim, device=dev)
    scene = torch.rand(B, 4, H, H, device=dev)
    tgt_vec = torch.randn(B, 5, device=dev)
    tgt_map = torch.rand(B, 2, H, H, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)
    params = sum(p.numel() for p in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    smi_peak = smi0
    times = []
    for i in range(a.warmup + a.iters):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp):
            vec, maps = model(scene, act)
            loss = F.mse_loss(vec.float(), tgt_vec) + F.binary_cross_entropy_with_logits(maps.float(), tgt_map)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        if i >= a.warmup:
            times.append(dt)
        if i == a.warmup:
            smi_peak = max(smi_peak, smi_used_mb())
    smi_peak = max(smi_peak, smi_used_mb())
    return dict(limit_gb=a._limit_gb, params_m=round(params / 1e6, 2),
                alloc_gb=round(torch.cuda.max_memory_allocated() / 2**30, 3),
                reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 3),
                smi_gb=round((smi_peak - smi0) / 1024, 3),
                step_s=round(sum(times) / len(times), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", default="full_image", choices=["full_image", "eef_image", "full_numeric", "eef_numeric"])
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--frames", type=int, default=99)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--act_dim", type=int, default=33)
    ap.add_argument("--steps_numeric", type=int, default=489)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--mem_limit_gb", type=float, default=0.0, help="0 = 開始時の空き VRAM - 0.3 GB")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.sweep:
        out = a.out or os.path.join("CaptureLogs", time.strftime("bench_%Y%m%d_%H%M%S.jsonl"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        confs = []
        for amp in (False, True):
            for res in (256, 128):
                for frames in (99, 25):
                    for batch in (8, 16, 32):
                        confs.append(["--rep", "full_image", "--res", str(res), "--frames", str(frames), "--batch", str(batch)] + (["--amp"] if amp else []))
        for amp in (False, True):
            for batch in (8, 16, 32):
                confs.append(["--rep", "full_numeric", "--res", "256", "--batch", str(batch)] + (["--amp"] if amp else []))
        with open(out, "w", encoding="utf-8") as f:
            f.write(json.dumps({"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "n_conf": len(confs), "start": time.strftime("%H:%M:%S")}) + "\n")
        for i, c in enumerate(confs):
            try:
                r = subprocess.run([sys.executable, __file__] + c, capture_output=True, text=True, timeout=300)
            except subprocess.TimeoutExpired:
                r = None
            line = json.dumps({"args": " ".join(c), "error": "timeout 300s"}) if r is None else r.stdout.strip().splitlines()[-1] if r.stdout.strip() else json.dumps({"args": " ".join(c), "error": r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else "no output"})
            with open(out, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            print(f"[{i + 1}/{len(confs)}] {line}", flush=True)
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps({"done": time.strftime("%H:%M:%S")}) + "\n")
        return

    res = dict(rep=a.rep, res=a.res, frames=a.frames if a.rep.endswith("image") else a.steps_numeric, batch=a.batch, amp=a.amp)
    try:
        res.update(run_one(a))
    except torch.OutOfMemoryError:
        res.update(oom=True, limit_gb=getattr(a, "_limit_gb", None), reserved_gb=round(torch.cuda.max_memory_reserved() / 2**30, 3))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
