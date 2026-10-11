"""
P10: キャッシュと Dataset の検証・速度測定

  python Learning/test_wm_dataset.py --episodes Episodes/_p8_20scenes
結果は <cache>_check.json にも書く。全部 PASS なら exit 0。
"""
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_dataset import WMDataset, make_splits, prepare_batch, REPS   # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": str(detail)})
    print(f"  [{'OK' if ok else 'NG'}] {name}  {detail}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="Episodes/_p8_20scenes")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = Path(args.episodes)
    if not root.is_absolute():
        root = project / root
    cache = Path(args.cache) if args.cache else root / "_cache" / "wm_128x25.h5"

    with h5py.File(cache, "r") as h:
        R, T = int(h.attrs["res"]), int(h.attrs["frames"])
        K = np.asarray(h.attrs["K"]); Tw = np.asarray(h.attrs["T_world_to_cam_cv"])
        pairs = [p.decode() for p in h["pairs/pair_id"][:]]
        sid = [s.decode() for s in h["scenes/scene_id"][:]]
        e = h["episodes"]
        nE, nA, nS = len(e["task"]), h["actions/eef_mask"].shape[0], len(sid)
        print(f"== キャッシュ {cache.name}: {cache.stat().st_size / 1e6:.0f} MB、ペア {len(pairs)}・scene {nS}・action {nA}・episode {nE}")

        # 1. 件数
        n_scene_dirs = len([d for d in root.glob("scene_*") if (d / "metadata.json").is_file()])
        check("件数: scene = フォルダの数", nS == n_scene_dirs, f"{nS} vs {n_scene_dirs}")
        check("件数: action = ペア × 16(A/B で共有)", nA == len(pairs) * 16, f"{nA}")
        check("件数: episode = ペア × 32", nE == len(pairs) * 32, f"{nE}")
        check("形: action 画像 (N, 25, 128, 128)", h["actions/full_mask"].shape == (nA, T, R, R), h["actions/full_mask"].shape)

        # 2. 元データと比べる(ペアをいくつか抜き取り)
        rng = np.random.default_rng(0)
        area_err, lab_bad, proj_hits, proj_n, init_err = [], 0, 0, 0, []
        ep_task, ep_scene, ep_act, ep_cand = e["task"][:], e["scene_idx"][:], e["action_idx"][:], e["cand_idx"][:]
        sample = rng.choice(nE, size=min(60, nE), replace=False)
        for i in sample:
            sd = root / sid[ep_scene[i]]
            cid = f"c{ep_cand[i]:03d}"
            cd = sd / "candidates" / cid
            # mask の面積(256 の画素数 ≒ 128 の値の合計 × 4 / 255)
            for rep, d in [("full", "fullbody_raster"), ("eef", "eef_raster")]:
                files = sorted((cd / "actions" / d).glob("*.png"))
                k = int(rng.integers(len(files)))
                a256 = (cv2.imread(str(files[k]), cv2.IMREAD_UNCHANGED) > 127).sum()
                a128 = h[f"actions/{rep}_mask"][ep_act[i], k].astype(np.float64).sum() * 4 / 255
                if a256 > 50:
                    area_err.append(abs(a128 - a256) / a256)
            # ラベル
            fs = json.loads((cd / "final" / "final_state.json").read_text(encoding="utf-8"))
            lab = fs["labels"]
            coll = -1 if lab.get("secondary_collision") is None else int(lab["secondary_collision"])
            if bool(e["goal"][i]) != bool(lab["goal_reached"]) or int(e["coll"][i]) != coll:
                lab_bad += 1
            tgt = next(o for o in fs["objects"] if o["name"] == "Target")
            if np.abs(e["final_pos"][i] - np.array(tgt["final_position_world"])).max() > 1e-5:
                lab_bad += 1
            # TCP を 128 px の K で投影して、EEF mask(少し太らせる)に入るか
            num = h["actions/eef_num"][ep_act[i]]
            steps = h["actions/raster_steps"][ep_act[i]]
            m = h["actions/eef_mask"][ep_act[i]] > 0
            for k in range(0, T, 6):
                p = Tw @ np.r_[num[steps[k], :3], 1.0]
                u = K[0, 0] * p[0] / p[2] + K[0, 2]; v = K[1, 1] * p[1] / p[2] + K[1, 2]
                ui, vi = int(round(u)), int(round(v))
                if 0 <= ui < R and 0 <= vi < R:
                    proj_n += 1
                    win = m[k, max(0, vi - 4):vi + 5, max(0, ui - 4):ui + 5]   # パッドの間(何もない所)に TCP が来るので ±4 px
                    proj_hits += bool(win.any())
            # 初期 depth(m)
            d256 = cv2.imread(str(sd / "initial" / "robot_free" / "depth.exr"), cv2.IMREAD_UNCHANGED)
            d256 = d256[:, :, 2] if d256.ndim == 3 else d256
            d128 = h["scenes/initial_depth_mm"][ep_scene[i]].astype(np.float64)
            blk = (d256.astype(np.float64) * 1000).reshape(R, 2, R, 2)
            full = (blk > 0).all(axis=(1, 3))          # 4 画素とも何かある所だけ比べる(縁は 0 を除いた平均なので別)
            init_err.append(np.abs(d128[full] - blk.mean(axis=(1, 3))[full]).max() / 1000)
        check("縮小: mask の面積の差 中央 < 3%", np.median(area_err) < 0.03, f"中央 {100 * np.median(area_err):.2f}%・最大 {100 * np.max(area_err):.1f}%(n={len(area_err)})")
        check("ラベル: goal・衝突・最終位置が final_state.json と同じ", lab_bad == 0, f"違う {lab_bad}/{len(sample)}")
        check("投影: 128 px の K で TCP が EEF mask の近く(±4 px)", proj_hits >= 0.95 * proj_n, f"{proj_hits}/{proj_n}")
        check("初期 depth: 2x2 の平均との差 ≤ 0.5 mm(mm に丸めた分)", max(init_err) <= 0.0005 + 1e-9, f"最大 {1000 * max(init_err):.2f} mm")

        # 3. A/B が同じ action を指す
        same = all(ep_act[i] == ep_act[i + 1] and ep_task[i] == 0 and ep_task[i + 1] == 1 for i in range(0, nE, 2))
        check("A/B: 同じ候補の A と B は同じ action を使う", same)
        # P14: matched pair の 2 本は同じ手先の動き。EEF 数値に枝の手がかり(quaternion の符号など)が残っていないか
        e = h["episodes"]; mp = e["mp"][:]; si = e["scene_idx"][:]; ai = e["action_idx"][:]; ln = h["actions/num_len"][:]
        grp = {}
        for i in np.nonzero(mp >= 0)[0]:
            grp.setdefault((int(si[i]), int(mp[i])), set()).add(int(ai[i]))
        mpd = []
        for s_ in grp.values():
            if len(s_) == 2:
                a_, b_ = sorted(s_); L_ = int(ln[a_])
                mpd.append(float(np.abs(h["actions/eef_num"][a_, :L_] - h["actions/eef_num"][b_, :L_]).max()))
        shared = bool(h.attrs.get("eef_shared_in_matched_pairs", False))
        lim = 0.0 if shared else 1e-3
        check(f"matched pair: EEF 数値の差 ≤ {lim:g}(枝が分からない。共有 {shared})", bool(mpd) and max(mpd) <= lim, f"最大 {max(mpd) if mpd else None:.2e}(n={len(mpd)})")
        if shared:
            img_same = all(np.array_equal(h["actions/eef_mask"][min(s_)], h["actions/eef_mask"][max(s_)]) and
                           np.array_equal(h["actions/eef_depth_mm"][min(s_)], h["actions/eef_depth_mm"][max(s_)]) for s_ in grp.values() if len(s_) == 2)
            check("matched pair: EEF 画像が同じ(共有)", img_same)

    # 4. 分け方
    sp = make_splits(cache)
    sets = {k: set(sp[k]) for k in ["train", "val", "test"]}
    check("分け方: train / val / test が重ならない", not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]),
          {k: len(v) for k, v in sets.items()})
    check("分け方: 全ペアを使う", set().union(*sets.values()) == set(pairs))
    for split in ["train", "test"]:
        ds = WMDataset(cache, rep="eef_num", task="both", split=split, splits=sp)
        pr = {ds.pair_ids[p] for p in ds.meta["pair_idx"][ds.idx]}
        ok_ab = all(((ds.meta["pair_idx"][ds.idx] == p) & (ds.meta["task"][ds.idx] == t)).sum() == 16
                    for p in {int(x) for x in ds.meta["pair_idx"][ds.idx]} for t in (0, 1))
        check(f"分け方({split}): ペアの A/B 16 本ずつが全部入る(matched pair も一緒)", pr <= sets[split] and ok_ab, f"ペア {len(pr)}")

    # 5. 4 表現 × Task
    shapes = {}
    for rep in REPS:
        for task in ["A", "B"]:
            ds = WMDataset(cache, rep=rep, task=task, split="train", splits=sp)
            x = prepare_batch({k: (v.unsqueeze(0) if torch.is_tensor(v) else v) for k, v in ds[0].items()}, "cpu")
            x = {k: (v[0] if torch.is_tensor(v) and v.dim() > 0 else v) for k, v in x.items()}
            shapes[f"{rep}/{task}"] = tuple(x["action"].shape)
            okx = x["obs"].shape == (4, 128, 128) and torch.isfinite(x["obs"]).all() and torch.isfinite(x["action"]).all()
            okx &= (task == "A") == (not bool(x["coll_valid"]))
            check(f"Dataset {rep} Task {task}: 形と値", okx, f"action {tuple(x['action'].shape)}, obs {tuple(x['obs'].shape)}, len {len(ds)}")
    # 同じ候補の A/B で action が同じ・obs が違う
    dA = WMDataset(cache, rep="full_img", task="A"); dB = WMDataset(cache, rep="full_img", task="B")
    xa, xb = dA[0], dB[0]
    check("Dataset: 同じ候補の A/B で action は同じ・初期画像は違う(円柱)",
          torch.equal(xa["action_mask"], xb["action_mask"]) and torch.equal(xa["action_depth"], xb["action_depth"]) and not torch.equal(xa["obs"], xb["obs"]))

    # 6. 速度(1 epoch 分を何秒で読めるか)
    speed = {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for rep in ["full_img", "full_num"]:
        for workers, mem in [(0, False), (args.workers, False), (0, True)]:
            ds = WMDataset(cache, rep=rep, task="both", split="train", splits=sp, in_memory=mem)
            dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=workers, persistent_workers=workers > 0)
            for epoch in range(2):   # 2 epoch 目を測る(Windows は worker の起動に数秒かかる)
                t0 = time.time(); n = 0
                for b in dl:
                    b = prepare_batch(b, dev)   # 学習と同じく GPU に送って float にするところまで測る
                    n += b["action"].shape[0]
                if dev == "cuda":
                    torch.cuda.synchronize()
                sec = time.time() - t0
            name = f"{rep} workers={workers}{' in_memory' if mem else ''}"
            speed[name] = round(n / sec, 1)
            print(f"  速度 {name}: {n} 件 / {sec:.2f} s = {n / sec:.0f} 件/s(2 epoch 目、{dev} へ送るまで)")
            del dl
    check("速度: full_img が 300 件/s 以上(どれかの設定で。GPU の 1 step 約 0.03 s/16 件 ≒ 530 件/s の目安)",
          max(v for k, v in speed.items() if k.startswith("full_img")) >= 300, speed)

    ok = all(r["pass"] for r in RESULTS)
    out = cache.with_name(cache.stem + "_check.json")
    out.write_text(json.dumps({"passed": ok, "checks": RESULTS, "shapes": shapes, "speed_items_per_s": speed}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"== {'PASS' if ok else 'FAIL'}({sum(r['pass'] for r in RESULTS)}/{len(RESULTS)})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
