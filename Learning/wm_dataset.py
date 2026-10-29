"""
P10: 学習用の PyTorch Dataset(wm_cache.py で作った HDF5 を読む)

    from wm_dataset import WMDataset, make_splits
    splits = make_splits("Episodes/_p8_20scenes/_cache/wm_128x25.h5")      # 初回に splits.json を作る
    ds = WMDataset(cache, rep="full_img", task="B", split="train", splits=splits)
    x = ds[0]

rep(action 表現。4 モデルで違うのはここだけ):
    "eef_num"  : x["action"] = (Lmax, 7)   TCP 位置 + quaternion。x["action_len"] = 本当の長さ
    "full_num" : x["action"] = (Lmax, 69)  関節角 6 + 9 点 × (位置 3 + quaternion 4)
      num_steps="raster" にすると、数値も画像と同じ 25 時刻(raster_steps)だけ → (T, 7)・(T, 69)、action_len = T
    "eef_img"  : x["action"] = (T, 2, R, R)  [mask 0〜1, depth m(0 = 何もない)]
    "full_img" : x["action"] = (T, 2, R, R)
    ※ 画像の action は、Dataset からは整数のまま(x["action_mask"] uint8・x["action_depth"] int16 mm)で出して、
      prepare_batch() で GPU に送ってから float にする(CPU で float にすると 1 件 3 MB の変換で 200 件/s 程度に詰まる)
共通の入力:
    x["obs"]      = (4, R, R)  robot-free の初期画像 RGB(0〜1)+ depth(m、0 = 背景)
    x["goal_xz"]  = (2,)       ゴールの中心(world の x, z)
    x["goal_uv"]  = (2,)       ゴールの中心(机の高さ)を画面に投影した位置(0〜1)
予測ターゲット:
    x["final_pos"] (3,)、x["goal"] ()、x["coll"] ()(A は NaN)、x["coll_valid"]、
    x["final_mask"] (1,R,R)、x["heat_target"]・x["heat_secondary"] (1,R,R)(0〜1)、
    x["contact_point"] (3,)・x["contact_valid"]
目印(評価用): x["fallback"](1 = 円柱方向の候補がランダムで埋まったペア)・x["ep"]・x["pair"]・x["scene"]・x["task"]・x["cand"]・x["kind"]・x["mp"]・x["branch"]

除外: 量産フォルダ直下の excluded_pairs.json にあるペアは、make_splits でも WMDataset でも自動で外す(exclude=False で無効)。
分け方: ペア(= 同じ seed の Task A/B)ごとに train / val / test に分ける。
同じシーンの 16 候補(matched pair を含む)と A/B は、必ず同じ側に入る。
"""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

REPS = ["eef_num", "full_num", "eef_img", "full_img"]


def load_excluded(cache_path, path=None):
    """excluded_pairs.json(量産フォルダ直下)を読んで、(除外するペア ID の集合, 円柱方向の候補がランダムで埋まったペア ID の集合)を返す。
    cache は <量産フォルダ>/_cache/xxx.h5 にある前提(path を渡せば別の場所も可)。ファイルが無ければ両方とも空。"""
    path = Path(path) if path else Path(cache_path).parent.parent / "excluded_pairs.json"
    if not path.is_file():
        return set(), set()
    d = json.loads(path.read_text(encoding="utf-8"))
    excl = {e["pair"] for e in d.get("excluded", [])}
    fb = set((d.get("cylinder_dir_fallback_pairs") or {}).get("pairs", []))
    return excl, fb


def make_splits(cache_path, ratios=(0.7, 0.15, 0.15), seed=0, out=None, overwrite=False, exclude=True):
    """ペア単位の分け方を作って JSON に保存する(既にあれば読むだけ)。戻り値 {"train": [pair_id...], ...}"""
    cache_path = Path(cache_path)
    out = Path(out) if out else cache_path.with_name(cache_path.stem + "_splits.json")
    if out.is_file() and not overwrite:
        return json.loads(out.read_text(encoding="utf-8"))
    with h5py.File(cache_path, "r") as h:
        pairs = [p.decode() for p in h["pairs/pair_id"][:]]
        seeds = h["pairs/seed"][:].tolist()
    ex_set = load_excluded(cache_path)[0] if exclude else set()      # excluded_pairs.json のペアは分けるときから外す
    if ex_set:
        keep_i = [i for i, p in enumerate(pairs) if p not in ex_set]
        pairs = [pairs[i] for i in keep_i]; seeds = [seeds[i] for i in keep_i]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    n = len(pairs)
    n_tr = int(round(ratios[0] * n)); n_va = int(round(ratios[1] * n))
    idx = {"train": order[:n_tr], "val": order[n_tr:n_tr + n_va], "test": order[n_tr + n_va:]}
    sp = {k: sorted(pairs[i] for i in v) for k, v in idx.items()}
    sp["_info"] = {"unit": "pair (= one seed / scene, Task A and B together; matched pairs never split)",
                   "ratios": list(ratios), "seed": seed, "num_pairs": n, "excluded_pairs": sorted(ex_set),
                   "seeds": {k: sorted(int(seeds[i]) for i in v) for k, v in idx.items()}}
    out.write_text(json.dumps(sp, indent=2, ensure_ascii=False), encoding="utf-8")
    return sp


def prepare_batch(b, device="cuda"):
    """DataLoader のバッチを device に送り、画像の action を (B, T, 2, R, R) の float にする"""
    out = {}
    for k, v in b.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    if "action_mask" in out:
        m = out.pop("action_mask").float() / 255.0
        d = out.pop("action_depth").float() / 1000.0
        out["action"] = torch.stack([m, d], 2)
    return out


class WMDataset(Dataset):
    def __init__(self, cache_path, rep="full_img", task="both", split=None, splits=None, in_memory=False, num_steps="all", exclude=True):
        assert rep in REPS, rep
        assert num_steps in ("all", "raster"), num_steps
        self.num_steps = num_steps
        assert task in ("A", "B", "both"), task
        self.path = str(cache_path)
        self.rep = rep
        self.in_memory = in_memory
        self._h = None
        with h5py.File(self.path, "r") as h:
            self.res = int(h.attrs["res"]); self.frames = int(h.attrs["frames"])
            self.K = np.asarray(h.attrs["K"]); self.T = np.asarray(h.attrs["T_world_to_cam_cv"])
            pair_ids = [p.decode() for p in h["pairs/pair_id"][:]]
            e = h["episodes"]
            self.meta = {k: e[k][:] for k in ["scene_idx", "action_idx", "pair_idx", "task", "cand_idx", "kind", "mp", "branch",
                                               "goal", "coll", "fell", "goal_dist", "final_yaw", "contact_valid"]}
            self.meta["final_pos"] = e["final_pos"][:]; self.meta["contact_point"] = e["contact_point"][:]
            self.goal_xz = h["scenes/goal_xz"][:]
            # 初期画像は scene の数しかない(1 万 episode でも 50 MB 程度)ので、最初に全部読んでおく
            self.scene_rgb = h["scenes/initial_rgb"][:]
            self.scene_depth = h["scenes/initial_depth_mm"][:]
            self.scene_task = h["scenes/task"][:]
            self.kinds = json.loads(e.attrs["kinds"])
            self.lmax = h["actions/eef_num"].shape[1]
            self.raster_steps = h["actions/raster_steps"][:]
            self.cache_version = str(h.attrs.get("version", ""))
        self.pair_ids = pair_ids
        ex_set, fb_set = load_excluded(self.path) if exclude else (set(), set())
        self.excluded_idx = {pair_ids.index(p) for p in ex_set if p in pair_ids}      # 除外ペア(excluded_pairs.json)
        self.fallback_idx = {pair_ids.index(p) for p in fb_set if p in pair_ids}      # 円柱方向の候補がランダムで埋まったペア(除外はしない。x["fallback"] で分かる)
        keep = np.ones(len(self.meta["task"]), bool)
        if self.excluded_idx:
            keep &= ~np.isin(self.meta["pair_idx"], list(self.excluded_idx))
        if task != "both":
            keep &= self.meta["task"] == (0 if task == "A" else 1)
        if split is not None:
            if splits is None:
                splits = make_splits(self.path)
            allowed = {pair_ids.index(p) for p in splits[split]}
            keep &= np.isin(self.meta["pair_idx"], list(allowed))
        self.idx = np.nonzero(keep)[0]
        if in_memory:
            self._load_all()

    # h5py のファイルは worker ごとに開く(fork / spawn どちらでも安全)
    def _file(self):
        if self._h is None:
            self._h = h5py.File(self.path, "r")
        return self._h

    def _load_all(self):
        h = self._file()
        need_a = np.unique(self.meta["action_idx"][self.idx])
        self._mem = {"actions": {}, "episodes": {}}
        keys = ["eef_num", "num_len"] if self.rep == "eef_num" else ["full_num", "num_len"] if self.rep == "full_num" else \
            (["eef_mask", "eef_depth_mm"] if self.rep == "eef_img" else ["full_mask", "full_depth_mm"])
        for k in keys:
            ds = h["actions/" + k]
            self._mem["actions"][k] = {int(i): ds[int(i)] for i in need_a}
        for k in ["final_mask", "heat_target", "heat_secondary"]:
            ds = h["episodes/" + k]
            self._mem["episodes"][k] = {int(i): ds[int(i)] for i in self.idx}

    def _get(self, group, key, i):
        if self.in_memory:
            return self._mem[group][key][int(i)]
        return self._file()[f"{group}/{key}"][int(i)]

    def project(self, p_world):
        """world 座標 → 画面(0〜1)。K は res 用"""
        p = self.T @ np.r_[p_world, 1.0]
        u = self.K[0, 0] * p[0] / p[2] + self.K[0, 2]; v = self.K[1, 1] * p[1] / p[2] + self.K[1, 2]
        return np.array([(u + 0.5) / self.res, (v + 0.5) / self.res], np.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        i = int(self.idx[j])
        m = self.meta
        si, ai = int(m["scene_idx"][i]), int(m["action_idx"][i])
        rgb = self.scene_rgb[si].astype(np.float32) / 255.0
        dep = self.scene_depth[si].astype(np.float32)[None] / 1000.0
        x = {"obs": torch.from_numpy(np.concatenate([rgb, dep], 0))}
        if self.rep.endswith("_num"):
            key = "eef_num" if self.rep == "eef_num" else "full_num"
            a = np.asarray(self._get("actions", key, ai), np.float32)
            if self.num_steps == "raster":
                x["action"] = torch.from_numpy(a[self.raster_steps[ai]])
                x["action_len"] = self.frames
            else:
                x["action"] = torch.from_numpy(a)
                x["action_len"] = int(self._get("actions", "num_len", ai))
        else:
            pre = "eef" if self.rep == "eef_img" else "full"
            x["action_mask"] = torch.from_numpy(np.asarray(self._get("actions", f"{pre}_mask", ai)))                       # (T, R, R) uint8
            x["action_depth"] = torch.from_numpy(np.asarray(self._get("actions", f"{pre}_depth_mm", ai)).astype(np.int16))  # (T, R, R) mm
            x["action_len"] = self.frames
        g = self.goal_xz[si]
        x["goal_xz"] = torch.from_numpy(g.astype(np.float32))
        x["goal_uv"] = torch.from_numpy(self.project(np.array([g[0], 0.773, g[1]])))
        coll = int(m["coll"][i])
        x.update({
            "final_pos": torch.from_numpy(m["final_pos"][i].astype(np.float32)),
            "goal": torch.tensor(float(m["goal"][i])),
            "coll": torch.tensor(float("nan") if coll < 0 else float(coll)),
            "coll_valid": torch.tensor(coll >= 0),
            "final_mask": torch.from_numpy(self._get("episodes", "final_mask", i).astype(np.float32)[None] / 255.0),
            "heat_target": torch.from_numpy(self._get("episodes", "heat_target", i).astype(np.float32)[None] / 255.0),
            "heat_secondary": torch.from_numpy(self._get("episodes", "heat_secondary", i).astype(np.float32)[None] / 255.0),
            "contact_point": torch.from_numpy(np.nan_to_num(m["contact_point"][i]).astype(np.float32)),
            "contact_valid": torch.tensor(bool(m["contact_valid"][i])),
            "ep": i, "pair": int(m["pair_idx"][i]), "scene": si, "task": int(m["task"][i]),
            "fallback": int(int(m["pair_idx"][i]) in self.fallback_idx),
            "cand": int(m["cand_idx"][i]), "kind": int(m["kind"][i]), "mp": int(m["mp"][i]), "branch": int(m["branch"][i]),
        })
        return x

    def __getstate__(self):   # DataLoader の worker に渡すとき、開いたファイルは渡さない
        d = self.__dict__.copy(); d["_h"] = None
        return d
