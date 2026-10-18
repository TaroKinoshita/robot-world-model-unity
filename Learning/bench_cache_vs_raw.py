'''
Report prep (2026-10-18): episode folders vs the single HDF5 cache - same data, same model, same batch.

What is measured (train split of the cache, Task A+B):
  1. loading only : samples/s through a DataLoader (workers 0 and 4), epoch 1 and epoch 2
  2. one training epoch : SmokeWM (the P14 smoke model), batch 16, AdamW, GPU - seconds per epoch
  3. sanity : raw and cache give the same tensors (obs, action, labels)
Raw path = what training would do without the cache: open the PNG/EXR/CSV/JSON files of each episode,
downscale 256 -> 128 px, parse CSVs - the same functions wm_cache.py uses to build the cache.

  python Learning/bench_cache_vs_raw.py --episodes Episodes/_p8_20scenes [--out Docs/report_20261018/bench.json]
'''
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wm_cache as C                                                   # noqa: E402
from wm_dataset import WMDataset, make_splits, prepare_batch            # noqa: E402
import wm_smoke_train as S                                             # noqa: E402
import h5py                                                            # noqa: E402

RES, FRAMES, TOL = 128, 25, 0.001


class RawDataset(Dataset):
    '''Same items as WMDataset, but read from the episode folders every time'''
    def __init__(self, root, cache_path, rep, split, splits):
        self.root = Path(root); self.rep = rep
        ref = WMDataset(cache_path, rep, task='both', split=split, splits=splits)
        self.idx = ref.idx; self.K = ref.K; self.T = ref.T; self.lmax = ref.lmax
        with h5py.File(cache_path, 'r') as h:
            self.scene_id = [s.decode() for s in h['scenes/scene_id'][:]]
            e = h['episodes']
            self.meta = {k: e[k][:] for k in ['scene_idx', 'cand_idx', 'pair_idx', 'task', 'kind', 'mp', 'branch']}

    def __len__(self):
        return len(self.idx)

    def project(self, p):
        q = self.T @ np.r_[p, 1.0]
        u = self.K[0, 0] * q[0] / q[2] + self.K[0, 2]; v = self.K[1, 1] * q[1] / q[2] + self.K[1, 2]
        return np.array([(u + 0.5) / RES, (v + 0.5) / RES], np.float32)

    def __getitem__(self, j):
        i = int(self.idx[j]); m = self.meta
        sd = self.root / self.scene_id[int(m['scene_idx'][i])]
        cd = sd / 'candidates' / ('c%03d' % int(m['cand_idx'][i]))
        rgb, dep = C.load_initial(sd, RES)
        x = {'obs': torch.from_numpy(np.concatenate([rgb.astype(np.float32) / 255.0, dep.astype(np.float32)[None] / 1000.0], 0))}
        if self.rep.endswith('_num'):
            if self.rep == 'eef_num':
                a = C.canon_quat_cols(C.read_numeric(cd / 'actions' / 'eef_numeric.csv', C.EEF_COLS), C.EEF_COLS)
            else:
                fc = C.full_cols(cd / 'actions' / 'fullbody_numeric.csv')
                a = C.canon_quat_cols(C.read_numeric(cd / 'actions' / 'fullbody_numeric.csv', fc), fc)
            pad = np.zeros((self.lmax, a.shape[1]), np.float32); pad[:len(a)] = a
            x['action'] = torch.from_numpy(pad); x['action_len'] = len(a)
        else:
            md, dd = ('eef_raster', 'eef_depth') if self.rep == 'eef_img' else ('fullbody_raster', 'fullbody_depth')
            mf = sorted((cd / 'actions' / md).glob('*.png')); df = sorted((cd / 'actions' / dd).glob('*.png'))
            mask = np.stack([np.round(C.down_area(C.read_gray(p).astype(np.float32), RES)).astype(np.uint8) for p in mf])
            depth = np.stack([np.round(C.down_depth(C.read_gray(p), RES)).astype(np.uint16) for p in df])
            x['action_mask'] = torch.from_numpy(mask); x['action_depth'] = torch.from_numpy(depth.astype(np.int16))
            x['action_len'] = FRAMES
        gen = C._json(sd / 'scene_generation.json')
        g = np.array([gen['goal']['center_world'][0], gen['goal']['center_world'][2]], np.float32)
        x['goal_xz'] = torch.from_numpy(g)
        x['goal_uv'] = torch.from_numpy(self.project(np.array([g[0], 0.773, g[1]])))
        f = C.load_final(cd, RES, TOL)
        coll = f['coll']
        x.update({
            'final_pos': torch.tensor(f['final_pos'], dtype=torch.float32),
            'goal': torch.tensor(float(f['goal'])),
            'coll': torch.tensor(float('nan') if coll < 0 else float(coll)),
            'coll_valid': torch.tensor(coll >= 0),
            'final_mask': torch.from_numpy(f['final_mask'].astype(np.float32)[None] / 255.0),
            'heat_target': torch.from_numpy(f['heat_target'].astype(np.float32)[None] / 255.0),
            'heat_secondary': torch.from_numpy(f['heat_secondary'].astype(np.float32)[None] / 255.0),
            'contact_point': torch.from_numpy(np.nan_to_num(np.array(f['contact_point'], np.float32))),
            'contact_valid': torch.tensor(bool(f['contact_valid'])),
            'ep': i, 'pair': int(m['pair_idx'][i]), 'scene': int(m['scene_idx'][i]), 'task': int(m['task'][i]),
            'cand': int(m['cand_idx'][i]), 'kind': int(m['kind'][i]), 'mp': int(m['mp'][i]), 'branch': int(m['branch'][i]),
        })
        return x


def loader(ds, workers, shuffle=True, bs=16, seed=0):
    g = torch.Generator(); g.manual_seed(seed)
    kw = dict(num_workers=workers, persistent_workers=workers > 0) if workers > 0 else {}
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=g, **kw)


def time_loading(ds, workers, dev, epochs=2):
    dl = loader(ds, workers)
    out = []
    for ep in range(epochs):
        torch.cuda.synchronize() if dev == 'cuda' else None
        t = time.perf_counter(); n = 0
        for b in dl:
            b = prepare_batch(b, dev); n += len(b['ep'])
        torch.cuda.synchronize() if dev == 'cuda' else None
        out.append(n / (time.perf_counter() - t))
    return out


def time_train(ds, rep, workers, dev, num_ms, pos_mean, epochs=2):
    torch.manual_seed(0)
    nd = None if num_ms is None else len(num_ms[0])
    mean, std = (None, None) if num_ms is None else num_ms
    model = S.SmokeWM(rep, FRAMES, nd, mean, std, pos_mean).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dl = loader(ds, workers)
    secs = []; t_wait = []
    for ep in range(epochs):
        model.train(); torch.cuda.synchronize() if dev == 'cuda' else None
        t = time.perf_counter(); wait = 0.0; tw = time.perf_counter()
        for b in dl:
            wait += time.perf_counter() - tw
            b = prepare_batch(b, dev)
            L = S.losses(model(b), b); opt.zero_grad(); L['total'].backward(); opt.step()
            torch.cuda.synchronize() if dev == 'cuda' else None
            tw = time.perf_counter()
        secs.append(time.perf_counter() - t); t_wait.append(wait)
    return secs, t_wait


def sanity(raw, cache, n=24):
    rng = np.random.default_rng(0); js = rng.choice(len(raw), n, replace=False)
    d = {'obs': 0.0, 'action': 0.0, 'final_pos': 0.0, 'final_mask': 0.0, 'labels_differ': 0, 'shared_eef_skipped': 0}
    for j in js:
        r = raw[int(j)]; c = cache[int(j)]
        d['obs'] = max(d['obs'], float((r['obs'] - c['obs']).abs().max()))
        if 'action' in r:
            d['action'] = max(d['action'], float((r['action'] - c['action']).abs().max()))
        else:
            if raw.rep == 'eef_img' and r['branch'] == 1:
                d['shared_eef_skipped'] += 1
            else:
                d['action'] = max(d['action'], float((r['action_mask'].int() - c['action_mask'].int()).abs().max()))
        d['final_pos'] = max(d['final_pos'], float((r['final_pos'] - c['final_pos']).abs().max()))
        d['final_mask'] = max(d['final_mask'], float((r['final_mask'] - c['final_mask']).abs().max()))
        if float(r['goal']) != float(c['goal']) or bool(r['coll_valid']) != bool(c['coll_valid']):
            d['labels_differ'] += 1
    return d


def folder_stats(root):
    nf = 0; size = 0
    for p in Path(root).glob('scene_*'):
        for f in p.rglob('*'):
            if f.is_file():
                nf += 1; size += f.stat().st_size
    return nf, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', default='Episodes/_p8_20scenes')
    ap.add_argument('--out', default='Docs/report_20261018/bench_cache_vs_raw.json')
    ap.add_argument('--reps', nargs='+', default=['full_img', 'full_num'])
    args = ap.parse_args()
    root = Path(args.episodes); cache = root / '_cache' / 'wm_128x25.h5'
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    splits = make_splits(cache)
    res = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'device': dev, 'gpu': torch.cuda.get_device_name(0) if dev == 'cuda' else '',
           'torch': torch.__version__, 'episodes_root': str(root), 'split': 'train (Task A+B)', 'batch': 16, 'res': RES, 'frames': FRAMES}
    nf, size = folder_stats(root)
    res['storage'] = {'raw_files': nf, 'raw_bytes': size, 'cache_files': 1, 'cache_bytes': cache.stat().st_size}
    print('storage', res['storage'], flush=True)
    t = time.perf_counter()
    with h5py.File(cache, 'r') as h:
        tr = WMDataset(cache, 'eef_num', split='train', splits=splits)
    pos_mean = np.mean(np.stack([tr[j]['final_pos'].numpy() for j in range(len(tr))]), 0).tolist()
    res['n_train'] = len(tr)
    for rep in args.reps:
        r = {}
        raw = RawDataset(root, cache, rep, 'train', splits)
        cch = WMDataset(cache, rep, task='both', split='train', splits=splits)
        r['sanity'] = sanity(raw, cch); print(rep, 'sanity', r['sanity'], flush=True)
        num_ms = S.num_stats(cch) if rep.endswith('_num') else None
        for name, ds, ws in [('raw', raw, [0, 4]), ('cache', cch, [0, 4]),
                             ('cache_in_memory', WMDataset(cache, rep, task='both', split='train', splits=splits, in_memory=True), [0])]:
            for w in ws:
                k = '%s_w%d' % (name, w)
                r['load_' + k] = time_loading(ds, w, dev)
                print(rep, 'load', k, ['%.1f' % v for v in r['load_' + k]], flush=True)
                secs, wait = time_train(ds, rep, w, dev, num_ms, pos_mean)
                r['train_' + k] = {'sec_per_epoch': secs, 'data_wait_sec': wait}
                print(rep, 'train', k, ['%.1f' % v for v in secs], 'wait', ['%.1f' % v for v in wait], flush=True)
        res[rep] = r
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1), encoding='utf-8', newline='\n')
    res['total_sec'] = time.perf_counter() - t
    Path(args.out).write_text(json.dumps(res, indent=1), encoding='utf-8', newline='\n')
    print('DONE', args.out, flush=True)


if __name__ == '__main__':
    main()
