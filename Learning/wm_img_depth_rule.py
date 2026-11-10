import h5py, numpy as np, sys
sys.path.insert(0, r'C:\UnityProjects\PerceptionTestHDRP\Learning')
from wm_dataset import make_splits, load_excluded
def run(C, which):
    h = h5py.File(C, 'r'); E = h['episodes']; ids = [p.decode() for p in h['pairs/pair_id'][:]]
    if which == 'all':
        ex, _ = load_excluded(C); P = [i for i, p in enumerate(ids) if p not in ex]
    else:
        P = [ids.index(p) for p in make_splits(C)[which]]
    task = E['task'][:]; mp = E['mp'][:]; br = E['branch'][:]; pidx = E['pair_idx'][:]; aidx = E['action_idx'][:]; coll = E['coll'][:]
    spair = h['scenes/pair_idx'][:]; stask = h['scenes/task'][:]; SD = h['scenes/initial_depth_mm']
    acc = {0: [], 1: []}; sc = {0: [], 1: []}
    for p in P:
        sB = np.nonzero((spair == p) & (stask == 1))[0][0]; sA = np.nonzero((spair == p) & (stask == 0))[0][0]
        dB = SD[sB].astype(int); dA = SD[sA].astype(int); cyl = (np.abs(dB - dA) > 5) & (dB > 0)
        for m in np.unique(mp[(pidx == p) & (mp >= 0) & (task == 1)]):
            es = np.nonzero((pidx == p) & (mp == m) & (task == 1))[0]
            if len(es) != 2 or (coll[es] < 0).any() or coll[es[0]] == coll[es[1]]: continue
            g = {}
            for e in es:
                fm = h['actions/full_mask'][aidx[e]] >= 128; fd = h['actions/full_depth_mm'][aidx[e]].astype(int); ov = fm & cyl[None]
                gs = [np.abs(fd[t][ov[t]] - dB[ov[t]]).min() for t in range(25) if ov[t].any()]
                g[int(br[e])] = min(gs) if gs else 9999
            hit = 1 if coll[es[br[es] == 1][0]] == 1 else 0
            pred = 1 if g[1] < g[0] else (0 if g[0] < g[1] else 0.5)
            acc[hit].append(float(pred == hit) if pred != 0.5 else 0.5)
    d, u = np.mean(acc[0]), np.mean(acc[1])
    print(f'{which}: down-hit {len(acc[0])} acc {d:.3f} | up-hit {len(acc[1])} acc {u:.3f} | balanced {(d+u)/2:.3f}')
run(r'C:\UnityProjects\PerceptionTestHDRP\Episodes\_main_v1v2\_cache\wm_128x25.h5', 'test')
run(r'C:\UnityProjects\PerceptionTestHDRP\Episodes\_main_v1v2\_cache\wm_128x25.h5', 'val')
run(r'C:\UnityProjects\PerceptionTestHDRP\Episodes\_h5_v1\_cache\wm_128x25.h5', 'all')
