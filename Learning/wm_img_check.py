import h5py, numpy as np, json, sys
sys.path.insert(0, r'C:\UnityProjects\PerceptionTestHDRP\Learning')
from wm_dataset import make_splits
C = r'C:\UnityProjects\PerceptionTestHDRP\Episodes\_main_v1v2\_cache\wm_128x25.h5'
h = h5py.File(C, 'r'); K = h.attrs['K']; T = h.attrs['T_world_to_cam_cv']
def proj(p):
    q = (T @ np.c_[p, np.ones(len(p))].T).T
    return np.c_[K[0,0]*q[:,0]/q[:,2]+K[0,2], K[1,1]*q[:,1]/q[:,2]+K[1,2]], q[:,2]
rng = np.random.default_rng(0)
NA = h['actions/num_len'].shape[0]
ais = rng.choice(NA, 200, replace=False)
# 1. TCP projection vs eef mask / depth, and keypoints vs full mask / depth
d_eef, z_eef, kp_hit, kp_z = [], [], [], []
for ai in ais:
    rs = h['actions/raster_steps'][ai]; e = h['actions/eef_num_raw'][ai][rs, :3]; f = h['actions/full_num'][ai][rs, 6:33].reshape(-1, 9, 3)
    em = h['actions/eef_mask'][ai] >= 128; ed = h['actions/eef_depth_mm'][ai]; fm = h['actions/full_mask'][ai] >= 128; fd = h['actions/full_depth_mm'][ai]
    uv, z = proj(e)
    for t in range(25):
        ys, xs = np.nonzero(em[t])
        if len(xs) == 0: d_eef.append(np.inf); continue
        dd = np.hypot(xs - uv[t,0], ys - uv[t,1]); j = dd.argmin(); d_eef.append(dd.min()); z_eef.append(abs(ed[t, ys[j], xs[j]] / 1000 - z[t]))
        uvk, zk = proj(f[t])
        for k in range(9):
            u, v = int(round(uvk[k,0])), int(round(uvk[k,1]))
            if 0 <= u < 128 and 0 <= v < 128:
                w = fm[t, max(0,v-2):v+3, max(0,u-2):u+3]; kp_hit.append(w.any())
d_eef = np.array(d_eef); z_eef = np.array(z_eef)
print(f'[1] EEF: TCP projection to nearest gripper pixel: median {np.median(d_eef):.2f}px, p95 {np.percentile(d_eef,95):.2f}px, max {d_eef.max():.2f}; empty frames {np.isinf(d_eef).mean():.3f}')
print(f'    depth vs TCP cam z: median {1000*np.median(z_eef):.1f} mm, p95 {1000*np.percentile(z_eef,95):.1f} mm')
print(f'    full-body keypoints within 2px of full mask: {np.mean(kp_hit):.3f} (n={len(kp_hit)})')
# 2. pixel sizes
fm_px = np.array([(h['actions/full_mask'][ai] >= 128).sum((1,2)).mean() for ai in ais[:50]]); em_px = np.array([(h['actions/eef_mask'][ai] >= 128).sum((1,2)).mean() for ai in ais[:50]])
print(f'[2] mean pixels per frame: full {fm_px.mean():.0f}, eef {em_px.mean():.0f} (of 16384)')
mmpx = 1.2 / K[0,0] * 1000
print(f'    1 px at 1.2 m depth = {mmpx:.1f} mm')
# 3. matched pairs: image difference between the two branches; overlap with the cylinder
E = h['episodes']; task = E['task'][:]; mp = E['mp'][:]; br = E['branch'][:]; pidx = E['pair_idx'][:]; aidx = E['action_idx'][:]; sidx = E['scene_idx'][:]; coll = E['coll'][:]
sec = h['scenes/secondary_pos'][:]; sech = h['scenes/secondary_height'][:]
res = []
for p in rng.choice(np.unique(pidx), 60, replace=False):
    for m in np.unique(mp[(pidx == p) & (mp >= 0) & (task == 1)]):
        es = np.nonzero((pidx == p) & (mp == m) & (task == 1))[0]
        if len(es) != 2: continue
        up = es[br[es] == 1][0]; dn = es[br[es] == 0][0]
        fu = h['actions/full_mask'][aidx[up]] >= 128; fdn = h['actions/full_mask'][aidx[dn]] >= 128
        eu = h['actions/eef_mask'][aidx[up]]; edn = h['actions/eef_mask'][aidx[dn]]
        diff = (fu ^ fdn).sum((1, 2))
        # cylinder footprint in the image: scene depth differs between task A and task B scenes of this pair
        sB = sidx[up]; sA = np.nonzero((h['scenes/pair_idx'][:] == p) & (h['scenes/task'][:] == 0))[0][0]
        cyl = np.abs(h['scenes/initial_depth_mm'][sB].astype(int) - h['scenes/initial_depth_mm'][sA].astype(int)) > 5
        ov_u = (fu & cyl).sum((1, 2)).max(); ov_d = (fdn & cyl).sum((1, 2)).max()
        res.append(dict(diff_max=int(diff.max()), diff_mean=float(diff.mean()), eef_same=bool((eu == edn).all()), cyl_px=int(cyl.sum()),
                        ov_up=int(ov_u), ov_dn=int(ov_d), c_up=int(coll[up]), c_dn=int(coll[dn])))
r = res
print(f'[3] matched pairs (task B): n={len(r)}; EEF masks identical in {np.mean([x["eef_same"] for x in r]):.2f}')
print(f'    full-mask pixel difference between branches: max per pair median {np.median([x["diff_max"] for x in r]):.0f}px, mean-over-time median {np.median([x["diff_mean"] for x in r]):.0f}px')
print(f'    cylinder footprint median {np.median([x["cyl_px"] for x in r]):.0f}px')
one = [x for x in r if x['c_up'] != x['c_dn']]
hit_ov = [ (x['ov_up'] if x['c_up'] else x['ov_dn']) for x in one]; miss_ov = [ (x['ov_dn'] if x['c_up'] else x['ov_up']) for x in one]
print(f'    pairs where exactly one collides: {len(one)}; max image overlap arm∩cylinder: colliding side median {np.median(hit_ov):.0f}px, non-colliding side median {np.median(miss_ov):.0f}px')
print(f'    overlap > 0: colliding {np.mean(np.array(hit_ov)>0):.2f}, non-colliding {np.mean(np.array(miss_ov)>0):.2f}')
# same with depth-aware overlap: arm depth within ±(radius) of cylinder depth
