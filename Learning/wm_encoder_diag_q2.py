"""
P15 診断 Q2: 数値の action に、結果を当てるのに必要な情報は入っているか（選ぶのには使わない）

数値 action（手先 eef_num、50 Hz 全 step）から手で特徴を取り出し、箱の初期位置・向き・ゴールと一緒に
小さいモデル（勾配ブースティング）に入れて、最終位置と goal を当てる。
- 当たる → 情報は数値に入っている。弱いのはエンコーダ
- 当たらない → 数値から取り出すこと自体が難しい
特徴: 押している区間（手先の高さが最低から 3 mm 以内）の始点・終点の xz、向き(sin, cos)、距離、そのときの手先の yaw
使い方: python Learning/wm_encoder_diag_q2.py [--out Docs/encoder_fix/results/q2.json]
"""
import argparse, json
from pathlib import Path
import numpy as np, h5py
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "Episodes/_p8_20scenes/_cache/wm_128x25.h5"


def yaw_from_quat(q):
    x, y, z, w = q
    # Unity: Y 上。Y 軸まわりの回転角
    return np.arctan2(2 * (w * y + x * z), 1 - 2 * (y * y + x * x))


def features(h):
    A = h["actions/eef_num_raw"][:]; L = h["actions/num_len"][:]
    F = []
    for a, n in zip(A, L):
        a = a[:n]; y = a[:, 1]
        low = np.nonzero(y < y.min() + 0.003)[0]
        s, e = a[low[0]], a[low[-1]]
        d = e[[0, 2]] - s[[0, 2]]; ln = np.linalg.norm(d) + 1e-9
        yw = yaw_from_quat(a[low[len(low) // 2], 3:7])
        F.append([s[0], s[2], e[0], e[2], d[0] / ln, d[1] / ln, ln, np.sin(2 * yw), np.cos(2 * yw)])
    return np.array(F, np.float32)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(ROOT / "Docs/encoder_fix/results/q2.json"))
    args = ap.parse_args()
    h = h5py.File(CACHE, "r"); E = h["episodes"]
    act = features(h)
    sc = E["scene_idx"][:]; ai = E["action_idx"][:]
    cp = h["scenes/target_init_pos"][:]; cy = np.deg2rad(h["scenes/target_init_yaw_deg"][:]); gz = h["scenes/goal_xz"][:]
    sc_feat = np.c_[cp[:, [0, 2]], np.sin(4 * cy), np.cos(4 * cy), gz]
    X_act = act[ai]; X_sc = sc_feat[sc]
    pair = E["pair_idx"][:]; task = E["task"][:]
    ids = [p.decode() for p in h["pairs/pair_id"][:]]
    sp = json.loads((CACHE.parent / "wm_128x25_splits.json").read_text())
    tr = np.isin(pair, [ids.index(p) for p in sp["train"]]); va = np.isin(pair, [ids.index(p) for p in sp["val"]])
    yA = task == 0                                    # Task A だけ（衝突なし。位置と goal の情報だけを見る）
    fp = E["final_pos"][:][:, [0, 2]]; goal = E["goal"][:].astype(int)
    res = {}
    for name, X in [("action_only", X_act), ("scene_only", X_sc), ("action+scene", np.c_[X_act, X_sc])]:
        r = {}
        for split, m in [("train", tr & yA), ("val", va & yA)]:
            r[split] = {}
        mt, mv = tr & yA, va & yA
        err_tr, err_va = [], []
        P_tr = np.zeros((mt.sum(), 2)); P_va = np.zeros((mv.sum(), 2))
        for k in range(2):
            g = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0).fit(X[mt], fp[mt, k])
            P_tr[:, k] = g.predict(X[mt]); P_va[:, k] = g.predict(X[mv])
        c = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0).fit(X[mt], goal[mt])
        r["train"] = {"pos_err_mm": float(1000 * np.linalg.norm(P_tr - fp[mt], axis=1).mean()),
                      "goal_auprc": float(average_precision_score(goal[mt], c.predict_proba(X[mt])[:, 1]))}
        r["val"] = {"pos_err_mm": float(1000 * np.linalg.norm(P_va - fp[mv], axis=1).mean()),
                    "goal_auprc": float(average_precision_score(goal[mv], c.predict_proba(X[mv])[:, 1]))}
        res[name] = r
    # 参考: 何もしない（箱が動かない）とした場合
    ip = cp[sc][:, [0, 2]]
    res["no_motion_baseline"] = {s: {"pos_err_mm": float(1000 * np.linalg.norm(ip[m & yA] - fp[m & yA], axis=1).mean())} for s, m in [("train", tr), ("val", va)]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(k, json.dumps(v))


if __name__ == "__main__":
    main()
