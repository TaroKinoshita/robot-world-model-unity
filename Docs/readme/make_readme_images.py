'''
README 用の画像を、手元の episode データから作る（Episodes/ は git に入っていない）。
- in_scene / in_hand / in_full.png：Task 1 でモデルに入る 3 つの入力（ロボットを消したシーン、手先・腕全体の action 画像）
- task2_pair / task3_pair.png：matched pair の 2 つの腕の形（オレンジ = 肩 A、青 = 肩 B）をシーンに重ねたもの
GIF は Docs/thesis/video/*.mp4 から ffmpeg で作った（README の末尾に書いたコマンド）。
使い方: python Docs/readme/make_readme_images.py
'''
import glob
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / 'Episodes'; OUT = Path(__file__).resolve().parent
COL_A = np.array([0x34, 0x68, 0xEB], float)  # #EB6834（BGR）
COL_B = np.array([0xD6, 0x78, 0x2A], float)  # #2A78D6


def steps(cand, folder='fullbody_raster'):
    return sorted(glob.glob(str(cand / 'actions' / folder / 'step_*.png')))


def overlay(scene, ca, cb, idx, name):
    out = cv2.imread(str(scene / 'initial' / 'robot_free' / 'rgb.png')).astype(float)
    for c, col in [(ca, COL_A), (cb, COL_B)]:
        m = cv2.imread(steps(scene / 'candidates' / c)[idx], 0) > 127
        out[m] = out[m] * 0.35 + col * 0.65
    cv2.imwrite(str(OUT / name), cv2.resize(out.clip(0, 255).astype(np.uint8), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))


overlay(E / '_task2_v1' / 'scene_0002', 'c000', 'c001', 20, 'task2_pair.png')
overlay(E / '_task3_v1' / 'scene_00001', 'c000', 'c001', 20, 'task3_pair.png')

s = E / '_main_v1' / 'scene_0001'; c = s / 'candidates' / 'c000'
ims = {'in_scene.png': cv2.imread(str(s / 'initial' / 'robot_free' / 'rgb.png'))}
for name, folder in [('in_hand.png', 'eef_raster'), ('in_full.png', 'fullbody_raster')]:
    m = cv2.imread(steps(c, folder)[15], 0)
    im = np.full((256, 256, 3), (0x2b, 0x22, 0x1c), np.uint8); im[m > 127] = (0xF0, 0xF4, 0xF7); ims[name] = im
for name, im in ims.items():
    cv2.imwrite(str(OUT / name), cv2.resize(im, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
print('written', sorted(p.name for p in OUT.glob('*.png')))
