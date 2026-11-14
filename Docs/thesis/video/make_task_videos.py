'''
Task 2・3 の説明動画（英語、1920×1080、50 fps）。Unity で撮ったコマ（Captures/video/、TaskVideo.cs）から作る。
構成：タイトル（3 s）→ 左右に肩 A / B を並べた実行（実時間）→ 結果を重ねて止める（3 s）。
使い方: python Docs/thesis/video/make_task_videos.py
'''
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
CAP = ROOT / 'Captures/video'
OUT = Path(__file__).resolve().parent
FPS = 50
W, H = 1920, 1080
RED = (210, 10, 17); INK = (30, 26, 25); PAPER = (250, 247, 244); SOFT = (85, 80, 77); OK = (26, 115, 72)
F = lambda size, bold=False: ImageFont.truetype(r'C:\Windows\Fonts\segoeuib.ttf' if bold else r'C:\Windows\Fonts\segoeui.ttf', size)

TASKS = [
    dict(name='task2_apple_shelf', a='scene_0002_c000', b='scene_0002_c001',
         eyebrow='NEW TASK · TASK 2', title='Apple into a shelf basket',
         lines=['The robot carries a held apple into a basket inside a shelf.',
                'If any part of the arm touches the shelf, the arm stops (protective stop).'],
         res_a=('Arm touches the shelf → stops', RED), res_b=('Apple placed in the basket', OK)),
    dict(name='task3_forearm_push', a='scene_00001_c000', b='scene_00001_c001',
         eyebrow='NEW TASK · TASK 3', title='Forearm pushing',
         lines=['The gripper moves past a tall box. It never touches the box itself.',
                'Depending on the arm pose, the forearm can push the box over.'],
         res_a=('Forearm pushes the box over', RED), res_b=('Box untouched', OK)),
]


def frames(folder):
    fs = sorted((CAP / folder).glob('f_*.png'))
    return [Image.open(f).convert('RGB') for f in fs]


def center(d, y, text, font, fill):
    w = d.textlength(text, font=font); d.text(((W - w) / 2, y), text, font=font, fill=fill)


def title_card(t):
    im = Image.new('RGB', (W, H), RED); d = ImageDraw.Draw(im)
    d.text((140, 330), t['eyebrow'], font=F(36, True), fill=(255, 235, 232))
    d.text((140, 390), t['title'], font=F(104, True), fill=(255, 255, 255))
    for i, l in enumerate(t['lines']):
        d.text((140, 560 + i * 62), l, font=F(42), fill=(255, 240, 238))
    d.text((140, 860), 'Same gripper path, two arm poses (shoulder A / shoulder B).', font=F(38, True), fill=(255, 255, 255))
    return im


def compose(t, fa, fb, result):
    im = Image.new('RGB', (W, H), PAPER); d = ImageDraw.Draw(im)
    center(d, 40, 'Same gripper path, different arm pose', F(52, True), INK)
    center(d, 112, t['title'], F(34), SOFT)
    pw, ph = 900, 675; y0 = 190
    for k, (fr, lab, res) in enumerate([(fa, 'Shoulder pose A', t['res_a']), (fb, 'Shoulder pose B', t['res_b'])]):
        x0 = 40 + k * (pw + 40)
        im.paste(fr.resize((pw, ph), Image.LANCZOS), (x0, y0))
        d.rectangle([x0, y0 + ph, x0 + pw, y0 + ph + 70], fill=INK)
        tw = d.textlength(lab, font=F(38, True)); d.text((x0 + (pw - tw) / 2, y0 + ph + 10), lab, font=F(38, True), fill=(255, 255, 255))
        if result:
            txt, col = res
            d.rectangle([x0, y0 + ph + 80, x0 + pw, y0 + ph + 170], fill=col)
            tw = d.textlength(txt, font=F(44, True)); d.text((x0 + (pw - tw) / 2, y0 + ph + 98), txt, font=F(44, True), fill=(255, 255, 255))
    return im


def write(t):
    A, B = frames(t['a']), frames(t['b'])
    n = max(len(A), len(B))
    A += [A[-1]] * (n - len(A)); B += [B[-1]] * (n - len(B))
    out = OUT / f"{t['name']}.mp4"
    p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                          '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', str(out)], stdin=subprocess.PIPE)
    tc = np.asarray(title_card(t)).tobytes()
    for _ in range(FPS * 4): p.stdin.write(tc)
    for i in range(n): p.stdin.write(np.asarray(compose(t, A[i], B[i], False)).tobytes())
    last = np.asarray(compose(t, A[-1], B[-1], True)).tobytes()
    for _ in range(FPS * 4): p.stdin.write(last)
    p.stdin.close(); p.wait()
    print('written', out, f'{(n + FPS * 8) / FPS:.1f} s')


for t in TASKS:
    write(t)
