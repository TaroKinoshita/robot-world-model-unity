# P4-B: 本番の入力仕様（mask＋depth の本描画・向き・contact heatmap・サイズ）

- 日付：2026-10-04
- コード：
  - `Assets/Scripts/RobotRasterizer.cs`（新規）
  - `ActionRepresentations.cs`（`actions_meta` v2）
  - `EpisodeRecorder.cs`（contact heatmap）
  - `validate_episode.py` / `check_task_pair.py` / `build_metadata.py`

## 1. 入力・ターゲットの一覧（1 episode あたり）

| 種類 | ファイル | 中身 |
|---|---|---|
| EEF-Numeric | `actions/eef_numeric.csv` | 毎 step の TCP 位置＋tool の回転（quaternion） |
| Full-Numeric | `actions/fullbody_numeric.csv` | 毎 step の関節角 6 ＋ キーポイント 11 個の位置 ＋ **向き（追加）** |
| EEF-Image | `actions/eef_raster/`（マスク）・`actions/eef_depth/`（depth） | グリッパーのメッシュだけを描いたもの、25 枚 |
| Full-Image | `actions/fullbody_raster/`・`actions/fullbody_depth/` | ロボット全体のメッシュを描いたもの、25 枚 |
| contact heatmap（予測ターゲット） | `final/contact_heatmap_target.png`・`final/contact_heatmap_secondary.png`・`final/contact_heatmaps.json` | 触れた接触点をカメラに投影したもの |

## 2. 画像表現：骨格線 → メッシュの本描画

- **描き方**：計画した関節角で、ロボットの見た目のメッシュ（15.2 万三角形）を CPU で z-buffer 描画する
  - GPU（HDRP）を使わないので、計画の段階で描ける
  - 結果はカメラの幾何だけで決まる
- **カメラ**：初期画像と同じ（256×256、OpenCV の画素の規約）
- **EEF**：グリッパーのメッシュだけ（腕の chain の外のリンク、約 3.0 万三角形）
- **Full**：ロボット全体
- 机や物体は描かない（物体に隠される処理もない）
- **マスク**：8bit PNG（255 = ロボット）
- **depth**：16bit PNG、カメラ座標の z を mm で（0 = 何も無い）
  - P1.5 で EXR の読み込みが一番のボトルネックだったので PNG にした
- 付随する変更：UR5e の見た目の `.dae` 7 個を Read/Write 可にした（`.meta` が変わる）

### depth の検証（カメラの幾何）

`validate_episode.py` に追加した検証：

| 検証 | 結果（`_ur5e_test/p4b_input1`） |
|---|---|
| マスク = (depth > 0)、枚数・大きさ | ✅ |
| グリッパーのマスク ⊂ 全身のマスク、全身の depth ≤ グリッパーの depth | ✅ はみ出し 0 px |
| TCP の投影から 5 px 以内にグリッパー、その depth と TCP のカメラ z の差 ≤ 3 cm | ✅ 最大 2.95 px / 6.8 mm |
| **step 0 の全身の描画 vs Perception が撮った初期画像（ロボットあり）**：マスク IoU ≥ 0.9、depth の差の中央値 ≤ 5 mm | ✅ **IoU 0.997、中央値 0.25 mm（90% 点 0.45 mm）** |

- 最後の検証は、開始姿勢が初期姿勢と同じ候補だけで行う（wrist-up は開始姿勢が違うので対象外）
- Perception の depth の EXR は、R チャンネル（cv2 で読むと index 2）にカメラ z が m で入っている

## 3. Full-Numeric に向きを追加

- キーポイント（chain のリンク原点 ＋ TCP）ごとに `_rx, _ry, _rz, _rw`（world の quaternion）
- 向きはリンクの回転、TCP は tool の回転

## 4. contact heatmap

- `executed/contacts.csv` の「触れた」接触（隙間 ≤ 1 mm）の行について、接触点（行の平均）をカメラに投影する
- ガウス（σ = 2 px、3σ まで）で足し合わせて、最大 = 255 にする（接触が無ければ全部 0）
  - **target**：ロボット → ターゲット
  - **secondary**：ロボット → 円柱、ターゲット ↔ 円柱（Task A では空）
- 検証：Python で `contacts.csv` と `camera.json` から作り直して一致（最大差 0）、「空でない ⇔ ラベルが True」

## 5. 入力のサイズ・枚数（P1.5 の結果を踏まえて）

| 項目 | 決めた値 | 理由 |
|---|---|---|
| 保存する解像度 | 256×256 | カメラと同じ。学習で 128 に縮めるのは後でできる（逆はできない） |
| 枚数 | **25 枚**（最初と最後を含めて等間隔、約 0.4 s おき） | P1.5 で、256 px・25 枚が 8 GB の GPU で batch 8 まで回った。99 枚は 8 GB だと OOM |
| depth の形式 | 16bit PNG（mm） | 読み込み：mask＋depth × 2 表現 × 25 枚 = 100 ファイルで 23 ms（Windows、cv2）。前の EXR は depth 99 枚で 240 ms |
| 1 episode の大きさ | 約 1.7 MB | 1 万 episode でも約 17 GB |

- 学習の初期設定：128 px・25 枚（P1.5 の「軽量」）。比較が成立しなければ 256 px にする

## 6. 検証の結果（`_ur5e_test/p4b_input1`、seed 1）

- Task A 564/564、Task B 548/548、ペア 107/107
- `check_task_pair.py` に depth の画像も加えた（A/B でバイト単位で同じ）
