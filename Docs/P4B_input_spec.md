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

## 7. 押し方の再調整（2026-10-04、git `7db0309`・`b3ce0da`）

- 閉じた指（0.6 rad）だと接触の幅が狭くて、向きを変えた箱が押す方向から 35〜40° それた
  - seed 2・3 では、ゴール方向に押した候補 6 本のうちゴールに入ったのが 0〜1 本
- **指を半分開く（0.42 rad、パッドの間隔は約 6 cm）＋ 傾き 0°** にした
  - 2 点で押すことになるので、箱が押す方向にまっすぐ進む
  - 半分開いた指で傾き 15° だと、8 本中 2 本が倒れた
- 腕の形が変わって、高さ 28 cm の円柱では「片方だけ当たる」場所が見つからなくなった
  - 円柱の高さもシーンごとに決めるようにした（24〜40 cm を低い方から試す）
  - 3 シーンとも 32 cm になった
- `validate_episode.py`：「TCP の近くにグリッパー」の許容を、パッドの間隔に合わせて広げた（指が開くと TCP はパッドの間の何も無い所に来るため）
- `validate_episode.py`：最終マスクの IoU は、円柱に隠れうる範囲を除いて比べる（Task B で円柱がターゲットを隠すことがある）

## 8. 量産と点検（3 シーン × 16 候補 × Task A/B = 96 episode、git `b3ce0da`）

- 検証：6 scene すべて PASS（A 564/564、B 548/548）、ペア 3 組とも 107/107
- 欠損 0、追従誤差（TCP）最大 0.3 mm
- 生成にかかった試行：どの seed も 1 回目で成功
  - 外した候補：`near_base` 15、`joint_step` 8、`ik_failed` 3

| 項目 | 結果 | 評価 |
|---|---|---|
| goal True | 14 / 48（29%） | ゴール方向 10/18、matched pair 4/12、ランダム 0/18 |
| 倒れた | 3 / 48 | matched pair の候補だけ（seed 1 の mp1 wrist-up、seed 3 の mp0 の 2 本） |
| 衝突（Task B） | 3 / 48 | 3 シーンとも、mp0 の片方だけが当たった（「片方だけ衝突」✅ × 3）。それ以外の衝突は 0 |
| matched pair | TCP の差 ≤ 0.12 mm、前腕の差 20〜24 cm | ✅ |

### 分かった偏り・問題

1. **衝突が少ない**：mp0 の片方以外は衝突 0。H2・H4 には足りない
   - 単独候補の一部を、円柱の方向へ振る必要がある
2. **Task A/B で、円柱に触れていない候補の結果がずれることがある**
   - 45 本中 2 本で、最終位置が 12.6 mm と 1.1 mm ずれた（ほかは 0.000 mm）
   - その候補では、円柱はロボットにもターゲットにも触れていない（`contacts.csv` で確認）
   - ずれ始めは、押している途中のターゲットの速度
   - 円柱がシーンにあるだけで、PhysX の計算の順番が変わるのかもしれない（Enhanced Determinism は on）
   - 次の候補：broadphase の種類を変える、円柱を別の物理シーンの判定から外す、など
3. **ランダム方向の単独候補は goal True が 0**（ゴールから遠いので当然）。goal の True/False は、全体で 29% / 71%
