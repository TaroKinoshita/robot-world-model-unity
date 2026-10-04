# P4-A: 10/16 の最低ライン（2026-10-04 時点で達成）

## 達成したこと

| 最低ラインの項目 | 状態 |
|---|---|
| 選定ロボットで | ✅ UR5e＋Robotiq 2F-140 |
| Task A/B の対応する候補を複数生成 | ✅ 1 シーン（pair_0000）× Task A/B × 8 候補 = 16 episode。matched pair 1 組（mp0）を含む |
| 入力・ラベル・自動検証が通る | ✅ scene_0000（A）182/182、scene_0001（B）174/174、A/B ペアの比較 59/59 |
| ラベル分布を点検できる | ✅ `Assets/Python/summarize_labels.py` → `Episodes/label_summary.md` |

データ：`Episodes/scene_0000`（A）・`scene_0001`（B）、git `e0f71d3`（作成時に未保存の変更なし）。手順：commit → Play → build_metadata → validate_episode → check_task_pair → summarize_labels。

## 設定

- Task B の secondary：カプセル 直径 9.6 cm × 高さ 28 cm、位置 (0.20, 0.913, 0.10)（P3 で前腕がかすめるように決めた）
- 候補（押す距離はどれも 10 cm）

| 候補 | 押す方向 | 中身 |
|---|---|---|
| c000 | −90° | ゴール方向（−X） |
| c001 | 130° | |
| c002 | 180° | ロボット側（−Z） |
| c003 | −45° | |
| c004 | −135° | |
| c005 | 150° | |
| c006 | −90° | matched pair mp0：wrist-down（開始 → 押す位置の上も手先の直線） |
| c007 | −90° | matched pair mp0：wrist-up（c006 と手先経路が同じ） |

- 外した方向：0°・90°（押し始めが土台に近すぎて IK が解けない）、45°（1 step で関節が 2.5° 跳ぶ）

## ラベル分布

| Task | episode | goal True / False | 衝突 True / False / 対象なし | ターゲット転倒 | goal かつ衝突なし |
|---|---|---|---|---|---|
| A | 8 | 4 / 4 | 0 / 0 / 8 | 0 | 4 |
| B | 8 | 4 / 4 | 3 / 5 / 0 | 0 | 2 |

Task B で衝突した候補：

- c000・c006（−90°、wrist-down）：前腕が step 207（descend）で接触 → 円柱が倒れて、ターゲットにも接触（step 341）。goal は True のまま
- c003（−45°）：指が step 72（transfer）で接触 → 円柱が倒れる。goal は False
- c007（wrist-up）：接触なし → **mp0 は「手先同じ（差 最大 0.064 mm）・腕違い（前腕 平均 21.1 cm）・片方だけ衝突」**

## 点検で分かった偏り・注意（10/30 までに直す候補）

1. **ゴールが 1 か所に固定**（(−0.10, 0, 0.30)、半径 3 cm）なので、goal True になるのは −90° と −135° 系だけ。c004 は 27.7 mm / 23.1 mm で境界ぎりぎり。→ ゴールの正式定義と、候補の自動生成（押す方向・距離をゴールに合わせて振る）が必要
2. **衝突の種類が偏っている**：前腕 2 件（matched pair の wrist-down を含む）、手（指）1 件。上腕・肘の衝突はこの配置だと出ない
3. **円柱が倒れたあと、ロボットやターゲットに当たり直す**ので、初回接触の後のラベル（target→secondary など）は「倒れた結果」を含む
4. **シミュレーションのブレ**：円柱に触れていない候補でも、A と B で最終位置が最大 4.8 mm 違う（c004）。H1・H3 のマージン 3 mm より大きい。量産前に直したい
5. 入力表現は骨格線のまま（mask＋depth の本描画はまだ）。候補は 16 本未満（8 本）。どちらも 10/16 の最低ラインとしては OK

---

## 作り直し（2026-10-04、git `ab6e261`）

上のパイロットは古い押し方で作っていて、**全候補で箱が横倒し（傾き 90°）**になっていた（`Docs/P4B_determinism_push.md` §4・§5）。新しい押し方とブレの修正を入れて作り直した。古いデータは `Episodes/_archive/p4a_oldpush_20261004_113316/` に退避した。

- 手順：commit（作業フォルダはきれい）→ Play → build_metadata → validate_episode → check_task_pair → summarize_labels
- metadata：`git=ab6e261d`、dirty なし
- 検証：Task A 206/206、Task B 198/198、ペア 59/59

| Task | episode | goal True / False | 衝突 True / False / 対象なし | 倒れた | goal かつ衝突なし | 追従誤差 最大 |
|---|---|---|---|---|---|---|
| A | 8 | 3 / 5 | 0 / 0 / 8 | 0 | 3 | 0.3 mm |
| B | 8 | 3 / 5 | 1 / 7 / 0 | 0 | 3 | 0.3 mm |

| 候補 | 押す方向 | goal（距離） | B 衝突 |
|---|---|---|---|
| c000 | −90° | ✅ 12.0 mm | — |
| c001 | 130° | ❌ 167.3 mm | — |
| c002 | 180° | ❌ 138.0 mm | — |
| c003 | −45° | ❌ 67.1 mm | ✅ robot→円柱 step 177（transfer 中） |
| c004 | −135° | ❌ 71.9 mm | — |
| c005 | 150° | ❌ 174.3 mm | — |
| c006 | −90° mp0 wrist-down（flip） | ✅ 8.6 mm | — |
| c007 | −90° mp0 wrist-up（flip） | ✅ 8.6 mm | — |

- A/B：円柱に触れない 7 候補は最終位置の差が 0.000 mm。c003 は円柱に当たるけど、ターゲットの結果は同じ
- matched pair mp0：TCP の差 最大 0.071 mm、前腕の差 平均 22.6 cm、手首 1 の差 平均 20.0 cm。**今の円柱の位置だと両方とも当たらない**（「片方だけ衝突」になっていない）
- 偏り：衝突が 1 件だけ。goal True は −90° 系だけ（ゴールが 1 か所に固定のため）→ 候補の自動生成で、円柱・ゴールの配置と一緒に直す

---

## 作り直し 2（2026-10-04、git `1701d06`、自動生成 seed 1）← 今のパイロット

候補の自動生成（`Docs/P4B_candidate_generation.md`）で 16 候補にして作り直した。1 つ前のデータ（傾き 30°・固定 8 候補）は `Episodes/_archive/p4a_pitch30_fixed8_20261004_124357/` に退避した。

- metadata：`git=1701d06c`、dirty なし
- 検証：Task A 422/422、Task B 406/406、ペア 107/107
- シーン：
  - ターゲット (−0.117, 0.313)、向き 22.4°
  - ゴール：方向 −64.5°、距離 8.6 cm
  - 円柱 (0.189, 0.476)：mp0 の wrist-up に 5.2 mm 重なる位置

| Task | episode | goal True / False | 衝突 True / False / 対象なし | 倒れた | goal かつ衝突なし | 追従誤差 最大 |
|---|---|---|---|---|---|---|
| A | 16 | 5 / 11 | 0 / 0 / 16 | 0 | 5 | 0.2 mm |
| B | 16 | 5 / 11 | 1 / 15 / 0 | 0 | 5 | 0.2 mm |

- matched pair：
  - mp0：wrist-up だけ衝突（step 250）＝ 片方だけ衝突 ✅
  - mp1：どちらも衝突なし
  - TCP の差は最大 0.11 mm / 0.076 mm、前腕の差は約 20 cm
- goal True の 5 本は、全部ゴール方向の単独候補（6 本中 5 本）
- A/B：円柱に触れない 15 候補は最終位置の差が 0.000 mm
- テスト出力（`_ur5e_test/p4b_gen4`）と全部同じ値（Play をまたいでも同じ）
- 偏り：衝突が 1 件だけ

---

## 作り直し 3（2026-10-04、git `b3ce0da`）← 今のパイロット

本番の入力仕様（`Docs/P4B_input_spec.md`）と、半分開いた指の押し方で作り直した。3 シーン（seed 1〜3）。

- `pair_0000`（`scene_0000`/`0001`、seed 1）
- `pair_0001`（`scene_0002`/`0003`、seed 2）
- `pair_0002`（`scene_0004`/`0005`、seed 3）
- 1 つ前のデータ：`Episodes/_archive/p4b_closedgrip_3seeds_20261004_140158/`（閉じた指）
- 2 つ前のデータ：`Episodes/_archive/p4a_auto_skeleton_20261004_132712/`（骨格線）
- 点検の結果は `Docs/P4B_input_spec.md` §8

---

## 作り直し 4（2026-10-05、git `9c76bc8`）← 今のパイロット

TGS ソルバーと円柱の方向の候補を入れて作り直した（`Docs/P4B_input_spec.md` §9–10）。seed 1〜3、`pair_0000`〜`pair_0002`。

- 1 つ前（TGS、ほかの候補への当たりの制限なし。seed 3 で 15/16 が円柱に当たった）：`Episodes/_archive/p4b_tgs_v2_*`
- 2 つ前（TGS、円柱が遠くて円柱の方向の候補が届かない）：`Episodes/_archive/p4b_tgs_far_cyl_*`
- 3 つ前（PGS ソルバー、作り直し 3）：`Episodes/_archive/p4b_pgs_3seeds_*`
- 検証：全部 PASS。goal 18/48、衝突 16/48（Task B）、倒れた 0、不安定 0

---

## 作り直し 5（2026-10-05、git `f46109c`）← 今のパイロット

P5（ドライブ値をそろえた、ロボット↔机の判定、SOLO の定義ファイル）を入れて作り直した。詳細は `Docs/P5_small_fixes.md`。1 つ前のデータは `Episodes/_archive/p4b_v4_preP5_*`。
