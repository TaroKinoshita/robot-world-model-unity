# 進捗報告書・引き継ぎ（2026-11-22）

*このセッション（2026-11-08〜11-22 ごろ）でやったことを、すべてまとめた。新しいセッションでは、まずこれを読む。前の詳しい経緯は `Docs/THESIS_HANDOFF.md`（10/12 まで）にある。*

---

## 0. 30 秒でわかる現状

- **テーマ：** world model に、ロボットの動き（action）をどう教えるか。範囲（手先だけ / 腕全体）× 形式（数値 / 画像）の 4 モデルを比べる。
- **実験は全部終わった。** タスクは 3 つで、どれも test を 1 回だけ使って判定まで済んでいる。
  - **Task 1：** 箱を押す。横に円柱があるかもしれない（元からの本実験。前回の打ち合わせのスライドの Task A / B）。
  - **Task 2：** 持ったりんごを、棚の区画の中のかごに入れる。腕が棚に触れたら止まる（保護停止）。新しいタスク。
  - **Task 3：** 前腕で背の高い箱を押す（腕が道具になる）。追加のタスクで、必須ではない。
- **結論：** 腕全体を見せる効果は、結果が腕にどれだけ依存するかで大きくなる。
- **論文：** 全章（アブストラクト・序論・関連研究・方法・結果・考察）の下書きと、図 2 枚、説明動画 2 本ができている（`Docs/thesis/`）。
- **打ち合わせの準備：** 4 枚のスライド「Thesis Progress — Week 12」（PDF は OneDrive の `MTG/Week12/`）と、日本語の控えができている。
- **git：** ブランチ `todo6-ur5e`。`main` も同じ位置（`98bbd36`）まで進めて、両方 push 済み。GitHub のトップに README が出る。
- **提出：** 2027-01-15。

---

## 1. 研究の前提（何も知らない人向け）

- **world model：** ロボットが動く前に、「その動きで周りの物がどうなるか」を予想する AI。
- **比べるもの：**
  - 手先だけ：グリッパーの通り道。
  - 腕全体：全部の関節とリンク。
  - それぞれを、数値（座標の列）と画像（カメラから見たロボットの絵）の 2 形式で渡す。
  - 4 モデルは、action を読む部分（エンコーダ）以外は全部同じ。
- **matched pair：** 手先の通り道は同じで（差 < 0.15 mm）、腕の形（IK の解）だけ違う 2 本。手先だけのモデルには同じ入力に見えるので、区別できない。
- **仮説：**
  - H1：腕が関係ないなら手先で十分。
  - H2：衝突の予想は腕全体が良い。
  - H3：画像の方が mask の IoU が良い。
  - H4：腕全体なら、選ぶ軌道が安全になる。
  - H5：腕全体なら、matched pair を区別できる（balanced accuracy で判定）。
- **公正さ：** 仮説と判定のルールは、データを作る前に `Docs/encoder_fix/protocol.md` に書いてコミットした。test は各タスク 1 回だけ使った。

---

## 2. このセッションでやったこと（時系列）

### 2.1 Task 1 の仕上げ（11/8〜11/11）
- RunPod の RTX 4090 で、重みを保存しながら学習し直した。seed を 3 → 5 に増やした（追記 7）。
- H5 の判定ルールを balanced accuracy に決めた（追記 8、test の前）。
- **test を 1 回だけ使った。** H1・H2・H5 は成立、H3・H4 は不成立。数値でも画像でも同じ判定。
  - H5：腕全体・数値 0.842 [0.785, 0.893]、腕全体・画像 0.719 [0.638, 0.799]。
- 追記 9（H3 を test で判定）・追記 10（接触位置の誤差、33〜39 mm）。
- **画像の入力の診断（探索的）：**
  - カメラは Unity と一致。位置のずれは中央値 1.45 px。
  - 学習なしのルールで H5 0.887。→ 情報は画像にある。
- **early fusion（追記 11、探索的、test は使わない）：** 初期画像と action 画像を 1 つのエンコーダで読む形にすると、H5 が +0.08（H5 セット 0.603 → 0.687、val 0.740 → 0.815）。
- 先行研究との比較を書いた（Ganapathi 2022 の表 I の数字は原典で確認した）。
- 進捗報告（英語）`Docs/report_20261108/progress_en.md`。画像の作り直しを断るための控え `meeting_prep_private.md`。
- ノートの RAM が足りなくなったので、学習を RunPod に移した（SAM3 と同時に動かしていたため）。

### 2.2 データの整理
- 要らないデータ（グループ A）を消した。グループ B は残した。

### 2.3 Task 2：りんご・棚・かご（11/12〜11/13）
- **何を足すかを、先行研究から決めた。**
  - Fishman et al. 2022：棚の区画は、机の上より難しい環境。
  - Ganapathi 2022：タスクによって、効く動作の表現が違う。
- **設計：** `Docs/task2/design_memo.md`（§0〜§15）
  - りんごは FixedJoint でグリッパーに固定（掴む物理は扱わない）。
  - 保護停止：腕かグリッパーが棚・かごの縁に触れたら止まる。
  - matched pair は**肩の左右**の解。手首の反転だと、片方（wrist-up）しか当たらなかった。肩の左右なら、計算で 180 条件中 A だけ 52 / B だけ 52 / 両方 36 / どちらも 40。
  - 手先を横向きにする案は捨てた（96 条件中 86 でどちらも当たらない）。
  - 見た目は CC0 のモデル（Poly Haven：りんご・かご・オークの木目）。当たり判定は単純な形のまま。
  - カメラは Task 1 と同じ向き・画角で 0.2 m 後ろ。
- **試作で直したこと：**
  - 静的バッチをオフにした（棚の見た目が動かず、depth も作れなかった）。
  - FBX の Read/Write をオンにした。
  - quaternion の符号のそろえ方を直した（手先が真下で w ≈ 0 になるため。Task 2 用のキャッシュだけ）。
  - 区画の幅の範囲を 0.28〜0.46 m に広げた。
- **本番（追記 12）：** 300 シーン、matched pair 1131 組。両方通る 207 / A だけ止まる 60 / B だけ止まる 188 / 両方止まる 676。
  - 学習はノートの RTX 4060 で約 50 分。
- **test の結果：**
  - H5：数値 0.900 [0.750, 0.986]、画像 0.623 [0.541, 0.707]。両方成立。
  - H2：数値だけ成立（+0.034）。画像は −0.009。
  - H1：判定つかず。
- **探索的な分析：**
  - 区画の狭さ別：はっきりした傾向はない（狭いと衝突率が約 90% になり、AUPRC が上限に張り付く）。
  - 画像モデルの偏り：AUROC は 0.86 だが、B 側に +0.09 ずれている。しきい値を補正すると 0.62 → 0.79。
- 結果：`Docs/task2/results.md`、`Docs/encoder_fix/results_t2/`

### 2.4 Task 3：前腕で押す（11/13〜11/14）
- **理由：** Task 1 と 2 は、どちらも腕が意図せず当たるタスクで、H1 が当然の結果になっていた。腕で物を動かす場面を確かめるために足した。
  - 根拠：Ganapathi 2022 の肘のタスク、Salisbury 1987 / Eberman 1989 の whole-arm manipulation、Cosgun et al. 2019 の前腕で押すタスク。
- **設計：** `Docs/task3/design_memo.md`
  - 背の高い箱（高さ 28〜42 cm）を、手先の経路の横 0.10〜0.20 m に置く。
  - グリッパーが箱から 10 cm 以内に入る配置は使わない。
  - 腕は止めない。箱は押されて倒れる。
- **試作で直したこと：**
  - 箱を、手首が通る範囲に置くようにした。
  - 机を奥へ延ばした（倒れた箱が机から落ちていた）。
  - シーンに保存されたコンポーネントの値が古いままだったので、付け直した。
- **本番（追記 13）：** 300 シーン、765 組。A だけ触れる 157 / B だけ 149 / どちらも触れない 459。グリッパーの接触は 0。
- **test の結果：**
  - **H1′（物体の位置、腕全体の方が良い）：成立。** 数値 +10.5 mm [+3.3, +18.6]、画像 +7.9 mm [+1.6, +15.7]。Task 1 と向きが逆。
  - H5：1.000（ただし、箱が左右どちらにあるかで決まるので、易しい問題）。
  - H2：+0.446 / +0.426。
- 結果：`Docs/task3/results.md`、`Docs/encoder_fix/results_t3/`

### 2.5 論文の下書き（11/13〜11/15）
- `Docs/thesis/`：
  - `00_abstract.md`、`01_introduction.md`、`02_related_work.md`、`03_methods.md`、`04_results.md`、`05_discussion.md`
- 図：
  - `fig/fig_gaps.png`：3 つのタスクの「腕全体 − 手先」の差と、H5。
  - `fig/fig_h5_bias.png`：H5 の枝ごとの正解率と、偏りの補正。
  - 作るスクリプト：`Learning/thesis_figures.py`
- 参考文献を arXiv で確認した：Cosgun 2019、iMaC（Wu ほか 2026）、EA-WM（Yang ほか 2026）、OSCAR（Wu & Gao 2026、今の版のタイトル）。

### 2.6 打ち合わせの準備（11/15〜11/19）
- **スライド 4 枚**（Artifact：https://claude.ai/artifact/3zxaUcn9ZpnHzsgeDUmb9b、private。PDF は OneDrive の `MTG/Week12/Thesis Progress — Week 12.pdf`）
  1. Overview：前回からの 3 つ（画像の確認・先行研究・新しいタスク）。
  2. Task 1 の画像の入力：確かめた 3 点、H3・H4 が成り立たなかった理由、+0.08、作り直さない理由（黒い帯）。
  3. Task 1 と先行研究：Ganapathi・Khansari・Wang 他・Yu の 4 行。
  4. 新しいタスク：Task 2 がメイン（動画つき）、Task 3 は追加として小さく。
- **説明動画 2 本**（英語、約 14 秒）：`Docs/thesis/video/task2_apple_shelf.mp4`、`task3_forearm_push.mp4`
  - Unity で、肩 A と肩 B を左右に並べて撮った。
  - 撮る仕組み：`Assets/Task2Pilot/TaskVideo.cs`。recorder の `videoDir` が空ならオフ。
  - 編集：`Docs/thesis/video/make_task_videos.py`
- **控え（日本語）：**
  - `Docs/report_20261108/meeting_brief_ja.md`：話す順番。
  - `Docs/report_20261108/meeting_details_ja.md`：スライドの範囲の詳細と数字。
  - スライドと動画の説明のしかた（前提から）は、チャットで説明した。必要なら、この報告書の §1 と meeting_details を使う。

### 2.7 README と GitHub（11/19〜11/22）
- `README.md`（英語）：問い、4 表現、matched pair、3 タスク（GIF）、結果、先行研究、フォルダ構成、再現の手順。
- 画像：`Docs/readme/`。作り直すスクリプトは `make_readme_images.py`。
- `todo6-ur5e` を push した。`main` を fast-forward で `98bbd36` まで進めた（前は `63447a4`）。

---

## 3. 3 つのタスクの結果のまとめ

| | Task 1（押す＋円柱） | Task 2（棚とりんご） | Task 3（前腕で押す） |
|---|---|---|---|
| 腕と物体の接触 | 意図しない | 意図しない（止まる） | 意図する |
| 物体の位置：腕全体の方が良いか | いいえ（手先が約 4 mm 良い） | 差なし | **はい（8〜11 mm）** |
| 衝突・接触の AUPRC（腕全体 − 手先） | +0.028 / +0.050 | +0.034 / −0.009 | +0.446 / +0.426 |
| H5 balanced（数値 / 画像） | 0.842 / 0.719 | 0.900 / 0.623 | 1.000 / 1.000（易しい） |
| test のシーンペア | 76 | 45 | 45 |

**画像が弱い理由（探索的）：**
- 情報はある（学習なしのルールで約 0.9）。
- 原因は、混ぜ方（late fusion）と、枝の偏り。

---

## 4. 決めたこと（論文に書く）

| 決めたこと | 理由 | 記録 |
|---|---|---|
| seed を 3 → 5 | H5 の seed のばらつきが大きかった | 追記 7 |
| H5 を balanced accuracy で判定（AUROC は副） | 手先は構造上 0.5。手首の枝だけでもかなり当たる。val を見たあと、test の前に自分で決めた | 追記 8 |
| 画像モデルは作り直さない | 問いは範囲であって、形式ではない。効果は両方の形式で出ている。test はもう使った | 追記 11、meeting_prep_private |
| Phase 3（動画予測）はやめて、新しいタスクにした | 「別のタスクで確かめて」の方が主張に効く | progress_en |
| Task 2 は保護停止、肩の左右のペア | 手首の反転では片側しか当たらない | 追記 12 |
| Task 3 を追加（意図した接触） | H1 が当然の結果になっていたため | 追記 13 |
| H3・H4 は Task 2・3 では判定しない | 候補が 4〜8 本で、選ぶ問題になっていない | 追記 12・13 |

---

## 5. 残っていること

### 次にやること（論文）
1. 下書きを仕上げる。数値を元のファイルと照合する。図を足す（予測の例、seed ごとのばらつき）。
2. 大学のテンプレート（LaTeX か Word）に移す。
3. 指導教員に全体の下書きを見せる時期を決める。
4. 打ち合わせ（Week 12）の結果を反映する。

### 気をつけること
- Task 3 の H5 = 1.0 は易しい問題だから、と論文で必ず断る。
- Task 2・3 は Task 1 の評価のあとで足した。事前登録はしたが、タスクを選んだのは自分なので、限界として書く（考察 §5.5）。
- Task 2・3 のキャッシュは、各シーンを task 0 / 1 の 2 行として書いている（学習で同じ episode が 2 回出る）。論文に書く。
- スライドの Artifact は private。Rekha に見せるなら、共有するか PDF を渡す。

### 後始末（あなたの作業）
- RunPod の pod の terminate と、ストレージの課金の確認（まだなら）。

---

## 6. ファイルの場所

| 内容 | 場所 |
|---|---|
| 事前登録（追記 1〜13） | `Docs/encoder_fix/protocol.md` |
| Task 1 の test 判定 | `Docs/encoder_fix/results_main_v1v2_ckpt/test_judgement.md` |
| 画像の診断・先行研究 | `Docs/analysis_20261111/image_diagnosis_and_related_work.md` |
| early fusion | `Docs/encoder_fix/results_fused_ss5/` |
| Task 2 | `Docs/task2/`、`Docs/encoder_fix/results_t2/`、`Assets/Task2Pilot/Task2Recorder.cs`、`Learning/wm_cache_t2.py` |
| Task 3 | `Docs/task3/`、`Docs/encoder_fix/results_t3/`、`Assets/Task2Pilot/Task3Recorder.cs`、`Learning/wm_cache_t3.py`、`Learning/wm_t3_judge.py` |
| 学習・評価 | `Learning/wm_encoder_study.py`（plan：main_ss5 / t2_ss5 / t3_ss5）、`wm_predict.py`、`wm_test_judge.py`（環境変数 WMJ_RUNS / WMJ_CACHE / WMJ_TAG） |
| 論文の下書き・図・動画 | `Docs/thesis/` |
| 打ち合わせの資料 | `Docs/report_20261108/`（progress_en、meeting_brief_ja、meeting_details_ja、meeting_prep_private） |
| README | `README.md`、`Docs/readme/` |
| Unity のシーン | Task 1：`Assets/OutdoorsScene.unity`。Task 2・3：`Assets/Task2Pilot.unity`（今は Task3Rec が本番の設定で有効、Task2Rec は無効） |
| データ（git に入っていない） | `Episodes/_main_v1v2`、`_h5_v1`、`_task2_v1`、`_task3_v1`（各 `_cache/wm_128x25.h5`） |
| 学習済みの重み（git に入っていない） | `Docs/encoder_fix/results_*/models/` |
