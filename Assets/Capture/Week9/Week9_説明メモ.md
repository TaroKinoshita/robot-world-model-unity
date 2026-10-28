# Week 9 MTG 用：動画 2 本の説明メモ

- 置いてあるもの（このフォルダ）
  - `video1_training_data_16_trajectories.mp4` … ① 学習データの動画
  - `video2_model_output_goal_highlighted.mp4` … ② 学習済みモデルの出力動画（ゴールをハイライト）
  - `video3_taskB_cylinder_whole_arm.mp4` … ③（任意）円柱ありの Task B 版。「なぜ腕全体を見るのか」を見せる用
  - このメモ
- 動画の作り方：記録済みデータ（`executed/executed_trajectory.csv` の各リンクと箱の位置・向き）を Unity でそのまま再生して描画（新しくシミュレーションはしていない）。ツールは `Assets/Capture/Tools/TrajectoryReplayRenderer.cs`、合成スクリプトは `Captures/week9_work/tools/`
- 英語のセリフは「そのまま読めばいい」用。日本語は自分用の中身の説明

---

## 0. 最初に言うこと（30 秒）

> Last time you asked for two videos: one showing how I generate the data, and one showing what the trained model outputs. Here they are. In both videos the green ring is the goal region, radius 3 cm. It is only drawn for the video — the images the model sees do not contain it.

（前回言われた「データの動画」と「モデル出力の動画」を持ってきた。緑の輪はゴール。動画にだけ描いていて、モデルの入力には入ってない）

---

## ① 学習データの動画（`video1_...mp4`、約 24 秒）

### 何が映ってるか

- 1 つのシーン（seed 4、Task A ＝円柱なし）で、**同じ初期配置から 16 本の候補軌道**をそれぞれ物理シミュレーションで実行したもの。4×4 に並べて同時に再生（等速）
- 黄色の線 ＝ 箱が実際に動いた跡
- 各タイルが終わると **GOAL（緑）／MISS（赤）** と、ゴール中心からの距離が出る
- 結果：**16 本中 6 本が GOAL、10 本が MISS**

| 候補 | 種類 | 結果 |
|---|---|---|
| c000〜c003 | matched pair 2 組（同じ手先の経路、手首の向きだけ違う） | 4 本とも MISS（88・150 mm） |
| c004〜c006 | ゴール方向に押す | 3 本とも GOAL（9〜11 mm） |
| c007〜c009 | ゴールの縁を狙う | 3 本とも GOAL（12〜27 mm） |
| c010〜c012 | ランダムな方向 | MISS（56〜115 mm） |
| c013〜c015 | 円柱の方向（Task A なので円柱は無い） | MISS（60〜77 mm） |

### 言うこと（英語）

> This is one scene from my dataset. The cube position, its orientation and the goal are random per scene. For every scene I plan 16 candidate push trajectories and run each of them in Unity physics — so one scene gives 16 episodes.
>
> The 16 candidates are a fixed mix: two matched pairs, three pushes toward the goal, three toward the edge of the goal, three random directions, and three toward the cylinder position. That way every scene has both successes and failures.
>
> The label comes from the final state: "goal reached" means the cube centre ends within 3 cm of the goal centre, it did not topple, and it stopped. Here 6 of 16 reached the goal and 10 missed.
>
> Every scene also has a Task B version: the same 16 trajectories, but with a cylinder near the arm, so some of them collide. Right now I have 20 scenes, that is 640 episodes. The plan is 200 to 320 scenes, around 10,000 episodes.

### 聞かれそうなこと

- **Q：なんで失敗も入れるの？** → The model has to predict the outcome of any trajectory, not only good ones. If the data only had successes, it could not learn what makes a push fail.
- **Q：matched pair が全部失敗してるけど？** → 手先の経路はランダム寄りなので、このシーンではたまたまゴールに届いてない。matched pair の役目は「手先が同じで腕の形だけ違う 2 本」を作ることで、Task B で片方だけ円柱に当たるように置いてある（H5 用）。成功させるための候補ではない
- **Q：どのくらい時間かかる？** → 1 シーン（A と B で 32 episode）約 8 分。16 倍速のバッチで 1 シーン約 2.6 分

---

## ② 学習済みモデルの出力動画（`video2_...mp4`、約 1 分）

### 先に自分が理解しておくこと（大事）

- **このモデルはロボットを直接動かすモデルではない**。world model なので、やってることは「この軌道を実行したら、箱はどうなるか」の予測
  - 入力：robot-free の初期画像 ＋ ゴール ＋ 候補軌道 1 本（action 表現）
  - 出力：ゴールに届く確率 P(goal)、箱の最終位置、最終の箱の mask、衝突の確率
- ロボットを動かすときは、**16 候補を全部予測 → P(goal) がいちばん高い候補を選ぶ → それを実行**。動画はこの流れを見せている
- 動画で使ったモデル ＝ **腕全体 × 画像（Full-Image）の小さいテストモデル**（P14、14 シーンで学習、30 epoch）。本番のモデルではない

### 何が映ってるか

1. 最初のカード：上の「モデルが何を出すか」の説明
2. **例 1：学習に使ったシーン（seed 4、①と同じシーン）**
   - 右の棒グラフ ＝ モデルが出した 16 候補の P(goal)。棒の色は実際の結果（緑＝届いた、赤＝外れた）
   - いちばん高い **c005（P = 0.99）を選ぶ** → 実行 → **GOAL（ゴール中心から 11 mm）**
   - 水色の × ＝ モデルが予測した「箱が止まる位置」。実際とのズレ 10 mm
   - 右下 ＝ モデル自身の視点（128 px の入力画像）に、予測した最終の箱（水色）と実際の箱（緑の線）を重ねたもの
3. **例 2：学習に使っていないシーン（seed 13）**
   - c012（P = 0.91）を選ぶ → **MISS（147 mm）**
4. 最後のカード：学習シーンでは 14/14 成功、未知シーンでは 0/3

### 言うこと（英語）

> The model is a world model, so it does not control the robot directly. It takes the robot-free scene image, the goal, and one candidate trajectory, and predicts the outcome: the probability that the cube reaches the goal, where the cube stops, the final cube mask, and the collision probability.
>
> To actually move the robot, I predict all 16 candidates, pick the one with the highest predicted goal probability, and execute it. The bars on the right are the model's predictions; the bar colour is what really happened.
>
> In the first example the model picks c005 with 0.99, and the cube really ends 11 mm from the goal centre. The cyan cross is where the model predicted the cube would stop — 10 mm away from where it actually stopped.
>
> I want to be clear about one thing: this is a small test model trained on only 14 scenes, so the first example is a scene it has seen. On the 3 unseen test scenes it picked the wrong trajectory every time — the second example is one of them. That is expected with 14 scenes: it memorises instead of generalising. The real result needs the mass-produced data, 200 to 320 scenes.

### 「95%」について（聞かれる前に自分から言う）

前回 Rekha が「whole robot model が 95% 成功してる」と言ってたのは、Week 8 スライド 5 の **「whole-robot models: 0.95–1.00（training set）」**のこと。中身は **H5：腕全体のモデルが、学習シーンで matched pair の 2 本を正しく区別できた割合**（数値 0.95、画像 1.00）。「ロボットが 95% 成功する」という意味ではない。

> The 0.95 on last week's slide was not a success rate. It was the matched-pair accuracy (H5) of the whole-robot models on the training scenes, with the small test model: they could tell the two trajectories of a matched pair apart. It shows that the pipeline works end to end, not that the model is good yet. On unseen scenes it is not reliable yet.

---

## ③ Task B 版（`video3_...mp4`、約 48 秒、任意）

### いつ出すか

- 頼まれた動画ではないので、**時間があるとき／「腕全体を見る意味は？」と聞かれたとき**に出す
- ①を見せた直後に「同じシーンで円柱を足すとこうなる」とつなげるのがいちばん自然

### 何が映ってるか

1. **前半（4×4）**：①と同じシーン（seed 4）・同じ 16 本。違いは腕の近くに青い円柱（高さ 40 cm）があることだけ
   - オレンジの枠と「HIT」＝ 円柱に初めて触れた瞬間（arm ＝ 腕が当たった、cube ＝ 押した箱が当たった）
   - 結果：**6 本が円柱に接触（腕 4・箱 2）**、ゴールは 6/16（①と同じ）
2. **後半（2 画面）**：matched pair 1 の 2 本を左右に並べたもの
   - 水色の線 ＝ 手先（グリッパー）の通り道。**2 本でほぼ同じ（差 0.2 mm 以下）**
   - 左（c000、手首が下向き）は円柱に触れない、右（c001、手首が上向き）は**腕が円柱に当たる**
   - matched pair 2（c002／c003）も同じで、上向きの方だけ当たる

### 言うこと（英語）

> This is the same scene and the same 16 trajectories as the first video. The only difference is the cylinder near the arm — this is Task B. The planned hand paths are exactly the same as in Task A.
>
> Six trajectories touch the cylinder: four times the arm hits it, twice the pushed cube does.
>
> The key part is the matched pair. These two trajectories have the same hand path — the difference is below 0.2 mm — only the wrist is flipped. The wrist-down one passes the cylinder, the wrist-up one hits it with the arm.
>
> A model that only sees the hand gets exactly the same input for both, so it cannot tell which one collides. A model that sees the whole arm can. That is hypothesis H5, and it is the reason the whole-arm representation should help in Task B — H2 and H4.

### 聞かれそうなこと

- **Q：円柱が当たったあと消えるのは？** → 腕に強くはじかれて、机の外に落ちてる（記録データのとおり。動画用に消してはいない）。ロボットは位置制御で、ぶつかっても止まらない設定なので、当たると円柱を押しのける。ラベルは「最初に触れたかどうか」で付けているので、そのあとの動きは結果に影響しない
- **Q：なんで箱が当たるケースも衝突に入るの？** → proposal の定義が「ロボット**または**押した物体が secondary に触れたら衝突」だから。腕の情報が要るのは腕が当たるケース
- **Q：Task A と B でゴールの数が同じなのは？** → このシーンでは、円柱に当たった 6 本はもともとゴールに届かない軌道だった。シーンによっては、当たったせいでゴールを外すケースもある

---

## 合わせて準備するもの（動画以外）

- 英語で詳しく書いた版：`Week9_metrics_and_model.md`（③④をそのまま Rekha に見せられる形）

### ③ 仮説 1〜5 の評価指標（Rekha が「見たい」と言ってたもの）

| 仮説 | 中身 | 使う指標 | supported の条件 |
|---|---|---|---|
| H1 | Task A では手先のみでも腕全体とほぼ同じ | 最終位置誤差、goal AUPRC、ranking accuracy | 手先のみの悪化の CI 上限がマージン以下（3 mm／0.05／5 pt） |
| H2 | Task B では腕全体の方が衝突予測が正確 | 衝突 AUPRC | 改善の CI 下限 > 0 |
| H3 | 画像は mask が良く、数値は位置が 3 mm 以上悪くならない | 最終 mask IoU、最終位置誤差 | IoU 改善の CI 下限 > 0 かつ 位置の悪化の CI 上限 ≤ 3 mm |
| H4 | Task B で腕全体の方がいい軌道を選ぶ | 選んだ軌道の衝突率、regret | 両方の改善の CI 下限 > 0、かつ H2 が成立 |
| H5 | 腕全体の方が matched pair を区別できる | matched pair の正解率（衝突する方に高い確率を付けたら正解） | 改善の CI 下限 > 0 |

- CI はシーン単位の paired bootstrap（95%）。Task A・B は別々に報告、3 seed の平均

### ④ モデルの種類と学習方法（メモに書いてた質問）

- **種類**：教師あり学習の CNN ベースの world model（4 つの action 表現で同じ構造）
  - 初期画像 → 5 層 CNN、ゴール → MLP、action → エンコーダ（数値は 1D conv 3 層、画像は 2D CNN 5 層）→ 統合 MLP → 出力ヘッド（最終位置・goal・衝突・接触点）＋ mask デコーダ
  - パラメータ 175〜197 万
- **学習方法**：記録済みデータでのオフラインの教師あり学習（強化学習ではない）
  - 損失：位置は smooth L1、goal・衝突・mask は BCE の合計（マルチタスク）
  - AdamW、lr 1e-3、batch 16、30 epoch、seed 0（本番は 3 seed）
  - データの分け方：シーン単位（matched pair を分けない）で train 70%／val 15%／test 15%

> It is a supervised world model, not reinforcement learning. A CNN encodes the scene image, an MLP the goal, and a separate encoder the action — 1D convolution for the numeric actions, a 2D CNN for the image actions. Everything after the action encoder is identical across the four models. It is trained offline on the recorded episodes with a multi-task loss: smooth L1 for the final position and binary cross-entropy for goal, collision and the final mask.

- 正直に言っておく点：今の数値エンコーダは弱い（学習データすら覚えきれてない）。本番前に直す（時間方向を 25 時刻にそろえる／GRU や Transformer にする、など）

---

## チェックリスト

- [ ] ① 動画を開いて、GOAL／MISS の表示と 6/16 を確認
- [ ] ② 動画を開いて、例 1・例 2 と最後のカードを確認
- [ ] （任意）③ 動画を開いて、matched pair の左右で当たる／当たらないを確認
- [ ] ③ の表をスライド 1 枚にする
- [ ] ④ をスライド 1 枚にする
