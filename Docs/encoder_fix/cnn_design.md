# action エンコーダの設計と、根拠にする先行研究（全部 CNN）

- 作成：2026-10-30。protocol 追記 3（全部 CNN に統一）の根拠をまとめたもの
- コード：`Learning/wm_smoke_train.py` の `NumEnc`・`ImgEnc`、それを使う `Learning/wm_encoder_study.py`（`enc=base`）

## 1. モデル全体の形

4 つのモデル（手先 / 腕全体 × 数値 / 画像）は、**action エンコーダの入力だけが違う**。ほかは全部同じ。

| 部品 | 入力 | 構造 | 出力 |
|---|---|---|---|
| 初期画像エンコーダ | robot-free の初期画像 RGB + depth（4ch、128×128） | 2D CNN 5 層（3×3、stride 2、GroupNorm、SiLU）→ 平均プーリング | 128 次元 |
| ゴール | ゴール中心の (x, z) と画面上の (u, v) | MLP 2 層 | 128 次元 |
| **action（数値）** | 手先 (25, 7) / 腕全体 (25, 69)。正規化してから | **1D CNN 3 層**（時間方向、カーネル 5、stride 1・2・2）→ 長さのマスク付き平均 | 128 次元 |
| **action（画像）** | 手先 / 腕全体の mask + depth、25 時刻 × 2ch（128×128） | **25 時刻をチャネルに並べて 50ch** → 2D CNN 5 層（初期画像と同じ形）→ 平均プーリング | 128 次元 |
| 統合 | 上の 3 つ（128 × 3）を連結 | MLP 2 層（256） | 256 次元 |
| 出力 | — | 最終位置・goal・衝突・接触点（線形）、最終 mask（アップサンプリングの CNN デコーダ） | — |

- パラメータ数（モデル全体）：数値 1.75 M（手先）・1.77 M（腕全体）、画像 1.97 M
- 数値も画像も同じ 25 時刻（`raster_steps`）。比べたいのは「数字か絵か」と「手先か腕全体か」で、時間の細かさではない

## 2. 部品ごとの根拠

| 設計の選択 | 根拠にする論文 | 何を言っている論文か（この研究との関係） |
|---|---|---|
| 数値の action 系列を **1D CNN（時間方向の畳み込み）** で読む | Bai, Kolter, Koltun (2018) *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*, arXiv:1803.01271 | 系列のタスクを幅広く比べて、単純な畳み込み（TCN）が LSTM・GRU と同等以上。系列データの最初の選択肢を RNN ではなく畳み込みにしてよい根拠 |
| action を **カメラにそろえた画像** にして CNN で読む | Khansari et al. (2020) *Action Image Representation*, arXiv:2005.06594 | 行動（把持の候補）を画像として描き、観測画像と同じ CNN で読む。数値ベクトルで渡すより成績が良かった（84% vs 53%）。この研究の画像表現そのものの先行例 |
| 画像の **空間構造を保ったまま** 畳み込む | Zeng et al. (2021) *Transporter Networks*, CoRL 2020（PMLR 155） | 物体の位置などを数値に直さず、画像の空間構造のまま畳み込むと学習効率が良い |
| action を条件にして、**画像から結果を予測する** モデルを CNN で作る | Finn, Goodfellow, Levine (2016) *Unsupervised Learning for Physical Interaction through Video Prediction*, arXiv:1605.07157；Finn & Levine (2017) *Deep Visual Foresight for Planning Robot Motion*, arXiv:1610.00696 | ロボットの押す動作を条件にした予測モデル（CNN）と、それを使ったプランニング。この研究の world model の位置づけ |
| 画像の特徴と action を **連結して** 結果を予測する | Agrawal et al. (2016) *Learning to Poke by Poking*, arXiv:1606.07419 | 押す前の画像を CNN で特徴にし、押す動作と合わせて結果を予測する順モデル。この研究の統合 MLP の形 |
| **手先だけか、全身か** を問う動機 | Yang et al. (2026) *DreamTrajectory*, arXiv:2608.01381；Xu et al. (2025) *Neural Robot Dynamics (NeRD)*, CoRL 2025, arXiv:2508.15755 | DreamTrajectory は手先の軌道を全身の action の手がかりとして使う。NeRD はロボット中心の全身の状態表現でダイナミクスを学ぶ。手先と全身のどちらを見せるべきか、の両側 |

## 3. まだ足りないもの・正直に書くこと

- **足りない根拠（1 本足す）**：画像の action の 25 時刻を **チャネルに並べて 1 つの 2D CNN に入れる**（early fusion）こと。定番の候補は、フレームを重ねて CNN に入れる Mnih et al. (2015) *Human-level control through deep reinforcement learning*（Nature）か、動画の early fusion を比べた Karpathy et al. (2014) *Large-scale Video Classification with Convolutional Neural Networks*（CVPR）。どちらか 1 本を読んで足す
- **TCN そのものではない**：`NumEnc` は dilation・残差接続のない単純な 1D CNN。論文には「TCN（Bai et al. 2018）の知見に基づく、単純な 1D 時間畳み込み」と書く
- **統合のしかたは Finn et al. (2016) と違う**：Finn et al. は action を空間に並べて CNN の特徴マップに重ねる。この研究は特徴ベクトルを連結して MLP に入れる（Agrawal et al. 2016 に近い）。4 モデルで統合のしかたは同じなので、比較の公平さには影響しない
- **関連研究に回す論文**：Salhotra et al. (2023) *Learning Robot Manipulation from Cross-Morphology Demonstration*（MAIL）, arXiv:2304.03833 は、形の違うロボット間で動作を移す話で、エンコーダの根拠にはならない。ロボットの体の表現の関連研究として触れる程度
- **数値に不利な方向の可能性**：protocol §3 では、数値は gru のほうが val loss が小さかった（追記 3）。全部 CNN に統一したことで、数値モデルが少し不利になっている可能性がある。§5 の再確認で、数値の E0 がフィットの基準を満たすか、val の成績がどれくらいかを記録する

## 4. 次に確かめること

1. 量産データ（`Episodes/_main_v1`）の 50〜100 ペアで、E0 の 4 モデルを protocol §1 の設定（100 epoch、seed 3 つ）で学習し、フィットの基準を満たすか確認する
2. 同じ設定で、データ量 25 / 50 / 100% の学習曲線を見る（`_main_v2` を作るかの判断に使う）
3. 候補の種類だけで当てる基準（`wm_eval.label_baseline`）を超えているか確認する
