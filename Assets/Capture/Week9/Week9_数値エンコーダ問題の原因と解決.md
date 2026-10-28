# 数値モデルの「誤差がやばい」問題：原因と解決（2026-10-24）

- 詳しいデータ：`Docs/encoder_fix/report.md`（全 100 run の表）、ルール：`Docs/encoder_fix/protocol.md`
- 後半に **Rekha 向けの英語版**あり

---

## 1. 何が問題だったか

Week 8 の小さいテスト学習（P14）で、**数値モデル 2 つが学習データすら覚えきれなかった**。

| | 学習データでの位置の誤差 |
|---|---|
| 画像モデル（手先・腕全体） | 13〜15 mm |
| 数値モデル（手先・腕全体） | **41〜47 mm** |

このまま本番に進むと、「数値モデルがただ下手なだけで、画像が勝ちました」という結果になって、**数値 vs 画像（H3）の比較が成り立たない**。

---

## 2. 原因：学習が短すぎた

4 つの問いで切り分けた（どれも数値モデル、seed 0）。

| # | 問い | やったこと | 学習データの位置の誤差 | わかったこと |
|---|---|---|---|---|
| Q1 | 学習が短かっただけ？ | 構造はそのまま、30 → **100 epoch**、学習率をだんだん下げる（cosine） | 手先 7.1 mm、腕全体 4.2 mm | **これが主な原因** |
| Q2 | 数値データに情報は入ってる？ | 手で取り出した特徴（押し始め・押し終わりの位置、向き、距離）だけで予測 | 3.7 mm | 情報はちゃんと入っている |
| Q3 | 時間の扱いのせい？ | 画像と同じ 25 時刻に揃える（100 epoch） | 手先 4.0 mm、腕全体 3.1 mm | 25 時刻でも問題なし |
| Q4 | 世界座標と画面の対応づけのせい？ | 箱の初期位置を直接渡す（診断用だけ） | 手先 5.8 mm、腕全体 3.9 mm | 原因ではない |

- 証拠：前回の学習ログで、数値モデルの loss は 30 epoch の時点でもまだ下がり続けていた（goal の loss は 20 → 30 epoch で 0.42 → 0.31）。画像モデルはそこでほぼ覚えきっていた（0.05）
- つまり：**数値モデルは覚えるのに時間がかかるのに、30 epoch で打ち切っていた**。エンコーダの構造が致命的にダメだったわけではない

---

## 3. どうやって解決したか

### 3.1 学習を全モデル共通で長くした

- 100 epoch ＋ cosine（学習率をだんだん 0 まで下げる）。数値だけでなく**全モデル同じ設定**

### 3.2 公平な選び方を「回す前に」文書で固定した（`protocol.md`）

数値側だけチューニングすると、それ自体が不公平になる。だから先にルールを決めた。

| ルール | 中身 |
|---|---|
| 候補 | 数値・画像に**同じ 4 種類**：base（今の構造）／attention／GRU／Transformer |
| 試す回数 | 両方とも 4 候補 × 学習率 2 つ（1e-3・3e-4）× seed 3 つ |
| 時間 | 数値も画像も同じ 25 時刻 |
| 大きさ | パラメータ数を全部 1.72〜2.24 M に揃える |
| フィットの基準 | 学習データで位置 ≤ 20 mm、goal AUPRC ≥ 0.95、衝突 AUPRC ≥ 0.95（Task A・B） |
| 選び方 | 基準を満たした候補の中で、**val の loss が一番小さいもの**。手先と腕全体は同じエンコーダ |
| test | 一切見ない |

- 途中で 1 か所だけルールを直した（**追記 1**）：「衝突 AUPRC ≥ 0.95」は、手先モデルだと構造上満たせない（matched pair は手先の入力がわざと同じだから）。そこで matched pair 以外で判定するようにした。数値・画像に同じ定義を使い、本番の比較の結果を見る前に変えた
- 記録の補足（**追記 2**）：診断のときの 4 run は追記 1 の前の計算方法のまま再利用した。調べたら選ばれる結果は変わらなかった

### 3.3 全部回して、ルールどおりに選んだ

- 全 100 run（本番の比較 96 ＋ 診断だけの 4）、エラー 0。Taro の PC（RTX 4060）で約 8 時間半

---

## 4. 結果

**フィットの基準を満たした候補**：数値 6/8、画像 7/8

**選ばれたエンコーダ**

| 形式 | エンコーダ | 学習率 | パラメータ | 学習データの位置の誤差（手先／腕全体） | goal AUPRC（手先／腕全体） |
|---|---|---|---|---|---|
| 数値 | **GRU** | 3e-4 | 1.88 M | 5.3 ／ 6.1 mm | 0.952 ／ 0.993 |
| 画像 | **base（今と同じ）** | 3e-4 | 1.97 M | 0.6 ／ 0.5 mm | 1.000 ／ 1.000 |

- パラメータ数の差は約 5%
- 参考（選ぶのには使っていない。学習に使っていない 3 シーンだけ）：位置の誤差は**数値 19〜24 mm、画像 32〜45 mm**。少なくとも「数値だけ精度が悪い」状態ではなくなった

---

## 5. まだ残っていること

1. **量産データでの再確認**：今回は 20 シーン（val は 3 シーン）で決めた仮決定。量産の最初の 50〜100 シーンで、同じ候補・同じルールでもう一度選ぶ
2. **本番の学習コードへの組み込み**：選んだエンコーダ・学習率・100 epoch を本番のコードに入れる
3. 手先 × 数値の goal AUPRC は 0.952 で、基準（0.95）ぎりぎり。量産データで余裕が出るか見る
4. 画像 × Transformer × 学習率 1e-3 は学習が不安定（手先は 3 回とも失敗）。結果として残してある

---
---

# English version (for Rekha)

## Problem

In the Week 8 test training, the two numeric models could not even fit the training data: final cube position error **41–47 mm** vs. **13–15 mm** for the image models. If left as is, any image advantage in H3 (numeric vs. image) could just mean "the numeric encoder is weak", so the comparison would not be meaningful.

## Cause: training was too short

| # | Question | Test | Train position error (EEF / Full) | Finding |
|---|---|---|---|---|
| Q1 | Just too short? | Same architecture, 30 → **100 epochs**, cosine learning-rate decay | 7.1 / 4.2 mm | **Main cause** |
| Q2 | Is the information in the numeric action? | Hand-crafted features (push start/end, direction, length) → gradient boosting | 3.7 mm | Yes |
| Q3 | Time resolution? | Same 25 time steps as the images (100 epochs) | 4.0 / 3.1 mm | Not a problem |
| Q4 | World-to-image mapping? | Give the initial cube pose directly (diagnostic only) | 5.8 / 3.9 mm | Not the cause |

The training log confirms it: at epoch 30 the numeric losses were still decreasing (goal loss 0.42 → 0.31 between epochs 20 and 30), while the image models had already converged. The numeric models simply learn more slowly and were stopped too early.

## Fix

1. **Longer training for all models**: 100 epochs with cosine decay.
2. **A fair encoder selection, fixed in writing before any run** (`Docs/encoder_fix/protocol.md`):
   - the **same four encoder types** for both formats: the current one, attention pooling, GRU, Transformer;
   - the same budget for both: 2 learning rates (1e-3, 3e-4) × 3 seeds, 100 epochs;
   - the same 25 time steps for numeric and image actions;
   - all models between 1.72 M and 2.24 M parameters;
   - fit criterion on the training set (Task A and B): position error ≤ 20 mm, goal AUPRC ≥ 0.95, collision AUPRC ≥ 0.95;
   - selection: among candidates that fit for both EEF and Full, the lowest validation loss; EEF and Full share the encoder; the test set is never used.
   - One amendment, made before seeing any comparison result: the collision criterion is computed **without the matched-pair candidates**, because an EEF-only model receives identical input for both trajectories of a pair and cannot fit them by design. The same definition applies to all four models.
3. **100 runs** (96 for the comparison + 4 diagnostic-only), no errors.

## Result

- Candidates meeting the fit criterion: numeric 6/8, image 7/8.
- Selected (lowest validation loss):

| Format | Encoder | Learning rate | Parameters | Train position error (EEF / Full) | Train goal AUPRC (EEF / Full) |
|---|---|---|---|---|---|
| Numeric | **GRU** | 3e-4 | 1.88 M | 5.3 / 6.1 mm | 0.952 / 0.993 |
| Image | **current CNN** | 3e-4 | 1.97 M | 0.6 / 0.5 mm | 1.000 / 1.000 |

- For reference only (not used for selection; 3 validation scenes): validation position error 19–24 mm (numeric) vs. 32–45 mm (image). The numeric models are no longer simply worse.

## Still to do

1. Repeat the same selection on the first 50–100 scenes of the main dataset (3 validation scenes are too few to be final).
2. Put the selected encoders and training schedule into the main training code.
3. The EEF-numeric goal AUPRC (0.952) is close to the 0.95 threshold; check it again on the larger dataset.
