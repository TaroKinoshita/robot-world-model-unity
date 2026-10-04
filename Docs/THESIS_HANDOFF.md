# 卒論ハンドオフ文書（Claude 引き継ぎ用）

- 最終更新：2026-10-05（前の版：2026-10-04）
- 対象：Taro Kinoshita（S1140213）の BSc AI 卒論（Radboud University, SOW-BKI300）
- 使い方：
  - 新しいチャットの最初に、このファイルを丸ごと貼る
  - Claude はこれを「現時点の正」として扱う。ここに書いていない数値・設定は推測せず、Taro に確認する
- 追記（2026-10-05 午後）：P6（バッチ実行）完了。§6.9・§8・§12・§14.2 を更新
- この版で変えたこと：
  - TODO6（UR5e）、P0〜P5 の結果を全部反映した
  - 現在の仕様（§5〜§7）と、全期間の経緯（§9）を 1 本にまとめた
- 細かい生データ（表・試行の全部）は、プロジェクトの `Docs/` にある経緯文書を見る：
  - `P1_P1.5_経緯_20261004.md`
  - `P2_P3_経緯_20261004.md`
  - `P4_P5_経緯_20261005.md`
- 優先順位：
  - 数値が食い違ったら、新しい方を正とする：`P4_P5` ＞ `P2_P3` ＞ `P1_P1.5`
  - 本書は全部をまとめた版なので、上の 3 本と食い違ったら本書を正とする

---

## 0. 30 秒で分かる現状

- **テーマ**：「How Much of the Robot Should a World Model See?」
  - world model に渡すロボットの action 表現を **範囲（手先だけ / 腕全体）× 形式（数値 / カメラ視点の画像）の 2×2** で比べる
- **proposal**：2026-09-30 付の版で、10/2 締切に提出済み（内容は §2）
- **データ生成は「本番の仕様」まで完成**
  - ロボット：UR5e＋Robotiq 2F-140
  - Play 1 回 ＝ seed 1 個 ＝ 1 シーン。次の 2 つが自動で作られる：
    - Task A（円柱なし）16 episode
    - Task B（円柱あり）16 episode
  - 16 候補の中に matched pair（同じ手先経路・腕の形だけ違う 2 本）が 2 組入る
  - action 表現 4 種：
    - 画像 2 種は、メッシュの本描画（mask＋depth）
    - 数値 2 種には、位置＋向きが入る
  - 予測ターゲット 5 種：最終位置・goal・衝突・最終 mask・contact heatmap
  - 自動検証の結果：全項目 PASS
- **パイロット**：3 シーン × 16 候補 × A/B ＝ 96 episode（git `f46109c` から作成）
  - 10/16 の最低ライン（P4-A）と、10/30 の理想形（P4-B）は、どちらも達成済み

| 項目 | 結果 |
|---|---|
| 自動検証 | A 580/580・B 564/564・ペア 107/107（3 シーンとも） |
| goal True | 18 / 48（38%） |
| 衝突 True（Task B） | 16 / 48（33%） |
| 倒れた | 0 |
| ロボット↔机 | 0 |
| 追従誤差（TCP） | 最大 2.8 mm |
| A/B の不安定（触れていないのにずれる） | 0 |

- **proposal から変えたこと**：6 つ（§2.10）。10/30 の中間報告で説明する
- **git**：
  - 作業ブランチは `todo6-ur5e`。この文書を commit する前の HEAD は `cf679b3`（push 済み）
  - `main` は `776fe3c`（UR3 版）のまま。28 commit 遅れている
- **次にやること**（詳しくは §14）：
  1. 量産用のまとめて回す仕組み（バッチ実行）を作る
  2. 量産と、偏りの点検
  3. 10/30 の中間報告
  4. GPU の相談（Rekha か Jordy へ）
  5. 学習モデルの設計。まだ何も決めていない

```
[proposal]   何を比べるか                          ← 提出済み（10/2）
     │
[TODO1〜5]   撮影環境・robot-free・1 episode        ← 完了（UR3 時代）
     │
[P0・P1]     TODO5 クローズ・UR5e＋2F-140 に交換    ← 完了（tag todo6-complete）
[P1.5]       学習コストの概算                      ← 完了
     │
[P2・P3]     Task A/B ペア・matched pair           ← 完了（matched pair＝手首の反転）
     │
[P4-A]       10/16 の最低ライン                    ← 達成
[P4-B]       ブレの修正・押し方・自動生成・本番入力  ← 達成
[P5]         小さい修正                            ← 完了
     │
[次]         バッチ実行 → 量産 → 学習 → 評価 → 執筆
```

---

## 1. 人・体制・経緯

| 役割 | 名前 | 連絡先 |
|---|---|---|
| 学生 | Taro Kinoshita（S1140213） | taro.kinoshita@ru.nl |
| First assessor（指導教員） | dr. Rekha Raja（Department of AI, Radboud） | rekha.raja@donders.ru.nl |
| Second assessor（second reader） | Pieter Wolfert（Department of AI, Radboud） | pieter.wolfert@ru.nl |
| 卒論コーディネーター | Jordy Thielen | bsc-thesis-coordinator@ai.ru.nl |

テーマが決まるまで：

- 元のテーマは、Wan2.1-VACE の LoRA で control video 表現（Arm A/B/C）を比べる、動画生成寄りの研究だった → **白紙に戻した**
- Rekha が指導を引き受ける条件は、**卒論のゴールをロボット制御（Robot Planning）にすること**
  - 動画生成・3DCG のスキルは過程で使ってよい。ただしゴールにはしない
- Rekha との MTG で、案 2「Robot Action Representation for Learned World Models」で進める許可が出た → prior art を調べて、今のテーマに確定
- 構成：
  - 必須：Phase 1（基盤構築）＋ Phase 2（4 条件の比較＝卒論の中心）
  - optional：Phase 3（短時間の video 予測）
- 進め方の約束：
  - git は Claude が commit まで。**push は Taro が自分でやる**（P1 の途中から）
  - proposal からの変更は、Taro が決めて、10/30 の中間報告で Rekha に伝える（P3 の matched pair など）

---

## 2. 研究設計（提出版 proposal の内容）

### 2.1 タイトル

How Much of the Robot Should a World Model See? Comparing End-Effector and Full-Body Action Representations for Robot–Object Prediction and Planning

### 2.2 4 つの action 表現

| 名前 | 範囲 | 形式 | proposal 上の本番仕様 | 今の実装（§6.4） |
|---|---|---|---|---|
| EEF-Numeric | 手先 | 数値 | 手先の位置＋向きの時系列 | TCP 位置＋tool の quaternion＋開閉（50 Hz 全 step） |
| EEF-Image | 手先 | 画像 | 手先の mask ＋ depth 画像 | グリッパーのメッシュの本描画 mask＋depth、25 枚 |
| Full-Numeric | 腕全体 | 数値 | 全リンクの位置＋向きの時系列 | 関節角 6 ＋ 9 点の位置＋向き（50 Hz 全 step） |
| Full-Image | 腕全体 | 画像 | 腕全体の mask ＋ depth 画像 | ロボット全体のメッシュの本描画 mask＋depth、25 枚 |

- すべて **planned trajectory から作る**
  - executed からは作らない。衝突などの結果が入力に漏れるのを防ぐため
- 画像表現は、シーン画像と同じカメラで描く → ピクセル単位で位置がそろう
- 初期シーン画像は **ロボットを消して撮る（robot-free）**
  - ロボットの姿勢・動きの情報は、action 表現からしか入らない
- 4 モデルは、action の入力以外をすべて同じにする

### 2.3 タスク

- **Task A**：物体 1 つをゴールへ押す。secondary object なし
- **Task B**：Task A と同じ物体配置・ゴール・初期姿勢・候補軌道に、**腕の近くに secondary object を足す**。腕がぶつかりうる
- どちらも押すだけなので、グリッパーの開きは固定
- 1 シーンあたり **候補軌道 16 本**
  - その中に **matched pair** を含む
  - proposal での定義：同じ手先経路を elbow-up / elbow-down の 2 つの IK 解で実行する。Task B では片方だけ衝突するように secondary を置く
  - → 実装では「手首の反転」に変更した（§2.10）

### 2.4 予測ターゲット（全モデル共通）

最終物体位置、goal 到達、意図しない衝突、最終物体 mask、contact heatmap。

### 2.5 RQ と仮説

- **Main**：同じ計画軌道を渡したとき、action 表現の範囲と形式は、結果予測・空間予測・候補選択にどう影響するか
- **RQ1（範囲）**：手先のみ vs 腕全体の差は、Task A と B でどう変わるか
  - H1：Task A では、手先のみが腕全体と事前に決めたマージン以内
  - H2：Task B では、腕全体の方が衝突予測が正確（Ganapathi et al., 2022）
- **RQ2（形式）**：同じ範囲で、数値 vs 画像は、数値予測・空間予測にどう影響するか
  - H3：画像は最終 mask IoU を改善し、数値は最終位置誤差で 3 mm 以上は悪化しない
- **RQ3（候補選択）**：予測の差は、選ばれる軌道の質にどう効くか
  - H4：Task B で、腕全体の方が、選んだ軌道の衝突率と regret が低い
- **RQ4（matched pair）**：手先経路が同じで腕の形だけ違う 2 候補を区別できるか
  - H5：腕全体の方が matched pair を正しく区別できる

### 2.6 評価

予測精度：

- 最終物体位置誤差（mm）
- goal・衝突の AUPRC
- 最終物体 mask の IoU
- 接触位置の誤差
- goal 到達 = 物体が goal から 3 cm 以内で静止（実装の定義は §6.6）
- 意図しない衝突 = ロボット or ターゲットが secondary に接触（衝突系の指標は Task B のみ）

意思決定の質：

- 各シーンで 16 候補の結果を予測する → 「衝突せず goal に届く確率」が最大の候補を選ぶ → シミュレーションで実行して成否を確認する
- 指標：成功率、Task B の衝突率、**regret**（選んだ候補と最良候補の差）、**pairwise ranking accuracy**
- 下限 = ランダム選択、上限 = 正解を知る oracle

統計：

- Task A / B は別々に報告する。テストシーンは学習から外す。3 seed の平均
- 95% CI は **シーン単位の paired bootstrap**
  - 全表現・両タスク・全 seed で、同じシーンを再サンプルする
  - matched pair は分割しない
- H5 の正解判定：衝突する方に高い衝突確率を付けたら正解

仮説の判定基準：

| 仮説 | supported の条件 |
|---|---|
| H1 | 手先のみの性能低下の CI 上限が、3 指標すべてでマージン以下：最終位置誤差 3 mm（goal 許容の 10%）、goal AUPRC 0.05、pairwise ranking accuracy 5 pt。いずれかの CI 下限がマージン超 → rejected。それ以外 → inconclusive |
| H2 | 腕全体の衝突 AUPRC 改善の CI 下限 > 0 |
| H3 | 画像の IoU 改善の CI 下限 > 0、かつ 数値の最終位置誤差超過の CI 上限 ≤ 3 mm |
| H4 | 腕全体の衝突率低下と regret 低下の CI 下限がどちらも > 0、かつ 同じ形式で H2 が成立 |
| H5 | 腕全体の matched pair 正解率改善の CI 下限 > 0 |

- 2 形式（or 2 範囲）それぞれで別々に支持を主張する場合は、Holm 補正をかける。その他の指標は探索的に扱う

### 2.7 Optional extension

- 条件：コア実験が終わって時間があり、指導教員が同意したとき
- 中身：初期シーン画像＋計画軌道の画像から、segmentation / depth の短期の未来フレームを予測する小さな時系列モデルで、EEF-Image vs Full-Image を比べる
- RGB 予測や、事前学習済みの動画モデルの LoRA / adapter は、さらに時間がある場合だけ
- RQ1〜4 には要らない

### 2.8 スケジュール（proposal 記載）と今の位置

| 作業 | 期限 | 状態（10/05） |
|---|---|---|
| proposal 提出 | 2026-10-02 | ✅ |
| 環境構築＋パイロットデータ（2 週） | **2026-10-16** | ✅ 前倒しで達成 |
| 中間チェックポイント | 2026-10-30 | 準備中（§14.3） |
| 本番データ生成＋学習（7 週） | 2026-12-04 | バッチ実行は完成（P6）。次は量産と点検（P7）。学習モデルは未設計 |
| 評価＋執筆（6 週） | 2027-01-14 | — |
| 最終提出 | 2027-01-15 | — |
| 発表準備（2 週） | 発表日前 | — |
| 発表 | 2027-01-18〜29 | — |

### 2.9 参考文献（proposal 掲載分）

- Alzayer et al. 2026（Masked visual actions, arXiv 2607.19343）
- Borkman et al. 2021（Unity Perception）
- Finn et al. 2016（NeurIPS）
- Ganapathi et al. 2022（Implicit kinematic policies, ICRA）
- Kim et al. 2026（Robot-factored world models, arXiv 2607.22535）
- Wang et al. 2025（Visual action prompts, ICCV）
- Yin et al. 2026（PlayWorld, arXiv 2603.09030）
- Yu et al. 2026（World model evaluation for decision-making, arXiv 2606.15032）
- Zhang & Du 2026（World Action Planner, arXiv 2607.27599）

### 2.10 proposal からの変更点（10/30 の中間報告で説明する）

| # | proposal | 実装 | 理由 | 決めた時期 |
|---|---|---|---|---|
| 1 | matched pair = elbow-up / elbow-down | **手首の反転**（wrist-down / wrist-up、どちらも elbow-up）。手先の差 < 0.15 mm、前腕・手首は約 20 cm 違う | ロボットを机と同じ高さに置いているので、elbow-down は肘が天板の下を通る。土台を下げる・グリッパーを傾ける、の別案もダメ（§9.8） | P3 |
| 2 | 押し方の指定なし | **指を半分開いて（0.42 rad、パッドの間隔約 6 cm）、グリッパーを 90° 回して、傾けずに 2 点で押す** | 元の押し方（開いた指 1 本）では、全候補で箱が横倒しになっていた。閉じた指だと、向きを変えた箱で 35〜40° それた（§9.11・§9.14） | P4-B |
| 3 | goal = 3 cm 以内で静止 | 3 cm 以内 **かつ 倒れていない（2 cm 下がる or 30° 超傾く）かつ 速さ < 5 mm/s** | 立方体は横倒しでも中心の高さが変わらず、高さだけの判定では見逃した | P4-B |
| 4 | 衝突 = ロボ or ターゲットが secondary に接触 | 同じ。ただし **隙間 ≤ 1 mm の「触れた」接触だけ**を数える | PhysX は contactOffset 以内に近づくだけで接触を報告する | P4-B |
| 5 | （物理の設定は記載なし） | **TGS ソルバー**、候補ごとに新しい物理シーン | 円柱に触れていないのに、円柱があるだけで Task B の結果がずれた（最大 12.6 mm）。同じ Play の中で、前の候補の履歴が残っていた | P4-B |
| 6 | 16 候補（内訳の記載なし） | **matched pair 2 組 ＋ ゴール方向 6 ＋ ランダム 3 ＋ 円柱の方向 3**。円柱の位置・高さ（24〜40 cm）はシーンごと | 1 か所の円柱では、全部のペアを片方だけ衝突させられない。衝突ラベルを確保するため | P4-B |

論文・報告で使う言い方（案）：

- matched pair: two candidates that share the same end-effector path (TCP position and orientation; max difference < 0.15 mm) but are executed with different inverse-kinematics solutions (wrist-down vs. wrist-up, both elbow-up), so that the forearm and wrist differ by about 20 cm.
- H5: Full-body representations distinguish matched-pair candidates (same end-effector path, different arm configuration) more accurately than end-effector-only representations.
- elbow-down was not used because, with the robot mounted at table height, the elbow of the elbow-down solution passes below the tabletop.
- collision geometry was convex-decomposed; residual bulge ≤ 1.8%, under-approximation ≤ 6 mm.
- pilot benchmark on an RTX 4060 Laptop GPU; estimates scaled to the target GPU class.

---

## 3. proposal の本番仕様と今のパイプライン（対応状況）

| proposal の本番仕様 | 今のパイプライン | 状態 | どこで |
|---|---|---|---|
| ロボット UR5e + Robotiq 2F-140 | UR5e＋2F-140。当たり判定は凸分割し直し | ✅ | P1 |
| 1 シーン 16 候補 | seed から自動生成（§6.2） | ✅ | P4-B |
| matched pair | 手首の反転で 2 組／シーン。Task B で片方だけ衝突する円柱を自動配置 | ✅（定義を変更） | P3・P4-B |
| Task A / B のペア | Play 1 回で両方。planned は 1 回だけ作って共有（バイト単位で同じ） | ✅ | P2 |
| 画像表現 = mask + depth の本描画 | CPU の z-buffer でメッシュを描く。Perception の画像と IoU 0.997・depth 差 0.25 mm | ✅ | P4-B |
| Full-Numeric = 全リンクの位置＋向き | 関節角 6 ＋ 9 点の位置＋quaternion | ✅ | P4-B |
| contact heatmap | 触れた接触点を投影してガウス（σ 2 px） | ✅ | P4-B |
| goal 定義 | シーンごとにランダム、半径 3 cm、倒れていない・止まっている | ✅ | P4-B |
| 衝突定義 | 触れた接触だけ | ✅ | P4-B |
| planned から action、executed と分離 | 実装済み | ✅ | TODO5 |
| robot-free 初期画像 | 実装済み。検証 A・B を UR5e でもやり直して合格 | ✅ | TODO4・P1 |
| 再現性（同じ入力 → 同じ結果） | 候補ごとの物理シーン＋TGS。Play をまたいでも 1 ビットも同じ | ✅ | P4-B |
| 本番データ（数千〜1 万 episode） | バッチ実行（P6）はできた。量産はこれから | 🔶 | §6.9・§14.2 |
| 学習モデル | 仮モデルでコストを測っただけ。本番の設計は未着手 | ❌ | §14.5 |
| GPU | 手元は RTX 4060 Laptop 8 GB。大学のクラスタが使えるか未確認 | ❌ | §14.4 |

---

## 4. 実験環境（確定）

| 項目 | 値 |
|---|---|
| Unity | 6000.0.84f1（Unity 6.0 LTS, DX12） |
| HDRP | 17.0.4 |
| Perception | 1.0.0-preview.1（最終版） |
| URDF Importer | 0.5.2-preview（Update は押さない） |
| Unity プロジェクト | `C:\UnityProjects\PerceptionTestHDRP` |
| シーン | `Assets/OutdoorsScene.unity` |
| リポジトリ | `TaroKinoshita/robot-world-model-unity`（private） |
| 作業ブランチ | `todo6-ur5e`（HEAD `cf679b3`、push 済み） |
| `main` | `776fe3c`（UR3 版のまま） |
| tag | `todo4-verified`（421a4a4）、`todo5-complete`（776fe3c、付け替え済み）、`todo6-complete`（9b6030b） |
| 解像度 | 256 × 256（Game View `capture_256`） |
| Asynchronous Shader Compilation | オフ |
| SOLO 出力 | `C:\Users\kinos\AppData\LocalLow\DefaultCompany\PerceptionTestHDRP\solo_N\sequence.0\`（Play ごとに番号が増える。パイロット作成時は solo_124 前後） |
| git 設定 | `core.autocrlf false` |
| `.gitignore` に入っているもの | `/Captures/`、`/Episodes/`、`/CaptureLogs/`、`/_import_backup/`、`__pycache__/`、`__pycache__.meta`、`/Claude outputs/`、`Library/`、`Temp/` など |
| GPU（手元） | NVIDIA GeForce RTX 4060 Laptop、8,188 MiB、ドライバ 610.74 |
| Python（Windows） | 3.11.9、PyTorch 2.5.1+cu121、cv2 あり。検証・ベンチマークはこれで回す |

### 4.1 Claude からの操作

| 道具 | 中身 | 注意 |
|---|---|---|
| Unity MCP | CoplayDev MCP for Unity **v10.2.0 で固定**（`manifest.json` の git URL 末尾 `#v10.2.0`）。有効ツールグループは `core` だけ。「Configure All Detected Clients」は押さない | `execute_code` は CodeDom（C# 6）。1 回の呼び出しは 60 秒で応答なしになる |
| 手元のシェル（device_bash） | Taro の PC の Linux VM。プロジェクトは `$HOME/mnt/PerceptionTestHDRP` に見える | Python 3.10 で cv2 なし。ファイルの読み書き・検索に使う。git はここから触らない |
| クラウド側のシェル | Claude のサンドボックス | 重い計算（IK の総当たりなど）を数分単位に分けて回す |
| git・Windows の Python | Unity の `execute_code` から `System.Diagnostics.Process` で起動 | 標準出力と標準エラーを同時に非同期で読む。`GIT_TERMINAL_PROMPT=0`。タイムアウト付き |
| 長い処理（パイプライン） | 別プロセスで起動して、結果を `Episodes/_pipeline_log.txt` に書かせる | MCP の 60 秒制限を避けるため |

- 方針：汎用の操作は CoplayDev、研究固有の操作だけ Custom Tool にする（候補：`run_robot_free_capture`、`run_verification_b`、`export_camera`。必要になったら作る）
- スクショは `Captures/` に保存する（`Assets/Screenshots` に入ると git が汚れる）
- スクリプトを書き換えたら、コンパイルが終わって idle になってから Play する
- 細かいルールは §11

---

## 5. シーン設定（現在値、git `f46109c`〜`cf679b3`）

### 5.1 ロボット

| 項目 | 値 |
|---|---|
| ロボット | `ur5e_with_gripper`（UR5e＋Robotiq 2F-140）。`Assets/ur5e_with_gripper/ur5e_with_gripper.urdf` から import |
| 旧ロボット | `ur3_with_gripper` は**無効化して残してある**（消していない） |
| ルート | `world`、`world_joint` z = 0.77（UR3 と同じ）。`base_link` は Immovable |
| 腕の関節 | `shoulder_link`〜`wrist_3_link` の 6 つ、全関節 ±180° |
| TCP | `left_inner_finger_pad` と `right_inner_finger_pad` の中点 |
| 初期姿勢（`RobotInitialPose`） | `[−63.5, −151.8, 122.3, −60.5, −90.0, −63.5]°`（手先 (0, 1.05, 0.15)、wrist-down の枝） |
| 指（`RobotInitialPose.gripperCloseRad`） | **0.42 rad**（パッドの間隔約 6 cm）。mimic の倍率 `{1, −1, 1, −1, −1, 1}` |
| wrist-up の枝の初期値 | `[−64, −122, 117, 95, 90, 116]°` |
| 当たり判定 | 肩・上腕・前腕・手首 1・手首 2 は凸分割（`<名前>_decomposed`、32 個）。元の凸包は無効化して残してある。土台・手首 3 は 1 個のまま |
| 衝突させない組（`UrdfRobot.collisionExceptions`） | 左右とも `inner_knuckle–outer_knuckle`、`inner_knuckle–inner_finger`、`inner_knuckle–robotiq_arg2f_base_link`（6 組） |
| 自己衝突 | 実行中はロボットの当たり判定どうしを全部無視（`ignoreRobotSelfCollision`） |
| 見た目のメッシュ | `.dae` 7 個は Read/Write on（CPU 描画で頂点を読むため） |

UR5e の寸法（公式 xacro、kinetic-devel から展開）：

| 項目 | 値 |
|---|---|
| d1（shoulder 高さ） | 0.163 |
| shoulder_offset / elbow_offset | 0.138 / −0.131 |
| upper_arm 長 / forearm 長 | 0.425 / 0.392 |
| wrist_1 / 2 / 3 | 0.127 / 0.100 / 0.100 |
| 質量 | base 4.0 / shoulder 3.7 / upper 8.393 / forearm 2.275 / wrist1 1.219 / wrist2 1.219 / wrist3 0.1879 kg |
| effort / velocity | 腕 3 軸 150 / 3.14、手首 3 軸 28 / 6.28 |
| tool0 | wrist_3 から xyz (0, 0.1, 0)、rpy (−π/2, 0, 0) |

### 5.2 物体

| | Target | SecondObject（Task B だけ） |
|---|---|---|
| 形 | 立方体 8 cm | カプセル、半径 0.048 m（直径 9.6 cm）、**高さはシーンごと 24〜40 cm**（パイロットは 3 シーンとも 32 cm） |
| 位置 | **seed で自動**：x −0.15〜0.15、z 0.22〜0.42、底面が天板にぴったり（y 中心 0.813） | **seed で自動**（§6.2）。y 中心 = 0.773 ＋ 高さ / 2 |
| 向き | yaw 0〜90°（自動） | 立てる |
| 質量 | 0.3 kg | 0.55 kg（高さを変えても質量は変えていない） |
| 色 | 赤 `TargetRed.mat`（HDRP/Lit） | 青 `SecondBlue.mat` |
| 衝突判定 | ContinuousSpeculative | 同左 |
| 摩擦 | Unity 既定（静・動 0.6） | 同左 |
| contactOffset | — | 0.001 m（実行中に設定） |

- 机の上面は y = 0.773（机 y = 0.748、厚さ 0.05）。机の範囲は XZ で x −0.5〜0.5、z −0.15〜0.65
- シーンファイル上の SecondObject は (0.20, 0.913, 0.10)・高さ 28 cm（P4-A のまま）。**自動生成が on のときは Play 中に上書きされる**
- Task A では、SecondObject を GameObject ごと無効化する

### 5.3 ラベル（Perception）

| 対象 | ラベル | semantic 色 |
|---|---|---|
| Table | `table` | `#808080` |
| Target | `target_object` | `#FF0000` |
| Secondary | `secondary_object` | `#FFFF00` |
| 腕＋付け根（UR5e は `world` に付けた） | `arm` | `#00FF00` |
| グリッパー（UR5e は `wrist_3_link` に付けた） | `hand` | `#0000FF` |
| 背景（Sky） | — | `#000000` |

- 見た目のパーツ 38 個すべてにラベルが付いている（arm 26・hand 12）
- instance の色は実行ごとに変わる → Python で決め打ちしない。`frame_data.json` の `instances` から引く
- Semantic Label Config は一度中身が消えた前歴あり。`.asset` を必ず commit する

### 5.4 カメラ（UR5e でも動かさずに確定）

| 項目 | 値 |
|---|---|
| position | (−0.031, 1.105, 1.083) |
| rotation | (10, 180, 0)（10° 見下ろし） |
| FOV | 60°（縦） |
| 解像度 | 256 × 256 |
| K | fx = fy = 221.70、cx = cy = 127.5 |
| Capture Trigger Mode | Manual |
| Labeler | Semantic / Instance / Depth（z-depth） |

- 4 候補（A 水平、B 45° 近め、C 35° 遠め、D 50°）を比べて、腕全体が写る A の位置を採用。角度は 0/10/15/20° を比べて 10°
- 解像度を 512 にしない理由：学習コストが増える（CNN 約 4 倍、attention 最大 16 倍）
- FOV 60° を変えない理由：上下ギリギリで、狭めると腕か机が切れる
- UR5e にしたとき、カメラを引く代わりに初期姿勢を低くして全部画面に入れた（§9.5）。カメラを引くと物体が小さく写り、mask IoU の評価に不利だから
- 行列の検証：Target の 8 頂点の投影と mask のズレは 1 px 以内

### 5.5 描画設定（決定性のため）

| 場所 | 項目 | 値 |
|---|---|---|
| Camera > Rendering | Post Anti-aliasing | なし |
| Camera > Rendering | Dithering | オフ |
| シーン Volume | SSAO Intensity | 0（HDRP Global の Default Volume にある SSAO 0.5 ＋ Temporal Accumulation を上書き） |
| シーン Volume | Exposure | Fixed、EV 14 |
| シーン Volume | Fog | コンポーネントごと無効 |

- それでも、遠くの空や机に 2 階調以下の横線状の差が出る（HDRP の間接光のゆれと思われる）。A/B の比較ではこれを許している（§6.8）

### 5.6 物理

| 項目 | 値 |
|---|---|
| dt | 0.02 s（FixedUpdate、50 Hz） |
| 重力 | −9.81（ロボットの ArticulationBody だけ重力オフ） |
| **ソルバー** | **TGS**（`m_SolverType: 1`）。P4-B で PGS から変更 |
| Enhanced Determinism | on（単体では効果なし。A/B が一致したときの設定なので残している） |
| broadphase | Sweep and Prune（`m_BroadphaseType: 0`） |
| ソルバーの反復 | 6 / 1（既定） |
| 既定の contactOffset | 0.01（ロボットは実行中に 0.002、円柱は 0.001 に上書き） |
| ドライブ（`DisableRobotGravity`、Play 開始〜実行の前） | 腕 **20000 / 200 / 5000**（stiffness / damping / forceLimit）。P5 で 10000 / 100 / 1000 からそろえた |
| ドライブ（`EpisodeRecorder.drives`、実行中） | 腕 20000 / 200 / 5000、指 1000 / 10 |
| 物理シーン | **候補ごとに新しい物理シーン**（`LocalPhysicsMode.Physics3D`）を作って、その中で回す（§6.5） |

- 元の URDF のドライブは stiffness 0。実行前に必ず設定する
- ロボットのリンクに衝突コールバック付きのスクリプトを後から付けると、PhysX が落ちる（§10）

### 5.7 座標系と規約（全ファイル共通）

| 対象 | 規約 |
|---|---|
| world | Unity world：左手系、Y 上、m |
| camera | OpenCV：x 右、y 下、z 前、m（`T = diag(1,-1,-1,1) · worldToCameraMatrix`） |
| pixel | OpenCV：(0,0) は左上ピクセルの中心、u 右、v 下 |
| 関節角 | `ArticulationBody.jointPosition`（rad）。符号 +1 を FK で確認（Play 開始時に自動） |
| 回転 | quaternion xyzw |
| 押す方向 | XZ 平面で、+Z から +X の向きに測った角度（°） |
| depth（Perception の EXR） | 32bit float、R チャンネル（cv2 で読むと BGRA 順で **index 2**）、z-depth、m、背景 0。`OPENCV_IO_ENABLE_OPENEXR=1` を cv2 の import 前に |
| depth（action の本描画） | 16bit PNG、カメラの z、**mm**、0 = 何もない |

---

## 6. データ生成パイプライン（今の仕様）

### 6.1 Play 1 回の流れ（`EpisodeRecorder`、GameObject `EpisodeController`）

```
Play
 ├─ 30 フレーム待つ（framesBeforeStart）
 ├─ 運動学（FK/IK）とグリッパーの頂点・描画用メッシュを作る   ← 指を閉じた結果が反映されてから
 ├─ autoScene on → SceneGenerator で seed からシーンを決める（§6.2）
 │     ターゲットの位置・向き → ゴール → 16 候補 → 円柱の位置・高さ
 ├─ 物体を落ち着かせる（物理 100 step、ズレ ≤ 1 mm、速さ ≤ 1 mm/s）→ 固定
 ├─ variant ごと（A → B）に：
 │     SelectVariant（A は SecondObject を無効化）→ 1 フレーム待つ
 │     初期シーンを撮る（ロボットあり → robot-free）、camera.json・scene_initial.json
 ├─ FK の検証 → 候補を 1 回だけ計画（PlanCandidates）
 ├─ 同じ計画を A・B の両方に書き出す（WriteCandidates：planned と action 表現）
 ├─ 実行：A の 16 候補 → B の 16 候補
 │     候補ごとに：新しい物理シーンへ移す → リセット → 実行 → 50 step 追加記録 → 最終画像 → 元のシーンへ戻す
 └─ 片付け（SecondObject を戻す、物体の固定を解く）
Play を止める（SOLO の annotation_definitions.json はここで書かれる）
 → build_metadata.py → validate_episode.py → check_task_pair.py → summarize_labels.py
```

- 1 シーン（A/B で 32 episode）で約 10 分
- scene ID：A = `sceneIndex`、B = `sceneIndex + 1`（`scene_0000` / `scene_0001`）。`pair_id` = `pair_XXXX`（`pairIndex`）
  - `scene_0000A` のような ID は使わない。番号を数値として扱う処理で壊れるため

### 6.2 シーンの自動生成（`SceneGenerator.cs`、結果は `scene_generation.json`）

| 項目 | 決め方 |
|---|---|
| ターゲット | x −0.15〜0.15、z 0.22〜0.42、yaw 0〜90°。箱の 8 つの角が、画面の端から 5% 以上内側に写ること |
| ゴール | ターゲットから 6〜12 cm。方向は「ターゲットの面の法線 ± 10°」で、計画できる方向。半径 3 cm |
| matched pair（2 組 = 4 本） | wrist-down と wrist-up。同じ手先経路を手先の直線で transfer（`cartesianTransfer`）。どちらかが計画できなければ `gripperYawFlip`（接近方向まわりに 180°）で計画し直す |
| ゴール方向（6 本） | ゴールの方向 ± 10°、距離 = ゴールまで ＋ 1.5 cm ± 3 cm |
| ランダム（3 本） | 方向ランダム、距離 5〜12 cm |
| 円柱の方向（3 本） | 円柱を置いたあとに作る。ターゲット → 円柱 ± 10°、距離 = 届く距離 ＋ 3〜6 cm（最大 22 cm） |
| 候補の並び | c000〜c003 = mp0（wrist-down, wrist-up）・mp1（同）、c004〜c009 = ゴール方向、c010〜c012 = ランダム、c013〜c015 = 円柱の方向 |
| 外す条件（`reject_reasons`） | `near_base`：押し始めが土台から 25 cm 未満／`target_end_off_table`：押した後のターゲットが机の端から 6 cm 以内／`ik_failed`／`joint_step`：1 step で関節が 2° 超／`secondary_too_far` |
| 試す回数 | シーン 20 回まで、候補 300 回まで |

円柱の置き場所（matched pair のどれか 1 組について「片方の腕だけが当たる」場所を探す）：

| 条件 | 値 |
|---|---|
| 当たる側の腕の重なり | 3〜10 mm（狙いは 6.5 mm） |
| もう片方の腕との隙間 | 3 cm 以上 |
| 手（tool0・指）との隙間 | 2 本とも 2.5 cm 以上 |
| ターゲットの通り道との隙間 | 3 cm 以上 |
| ゴールとの距離 | 6 cm 以上 |
| ほかの候補が当たると予測される数 | 3 本以下（`maxOtherRobotHits`） |
| 高さ | 24〜40 cm を 2 cm 刻みで、低い方から試す |
| ターゲットに近い所を優先 | 15 cm より遠いと 10 cm ごとに 5 mm 分の減点（`secondaryNearTargetWeight` 0.05） |
| 机・画面 | 机の端から余裕、画面に写る |

- 計算方法：計画した各リンクの姿勢（2 step おき）× 当たり判定の頂点
  - 1 cm の XZ 格子で候補を絞る
  - 良さそうな所だけ「頂点とカプセルの正確な距離」で測る
  - 重なりが 6.5 mm になるように中心をずらして測り直す
- 生成は Unity のメインスレッドで回る。その間、MCP が応答しなくなる（待てば終わる）

### 6.3 運動学と軌道（`ArmKinematics.cs`・`TrajectoryPlanner.cs`）

- FK は自作：`子の姿勢 = 親 × parentAnchor × Rx(q) × anchor⁻¹`
  - 検証結果：最大誤差 0.002 mm、符号 +1
- IK：減衰最小二乗、数値ヤコビアン、6 DoF、回転の重み 0.2
  - 前の step の解から続けて解く（途中で枝が変わらない）
- 軌道の形：

```
start → transfer（3 s、関節空間の min-jerk。matched pair は手先の直線＋slerp）
      → descend（押す位置の真上から直線で降りる）
      → push（平均 5 cm/s）
      → lift
      → hold（0.5 s）
```

- 1 候補 約 480 step（例：483 step ＝ 9.7 s）＋ 計画終了後 50 step

Planner の設定（現在値）：

| 項目 | 値 | 意味 |
|---|---|---|
| approachHeight | 0.16 m | 押す位置の上の高さ |
| preContactGap | 0.05 m | 押し始めの手前の隙間 |
| contactPadding | 0.005 m | 形から決めた接触距離に足す余裕 |
| contactFromGeometry | on | グリッパーの当たり判定の形から、押し始めの距離を計算する |
| pushHeightOffset | 0 | |
| transferDuration | 3.0 s | |
| moveSpeed / pushSpeed | 0.10 / 0.05 m/s | |
| holdDuration | 0.5 s | |
| gripperYawDeg | 90 | 指の開閉方向を、押す方向と直角にする（2 本の指の面で押す） |
| gripperPitchDeg | 0 | 傾けない |

- ターゲットは向きのある箱として扱う（`Support()`）。前は AABB だったので、回した箱だと押し始めがずれた

### 6.4 action 表現（`ActionRepresentations.cs`・`RobotRasterizer.cs`、`actions_meta.json` は v2）

| 表現 | ファイル | 中身 | 頻度 |
|---|---|---|---|
| EEF-Numeric | `eef_numeric.csv` | `step,t,phase,tcp_x/y/z,rot_x/y/z/w,gripper_open` | 50 Hz 全 step |
| Full-Numeric | `fullbody_numeric.csv` | 関節角 6（`q_*`）＋ 9 点（base_link〜tool0・tcp）の位置 `_x/_y/_z` ＋ 向き `_rx/_ry/_rz/_rw` | 50 Hz 全 step |
| EEF-Image | `eef_raster/step_XXXX.png`（mask）＋ `eef_depth/step_XXXX.png`（depth） | グリッパーのメッシュだけ（腕の chain の外のリンク、30,386 三角形） | **25 枚**（最初と最後を含めて等間隔） |
| Full-Image | `fullbody_raster/`＋`fullbody_depth/` | ロボット全体の見た目のメッシュ（152,175 三角形） | 25 枚 |

- 描き方：計画した関節角で、**CPU の z-buffer** で描く
  - 画素の中心で描く。depth は 1/z で補間する
  - 物体は描かない（物体による隠れもない）
- 形式：mask は 8bit PNG（255 = ロボット）、depth は 16bit PNG（カメラ z、mm、0 = 何もない）。256 px
- GPU（HDRP）の描画にしなかった理由：depth を取るのが大変。CPU なら計画の段階で描けて、結果がカメラの幾何だけで決まるので検証しやすい
- 検証：step 0 の全身の描画と、Perception が撮った初期画像（ロボットあり）を比べた → IoU 0.997、depth の差の中央値 0.25 mm
- 読み込み：mask＋depth × 2 表現 × 25 枚 = 100 ファイルで 23 ms（Windows、cv2）
  - 前の EXR は depth 99 枚で 240 ms だった
- 1 episode 約 1.7 MB
- 入力サイズ：
  - 保存は 256 px・25 枚
  - 学習の初期設定は 128 px・25 枚にする予定（P1.5 の「軽量」）。縮めるのは後でできる

### 6.5 実行と再現性

- 候補ごとに：
  1. `CreateScene(LocalPhysicsMode.Physics3D)` で新しい物理シーンを作る
  2. 移す前に、ロボットのリンクの Transform を「`Awake` で覚えた、シーンに置いたときの値」へ直接戻す
     - ArticulationBody は、移したときの Transform から作り直されるため
  3. ロボットと `Objects` をそのシーンへ移す
  4. 自己衝突の無視・contactOffset をかけ直す
  5. 全関節（指も）の位置・速度・力・ドライブの目標を、候補の開始姿勢に戻す
  6. 物体は落ち着いた位置に戻す
  7. `FixedUpdate` で `PhysicsScene.Simulate(dt)` を回す
  8. 終わったら元のシーンへ戻して `UnloadSceneAsync`
- 結果：
  - 同じ候補を何回回しても、全 step が 1 ビットも同じ
  - Play をまたいでも同じ
  - TGS と合わせると、円柱に触れていない候補は A/B の最終位置の差が 0.000 mm
- テスト用：`debugOnlyCandidates`（書いた候補だけ実行。本番では空にする）

### 6.6 記録とラベル

記録（`executed/`）：

| ファイル | 中身 |
|---|---|
| `executed_trajectory.csv` | 指令の関節角、実測の関節角・速度、実測 TCP pose、各リンクの pose、各物体の pose と速度。行 i ＝ planned_step i の指令で物理 step i を回した後、t = (i+1) × dt |
| `contacts.csv` | `step,t,event,body,body_kind,other,other_kind,num_points,px,py,pz,nx,ny,nz,impulse,rel_speed,min_separation` |
| `executed_meta.json` | ドライブ設定、リセット誤差、追従誤差、接触のまとめ（`contact_summary.robot_table` を含む） |

- `ContactRecorder` は**物体側だけ・`Start` で付ける**（ロボットに付けると PhysX が落ちる）
- ロボット↔机は、物理 step ごとに `Physics.ComputePenetration` で重なりを調べる（`world`・`base_link` は数えない）

ラベル（`final/final_state.json` の `labels`）：

| ラベル | 定義 |
|---|---|
| `goal_reached` | ターゲットの最終中心がゴール中心から XZ で半径 3 cm 以内、**かつ** 倒れていない、**かつ** 速さ < 5 mm/s |
| `goal_distance_m` | XZ の距離 |
| `target_fell` | 最終の高さが初期より 2 cm 以上低い、**または** 上向きの軸の傾きが 30° 超 |
| `target_tilt_deg`・`target_final_speed_mps` | 判定に使った値 |
| `target_contact`・`first_robot_target_step`・`robot_target_contact_rows` | ロボット→ターゲットの接触 |
| `secondary_applicable` | Task A は false |
| `secondary_collision` | ターゲットが secondary に触れた、またはロボットが secondary に触れた。**隙間 ≤ 1 mm の接触だけ数える**。Task A は null |
| `first_target_secondary_step`・`first_robot_secondary_step` | 初回の step（なし = −1、Task A は null） |
| `robot_table_contact`・`first_robot_table_step` | ロボット（土台以外）と机の重なり |

- `label_params`：`fall_drop_threshold_m` 0.02、`fall_tilt_deg` 30、`goal_max_speed_mps` 0.005
- 最終画像：50 step 後に物体を固定して撮る（ロボットあり・robot-free）。`target_mask.png` は 8bit（255 = ターゲット）
- contact heatmap：`final/contact_heatmap_target.png`・`contact_heatmap_secondary.png`・`contact_heatmaps.json`
  - 触れた接触点をカメラに投影し、ガウス（σ = 2 px）で足して、最大 = 255 にする
- 注意：円柱が倒れたあと、ロボットやターゲットに当たり直すことがある。初回接触より後のラベルには「倒れた結果」が混ざる

### 6.7 出力フォルダ

```
Episodes/
├─ scene_0000/                     （Task A、pair_0000）
│   ├─ metadata.json               （ID、seed、単位、ロボット、カメラ、物理、planner、判定基準、バージョン、git、全パス）
│   ├─ validation.json
│   ├─ camera.json
│   ├─ scene_initial.json          （task ブロック：task_variant、pair_id、paired_scene_id、secondary_present など）
│   ├─ scene_generation.json       （seed、ターゲット、ゴール、円柱、試行回数、外した理由、設定）
│   ├─ candidates.json             （16 候補：押す方向・距離、matched_pair_id、branch、gripper_yaw_flip、FK 検証）
│   ├─ solo_annotation_definitions.json
│   ├─ initial/
│   │   ├─ robot_free/  rgb.png, depth.exr, semantic.png, instance.png, frame_data.json
│   │   └─ with_robot/  （同じ 5 種類）
│   └─ candidates/c000 … c015/
│       ├─ episode.json
│       ├─ planned_trajectory.json / .csv
│       ├─ actions/  actions_meta.json, eef_numeric.csv, fullbody_numeric.csv,
│       │            eef_raster/, eef_depth/, fullbody_raster/, fullbody_depth/（各 25 枚）
│       ├─ executed/ executed_trajectory.csv, contacts.csv, executed_meta.json
│       └─ final/    final_state.json, target_mask.png, contact_heatmap_target.png,
│                    contact_heatmap_secondary.png, contact_heatmaps.json, robot_free/, with_robot/
├─ scene_0001/                     （Task B、pair_0000。同じ構成）
├─ pairs/pair_0000/  pair_check.json, comparison.md, initial_robot_free.png, cXXX_final.png
├─ label_summary.md / .json
├─ _pipeline_log.txt
├─ _archive/                       （古いデータ。§13）
└─ _ur5e_test/                     （テスト出力。§13）
```

学習コードからの読み込み：

```python
from validate_episode import load_episode, find_pair, load_pair_episode
ep = load_episode("Episodes/scene_0001/candidates/c000")
ep["initial_rgb"]          # robot-free 初期画像（モデル入力）
ep["eef_raster"]           # (25, 256, 256) uint8 mask
ep["eef_depth"]            # (25, 256, 256) uint16 mm
ep["fullbody_raster"], ep["fullbody_depth"]
ep["eef_numeric"], ep["fullbody_numeric"]
ep["final"]["labels"]      # goal_reached など
ep["final_mask"], ep["heatmap_target"], ep["heatmap_secondary"]
ep["task_variant"], ep["pair_id"]
find_pair("Episodes", "pair_0000")                   # {"A": scene_0000, "B": scene_0001}
load_pair_episode("Episodes", "pair_0000", "B", "c001")
```

### 6.8 自動検証（Python）

| スクリプト | 何を確かめるか |
|---|---|
| `build_metadata.py` | metadata・episode.json を作る。git hash と dirty を記録。SOLO の定義ファイルをコピー。欠けたファイルを数える |
| `validate_episode.py` | 1 シーン分。下の表 |
| `check_task_pair.py` | A/B ペア：planned・action（depth も）がバイト単位で同じ／条件（カメラ・ターゲット・初期関節角・ゴール）が同じ／初期画像の差が secondary だけ（semantic・depth は secondary の画素の中だけ、RGB は secondary から 40 px より外で 3 階調以下）／結果の比較。触れていないのに A/B が 1 mm 超ずれた候補を `sim_unstable_candidates` に書く（FAIL にはしない） |
| `summarize_labels.py` | ラベル分布と matched pair の差（TCP・向き・前腕・手首 1）を集計 |

`validate_episode.py` の主な中身：

- ファイルがそろっているか、枚数・大きさ・時系列が planned と一致するか
- カメラ行列で、ターゲットの投影と mask が一致するか（初期 IoU・最終 IoU。円柱に隠れうる範囲は除く）
- ラベルを最終 pose・contacts から計算し直して一致するか（傾き・倒れた・速さ・goal・触れた接触だけ・Task A は null）
- 本描画：
  - mask = (depth > 0)
  - グリッパー ⊂ 全身、全身の depth ≤ グリッパーの depth
  - TCP の近くにグリッパーがある（許容は 5 px ＋ パッドの間隔の半分を投影した長さ）
  - step 0 の全身の描画と Perception の画像の IoU・depth の差（円柱に隠れうる範囲は除く）
- heatmap を contacts.csv から作り直して一致するか、空でない ⇔ ラベルが True か
- ロボット↔机のラベルと `executed_meta` が一致するか
- task の整合（scene / metadata / candidates / episode / final_state で `task_variant`・`pair_id` が同じ）

### 6.9 量産の手順（P6：バッチ実行）

- 仕組み（`Assets/Editor/BatchRunner.cs`・`Assets/Scripts/BatchJob.cs`・`Assets/Python/run_pipeline.py`）：

```
BatchRunner（Editor、EditorApplication.update で動く）
 ├─ Temp/p6_batch/job.json を書く（seed・sceneIndex・pairIndex・outputRoot）→ Play
 ├─ EpisodeRecorder.Start が job.json を読んで消す → Play 中のインスタンスだけ値が変わる（シーンファイルは変わらない）
 ├─ EpisodeRecorder が終わったら Temp/p6_batch/result.json（ok / stage / reason）を書く
 ├─ BatchRunner が result を見て Play を止める → 3 秒待つ（SOLO の定義ファイル）
 ├─ ok → run_pipeline.py を別プロセスで起動（build_metadata ×2 → validate_episode ×2 → check_task_pair → summarize_labels）
 │        → <root>/_batch/logs/pair_XXXX.txt / .json
 └─ 失敗（生成失敗・Error Pause・タイムアウト）→ 途中のフォルダを <root>/_failed/<jobId>/ へ移す → 次の seed（scene・pair の番号は詰める）
```

- 状態は `Temp/p6_batch/state.json`（Play の開始・終了のドメインリロードをまたいで続く。Unity を閉じると消える）
- ログ：`<root>/_batch/batch_log.txt`、終わったら `<root>/_batch/batch_summary_<日時>.json`
- 使い方：
  1. commit して clean にする（`requireCleanGit` が true だと、dirty なら始まらない）
  2. `Episodes/_batch_config.json` を書く（無ければ `Tools > P6 Batch > Start` で雛形ができる）
  3. `Tools > P6 Batch > Start`
  - 止める：`Stop after current`（今のシーンの後）／`Abort now`（すぐ）。状態：`Status`
- 設定（`Config`）：`outputRoot`（相対 or 絶対）、`numScenes`（成功させるペア数）、`seedStart` か `seeds`、`maxSeedAttempts`（0 = numScenes×2＋5）、`sceneIndexStart`・`pairIndexStart`、`playTimeoutMin` 30、`pythonTimeoutMin` 30、`pythonExe`（Python311）、`requireCleanGit`、`runPython`
- 安全装置：出力先に同じ番号の `scene_XXXX`・`pairs/pair_XXXX` があると始めない（上書きしない）。`debugOnlyCandidates` が空でないと始めない
- EpisodeRecorder の変更：scene フォルダはシーンの生成に成功してから作る。生成失敗は `LogWarning`
- 実測：1 シーン（32 episode）Play 約 8 分＋Python 約 10 秒。1 万 episode なら約 42 時間
- 検証（git `198d0d6`、clean）：`_ur5e_test/p6_accept` に seed 1〜3 を無人で連続 → 3/3 PASS（A 580/580・B 564/564・ペア 107/107）。executed と labels はパイロット（`f46109c`）と 96/96 で 1 ビットも同じ
- 失敗の経路は、タイムアウト 1 分で確認（`_ur5e_test/p6_failpath`：途中のフォルダを `_failed/` へ移して次の seed へ、上限で終了）。生成失敗（stage = generate）は同じ経路を通るが、実際に失敗する seed ではまだ試していない

---

## 7. 今のパイロット（git `f46109c`、作り直し 5 回目）

### 7.1 シーン

| pair | scene（A / B） | seed | ターゲット (x, z)・向き | ゴール（方向・距離） | 円柱 (x, z)・高さ | 重なる枝・深さ | ほかの候補への当たり予測 |
|---|---|---|---|---|---|---|---|
| pair_0000 | 0000 / 0001 | 1 | (−0.117, 0.313)・22.4° | −64.5°・8.6 cm | (0.125, 0.310)・32 cm | mp0 wrist_down・6.1 mm | 0 |
| pair_0001 | 0002 / 0003 | 2 | (−0.029, 0.253)・69.4° | 74.7°・6.2 cm | (0.136, 0.161)・32 cm | wrist_down・6.1 mm | 3 |
| pair_0002 | 0004 / 0005 | 3 | (0.059, 0.393)・26.4° | 125.4°・8.1 cm | (0.276, 0.235)・32 cm | wrist_up・6.2 mm | 1 |

- 生成はどれも 1 回目で成功（seed 1：計画した候補 22、外した 5 ＝ near_base 3・ik_failed 2）

### 7.2 検証

- 3 シーンとも：A 580/580、B 564/564、ペア 107/107
- 不安定 0、欠損 0
- SOLO の定義ファイル：6 シーンにコピー済み

### 7.3 ラベル

| 候補の種類 | 本数 | goal True | 衝突（Task B） | 倒れた |
|---|---|---|---|---|
| matched pair | 12 | 4 | 3（3 組とも片方だけ ✅） | 0 |
| ゴール方向 | 18 | 14 | 0 | 0 |
| ランダム | 9 | 0 | 4（ロボット → 円柱） | 0 |
| 円柱の方向 | 9 | 0 | 9（ターゲット → 円柱） | 0 |
| **合計** | **48** | **18（38%）** | **16（33%）** | **0** |

- goal True は Task A・B とも 18/48（Task B で衝突した候補は、Task A でも goal に届いていなかった）
- ロボット↔机 0、追従誤差（TCP）最大 2.8 mm
- matched pair：
  - TCP の差 最大 0.062〜0.141 mm、向きの差 0.0°
  - 前腕の差 平均 19.5〜24.0 cm、手首 1 の差 平均 20.0 cm
  - 片方だけ衝突するのは、各シーン mp0 の 1 組だけ（mp1 は当たらない）
- 例（seed 1、Task B）：
  - c000（mp0 wrist-down）：ロボット → 円柱 step 216
  - c013〜c015（円柱の方向）：ターゲット → 円柱 step 409〜419

### 7.4 気になっている偏り（量産で見る）

1. 「片方だけ衝突」の matched pair は 1 シーン 1 組だけ
2. ランダム方向の候補は goal True が 0
3. ゴール方向の候補は衝突 0、円柱の方向の候補は goal 0 → 候補の種類とラベルがほぼ 1 対 1。モデルが「方向」だけで当てられてしまう可能性
4. TGS にしてから、追従誤差の最大が 0.3 mm → 2.8 mm に増えた
5. 円柱の高さが 3 シーンとも 32 cm（24〜40 cm のうち、低い方から試して最初に見つかった高さ）

---

## 8. スクリプト一覧

Unity（`Assets/Scripts/`）：

| ファイル | 行数 | 役割 | 状態 |
|---|---|---|---|
| `EpisodeRecorder.cs` | 1433 | 本体（§6.1・§6.5・§6.6） | 有効 |
| `SceneGenerator.cs` | 542 | シーンの自動生成（P4-B で新規） | 有効 |
| `RobotRasterizer.cs` | 116 | メッシュの CPU 描画（P4-B で新規） | 有効 |
| `TrajectoryPlanner.cs` | 270 | 候補の指定・軌道の生成。yaw・pitch・flip、形から決める押し始め、向きのある箱 | 有効 |
| `ArmKinematics.cs` | 293 | FK・数値 IK・腕の点・`GripperPointsLocal`・`LinkNames`・`LinkPoses` | 有効 |
| `ActionRepresentations.cs` | 237 | 4 種の action 表現（v2） | 有効 |
| `ContactRecorder.cs` | 53 | 接触の記録（物体に付ける）。`min_separation` | 有効 |
| `CameraExporter.cs` | 153 | カメラ行列の書き出し（Play ごとに `CaptureLogs/` へ書く） | 有効 |
| `RobotInitialPose.cs` | 61 | Play 開始時の初期姿勢・指の角度（P1 で新規、`[DefaultExecutionOrder(-100)]`） | 有効 |
| `DisableRobotGravity.cs` | 28 | 重力オフ＋ドライブ | 有効 |
| `BatchJob.cs` | 77 | P6：BatchRunner との受け渡し（job.json / result.json） | 有効 |
| `RobotFreeCapture.cs` | 248 | TODO4 のペア撮影（`freezeObjectsDuringCapture` 既定 on、UR5e 用 16 姿勢） | **無効**（検証 B をやり直すときだけ on にして、EpisodeController を off） |

Python（`Assets/Python/`）：

| ファイル | 役割 |
|---|---|
| `build_metadata.py` | metadata・episode.json・SOLO 定義のコピー（`--scene`） |
| `validate_episode.py` | 自動検証、`load_episode`・`find_pair`・`load_pair_episode`（`--scene`） |
| `check_task_pair.py` | A/B ペアの比較（`--pair`、`--shadow-px 40`、`--far-tol 3`） |
| `summarize_labels.py` | ラベル分布・matched pair の集計 → `Episodes/label_summary.md/.json` |
| `check_camera.py` | カメラ行列の検証 |
| `check_verification_a.py`・`check_verification_b.py` | 検証 A・B |
| `bench_training_cost.py`・`bench_data_loading.py` | P1.5 の計測 |
| `run_pipeline.py` | P6：1 ペア分の 4 本を順に回して、ログと結果 JSON を書く |

Editor（`Assets/Editor/`）：`BatchRunner.cs`（P6、`Tools > P6 Batch`）

`EpisodeController` の主な設定（今の値）：

| 項目 | 値 |
|---|---|
| `sceneIndex` / `seed` / `pairIndex` | 0 / 1 / 0 |
| `outputRoot` | 空（= `Episodes`）。テストのときだけ `Episodes/_ur5e_test/...` |
| `generateTaskPair` | on |
| `autoScene.enabled` | on（このとき `numCandidates`・`candidateSpecs`・`goalCenter` は使われない。シーンには P4-A の 8 候補が残っている） |
| `settleFixedSteps` / `framesBeforeStart` / `hideSettleFrames` | 100 / 30 / 1 |
| `validateKinematics` / `fkToleranceM` | on / 0.001 |
| `postSettleSteps` / `resetHoldSteps` | 50 / 3 |
| `actions.rasterFrames` / `useMeshRaster` | 25 / on |
| `contactHeatmapSigmaPx` | 2 |
| `ignoreRobotSelfCollision` / `freshPhysicsScenePerCandidate` | on / on |
| `contactTouchTolerance` / `robotContactOffset` / `secondaryContactOffset` | 0.001 / 0.002 / 0.001 |
| `goalRadius` / `fallDropThreshold` / `fallTiltDeg` / `goalMaxSpeed` | 0.03 / 0.02 / 30 / 0.005 |
| `debugOnlyCandidates` | 空 |
| `robotRoot` | UR5e（空なら `ur5e_with_gripper` → `ur3_with_gripper` の順に探す） |

---

## 9. 全期間の経緯（詰まったところ込み）

各節の並び：やったこと → 詰まったところ → 決めたこと → 結果。表をもっと細かく見たいときは、`Docs/` の経緯文書を見る。

### 9.1 TODO1〜3：環境構築・基本撮影

- Unity 6 ＋ HDRP ＋ Perception ＋ URDF Importer を入れて、UR3＋グリッパー（`ur3_with_gripper`）を机の上に置いた
- Perception で RGB・semantic・instance・depth を撮れるようにした
- カメラは 4 候補・4 角度を比べて決めた（§5.4）

### 9.2 TODO4：robot-free 撮影（完了、tag `todo4-verified`）

方法：

- ロボットの全 Renderer（付け根を含む、`GetComponentsInChildren<Renderer>(true)`）を `forceRenderingOff = true` にする → 待ってから撮る → 戻す
- GameObject は非アクティブにしない。Culling Mask も使わない。影も消える

検証：

| 検証 | 中身 | 結果（UR3） |
|---|---|---|
| B | 大きく違う姿勢（5 → のちに 16）で robot-free 画像を撮って比べる | RGB・semantic・depth すべて 65,536 px 完全一致 |
| A | ターゲットを付け根の真後ろ (0, 0.7978, −0.12) に置く | ロボットありで 0 px でも、robot-free では全体（108 px）が写る。depth の穴・NaN ゼロ、16 姿勢で完全一致 |

詰まったところ：

- robot-free どうしで RGB が少しずれた
  - 主犯は SSAO の Temporal Accumulation、全体のノイズは Dithering → §5.4 の設定で解決
- semantic が全部黒になった
  - Label Config が空になっていた → 登録し直して `File > Save Project`、`.asset` を commit
- 撮影で姿勢を切り替えたとき、物体が吹き飛んだ
  - Rigidbody を付けたあと、腕が瞬間移動していた → 撮影中は物体を kinematic にする

### 9.3 TODO5：1 episode の自動生成（UR3 時代）

- Play 1 回で「1 シーン・候補 2 本」の episode フォルダができるようにした（今の §6 の原型）
- 初期化：
  - 30 F 待つ → 物理 100 step で物体を落ち着かせる → kinematic
  - 初期シーンを 2 回撮って、全桁一致（再現性あり）
- 運動学：
  - FK を自作した（最大誤差 0.001 mm、符号 +1）
  - IK は減衰最小二乗
  - TCP は指パッドの中点（tool0 から 17.7 cm）
- 軌道：
  - グリッパー下向き。指の開閉方向を押す方向に合わせて、前側の指で押す（開いた指の間に箱が入るのを防ぐため）
  - → **P4-B で、これが箱を横倒しにしていたと分かった**
- action 表現（当時）：

| 表現 | 中身 |
|---|---|
| EEF raster | TCP を白丸（半径 3 px）で描く。10 Hz、99 枚 |
| Full-body raster | 9 点を骨格線で描く。前後関係なし |
| Full-numeric | 関節角 6 ＋ 9 点の位置。向きなし |

- 候補：
  - c000：SecondObject の反対側（−X、−90°）へ 10 cm
  - c001：SecondObject の方（130°）へ 10 cm

結果：

| | c000 | c001 |
|---|---|---|
| steps | 489 + 50 | 489 + 50 |
| goal | True（9.5 mm） | False（201.9 mm） |
| 衝突 | False | True（robot→target 270、target→Second 308、robot→Second 384） |

詰まったところ：

| 症状 | 原因 | 対処 |
|---|---|---|
| approachHeight 0.12 で、指が円柱の上端（0.933 m）をかすめた。0.20 だと c001 が UR3 の届く範囲外 | — | **0.16 を妥協点にした**（今も 0.16） |
| c000 だけ追従誤差 60 mm | 押し始めが土台から 13 cm で、腕が折りたたまれて自己干渉 | −X 方向へ変更（土台から 32 cm）。**押し始めが土台に近い候補は作らない**（→ P4-B の `near_base` 25 cm） |
| raster の枚数が合わない（40/42） | 古い step 画像が残っていた | 書き出し前に `actions/` を消す |
| 実測の関節角が目標から最大 1.2° ずれる | グリッパーが机か自分に接触 | action は planned、実測は executed として別に保存 |
| PhysX クラッシュ（`PxsSolverStartTask::setupDescTask`） | シミュレーションの途中で、ArticulationBody に衝突コールバック付きスクリプトを AddComponent した | `ContactRecorder` は物体側だけ・`Start` で付ける |
| Play 中にコルーチンが止まる | 遅れてリコンパイルが走った | idle を待ってから Play |
| エディタが止まる | `Debug.LogError` で Error Pause | 計画失敗は `LogWarning` に（P4-B で対応） |
| ドライブを 10000/100 → 20000/200/5000 に上げた | 追従誤差の切り分けのため（原因は力不足ではなかった） | そのまま使っていた（P5 でそろえた） |

### 9.4 P0：TODO5 のクローズ

- 確認したら、tag `todo5-complete` が 1 つ前の commit（`9668d2d`）に付いていた
  - その commit には、古い raster を消す修正が入っていなかった
- `build_metadata.py` が、毎回 dirty と判定していた
  - 原因：`CameraExporter` が Play のたびに `CaptureLogs/camera_日時.json` を書く
  - → `.gitignore` に `/CaptureLogs/` を追加（`776fe3c`）
- 古いデータは `_archive/scene_0000_20261004_043223` に退避して、clean な commit から作り直した
  - 結果：42/42、c000 の追従誤差 最大 0.52°・TCP 5.6 mm
- Taro の判断で、`todo5-complete` を `776fe3c` に付け替えて force push
  - `main` を push、`todo6-ur5e` ブランチを作った

### 9.5 P1：UR5e＋Robotiq 2F-140（完了、tag `todo6-complete`）

#### 分かったこと

- **グリッパーはもともと 2F-140 だった**。替えるのは腕だけ
- 指パッド名・リンク名は UR5e でも同じ → 運動学の設定はそのまま使える

#### データと URDF

- 取得元：`ros-industrial/universal_robot`（kinetic-devel、BSD-3）
  - `ur5e.urdf.xacro` を読んで数値を展開した
  - メッシュ 14 個、合計 9,712,622 B
- ダウンロードは Taro が PowerShell で実行した
  - 最初の 2 ファイルが「接続が切断されました」で失敗した
  - → `[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12` で再取得
- URDF の作り方：UR3 の URDF から腕の 27 要素と `<gazebo>` を除いて、UR5e の腕を足した
  - 結果：リンク 22・関節 21、構造エラー 0
  - 関節の範囲は全関節 ±π にした（公式は肘以外 ±2π。UR3 も ±π で、IK もそれが前提のため）
- グリッパーの元データ（`.stl` 12 個）をコピーした
  - **`.meta`・`.prefab`・`.asset` はコピーしない**（GUID の重複を防ぐため）

#### import のトラブル

| # | 起きたこと | 原因 | 対処 |
|---|---|---|---|
| 1 | MCP が 4 分でタイムアウト、Unity が固まって見えた | 「Asset Not Found」ダイアログが待っていた（`.stl` を変換した `.prefab` がなかった） | Taro がダイアログで No を押して中止 |
| 2 | 予約した import が走らない | `EditorApplication.delayCall` は、Unity がバックグラウンドだと発火しにくい | 直接実行に切り替え（予約はあとで勝手に発火した → #5） |
| 3 | VHACD で NullReference | 中止した import が作った空の `.prefab` | `_import_backup/` に退避（消さない）、`CreateStlPrefab` で作り直し |
| 4 | `.prefab` が足りない | 同上 | 不足の 11 個を作った（19 個そろった） |
| 5 | 不完全なロボットがもう 1 体と、迷子の `base` | #2 の予約が後から発火した | 正しい 1 体を残して片付け |
| 6 | `CreateCollisionExceptions` で NullReference | 衝突させない組が、消した部品を指していた | URDF の `<disable_collision>` から 6 組を作り直し |

- 最終的な import は 2.4 秒。ArticulationBody 21・当たり判定 19・見た目 38

#### 当たり判定の作り直し（凸分割）

- 問題：UR5e は全リンクが凸包 1 個だった。上腕・前腕の凸包が見た目から最大 90〜97 mm 離れていた
  - → 「画像では隙間があるのに衝突 True」になる → H2・H4・H5 がずれる → **Taro の判断で今すぐ直す**
- 測るときに詰まったこと：
  - import した当たり判定は、編集モードでは物理エンジンに未登録だった（bounds ゼロ）。`ClosestPoint` が常に「内側」と答えた
  - → 新しく作った collider どうしで比べた
- VHACD は、設定を変えても 1 個にしかまとめなかった
- 採用した方法：STL の三角形を「軸方向の輪切り」か「立方体のマス目」で分けて、凸の MeshCollider にする
  - リンクごとに候補を自動で比べた

| リンク | 採用 | 個数 | 膨らみ ≥1 cm（前 → 後） | 小さい側 p99 / 最大 |
|---|---|---|---|---|
| 肩 | Z 方向 8 枚 | 8 | 29.7% → 0.1% | 3.3 / 5.5 mm |
| 上腕 | Z 方向 4 枚 | 4 | 22.5% → 0.0% | 3.2 / 5.3 mm |
| 前腕 | X 方向 8 枚 | 8 | 3.0% → 0.0% | 3.8 / 6.0 mm |
| 手首 1 | マス目 2×2×2 | 8 | 40.3% → 1.8% | 4.5 / 5.2 mm |
| 手首 2 | X 方向 4 枚 | 4 | 33.6% → 0.6% | 4.9 / 6.0 mm |
| 土台・手首 3 | そのまま | 1 | 0% | — |

- 記録：`Assets/ur5e_with_gripper/collision_parts/decomposition.json`

#### git が固まった（重要な教訓）

- 症状：MCP から実行した `git add -A` が戻らず、Unity が「Hold on (busy for 07:17)」で固まった
- 原因：標準出力を読み終わるまで標準エラーを読まない書き方だった
  - CRLF の警告で標準エラーの受け皿が満杯になり、デッドロックした
- 復旧（Taro が PowerShell で実施）：
  1. git のプロセスを止めた
  2. `.git\index.lock` を消した
  3. add → commit → push
  - 途中で、`PS C:\...>` の行ごと貼って全行エラーになった。`PS` は `Get-Process` の別名なので無害だった
- 以後：標準出力と標準エラーを同時に非同期で読む、`GIT_TERMINAL_PROMPT=0`、タイムアウトで強制終了
- **この件のあと、push は Taro が自分でやる運用になった**

#### 初期姿勢・ラベル・カメラ

- UR5e の全関節 0 は、腕が水平に約 1 m 伸びる → `RobotInitialPose.cs` を作った
  - 1 回目は `[0, −90, 0, −90, 0, 0]°`（真上に立つ形）
- 1 候補のテストで分かった問題 2 つ：
  1. semantic で腕もグリッパーも 0 px（ラベルがなかった）→ `world` に `arm`、`wrist_3_link` に `hand` を付けた
  2. EEF raster 99 枚中 18 枚が真っ白（手先が画面の上にはみ出した）
- カメラは動かさず、初期姿勢を低くする方を選んだ
  - 待機位置を 6 通り比べた → 手先 (0, 1.05, 0.15)、余白 0.126 を採用
  - 結果：**`[−63.5, −151.8, 122.3, −60.5, −90, −63.5]°`**
- 2 候補のテスト：42/42、raster も 99/99 枚すべて画面内
  - ターゲット 657 px（UR3 と同じ）

#### 検証 A・B（UR5e でやり直し）

- 検証 B 用の 16 姿勢を UR5e 用に作り直した（UR3 の姿勢だと、机にめり込むものが混ざるため）
  - 形 1：`[pan, −151.8, 122.3, −60.5, −90, pan]`
  - 形 2：`[pan, −135, 135, −90, −90, pan]`
- `check_verification_b.py` を作った
- 結果：

| 検証 | 結果 |
|---|---|
| B（solo_43） | robot-free 16 枚の差 0 px。ロボットありではロボットが 4,582〜7,461 px 写った |
| A（solo_44） | ロボットありでターゲット 0/224 px、robot-free で 224/224 px |

#### 正式データと tag

- UR3 の正式データを `_archive/scene_0000_ur3_20261004_090146` に退避した
- UR5e で作り直した：42/42
  - c000：goal True（12.7 mm）
  - c001：衝突あり
- tag `todo6-complete`（`9b6030b`）

### 9.6 P1.5：学習コストの概算

- 計算資源：
  - Radboud の Science Cluster（Slurm）に、BSc/MSc 卒論生用のアカウント `cseduproject` がある
    - 申請先：Kasper Brink（kbrink@cs.ru.nl）か、セクションの scientific programmer。指導教員を CC
  - ただし Computer Science 学科向けの記載。**AI 学科の卒論生が対象かは未確認**
  - ほかの選択肢：SURF Research Cloud（dcc@ru.nl）、RunPod（自費。最後の手段）
- 仮モデル：
  - 画像版（2.0 M パラメータ）：フレームごとの CNN → GRU → FiLM → 数値ヘッド＋画像デコーダ
  - 数値版（1.6 M パラメータ）：初期シーン CNN＋GRU
  - 30 条件を、1 条件 1 プロセスで測った
- 詰まったところ：
  - 1 条件目が終わらなかった
  - Windows の NVIDIA ドライバは、VRAM が足りないと OOM で止まらず、メインメモリに溢れて極端に遅く動き続ける
  - → `set_per_process_memory_fraction` で上限を付けた

結果（Full-Image、fp32、1 step 秒）：

| 解像度・枚数 | batch 8 | batch 16 | batch 32 |
|---|---|---|---|
| 256 px・99 枚 | OOM | OOM | OOM |
| 256 px・25 枚 | 4.1 GB・0.115 s | 6.2 GB・0.372 s | OOM |
| 128 px・99 枚 | 4.0 GB・0.117 s | 6.4 GB・0.222 s | OOM |
| 128 px・25 枚 | 1.1 GB・0.031 s | 2.1 GB・0.059 s | 4.1 GB・0.118 s |

- メモリは、batch × 枚数 × 解像度² にほぼ比例する
- AMP ではほとんど減らなかった
- 数値表現は batch 32 でも 1.3 GB 以下
- データ読み込み：
  - depth EXR 99 枚の読み込みは 240 ms。GPU の 1 サンプル処理の 20〜75 倍遅い
  - → P4-B で 16bit PNG・25 枚にした（23 ms）
- 学習時間の目安（1 万 episode × 50 epoch × 12 run）：

| 入力 | 8 GB（今の PC） |
|---|---|
| 軽量（128 px・25 枚） | 約 4 時間 |
| 中間（256 px・25 枚） | 約 13 時間 |
| 重い（256 px・99 枚） | 約 55〜70 時間 |

- A/B 別々に学習・試行錯誤込みの、中間の設定：8 GB で約 40〜52 時間
- 大学に申請するときの目安：

| 項目 | 目安 |
|---|---|
| GPU | 24 GB 以上（16 GB でも可） |
| GPU 時間 | 約 100 時間 |
| ストレージ | 100〜200 GB |
| CPU / RAM | 8 コア / 32 GB |

- 暫定方針：軽量〜中間なら手元の 8 GB で進められる。クラスタは保険として申請する
- 結果の文書：`Docs/P1.5_training_cost.md`、生データ `CaptureLogs/bench_20261004_095618.jsonl`

### 9.7 P2：Task A/B のペア

作ったもの：

- Play 1 回で A（`sceneIndex`）と B（`+1`）を作る
- 計画は 1 回だけ作って、両方で使う
- Task A は SecondObject を GameObject ごと無効化する
  - `GetComponentsInChildren<Rigidbody>()` は無効な物を含まないので、variant ごとに `bodies` を絞り直した（`SelectVariant`）
- Task A の衝突ラベルは null（対象なし）
- Python 3 本を対応させて、`check_task_pair.py` を新しく作った
  - 前のデータ（task 情報なし）も 45/45 で通る

詰まったところ：

1. **ペア比較の RGB が 22/23 で落ちた**
   - 円柱から 200 px 離れた所にも差があった（12,171 px）
   - 調べると、遠くの差は 2 階調以下の横線（HDRP の間接光のゆれ）。円柱の近くは影で最大 82 階調
   - → 判定を「40 px より外は 3 階調以下」にした → 23/23
2. **円柱に触れていない c000 でも、A/B の最終位置が 2.4 mm 違った**
   - 関節角は step 93 からずれ始めていた。同じ順番なら毎回同じ値 → 実行の履歴に依存している
   - → P4-B①で解決
3. **円柱（高さ 16 cm、上端 0.933 m）では肘・前腕に届かない**（前腕は天板から 35 cm 以上の高さを通る）
   - Taro の判断で「土台の横で上腕をかすらせる」にした
   - 当たり判定の頂点を書き出して（`collider_points_local.json`）、円柱との距離を計算した

| 位置 (x, z) | 予測 | 実際 |
|---|---|---|
| (−0.20, −0.05) | 上腕が 33 mm 重なる | 円柱が机から吹っ飛んだ（最終 y = −332 m） |
| (−0.20, −0.09) | 5.4 mm | 机から落ちた（奥の端まで 1 cm しかなかった） |
| **(−0.24, −0.03)** | 4.9 mm | c001 の上腕だけ当たり、円柱は倒れて机に残った ✅ |

4. 送った画像のコピー（`Claude outputs/`）がプロジェクト直下に置かれて dirty になった → `.gitignore` に追加

結果（pair_0000）：

- A 50/50、B 48/48、ペア 23/23
- c000 は両方 goal、c001 は B だけ衝突（上腕 step 90）
- `950bf95`・`e8e5f3f`・`15a8416`

### 9.8 P3：matched pair

IK の枝を調べた：

- 各通過点で初期値 2,592 通りから IK を解いた
- elbow-down は、押す位置（pre・end）で**肘が 0.69〜0.75 m（天板 0.773 m より下）**になる
- 成り立たせるには、手先を土台から約 0.7 m 離す必要がある
  - 机は z ≤ 0.65 m まで。カメラにも入らない

同じ手先経路を追った結果：

| 開始の枝 | 結果 |
|---|---|
| 今の初期姿勢（wrist-down） | ✅ 1 step 最大 0.78° |
| `[−64, −122, 117, 95, 90, 116]°`（**wrist-up**） | ✅ 最大 0.62°、肘の差 平均 21.1 cm |
| `[−64, −38, −122, 70, −90, −64]°`（elbow-down） | 追えるが、肘が 0.716 m（天板の下） |
| 肩を反対に回す枝（5 通り） | ❌ ±180° に引っかかって途切れる |

実装：

- `CandidateSpec` に `startBranchSeedDeg`・`cartesianTransfer`・`matchedPairId`・`branchLabel` を追加した
- `ResetScene(qStart)` で、候補ごとの開始姿勢に戻すようにした

1 ペアのテスト：

- TCP の差 最大 0.064 mm、向きの差 0°、肘の差 21.1 cm
- wrist-up にリセットした直後、指が約 3.4° ずれていた（指はリセットしていなかった）→ P4-B①で解決

片方だけ衝突する配置：

- 計算が重く、デバイスでは 3 分で打ち切られた → クラウド側で回した
- 高さ 28 cm、(0.20, 0.10) の円柱：
  - wrist-down の前腕が step 207 で接触 → 円柱が倒れてターゲットにも接触
  - wrist-up は接触なし ✅

elbow-down の別案（Taro の依頼で検討）：

- 運動学の鎖を `kin_chain.json` に書き出して、Python で再現した
  - 最初は配置を詰め込みすぎて、Unity が 60 秒で応答なし・Python は 10 分で打ち切られた
- 方法 1：土台を 15〜35 cm 下げて、奥に置く → elbow-down はめり込む。elbow-up まで当たる配置もある。35 cm 以上下げると IK が解けない
- 方法 2：グリッパーを 30°・50° 傾ける → 肘は上がらず、むしろ下がる
- 残る手は「机を大きくして、ターゲットを土台から約 0.7 m 離す」。カメラと検証 A/B のやり直しが必要

手首の反転は、押す方向 6 つのうち 5 つで作れた（−45° だけ wrist-up が途切れる）。

**決定（Taro）**：

- H5 の matched pair は「手首の反転（wrist-down / wrist-up、どちらも elbow-up）」にする
- 今のシーンのまま量産する
- Rekha には中間報告で伝える（相談の前に方針を決めた）

commit：`170044f`・`9f700db`・`d3b17aa`

### 9.9 P4-A：10/16 の最低ライン（最初の版）

- 8 候補（−90・130・180・−45・−135・150° と mp0 の 2 本）× A/B、円柱は 28 cm を (0.20, 0.10) に置いた
- 0°・90°・45° は外した（押し始めが土台に近い、関節が 2.48° 跳ぶ）
- 結果：
  - A 182/182・B 174/174・ペア 59/59
  - goal 4/8、B の衝突 3/8
  - mp0 は片方だけ衝突 ✅
- 点検で見つかったこと：
  - ゴールが固定なので、goal True が −90°・−135° 系だけ
  - 衝突は前腕・指だけ
  - 円柱が倒れたあとに当たり直す
  - **触れていない c004 で A/B が 4.8 mm ずれる**（H1・H3 のマージン 3 mm より大きい）
- → 量産の前に、ブレの修正を最優先にした
- `e0f71d3`・`14d2a6e`

### 9.10 P4-B①：シミュレーションのブレ

再現テスト：

- 1 回の Play の中で、同じ c004 を 3 回回した（`[c004, c004, c007, c004]`）
- 比べたもの：最終位置と、`executed_trajectory.csv` で最初にずれる step

| 試したこと | 同じ候補の差（最大） | 最初にずれる step | フォルダ |
|---|---|---|---|
| 元のまま | 9.2 mm | 0（1e-11） | `p4b_det1` |
| ① 指も含めて全関節をリセット | 16.6 mm | **219** | `p4b_det2` |
| ①＋ Enhanced Determinism | 効果なし | 219 | `p4b_det3` |
| ↑ エディタ再起動後 | 効果なし | 219 | `p4b_det5` |
| ①＋ 当たり判定を外して付け直す | 0.003〜6 mm | 0 | `p4b_det4` |
| ①＋ 物体だけ付け直す | 効果なし | 219 | `p4b_det6` |
| ①＋ ソルバーの反復 32/8 | 3.9〜9.4 mm | 219 | `p4b_det7_it32_8` |
| ①＋ **自己衝突を全部無視** | 13.6 mm | **288**（ターゲットに触れた後） | `p4b_v4_open_noself` |
| ↑＋ 物体・机を 1 step 外して付け直す | 効果なし | 288 | `p4b_v5`・`v6` |
| ↑＋ ロボットも付け直す | 0.003〜6 mm | 0 | `p4b_v7_recreate_all` |
| ↑＋ 候補ごとに新しい物理シーン（失敗版） | 腕が暴れる | — | `p4b_v9`・`v10` |
| ↑＋ 移す前に全関節 0 へ物理で動かす | 2〜3 mm | 0（1e-6 m） | `p4b_v11` |
| ↑＋ **移す前に Transform をシーンに置いた値へ直接戻す** | **0（全 step 同じ）** | なし | `p4b_v12` |

分かったこと：

- Play をまたぐと、最初から 1 ビットも同じになる
- **同じ Play の中で続けて回すときだけ、PhysX の中に前の候補の履歴が残る**
- step 219 のずれは、指先が机やターゲットに近づいて、新しい「接触の組」ができるあたり（推測。特定はしていない）

詰まったところ：

1. Enhanced Determinism は、Unity を再起動しても効果がなかった
2. 別のシーンに移すと、上腕の関節が −9.7 rad になり、追従誤差が 1 m 近くになった
   - 原因：ArticulationBody は、移したときの Transform から作り直される（関節の 0 がずれる）
   - → `Awake` で覚えた Transform を書き戻してから移す
3. wrist-up の計画失敗で `LogError` → Error Pause で Play が止まった（`isPaused = true`）→ `LogWarning` に
4. `OpenScene` で今のシーンを開いても、メモリ上の変更が捨てられなかった
   - テスト用の `outputRoot` が、シーンファイルに保存されてしまった
   - → `git checkout` で戻した。以後は `NewScene` → `OpenScene` で読み込み直す
5. VM から `git diff --stat` を打つと、180 ファイル変更と出た（改行・権限の差）→ git は Unity からだけ使う

採用したもの（`ab6e261`）：

- 全関節のリセット
- `ignoreRobotSelfCollision`
- `freshPhysicsScenePerCandidate`
- Transform の書き戻し
- 付け直すコードは効果がなかったので消した

### 9.11 P4-B②：押し方（1 回目）と、倒れた判定のバグ

| 押し方 | 箱の動き | 同じ候補の差 |
|---|---|---|
| 元（開いた指、開閉方向 = 押す方向、前の指 1 本の曲面） | **90° 回る** | 5〜16 mm |
| 閉じた指（0.6 rad） | 90° 回る | 1.5〜4 mm |
| 閉じた指 ＋ 手先を 90° 回す（2 本の側面） | 2〜3° | 0.05〜0.8 mm |
| ↑＋ 指を硬く（20000/200） | 3° | 0.3〜5 mm（悪化） |

- Taro の判断：「押し方を安定させる」「まず物理シーンの作り直しを試して、ダメなら Play し直し」

詰まったところ：

1. **押し始めが 10 cm ずれた**
   - `contactFromGeometry` で、前に出ている距離が 106 mm と出た
   - 原因：運動学とグリッパーの頂点を `Start()` で作っていた。指を閉じた結果が、まだ Transform に反映されていなかった
   - → `Run()` の最初に作るように移した
   - `robotiq_arg2f_base_link` という名前の Transform が複数あった → ArticulationBody を持つ方を探す
2. **大発見：P4-A の全候補で、立方体が横倒しになっていた**
   - 立方体は横倒しでも中心の高さが変わらないので、「2 cm 下がったら倒れた」では見逃していた
   - goal True の 4 件も、倒れた箱がゴールに入っただけだった
3. 閉じた指 ＋ 90° 回すでも、6 方向中 2 方向で倒れた
   - ナックルが前に出て、箱の上の方を押していた（TCP の高さ 30〜40 mm で 19〜20 mm 前に出る）
   - → 指先を 30° 傾けた → 立ったまま
4. 傾けると、wrist-up の計画が transfer の途中で失敗した → `gripperYawFlip`（180° 回す）を追加

ラベルの修正：

- `target_tilt_deg` を追加
- `target_fell` = 2 cm 下がる or 30° 超傾く
- 検証で、最終 pose から計算し直して照合する

P4-A を作り直した（`ed0426d`）：

- 古いデータは `_archive/p4a_oldpush_20261004_113316`
- 結果：A 206/206・B 198/198・ペア 59/59、goal 3/8、倒れた 0
- ただし、新しい押し方では、今の円柱の位置に mp0 のどちらも当たらない → 自動生成へ

### 9.12 P4-B③：候補の自動生成・ゴールの正式定義（`1701d06`）

Taro が決めたこと：

- ゴールはシーンごとにランダム、半径 3 cm、倒れていない・止まっている
- 16 候補 = matched pair 2 組 ＋ 単独 12
- ターゲットの位置はシーンごとに変える

詰まったところ：

1. **斜めの面を押すと、箱が約 40° それた** → ゴールの方向を「面の法線 ± 10°」にした
2. **ターゲットの向きを変えると、傾き 30° では 6 本中 4 本が倒れた**
   - 指先を前に傾けると、押す力に上向きの成分が出る
   - → 15° にした
3. **円柱の近く（2 cm 以内）を通るだけで、Task B が最大 3 mm ずれた**
   - PhysX が、円柱をロボットと同じ計算のまとまりに入れていた
   - → contactOffset をロボット 2 mm・円柱 1 mm にした → 0.000 mm
4. **「衝突」に、触れていない接近が混ざっていた**
   - PhysX は contactOffset（既定 1 cm ずつ）以内なら、隙間 > 0 でも接触として報告する
   - → `min_separation` を記録して、1 mm 以下だけ数える
   - **P3 までの衝突ラベルにも、近づいただけのものが混ざっていた可能性がある**
5. **円柱の置き場所の誤差**
   - 1 cm の格子だと最大 7 mm ずれた
   - → 格子で絞ってから、正確な距離で測り直す
6. 生成中、MCP が 60 秒で応答なしになった（待てば終わる）
7. `executed_meta.json` が壊れていた
   - 補間しない文字列に `}}` と書いていた → 直した
8. matched pair 0 組のテストで、円柱が原点へ飛んだ
   - → 置き場所が決まらないときは動かさない

結果（`8d62049`）：

- seed 1・16 候補：A 422/422・B 406/406・ペア 107/107
- goal 5/16、衝突 1/16（mp0 の片方だけ ✅）

### 9.13 P4-B④：本番の入力仕様（`acd741d`）

- 骨格線をやめて、メッシュの本描画（§6.4）にした
  - Full-Numeric に向きを追加、contact heatmap を追加、depth の比較を `check_task_pair.py` に足した
- 詰まったところ：
  1. UR5e の見た目の `.dae` が Read/Write 不可で、頂点を読めなかった → on にした（`.meta` 7 個）
  2. Perception の画像との depth の差が 980 mm と出た
     - EXR の index 0 を読んでいた。正しくは index 2（R チャンネル）
  3. VM から読み込み速度を測ると 1.6 s 出た（共有フォルダのせい）→ Windows で測り直して 23 ms
- 入力サイズは、保存 256 px・25 枚、学習は 128 px・25 枚から始める、に決めた

### 9.14 P4-B⑤：押し方（2 回目）と円柱の高さ（`7db0309`・`b3ce0da`・`09956cf`）

- 3 シーンにしたら、seed 2・3 ではゴール方向 6 本のうちゴールに入ったのが 0〜1 本
  - 押す方向から 35〜40° それていた
  - 閉じた指の接触は幅が約 5 cm と狭く、面が少し斜めだと回る

| 指 | 傾き | seed 2 の結果 |
|---|---|---|
| 半分開く（0.42 rad） | 15° | goal 5/8、倒れた 2 |
| **半分開く（0.42 rad）** | **0°** | **goal 6/8、倒れた 0、それ ≤ 18°** |
| 同上（seed 3） | 0° | goal 3/4、倒れた 0 |

- パッドの間隔約 6 cm で、箱の面（8 cm）を 2 点で押す → まっすぐ進む → 採用

詰まったところ：

1. **高さ 28 cm の円柱では「片方だけ当たる」場所が見つからない**（20 回とも失敗）
   - 理由をログに出した：片方だけの最大の重なりが −14〜−17 mm。腕の差が出る場所が、円柱の上端より上にあった
   - → 高さもシーンごとに決める（24〜40 cm）。3 シーンとも 32 cm になった
2. 置換で `s.secondaryHeightRange` まで `HRange` になって、コンパイルエラー → 直した
3. 「TCP の近くにグリッパー」の検証が失敗した
   - 指が開くと、TCP はパッドの間の何もない所に来る
   - → 許容を「5 px ＋ パッドの間隔の半分」にした
4. Task B で円柱がターゲットを隠して、最終 mask の IoU が落ちた → 隠れうる範囲を除いて比べる

3 シーンの結果：

- 検証は全部 PASS。goal 14/48、倒れた 3、衝突 3/48
- 残った問題：
  1. 衝突が少ない
  2. **円柱に触れていない候補でも、A/B が 45 本中 2 本ずれた（12.6 mm・1.1 mm）**

### 9.15 P4-B⑥：A/B のずれと衝突の少なさ（`8e986e0`・`e435a6e`・`9c76bc8`・`ca8026b`）

A/B のずれ（seed 1 の c002）：

| 試したこと | A/B の差 |
|---|---|
| 元（PGS、Sweep and Prune） | 12.6 mm |
| broadphase = Automatic Box Pruning | 12.0 mm |
| 既定の物理シーンで実行 | 3.4 mm（ただし最終位置が 8 cm 違って横倒し） |
| 指を硬く（50000/500） | 12.0 mm |
| **ソルバー = TGS** | **0.000 mm** |

- c002（mp1 の wrist-down、ロボットの方へ押す、ナックルも当たる）は、ほんのわずかな計算の違いが大きく育つ押し方だった
- 円柱に触れていないのに、円柱があるだけで計算に違いが出ていた
- TGS にした。3 シーン × 16 候補で、触れていない候補は全部 0.000 mm
- 代わりに、追従誤差の最大が 0.3 → 2.8 mm に増えた
- `sim_unstable_candidates` を追加した

衝突を増やす：

- 単独 12 本を「ゴール方向 6 ＋ ランダム 3 ＋ 円柱の方向 3」にした
- 詰まったところ：
  1. **円柱が遠すぎて、円柱の方向の候補が届かない**（`secondary_too_far` 146〜150 回）→ ターゲットに近い所を優先した
  2. **seed 3 で、円柱が腕の通り道の真ん中に立った**
     - 16 本中 15 本がロボットで当たり、mp1 は両方当たった
     - 本描画の検証も IoU 0.812 で失敗した（円柱がロボットを隠した）
     - → ほかの候補の当たり予測を 3 本以下にした。検証も隠れを考慮するようにした
  3. **検証を同期で回すと 60 秒を超えて、MCP が応答なし**
     - → 別プロセスで起動して、`_pipeline_log.txt` に書かせた

結果：goal 18/48、衝突 16/48、倒れた 0、不安定 0

### 9.16 P5：小さい修正（`f46109c`・`9f89f72`）

| 項目 | 中身 |
|---|---|
| 計画失敗のログ | `LogWarning` に（P4-B①で対応）。残っている LogError は、設定が足りなくて続けられないときだけ |
| ドライブ値 | `DisableRobotGravity` を 10000/100/1000 → 20000/200/5000 にそろえた。落ち着いた姿勢がわずかに変わり、同じ seed でも最終位置が約 3 mm 変わった → パイロットを作り直した |
| ロボット↔机 | `Physics.ComputePenetration` で毎 step 調べる |
| SOLO の定義ファイル | Play 終了時に書かれるので、`build_metadata.py` がコピーする |

詰まったところ：

- ロボット↔机が、全候補 step 0 から「接触あり」になった
  - `base_link` が天板に 2.6 mm めり込んでいた（机と同じ高さに固定しているため）
  - → `world`・`base_link` は数えない
- わざと当てるテスト（TCP を 3 cm 下げる）：step 207 から指パッドで 1.7 mm を検出した（`p5_table_test`）

その後：

- 経緯の文書 `Docs/P4_P5_経緯_20261005.md`（`cf679b3`）
- 本書（`THESIS_HANDOFF.md` 2026-10-05 版）

---

## 10. ハマりどころ（全期間、再発防止）

| 症状 | 原因 | 対処 |
|---|---|---|
| semantic が全部黒 | Label Config が空になっていた | 登録し直して `File > Save Project`、`.asset` を commit |
| ロボットのラベルが 0 px | UR5e にラベルがなかった | `world` に `arm`、`wrist_3_link` に `hand` |
| Play でロボットが倒れる | ドライブなし＋重力 | 重力オフ＋ドライブ＋ルート Immovable |
| robot-free どうしで RGB が少しずれる | SSAO Temporal Accumulation／Dithering | §5.5 |
| 撮影の姿勢切替で物体が吹き飛ぶ | 腕の瞬間移動 | 撮影中は物体を kinematic |
| Play 中にコルーチンが止まる | 遅れてリコンパイル | idle を待ってから Play |
| エディタが止まる | `Debug.LogError` で Error Pause | 続けられる失敗は `LogWarning`。止まったら `isPaused` を確認 |
| **PhysX クラッシュ**（`PxsSolverStartTask::setupDescTask`） | 実行中に ArticulationBody へ衝突コールバック付きスクリプトを AddComponent | `ContactRecorder` は物体側だけ・`Start` で。ロボット↔机は重なり判定で |
| 追従誤差 60 mm | 押し始めが土台に近く、腕が折りたたまれる | 押し始めは土台から 25 cm 以上 |
| raster の枚数が合わない | 古い step 画像が残る | 書き出し前に `actions/` を消す |
| 毎回 dirty になる | `CameraExporter` が `CaptureLogs/` に書く／`Claude outputs/` | `.gitignore` |
| 衝突 True なのに画像では隙間がある | 当たり判定が凸包 1 個で膨らんでいた | 凸分割（§9.5） |
| 同じ候補でも結果が変わる | 同じ Play の中で PhysX に履歴が残る、指を戻していない | 全関節リセット＋候補ごとの物理シーン（§6.5） |
| 別シーンに移すと腕が暴れる | ArticulationBody が今の Transform から作り直される | 移す前に Transform をシーンに置いた値へ戻す |
| 触れていない円柱で Task B がずれる | 2 cm 以内を通ると同じ計算のまとまりになる／PGS の計算の違いが育つ | contactOffset を小さく＋TGS |
| 衝突ラベルに「近づいただけ」が混ざる | contactOffset 以内なら隙間 > 0 でも接触を報告 | `min_separation` ≤ 1 mm だけ数える |
| 箱が横倒しなのに「倒れていない」 | 立方体は横倒しでも中心の高さが同じ | 傾き 30° 超も「倒れた」 |
| 箱が横倒しになる | 前の指 1 本の曲面・ナックルで押していた | 指を半分開いて、90° 回して、2 点で押す |
| 押す方向から 35〜40° それる | 閉じた指は接触の幅が狭い／斜めの面を押す | 2 点で押す＋ゴールは面の法線 ± 10° |
| 押し始めが 10 cm ずれる | 運動学を `Start()` で作っていて、指を閉じる前の形を使った | `Run()` の最初に作る |
| 円柱の置き場所が見つからない | 格子の誤差／腕の差が円柱の上端より上 | 正確な距離で測り直す／高さもシーンごと |
| 円柱が吹っ飛ぶ・机から落ちる | 重なりが大きい／机の端に近い | 重なりは数 mm、机の端から余裕 |
| ロボット↔机が常に接触あり | `base_link` が天板に 2.6 mm めり込んでいる | `world`・`base_link` は数えない |
| depth の差が 980 mm | EXR の index 0 を読んでいた | index 2（R チャンネル） |
| 頂点が読めない | `.dae` が Read/Write off | on にする |
| `executed_meta.json` が壊れる | 補間しない文字列に `}}` | `}` 1 個にする |

クラッシュログの確認：

```powershell
$d = Get-ChildItem "C:\Users\kinos\AppData\Local\Temp\Unity\Editor\Crashes" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Select-String -Path "$($d.FullName)\Editor.log" -Pattern "\[EpisodeRecorder\]|OUTPUTTING STACK|physx::" | Select-Object -Last 15 | ForEach-Object { $_.Line }
```

Unity 6 の API：`Rigidbody.linearVelocity`（`velocity` は非推奨）、`FindFirstObjectByType`、`jointAcceleration` の setter は使えない（CS0619）

---

## 11. MCP 作業のルール（全期間の教訓）

| 教訓 | ルール |
|---|---|
| Unity から外部プロセスを動かすと、出力の読み方しだいでデッドロックする | 標準出力と標準エラーを同時に非同期で読む。`GIT_TERMINAL_PROMPT=0`。タイムアウトで強制終了 |
| `EditorApplication.delayCall` は後から勝手に発火する | 予約処理は使わない |
| import 済みの当たり判定は、編集モードでは物理エンジンに未登録 | 測定は新しく作った collider か、Play 中に行う。物体の大きさは Play 中か `scene_initial.json` で確認 |
| URDF Importer は `.stl` が `.prefab` に変換済みだと思っている | 新しい `.stl` を入れたら `CreateStlPrefab` で先に変換 |
| 中止した import は、壊れたファイルや余分なロボットを残す | 保存済みのシーンを読み込み直して片付ける。壊れたファイルは退避（消さない） |
| `OpenScene` で今開いているシーンを開いても、メモリ上の変更が捨てられない | `NewScene`（空）→ `OpenScene`。commit の前に、シーンに `outputRoot` などのテスト設定が入っていないか確認 |
| Play 前に reflection や SerializedObject で設定を変えると、その Play に効く（シーンは未保存のまま） | テストは出力先を `Episodes/_ur5e_test/…` にして、終わったら読み込み直す |
| `execute_code` は CodeDom（C# 6） | タプル不可、`UnityEngine.Object` と書く |
| Unity のメインスレッドで重い処理をすると、MCP が 60 秒で応答なし | 待つか、別プロセスで起動してログをファイルに書かせる。重い計算はクラウド側で、数分で終わる量に分ける |
| `read_console` は古いログから返す | 結果はファイル（final_state.json など）で確認する |
| 物理の設定（ソルバーなど）は、候補ごとの物理シーンにも反映される | 変えたら、再起動しなくても次の Play で試せる |
| Windows の GPU は、VRAM 不足で OOM せずに RAM に溢れる | ベンチマークは `set_per_process_memory_fraction` で上限を付ける |
| VM の Python 3.10 には cv2 がない | 検証は Unity から Windows の Python 3.11 を起動して回す |
| VM から git を触ると、改行・権限で差分が出る | git は Unity からだけ起動する |
| VM の時刻は UTC | 退避フォルダ名の時刻は UTC |
| 共有フォルダ越しの VM は遅い・ファイル一覧が遅れて見える | 速度は Windows で測る。移動の直後はもう一度確認 |
| PowerShell に画面ログ（`PS C:\...>` 付き）を貼ると全行エラー | コードブロックの中身だけをコピーする |
| 送ったファイルのコピーがプロジェクト直下（`Claude outputs/`）に置かれる | `.gitignore` 済み |
| 一時的な設定変更が正式データに混ざる危険 | 出力先をテスト用にする。データは必ず clean な commit から作る（commit → Play → 止める → Python） |

---

## 12. git の記録（`todo6-ur5e`、上が古い）

| commit | 内容 |
|---|---|
| `776fe3c` | P0：`.gitignore` に `CaptureLogs/`（`main`、tag `todo5-complete`） |
| `a4f0784` | TODO6-1：UR5e 導入・当たり判定の凸分割・初期姿勢（1 回目）・UR3 無効化 |
| `35f2783` | TODO6-2：ラベル・初期姿勢 `[−63.5,…]`・カメラ据え置き |
| `9b6030b` | TODO6-3：検証 A・B（tag `todo6-complete`） |
| `4a9288f` | P1.5：ベンチマークと結果 |
| `cbf0241` | `.meta` 2 つ、`__pycache__.meta` を無視 |
| `950bf95` | P2-1：Task A/B ペアの仕組み |
| `e8e5f3f` | P2-2：円柱を土台の横へ |
| `15a8416` | `/Claude outputs/` を無視 |
| `170044f` | P3：matched pair の機能と報告 |
| `e0f71d3` | P4-A：パイロットの設定（8 候補・28 cm） |
| `14d2a6e` | P4-A：結果の記録 |
| `9f700db` | P3 追加：elbow-down の別案、他の方向 |
| `d3b17aa` | P3 決定：matched pair = 手首の反転 |
| `ab6e261` | P4-B：候補ごとの物理シーン・押し方（閉じた指、yaw 90、pitch 30） |
| `ed0426d` | P4-A：作り直し（8 候補） |
| `1701d06` | P4-B：自動生成・ゴールの正式定義・触れた接触だけ |
| `8d62049` | P4-A：自動生成で作り直し（seed 1、16 候補） |
| `acd741d` | P4-B：本番の入力仕様 |
| `7db0309` | P4-B：指を半分開く・傾き 0 |
| `b3ce0da` | P4-B：円柱の高さをシーンごと |
| `09956cf` | P4-B：3 シーンのパイロットと点検 |
| `8e986e0` | P4-B：TGS・円柱の方向の候補 |
| `e435a6e` | P4-B：円柱をターゲットの近くに |
| `9c76bc8` | P4-B：ほかの候補の通り道に立たない・隠れの考慮 |
| `ca8026b` | P4-B：パイロット v4 と記録 |
| `f46109c` | P5：ロボット↔机・SOLO・ドライブ値（**今のパイロットの元**） |
| `9f89f72` | P5：記録・パイロット v5 |
| `cf679b3` | `Docs/P4_P5_経緯_20261005.md` |
| `b2c6a16` | `Docs/THESIS_HANDOFF.md`（push 済み） |
| `198d0d6` | P6：バッチ実行 |

- 10/05 の時点で `cf679b3` まで push 済み。`main` より 28 commit 先にある
- この文書（`Docs/THESIS_HANDOFF.md`）は、その次の commit で追加する
- tag：`todo4-verified`（421a4a4）、`todo5-complete`（776fe3c）、`todo6-complete`（9b6030b）。P2 以降の tag はまだない
- 運用：
  - Claude は commit まで。**push は Taro**
  - 通常：`git push origin todo6-ur5e`
  - tag：`git push origin <tag名>`
  - 未 push の確認：`git log --oneline origin/todo6-ur5e..HEAD`
- GitHub に上げないもの：`Episodes/`・`CaptureLogs/`・`Captures/`・`Library/`・`Temp/` など
  - どの commit から作ったデータかは、`metadata.json` の git 欄で追える

---

## 13. 退避・テスト用のデータ（消していない）

`Episodes/_archive/`：

| フォルダ | 中身 |
|---|---|
| `scene_0000_20261004_043223` | P0 前の、混ざった UR3 データ |
| `scene_0000_ur3_20261004_090146` | UR3 版の正式データ（42/42） |
| `scene_0000_ur5e_preP2_20261004_014502` | P2 前の UR5e 正式データ |
| `p2_pair_0000_20261004_022510` | P2 の本番ペア |
| `p4a_oldpush_20261004_113316` | P4-A の元のパイロット（全部横倒し） |
| `p4a_pitch30_fixed8_20261004_124357` | 閉じた指・傾き 30°・固定 8 候補 |
| `p4a_auto_skeleton_20261004_132712` | 自動生成 seed 1・骨格線 |
| `p4b_closedgrip_3seeds_20261004_140158` | 閉じた指・本描画・3 シーン |
| `p4b_pgs_3seeds_20261004_153118` | 半分開いた指・PGS（A/B のずれ 2 本） |
| `p4b_tgs_far_cyl_20261004_160223` | TGS・円柱が遠い |
| `p4b_tgs_v2_20261004_164119` | TGS・seed 3 で円柱が通り道に立った |
| `p4b_v4_preP5_20261004_172613` | P5 の前 |

`Episodes/_ur5e_test/`（テスト出力）：

- P1・P2・P3：`_old_scene_0000_upright_pose`、`_p2_regress`、`p2_infra`、`p2_place1〜3`、`p3_run1・2`
- P4-B①：`p4b_det1〜7`、`p4b_v1〜v17`
- P4-B②〜⑤：`gen*`、`tip_*`、`face_*`、`grip042_*`、`input1`
- P4-B⑥：`bp_sap`・`bp_abp`、`default_scene*`、`stiff_fingers`、`tgs*`
- P5：`p5_test`、`p5_table_test`
- 書き出した資料：`collider_points_local.json`（当たり判定の頂点）、`kin_chain.json`（Python で FK を再現する用）

その他：

- `_import_backup/<日時>/`：壊れた `inner_knuckle` の `.prefab`・`.asset`
- `CaptureLogs/bench_*.jsonl`：P1.5 の生データ

---

## 14. 次にやること（優先順）

### 14.1 すぐ

- [ ] 区切りを付ける。どちらかを Taro が決める：
  - `main` を `todo6-ur5e` まで進める（fast-forward）
  - tag を付けるだけ（例：`p4-complete`・`p5-complete`）
- [ ] Rekha か Jordy に「AI の卒論生が GPU を使うならどこに頼めばいいか」を聞く
  - `cseduproject` が使えるか。申請の数字は §9.6
  - 返事に時間がかかるかもしれないので、早めに

### 14.2 量産（12/4 に向けて）

1. ~~**バッチ実行の仕組みを作る**~~ → ✅ P6 完了（§6.9）
   - seed・`sceneIndex`・`pairIndex` を変えながら、Play → 止める → Python を繰り返す
   - シーンファイルを汚さない（Play 中だけ値を変える）
   - 1 シーン約 10 分。1 万 episode で約 50 時間
2. 生成に失敗した seed の扱いを決める（飛ばす／記録する）
3. 量産したデータで偏りを点検する（§7.4）
   - matched pair の「片方だけ衝突」が 1 シーン 1 組で足りるか（H5 の検出力）
   - 候補の種類とラベルがほぼ 1 対 1 になっていないか
   - 追従誤差（TGS で 2.8 mm）が増えすぎないか
   - 円柱の高さが 32 cm ばかりにならないか
4. 学習・テストの分け方（シーン単位、matched pair は分けない）

### 14.3 10/30 の中間報告

- proposal からの変更点 6 つ（§2.10）と、その理由・証拠
- パイロットの数字（§7）
- 今後の予定（量産・学習・評価）

### 14.4 計算資源

- 手元：RTX 4060 Laptop 8 GB。軽量〜中間の入力なら回る（学習中は Unity を閉じる）
- 大学：Science Cluster `cseduproject`（AI 学科が対象か未確認）、SURF Research Cloud
- 最後の手段：RunPod（自費）

### 14.5 学習モデル（まだ何も決めていない）

- 4 表現で、action 入力の部分以外を同じにする
- 入力：robot-free 初期画像（RGB＋depth）＋ action 表現
- 出力：最終位置・goal・衝突・最終 mask・contact heatmap
- P1.5 の仮モデル（フレームごとの CNN → GRU → FiLM）は、メモリを一番食う構造。時間方向を先に縮める構造（チャンネル方向に重ねる、時間方向の stride など）なら軽くなる見込み
- 3 seed × 4 表現 × Task A/B

---

## 15. 未決の設計判断

| 項目 | 今の状態 |
|---|---|
| 学習モデルのアーキテクチャ・学習設定 | 未着手 |
| 本番の episode 数・epoch 数 | 未定（P1.5 の見積もりは 1 万 episode × 50 epoch） |
| 学習の入力サイズ | 128 px・25 枚から始める予定（保存は 256 px・25 枚） |
| 数値表現の時間方向の扱い | 全 step（50 Hz、約 480 step）のまま。間引くかは未定 |
| 候補の内訳の最終形 | 今は 2 組＋6＋3＋3。量産の点検しだいで見直す |
| `main` の扱い・tag | 未決 |
| TGS で増えた追従誤差（最大 2.8 mm）の影響 | 計画（モデルの入力）と実際の動き（ラベル）のずれ。影響はまだ評価していない。量産で増えないか見る |
| P3 までの衝突ラベル（近づいただけが混ざっていた） | 今のデータは全部新しい定義で作り直し済み。古いデータは使わない |

---

## 16. Claude への指示

- このファイルを現状の正として扱う。ここにない数値・設定・判断は推測で埋めず、Taro に聞く
- 古いテーマ（Wan2.1-VACE の Arm A/B/C、LoRA の比較）は破棄済み。今のテーマと混同しない
- 研究のゴールはロボット制御・planning。動画生成はゴールではない（optional extension のみ）
- 説明の仕方：
  - 素人前提で、結論から、手順を細かく分解して、図やイメージを使う
  - カジュアルな口調で、誇張なし、現実的な案だけ
- git：
  - Claude は commit まで。push は Taro
  - commit したら、ブランチ・commit・push のコマンドをコードブロックで渡す（「中身だけコピーしてね」と添える）
- データ：
  - 消さずに `_archive/` へ退避する
  - テストは `_ur5e_test/` に出す
  - 正式データは clean な commit から作る
- 重い計算は、数分で終わる単位に分ける
- 作業後にこのファイルを更新するときは、変更した箇所と最終更新日を書き換える。細かい経緯は `Docs/` に経緯文書を足す
