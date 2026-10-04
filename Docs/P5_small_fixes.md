# P5：小さい修正

- 日付：2026-10-05
- git：`f46109c`（コード）

| 項目 | 状態 | 中身 |
|---|---|---|
| 計画失敗のログ：LogError → LogWarning | ✅（P4-B の途中で対応） | 失敗が 1 つあると Error Pause で Play が止まっていた。残っている LogError は、設定が足りなくて続けられないときだけ |
| ドライブ値の決め直し | ✅ | 下の表 |
| ロボット↔机の接触を重なり判定で追加 | ✅ | 下の説明 |
| SOLO の `annotation_definitions.json` をコピー | ✅ | `build_metadata.py` でコピー |

## ドライブ値

| 場所 | 前 | 後 |
|---|---|---|
| `DisableRobotGravity`（Play 開始〜実行の前） | 10000 / 100 / 1000 | **20000 / 200 / 5000** |
| `EpisodeRecorder.drives`（実行中、腕） | 20000 / 200 / 5000 | 変えない |
| `EpisodeRecorder.drives`（実行中、指） | 1000 / 10 | 変えない |

- 実行中の値は前から 20000 / 200 / 5000 だった。Play を始めてから実行の前までだけ、別の値になっていた。これを実行中と同じにそろえた
- 腕の追従誤差（TCP）は最大 2.8 mm（TGS ソルバー）
- 指を硬くする（50000 / 500）のも試したけど、結果は良くならなかったので 1000 / 10 のまま（`Docs/P4B_input_spec.md` §9）
- この変更で、落ち着かせたあとの姿勢がわずかに変わる → 同じ seed でも、前のデータとは最終位置が数 mm 違う。パイロットは作り直した

## ロボット↔机の接触

- ロボットのリンクに `ContactRecorder` を付けると PhysX が落ちる。そこで、物理の 1 step ごとに `Physics.ComputePenetration` で、ロボットの当たり判定と机の当たり判定が重なっているかを調べる
- 土台（`world`・`base_link`）は数えない。机と同じ高さに固定してあって、天板に 2.6 mm めり込んでいるため
- 書き出し：
  - `executed_meta.json` の `contact_summary.robot_table`：最初の step、step 数、最大のめり込み、リンク
  - `final_state.json` の `labels.robot_table_contact`・`first_robot_table_step`
- `validate_episode.py`：ラベルと `executed_meta` が一致するかを確かめる
- わざと当てたテスト（TCP を 3 cm 下げる）：step 207 から指パッドが当たって、めり込み 1.7 mm を検出した
- 今のパイロット（96 episode）では 0 本

## SOLO の annotation_definitions.json

- Perception は、このファイルを Play が終わるときに書く。だから Play 中の `EpisodeRecorder` ではコピーできなかった
- `build_metadata.py` が `scene_initial.json` の `capture.solo_dir` からコピーする（無いときは警告を出す）
- パイプラインの順番（commit → Play → **Play を止める** → build_metadata → …）は前と同じ

## パイロット（作り直し 5、git `f46109c`）

- 3 シーン（seed 1〜3）× 16 候補 × Task A/B = 96 episode
- 検証：A 580/580、B 564/564、ペア 3 組とも 107/107、不安定 0
- goal 18/48、衝突（B）16/48、倒れた 0、ロボット↔机 0、追従誤差 最大 2.8 mm
- 1 つ前のデータ：`Episodes/_archive/p4b_v4_preP5_*`
