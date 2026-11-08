# Session log — 2026-11-07 to 2026-11-09

This log covers one working session: from re-training on a cloud GPU to the final, one-time test evaluation of Phase 2. It records what was done, in what order, which problems came up, and what remains.

## 0. Starting point

- **Last week's state** (`HANDOFF_2026-11-03.md`, slides "Thesis Progress for Rekha"):
  - The 4 world models (EEF / Full × Numeric / Image) were trained on the main data (v1 + v2) with 3 seeds. The image encoder is SS: CoordConv + spatial softmax. The numeric encoder is a 1D CNN.
  - Validation verdicts: H1 supported, H2 supported, H4 not supported.
  - A dedicated H5 evaluation set (`Episodes/_h5_v1`) had been generated.
  - The trained weights had **not** been saved, so the H5 set could not be evaluated without retraining.
- **Constraint:** the lab PC was unavailable, so training had to move to a cloud GPU.
- The Unity MCP connection was checked at the start of the session.

## 1. Training on RunPod (seeds 0–2, with saved weights)

- **Setup:**
  - GPU: RTX 4090 (48 GB), Community Cloud.
  - Template: PyTorch 2.8.0 / CUDA 12.8.1.
  - The data (`rp_data.tar`, 2.5 GB) was moved with `runpodctl send/receive`.
  - Training ran in `tmux`.
- **Problems and fixes:**
  - **`runpodctl receive` on the pod failed** ("config not found"). Fix: `runpodctl config --apiKey "$RUNPOD_API_KEY"`, then receive again.
  - **Training was very slow, with GPU utilisation at 0%.** The host reports 256 shared cores, so PyTorch spawned too many CPU threads. Fix: `export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8`. After that, training was about 1.7× faster than the laptop RTX 4060.
  - **Two training processes ran at the same time**, because Ctrl+C did not stop the first one. Fix: `pkill`, delete the output folder, start a single process.
- **Run:** `wm_encoder_study.py sweep --plan main_ss --epochs 40`, 12 runs in about 1.5 h. Weights are saved in `Docs/encoder_fix/results_main_v1v2_ckpt/models/`; the folder is gitignored.
- **Reproducibility:** the validation results match the earlier laptop run. Hypothesis verdicts are unchanged, and the only notable difference is that the Full-Num position error moved from 12.1 to 10.1 mm.
- **H5 set, 3 seeds:** Full-Num balanced accuracy 0.725, Full-Img 0.598.
- **Commits:** `337210c` saves the weights and adds `wm_predict.py`; `c918bf8` contains the results.

## 2. H5 deep dive (exploratory)

- **Script:** `Learning/wm_h5_deep.py`. It computes per-seed accuracy on the wrist_down and wrist_up sides, balanced accuracy, AUROC within matched pairs, the bias, and an ensemble.
- **Findings:**
  - Full-Num varied strongly by seed: 0.893 / 0.683 / 0.601.
  - All models are biased towards "wrist_up collides". Even in pairs where both candidates have the same outcome, c_up − c_down is +0.16 to +0.41. This matches the training data, where about 73% of collisions are on the wrist_up side.
  - AUROC within matched pairs, which a constant bias does not affect, is well above 0.5 for both full-body models.
- **Commit:** `3d457c7` (`h5_analysis.md`).
- **Decision:** because of the seed variance, increase to 5 seeds.

## 3. Seeds 3 and 4 (protocol entry 7)

- **Protocol entry 7** was written and committed before running (`cb6f280`): 5 seeds, the same environment, and results reported as the 5-seed mean together with per-seed values.
- **Run:** `--plan main_ss5`. Runs that already exist are skipped, so only 8 new runs were trained (about 45 min).
- **Problems and fixes:**
  - **The new pod reused the `/workspace` volume**, so `git clone` failed. Fix: move the old results aside, `git pull`, copy the models back.
  - **The tar was created before `wm_predict` had finished**, so the H5-set predictions for the image models' seeds 3 and 4 were missing. Fix: regenerate them locally from the saved weights. The seed 0–2 values are identical to the RunPod values to 3 decimals.
  - **`wm_predict.py` crashed when printing** (cp932 encoding). Fix: `PYTHONIOENCODING=utf-8`.
- **H5 set, 5 seeds:**
  - Full-Num balanced 0.747 [0.689, 0.803], AUROC 0.833.
  - Full-Img balanced 0.603 [0.543, 0.665], AUROC 0.717.
  - All 5 seeds are above 0.5 for both models. The conclusions did not change.
- **Commit:** `2c88e70`.

## 4. H5 decision rule (protocol entry 8)

- The student decided not to consult the supervisor and to fix the rule themself.
- **Rule:**
  - Primary metric is **balanced accuracy**: the mean of the accuracy on pairs where the wrist_down side collides and on pairs where the wrist_up side collides. The wrist-branch-only baseline scores 0.5.
  - H5 is supported if the lower bound of the 95% CI of the 5-seed mean is above 0.5.
  - Holm correction is applied over numeric and image.
  - AUROC is reported as secondary only and is not used for the verdict.
- The protocol states that the rule was chosen after seeing the validation and H5-set results.
- **Commit:** `864fdbb`, pushed before test.

## 5. Test evaluation (used once)

- **Run:** `wm_predict.py --split test --name test` with 20 models (4 × 5 seeds) on 76 scene pairs. Hypotheses were judged with the new `Learning/wm_test_judge.py`.
- **H1–H5 verdicts:**
  - H1 supported (num, img).
  - H2 supported: num +0.028 [+0.009, +0.047], img +0.050 [+0.025, +0.074].
  - H4 not supported (num, img).
  - H5 supported: balanced accuracy num 0.842 [0.785, 0.893], img 0.719 [0.638, 0.799]; AUROC 0.941 and 0.898.
- **Commits:** `8ae914c` and `66ca677`.

### 5.1 H3 (protocol entry 9)

- **Missing piece:** `wm_predict.py` did not keep the predicted masks, so H3 had not been judged. Entry 9 was committed first (`7e6b752`).
- **Problems and fixes:**
  - **Keeping the full masks in memory made the run stall.** Fix: keep only the per-episode IoU. `wm_eval.py` now accepts `mask_iou_ep`.
  - **Another program (`vrecon` inference) was using the GPU at 100%.** We waited until it finished, then re-ran.
- **Result: H3 not supported in all 4 cells.** The image models have *lower* mask IoU than the numeric ones. Image − numeric is −0.021 to −0.029, with every CI below 0.
  - The validation IoU in the training logs shows the same order, so this is not an evaluation bug.
  - The H1, H2, H4 and H5 numbers were identical after regeneration.
- **Commit:** `3253b11`.

### 5.2 Contact point error (protocol entry 10)

- This is the last proposal metric that had not been evaluated on test; it is exploratory. Entry 10 was committed first (`052ccf3`).
- **Change:** `evaluate_split` now also collects the predicted contact point.
- **Result** (Task A): 33.1 / 37.3 / 38.4 / 38.8 mm for EEF-Num / Full-Num / EEF-Img / Full-Img. This is much larger than the final position error; no model predicts the contact point well. All other numbers were unchanged.
- **Commit:** `80b7c08`.

## 6. Reports

- `Docs/encoder_fix/results_main_v1v2_ckpt/summary.md`: the full result summary in Japanese, with validation, H5 set, test, H3 and contact error.
- `Docs/encoder_fix/results_main_v1v2_ckpt/h5_analysis.md`: the 5-seed H5 analysis.
- `Docs/encoder_fix/results_main_v1v2_ckpt/test_judgement.md` and `eval_test.md`: the test verdicts and all metrics.
- `Docs/report_20261108/progress_en.md`: an English progress report for the next meeting, commit `5185cfc`.
- Slides: not made yet.

## 7. Final state

| Hypothesis | Numeric | Image |
|---|---|---|
| H1 | supported | supported |
| H2 | supported | supported |
| H3 | not supported (all 4 task × scope cells) | — |
| H4 | not supported | not supported |
| H5 (balanced accuracy) | supported | supported |

- **Phase 2 is complete except for writing.** All proposal metrics have been evaluated on test.
- **Schedule:** about 4 weeks ahead. Training was due on 2026-12-04, and about 9 weeks remain before the 2027-01-15 submission.
- **Phase 3** (optional video prediction) is not started. It needs new data recording and the supervisor's agreement.

## 8. Open items

- [ ] `git push` of the latest commits (`052ccf3`, `80b7c08`, `5185cfc`, and this log).
- [ ] Check RunPod: pod terminated, and no Storage / Network Volume left billing.
- [ ] Decide whether to delete `ckpt.tar`, `ckpt5.tar` and `rp_data.tar` in the repo root. They are untracked, and their contents are already extracted.
- [ ] Make the slides for the next meeting from `progress_en.md`.
- [ ] Start writing: methods and results chapters, plus figures.
- [ ] Decide on Phase 3 after the first draft.

## 9. Lessons for next time

- On shared cloud hosts, always set `OMP_NUM_THREADS` and `MKL_NUM_THREADS`.
- Before packing results on the pod, make sure that every chained step, including evaluation, has finished.
- Before a long GPU job on the laptop, check whether another process is already using the GPU.
- Write each protocol entry and commit it *before* running anything that touches the test set.
