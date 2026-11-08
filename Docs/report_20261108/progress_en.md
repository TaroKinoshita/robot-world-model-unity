# Progress report — 2026-11-08

**Thesis:** How Much of the Robot Should a World Model See? Comparing End-Effector and Full-Body Action Representations for Robot–Object Prediction and Planning

## 1. Summary

- **The core experiment (Phase 2) is complete.** The 4 world models (end-effector / full-body × numeric / image) were trained with 5 seeds and evaluated **once** on the held-out test set.
- **Results:** H1, H2 and H5 are supported, and H3 and H4 are not supported. Each verdict holds for both the numeric and the image format.
- **Schedule:** about 4 weeks ahead. Data generation and training were planned to finish by 2026-12-04, and evaluation is already done as well. From now on, the time goes to writing.

## 2. What was done since the last report

1. **Training on a cloud GPU (RunPod, RTX 4090).** The lab PC was not available. The weights of all models are now saved, so any later evaluation runs without retraining.
   - Setup: torch 2.8.0 + CUDA 12.8; CPU threads limited to 8, because the host is shared.
   - The validation results reproduce the earlier run on the laptop (RTX 4060).
2. **Seeds increased from 3 to 5** (protocol entry 7, written before running). With 3 seeds, the H5 result varied strongly between seeds.
3. **The H5 decision rule was fixed before test** (protocol entry 8; see §4).
4. **The test set was used once.** H3 and the contact-point error needed the predicted masks and contact points, which the prediction script did not save at first. The script was extended (protocol entries 9 and 10, written before running) and the test predictions were regenerated with the same weights. All other numbers stayed identical.

## 3. Test results (76 scene pairs, mean of 5 seeds, 95% CI from a scene-level paired bootstrap)

### 3.1 Hypotheses

| Hypothesis | Numeric | Image |
|---|---|---|
| **H1** (Task A): end-effector is within the margin of full-body | **supported** | **supported** |
| **H2** (Task B): full-body has higher collision AUPRC | **supported**: +0.028 [+0.009, +0.047] | **supported**: +0.050 [+0.025, +0.074] |
| **H3**: image improves mask IoU, and numeric is not worse by > 3 mm in position | **not supported** (all 4 task × scope cells) | — |
| **H4** (Task B): full-body has lower collision rate **and** lower regret | **not supported** | **not supported** |
| **H5**: full-body distinguishes matched pairs (balanced accuracy) | **supported**: 0.842 [0.785, 0.893] | **supported**: 0.719 [0.638, 0.799] |

- **H1:** the end-effector models are not worse than the full-body models; they are actually better. For example, the final position error is about 4 mm lower.
- **H2 and H5** pass the Holm correction over the two formats.
- **H3:** the image models have *lower* mask IoU than the numeric models; image − numeric is −0.021 to −0.029, with every CI below 0. The position condition is met: the numeric models are 3–4 mm *better*. The earlier pilot result (H3 supported) came from a stage where the numeric encoder could not even fit the training data. Once the encoders were fixed, numeric is better in both position and mask.
- **H4:** full-body predicts collisions better, but the trajectories it selects do not have a significantly lower collision rate or regret.

### 3.2 Main metrics (test)

| Metric | EEF-Num | Full-Num | EEF-Img | Full-Img |
|---|---|---|---|---|
| Final position error, Task A (mm) | **6.5** | 10.3 | 10.1 | 14.2 |
| Collision AUPRC, Task B | 0.796 | 0.824 | 0.777 | **0.828** |
| Mask IoU, Task A | **0.853** | 0.815 | 0.825 | 0.794 |
| Contact point error, Task A (mm, exploratory) | **33.1** | 37.3 | 38.4 | 38.8 |
| Success rate, Task B | **0.950** | 0.932 | 0.921 | 0.903 |
| Matched-pair accuracy (h5_acc) | 0.500 | **0.883** | 0.500 | 0.798 |

- End-effector models score exactly 0.5 on matched pairs. This is by construction: both candidates of a pair share the same end-effector path, so the input is identical.
- Contact point error is 33–39 mm for all models, much larger than the final position error. No model predicts the contact point well.

## 4. A closer look at H5

- **Metric.** The pre-registered H5 metric (full − EEF accuracy) is weak, because EEF is always 0.5. A stronger baseline is the **wrist branch alone**: in the training data about 73% of collisions are on the wrist_up side, so "always pick wrist_up" already does well. To control for this, we use **balanced accuracy**, the mean of the accuracy on pairs where the wrist_down side collides and on pairs where the wrist_up side collides. The branch-only baseline scores exactly 0.5.
- **Decision rule (protocol entry 8, fixed before test).** H5 is supported if the lower bound of the 95% CI of the 5-seed mean balanced accuracy is above 0.5. AUROC within matched pairs is reported as a secondary, exploratory metric that is not affected by a constant bias.
  - I chose this rule myself, without consulting the supervisor, *after* seeing the validation and H5-set results. This is stated in the protocol.

**Per seed, balanced accuracy on test:**

| Model | seed 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Full-Num | 0.918 | 0.877 | 0.780 | 0.856 | 0.779 |
| Full-Img | 0.710 | 0.705 | 0.711 | 0.745 | 0.723 |

**Dedicated H5 evaluation set** (126 scene pairs; the wrist_down side collides more often than in training):

| Model | wrist_down side | wrist_up side | Balanced | AUROC |
|---|---|---|---|---|
| Full-Num | 0.590 | 0.904 | **0.747** [0.689, 0.803] | 0.833 |
| Full-Img | 0.325 | 0.881 | **0.603** [0.543, 0.665] | 0.717 |

- **Caveat.** Both full-body models still carry a bias towards "wrist_up collides". Even in pairs where both candidates have the same outcome, the predicted collision probability is higher on the wrist_up side (+0.16 to +0.46). The image model in particular is near or below chance on the wrist_down side (test 0.500, H5 set 0.325). The claim we can make is "full-body models capture information beyond the wrist branch", **not** "they predict both branches equally well".

## 5. Changes from the proposal (to be stated in the thesis)

| Item | Proposal | Now | Reason |
|---|---|---|---|
| Seeds | 3 | 5 | High seed variance on H5 (protocol entry 7) |
| H5 decision | full − EEF accuracy | Balanced accuracy vs. the wrist-branch baseline; AUROC secondary | EEF is 0.5 by construction; the branch alone is a stronger baseline (protocol entry 8) |
| Matched pair | elbow-up / elbow-down IK | Wrist flip (wrist_up / wrist_down) | Decided earlier (P3) |
| Compute | Lab PC | RunPod RTX 4090 + laptop RTX 4060 | Lab PC unavailable; results reproduce |

## 6. Schedule

| Task | Planned deadline | Status |
|---|---|---|
| Environment and pilot data | 2026-10-16 | ✅ |
| Interim checkpoint | 2026-10-30 | ✅ |
| Data generation + training | 2026-12-04 | ✅ (done ~4 weeks early) |
| Evaluation + writing | 2027-01-14 | Evaluation ✅, writing next |
| Final submission | 2027-01-15 | — |

## 7. Next steps

1. **Writing:** the methods and results chapters first (2–3 weeks), with figures: bar charts with CIs, per-seed spread, the H5 branch breakdown, and qualitative prediction examples.
2. **Phase 3 (optional, short-horizon video prediction):** decide after the draft. It needs new data recording, because intermediate frames were not saved, plus the supervisor's agreement.

## Questions for the meeting

- Is the H5 framing acceptable: balanced accuracy against the wrist-branch baseline, with AUROC as secondary?
- Is H3 being not supported (numeric beats image even on mask IoU) a reasonable result to report as-is?
- Is there interest in Phase 3, given about 9 weeks remain before submission?
