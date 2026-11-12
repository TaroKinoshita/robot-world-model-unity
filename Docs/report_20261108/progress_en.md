# Progress report — 2026-11-08 (updated 2026-11-13)

**Thesis:** How Much of the Robot Should a World Model See? Comparing End-Effector and Full-Body Action Representations for Robot–Object Prediction and Planning

## 1. Summary

- **The core experiment (Phase 2) is complete.** The 4 world models (end-effector / full-body × numeric / image) were trained with 5 seeds and evaluated **once** on the held-out test set.
- **Results:** H1, H2 and H5 are supported, and H3 and H4 are not supported. Each verdict holds for both the numeric and the image format.
- **Why the image models are weaker (new, exploratory):** the image input is correct and contains the needed information. One confirmed cause is late fusion: switching to early fusion raised the image model's H5 balanced accuracy by about +0.08 on val and on the H5 set (test not used). See §5.
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

## 5. Why are the image models weaker? (exploratory, test not used)

### 5.1 Is the image input correct?

Yes. I checked the rendered action images against the Unity scene.

- **Camera:** the camera in the dataset matches the Unity camera exactly (position, 10° pitch, 60° field of view).
- **Alignment:** the projected tool-centre point lies a median of 1.45 px from the rendered gripper mask, and 98% of the arm keypoints fall on the rendered arm mask.
- **Information content:** a simple rule with **no learning**, using only the action image and the depth of the initial image, reaches a balanced accuracy of 0.887 on test, 0.914 on val and 0.895 on the H5 set. The trained full-body image model reaches 0.719 / 0.740 / 0.603 on the same sets.

So the information is in the images; the model does not use it fully.

### 5.2 Causes

| Result | Cause |
|---|---|
| H5 and H2: image weaker than numeric | **Late fusion.** The initial image and the action image are each compressed to a vector by separate encoders and only then concatenated. The pixel-wise overlap of arm and cylinder, including depth, is lost before the two are compared. |
| Position error 3.6–4 mm worse for image | **Resolution:** at 128 px, 1 px ≈ 10.8 mm at the scene distance. |
| H3: mask IoU lower for image | Mask IoU is driven by position error, and the mask decoder has no skip connection from the action image. |
| H4 not supported (both formats) | **Floor effect:** only 5.4% of goal-reaching candidates collide, so better collision prediction rarely changes the selected trajectory. |

### 5.3 Early fusion check (protocol entry 11, written before running)

Only the fusion was changed: the observation (4 ch), goal (1 ch) and action image (50 ch) are stacked and read by **one** encoder. Everything else is as in the main runs (5 seeds). Evaluated on val and the H5 set only.

| Full-body image, H5 balanced | Numeric (ref.) | Image, late | **Image, early** | Early − late [95% CI] | No-learning rule |
|---|---|---|---|---|---|
| H5 set (126 scene pairs) | 0.747 | 0.603 | **0.687** | **+0.084** [+0.052, +0.120] | 0.895 |
| val (77 scene pairs) | 0.836 | 0.740 | **0.815** | **+0.075** [+0.019, +0.133] | 0.914 |

- Early ≥ late for all 5 seeds on both sets. AUROC moves in the same direction.
- Val position error improves from 14.2 to 12.0 mm, and collision AUPRC from 0.837 to 0.861.
- About 60% of the gap to the numeric model closes on the H5 set. The wrist_down side is still below 0.5 (0.452), so the bias towards wrist_up remains.
- **Interpretation:** late fusion is part of the cause, now shown experimentally. It is not the whole cause; the remaining candidates are resolution, data size and the training-set bias.
- **Status in the thesis:** the main verdicts (pre-registered design, test used once) are unchanged. This result goes into the discussion as an exploratory finding.

### 5.4 Comparison with prior work

| Work | Their finding | Mine |
|---|---|---|
| Ganapathi et al. 2022 (IKP) | Joint-space wins when the elbow matters (box flipping 98.4% vs 38.6%); Cartesian wins for sweeping (79.4% vs 44.3%) | **Same direction:** full-body helps where the arm shape decides the outcome (H2, H5); end-effector suffices where it does not (H1) |
| Khansari et al. 2020 (Action Image) | Action as an image, stacked with the observation in one CNN: 84% vs 53% grasp success | Late fusion: image worse. Stacking (early fusion) improves it by +0.08, which partly reproduces their benefit |
| Wang et al. 2025; Kim et al. 2026; Alzayer et al. 2026 | Image actions win in large pretrained video models that output pixels and target new robots or cameras. Wang notes numeric states are nearly lossless with a fixed camera | Fixed camera, one robot, small model trained from scratch, low-dimensional outputs: the condition where numeric is expected to be strong |
| Yu et al. 2026 | Better prediction does not imply better decisions | H2 improves but H4 does not, partly because of the floor effect |

## 6. Changes from the proposal (to be stated in the thesis)

| Item | Proposal | Now | Reason |
|---|---|---|---|
| Seeds | 3 | 5 | High seed variance on H5 (protocol entry 7) |
| H5 decision | full − EEF accuracy | Balanced accuracy vs. the wrist-branch baseline; AUROC secondary | EEF is 0.5 by construction; the branch alone is a stronger baseline (protocol entry 8) |
| Matched pair | elbow-up / elbow-down IK | Wrist flip (wrist_up / wrist_down) | Decided earlier (P3) |
| Compute | Lab PC | RunPod RTX 4090 + laptop RTX 4060 | Lab PC unavailable; results reproduce |

## 7. Schedule

| Task | Planned deadline | Status |
|---|---|---|
| Environment and pilot data | 2026-10-16 | ✅ |
| Interim checkpoint | 2026-10-30 | ✅ |
| Data generation + training | 2026-12-04 | ✅ (done ~4 weeks early) |
| Evaluation + writing | 2027-01-14 | Evaluation ✅, writing next |
| Final submission | 2027-01-15 | — |

## 8. Next steps

1. **Writing:** the methods and results chapters first (2–3 weeks), with figures: bar charts with CIs, per-seed spread, the H5 branch breakdown, and qualitative prediction examples.
2. **Discussion chapter:** why the image models are weaker (§5) and the comparison with prior work.
3. **Future work (not planned in this thesis):** higher image resolution, and a skip connection from the action image to the mask decoder. Early fusion as the main design would need a fresh test set.
4. **Phase 3 (optional, short-horizon video prediction):** decide after the draft. It needs new data recording, because intermediate frames were not saved, plus the supervisor's agreement.

## Questions for the meeting

- Is the H5 framing acceptable: balanced accuracy against the wrist-branch baseline, with AUROC as secondary?
- Is H3 being not supported (numeric beats image even on mask IoU) a reasonable result to report as-is?
- I plan to report the early-fusion result as an exploratory finding in the discussion, and to leave a redesign of the image model to future work, because the test set has already been used and the research question concerns the scope of the robot representation, not the best input format. Is that acceptable?
- Is there interest in Phase 3, given about 9 weeks remain before submission?
