# Interim checkpoint (30 Oct 2026) — materials

Taro Kinoshita (S1140213) · BSc AI thesis · supervisor dr. Rekha Raja

All figures are regenerated from the stored episodes by `make_figures.py` (numbers in `numbers.json`).
Captions and tables are in English so they can be pasted into the report or slides.

---

## 1. Changes with respect to the proposal (overview)

| # | Proposal | Implementation | Why | Evidence |
|---|---|---|---|---|
| 1 | Matched pair = elbow-up / elbow-down | **Wrist flip**: wrist-down vs wrist-up, both elbow-up. Same TCP path (max difference < 0.15 mm), forearm and wrist differ by ~20 cm | With the robot mounted at table height, the elbow-down solution moves the elbow below the tabletop (elbow at 0.69–0.75 m, tabletop 0.773 m) | Fig. 1 |
| 2 | Push style not specified | Fingers **half open (0.42 rad, ~6 cm between pads)**, gripper **rotated 90°**, no tilt → the box is pushed at two points | With one open finger every box tipped over; with closed fingers boxes rotated and drifted 35–40° off the push direction | Fig. 2, video `Assets/Capture/push_methods_comparison.mp4` |
| 3 | Goal = within 3 cm and at rest | Within 3 cm **and not fallen** (tilt ≤ 30° and drop < 2 cm) **and speed < 5 mm/s** | A cube on its side keeps its centre height, so the height-only rule missed tipped boxes | Fig. 2 |
| 4 | Collision = robot or target contacts the secondary object | Same, but only **touching contacts (gap ≤ 1 mm)** count | PhysX reports a contact as soon as two shapes are within their contact offsets, before they touch | Fig. 3 |
| 5 | (physics settings not specified) | **TGS solver** and a **fresh physics scene per candidate** | The mere presence of an untouched cylinder shifted Task B outcomes (up to 12.6 mm); running candidates back-to-back left solver history behind | Fig. 4, Table 2 |
| 6 | 16 candidates (composition not specified) | **2 matched pairs + 3 toward the goal + 3 at the goal edge + 3 random + 3 toward the cylinder**; cylinder position and height (32–40 cm) per scene, placed out of the camera's line of sight to the target and goal | One fixed cylinder cannot make every pair collide one-sidedly; goal labels were almost fully predictable from the candidate type; the cylinder sometimes hid the target | Fig. 5, Fig. 6 |

Suggested wording for the matched pair (H5):
*Two candidates that share the same end-effector path (TCP position and orientation; max difference < 0.15 mm) but are executed with different inverse-kinematics solutions (wrist-down vs. wrist-up, both elbow-up), so that the forearm and wrist differ by about 20 cm. Elbow-down was not used because, with the robot mounted at table height, the elbow of the elbow-down solution passes below the tabletop.*

---

## 2. Figures and captions

### Fig. 1 — `fig1_matched_pair.png`
**Matched pair by wrist flip.** (a) Rendered full-body masks of the two candidates of one matched pair at the same time step, drawn on the robot-free initial image (scene_0001, Task B). (b) The TCP paths differ by at most 0.073 mm. (c) The forearm differs by 21.3 cm on average and wrist 1 by 20.0 cm; in this scene only the wrist-up arm touches the cylinder (dashed line), so the two candidates have identical end-effector input but different collision labels.

### Fig. 2 — `fig2_push_and_fall.png`
**Push style and the "fallen" rule.** Final tilt and centre-height drop of the target. With the original push (one open finger, P4-A pilot) all 16 episodes ended with the cube on its side (tilt ≈ 90°) while the centre height did not change, so the height-only rule labelled none of them as fallen and 8 of them as goal reached. With the current push (half-open fingers, 90° yaw, two contact points) none of the 640 episodes tipped (max tilt 0.3°). The goal label now also requires tilt ≤ 30°.

### Fig. 3 — `fig3_touch_only.png`
**Collision = touching contacts only.** For every Task B episode with any reported contact with the cylinder (111 of 320), the smallest gap between the robot/target and the cylinder. 13 episodes had contacts reported only at a positive gap (up to 11.5 mm) and never touched; counting them would label near-misses as collisions. Negative values are penetration while pushing into the cylinder.

### Fig. 4 — `fig4_ab_stability.png`
**Task A vs. B for candidates that never touch the cylinder.** Difference in final target position between the two tasks. With the PGS solver one candidate differed by 12.6 mm (and another by 1.1 mm) although the cylinder was never touched. With TGS and a fresh physics scene per candidate, 220 of 222 untouched candidates are bit-identical (0.000 mm); two differ by 1.7 and 2.1 mm, below the 3 mm margin used in H1/H3. Note: the two groups also differ in scenes and candidate composition; the controlled comparison is Table 2.

### Fig. 5 — `fig5_candidate_mix.png`
**Outcome by candidate type (Task B, 20 scenes each).** Before (P7), 88% of goal-directed candidates reached the goal and the other types almost never did, so the goal label was largely predictable from the candidate type. Adding goal-edge candidates (push direction ±10–20° or length ±1.5–4.5 cm off the goal push) gives a near 50/50 split (47% goal, 43% neither) within one type. Predicting the label from the type alone: goal 64% → 93% before, 71% → 85% now (baseline = majority class); collision 76% → 84% before, 69% → 80% now.

### Fig. 6 — `fig6_occlusion_height.png`
**Cylinder placement.** (a) Before the fix the cylinder could stand between the camera and the target, so the robot-free input image showed only part of the target (initial mask IoU 0.57). (b) The same seed after excluding positions that occlude the target or the goal region in the camera view. (c) Heights were tried from the lowest, so 19 of 20 scenes got 32 cm (lower heights have no valid position because the arm passes above them); trying 32–40 cm in a random order per seed spreads the heights.

---

## 3. Pilot numbers

**Table 1. Data generated with the final specification.**

| | 3-scene pilot (P5, seeds 1–3) | 20-scene batch (P8, seeds 4–23) |
|---|---|---|
| Scenes (Task A + B) / episodes | 6 / 96 | 40 / 640 |
| Generation | manual Play per scene | unattended batch, 20/20 seeds succeeded, 7.8 min per scene (max 8.1) |
| Automatic checks passed (A / B / pair) | 1740/1740 · 1692/1692 · 321/321 | 11600/11600 · 11280/11280 · 2140/2140 |
| Goal reached (Task A / Task B) | 18/48 (38%) · 18/48 | 93/320 (29%) · 93/320 |
| Collision (Task B): robot → cylinder / target → cylinder | 16/48 (33%): 7 / 9 | 98/320 (31%): 60 / 40 |
| Matched pairs with a one-sided collision (Task B) | 3/6 (1 per scene) | 28/40 (2 in 9 scenes, 1 in 10, 0 in 1) |
| Target fell / robot–table contact | 0 / 0 | 0 / 0 |
| TCP tracking error: median / 95% / max | 2.0 / 2.8 / 2.8 mm | 2.0 / 2.9 / 7.9 mm (all > 5 mm: arm pushed back by the cylinder) |
| Untouched candidates with A/B difference > 1 mm | 0 | 2 (1.7, 2.1 mm) |
| Cylinder heights | 32 cm ×3 | 32×2, 34×3, 36×6, 38×3, 40×6 |
| Training cache (128 px, 25 frames) | — | 76 MB, built in 37 s |

**Table 2. Repeatability fixes (controlled tests on the same candidate).**

| Test | Before | After |
|---|---|---|
| Same candidate run 3× in one Play (max final-position difference) | 9.2 mm (diverges from step 0) | 0 mm, all steps bit-identical (reset all joints, fresh physics scene per candidate, restore link transforms before moving) |
| Same candidate across Plays | — | bit-identical |
| Task A vs B, untouched candidate (seed 1, c002) | PGS: 12.6 mm; box pruning: 12.0; stiffer fingers: 12.0 | TGS: 0.000 mm |

---

## 4. Pipeline status

| Part | Status |
|---|---|
| Scene generation (seed → target, goal, 16 candidates, cylinder) | done |
| Episode recording, 4 action representations, labels, automatic checks | done |
| Unattended batch runner (Play → stop → checks, failed seeds skipped) | done |
| Training cache + PyTorch dataset (4 representations × Task A/B, scene-level split, matched pairs never split) | done |
| Evaluation (AUPRC, position error, IoU, decision metrics, H5, scene-level paired bootstrap, H1–H5 verdicts) | done, tested with dummy predictions |
| Full dataset generation (~10k episodes ≈ 42 h of unattended generation) | next |
| Model architecture and training | not started |

---

## 5. Open points to mention

- Two untouched candidates (out of 222) still differ by 1.7–2.1 mm between Task A and B. Below the 3 mm margin; will be monitored in the full dataset.
- In 5 of 20 scenes the cylinder was too far to push the target into it; those three candidates are filled with random directions.
- The collision label remains partly predictable from the candidate type (toward-cylinder pushes collide 87% of the time). This is physically expected; the models see trajectories, not types.
- Decision metrics (success rate, regret, collision rate) have one value per scene, so their confidence intervals depend on the number of test scenes (±0.1 with 20 scenes in the dummy test). The test split size should be chosen with H4 in mind.
- Provisional definitions to confirm: regret with binary success utility, position error in the table plane (XZ), contact point = first robot→target contact, mask threshold 0.5.
