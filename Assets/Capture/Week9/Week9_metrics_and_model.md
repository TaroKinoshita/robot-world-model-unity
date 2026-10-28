# Evaluation Metrics (H1–H5) and the World Model

*Bachelor thesis "How Much of the Robot Should a World Model See?" — status 2026-10-24*

This note answers two questions from the last meeting:

1. **Which metric is used to evaluate each hypothesis (H1–H5)?** → Section 2
2. **What kind of deep-learning model is used, and how is it trained?** → Section 3

Important: the model described in Section 3 is the **small test model** used to check the full pipeline (training → prediction → evaluation → H1–H5 verdicts). The final model will keep the same inputs, outputs, loss and evaluation, but its encoders may change before the main training run (see Section 3.8).

---

## 1. Study design in one paragraph

A world model predicts the outcome of a robot push **before** it is executed. Its input is the robot-free scene image, the goal, and one planned trajectory (the *action*). I compare four ways of giving the action to the model, in a 2 × 2 design:

| | **Numeric** | **Image (camera-aligned raster)** |
|---|---|---|
| **End-effector only (EEF)** | EEF-Numeric: gripper position + orientation per time step | EEF-Image: mask + depth of the gripper only, 25 frames |
| **Whole arm (Full)** | Full-Numeric: 6 joint angles + position + orientation of 9 links per time step | Full-Image: mask + depth of the whole robot, 25 frames |

- **Task A**: push a cube to a goal; nothing else on the table.
- **Task B**: identical scene, goal and trajectories, plus a cylinder near the arm, so some trajectories collide.
- Each scene has **16 candidate trajectories**, including **2 matched pairs** (same end-effector path, wrist flipped → the arm differs by ~20 cm). In Task B the cylinder is placed so that only one trajectory of each matched pair touches it.

---

## 2. Evaluation metrics

All metrics are computed on **test scenes that are never used for training**, separately for **Task A and Task B**. Collision metrics exist only for Task B.

### 2.1 Prediction metrics (one value per episode, averaged)

| Metric | What it measures | How it is computed | Better |
|---|---|---|---|
| **Final position error** (mm) | Where the cube ends up | Distance between predicted and true final cube centre, in the table plane (XZ) | lower |
| **Goal AUPRC** | Whether the cube reaches the goal | Area under the precision–recall curve of predicted P(goal) vs. the true label | higher |
| **Collision AUPRC** (Task B) | Whether the robot or the pushed cube touches the cylinder | AUPRC of predicted P(collision) vs. the true label | higher |
| **Final mask IoU** | Spatial prediction of the final cube | IoU between the predicted final cube mask (thresholded at 0.5) and the true mask, 128 × 128 px | higher |
| Contact position error (mm) | Where the robot first touches the cube | Distance between predicted and true first contact point (episodes with contact only). Exploratory | lower |

**Label definitions**

- *Goal reached* = final cube centre within **3 cm** of the goal centre, **and** not toppled (dropped < 2 cm and tilted ≤ 30°), **and** stopped (speed < 5 mm/s).
- *Collision* = the robot or the target cube touches the cylinder (only real contacts with a gap ≤ 1 mm are counted).

### 2.2 Decision metrics (one value per scene)

The model is used for planning: for each scene it scores all 16 candidates and the best-scoring one is executed.

- Score = P(goal) in Task A, and P(goal) × (1 − P(collision)) in Task B.
- A pick is a *success* if it reaches the goal **without** a collision.

| Metric | What it measures | How it is computed | Better |
|---|---|---|---|
| **Success rate** | Quality of the picked trajectory | Fraction of scenes in which the picked candidate is a success | higher |
| **Collision rate** (Task B) | Safety of the picked trajectory | Fraction of scenes in which the picked candidate collides | lower |
| **Regret** | How far the pick is from the best possible pick | (success of the best candidate in the scene) − (success of the picked candidate), 0 or 1 per scene | lower |
| **Pairwise ranking accuracy** | Whether the model orders candidates correctly | Over all pairs of candidates in a scene where one succeeds and the other fails: fraction where the successful one gets the higher score (ties = 0.5) | higher |
| **Matched-pair accuracy** (Task B, H5) | Whether the model can tell the two trajectories of a matched pair apart | Over matched pairs where only one trajectory collides: fraction where the colliding one gets the higher P(collision) (ties = 0.5) | higher |
| Random / oracle success | Lower and upper bound | Success of a random pick / of the best candidate | — |

Two **label-only baselines** are always reported next to the models, because they show how much can be guessed without looking at the trajectory:

- *Candidate type only*: success rate of each candidate type (toward goal, goal edge, random, …) in the training data.
- *Type + matched-pair branch*: the same, plus wrist-down / wrist-up.

### 2.3 Hypotheses and the metric used for each

| | Research question | Hypothesis | Compared models | Task | Metric(s) | **Supported if** (95 % CI of the difference) |
|---|---|---|---|---|---|---|
| **H1** | RQ1 — scope | In Task A, EEF-only is **not worse** than whole-arm by more than a pre-set margin | EEF vs. Full (same format) | A | Final position error, goal AUPRC, pairwise ranking accuracy | The upper CI bound of the EEF degradation is below the margin for **all three**: 3 mm, 0.05 AUPRC, 5 percentage points. *Rejected* if any lower bound exceeds its margin; otherwise *inconclusive* |
| **H2** | RQ1 — scope | In Task B, whole-arm predicts collisions **better** | Full vs. EEF (same format) | B | Collision AUPRC | Lower CI bound of (Full − EEF) > 0 |
| **H3** | RQ2 — format | Images improve the spatial prediction, while numeric actions are **not worse** on position by more than 3 mm | Image vs. Numeric (same scope) | A and B separately | Final mask IoU **and** final position error | Lower CI bound of the IoU gain (Image − Numeric) > 0 **and** upper CI bound of the position-error excess (Numeric − Image) ≤ 3 mm |
| **H4** | RQ3 — decisions | In Task B, whole-arm leads to **better choices** | Full vs. EEF (same format) | B | Collision rate and regret of the picked trajectory | Lower CI bounds of both reductions (EEF − Full) > 0, **and** H2 holds for the same format |
| **H5** | RQ4 — matched pairs | Whole-arm can tell matched-pair trajectories apart (EEF-only cannot, because both trajectories give it identical input) | Full vs. EEF (same format) | B | Matched-pair accuracy | Lower CI bound of (Full − EEF) > 0 |

Notes:

- Each hypothesis is tested **separately for both formats** (numeric and image). When a claim is made for both formats together, a **Holm correction** is applied (H2, H5).
- All other metrics (contact error, success rate in Task A, etc.) are reported as **exploratory**.
- Why H5 is a fair test: in a matched pair the end-effector input is made identical (the two planned EEF paths differ by < 0.2 mm; the shared values are used). So an EEF-only model is expected to score 0.5 by construction; the question is whether the whole-arm model is clearly above it.

### 2.4 Statistics

- **Unit of resampling = scene.** 95 % confidence intervals come from a **scene-level paired bootstrap** (2,000 resamples). In every resample, the same scenes are used for all four models, both tasks and all seeds, so the comparisons are paired.
- Matched pairs and the Task A / Task B versions of a scene are never split.
- Each model is trained with **3 random seeds**; metrics are averaged over seeds inside every bootstrap resample.
- Data split: by scene, **70 % train / 15 % validation / 15 % test**.

---

## 3. The model and how it is trained

### 3.1 What kind of model

- A **supervised, feed-forward world model** (an *outcome predictor*): it maps (scene, goal, action) → predicted outcome of that action.
- It is **not a policy** and **not reinforcement learning**. It never controls the robot directly; it only predicts. The robot is moved by picking the best-scoring candidate trajectory (Section 2.2).
- Built from **convolutional neural networks (CNNs)** for images, a **1D convolutional network** for numeric action sequences, and **multi-layer perceptrons (MLPs)** for fusion and outputs. Implemented in PyTorch.
- The **four models are identical except for the action encoder**, so that differences in the results can be attributed to the action representation.

### 3.2 Inputs and outputs

| | Name | Shape | Content |
|---|---|---|---|
| **Input** | Scene image | 4 × 128 × 128 | Robot-free initial image from the fixed camera: RGB (0–1) + depth (m) |
| | Goal | 4 | Goal centre in world coordinates (x, z) and its position in the image (u, v) |
| | Action — EEF-Numeric | 604 × 7 | Gripper position (3) + orientation quaternion (4) at every 50 Hz step, zero-padded, with the true length |
| | Action — Full-Numeric | 604 × 69 | 6 joint angles + 9 links × (position 3 + quaternion 4) per step |
| | Action — EEF-Image | 25 × 2 × 128 × 128 | Mask + depth of the gripper only, 25 time steps, rendered from the same camera |
| | Action — Full-Image | 25 × 2 × 128 × 128 | Mask + depth of the whole robot, 25 time steps |
| **Output** | Final position | 3 | Final cube centre (world, m) |
| | P(goal) | 1 | Probability that the cube reaches the goal |
| | P(collision) | 1 | Probability of touching the cylinder (trained on Task B only) |
| | Final mask | 128 × 128 | Per-pixel probability of the cube in the final image |
| | First contact point | 3 | Where the robot first touches the cube |

What is deliberately **not** given to the model: anything recorded during execution (it would leak the outcome), the image with the robot in it (the robot's shape must come only from the action representation, otherwise the four representations would become equal), and the candidate type / matched-pair branch (they alone predict the label surprisingly well and would be a shortcut).

### 3.3 Architecture

```
scene image (4×128×128) ──► Image CNN (5 layers) ─────────► 128-d ─┐
goal (4)                ──► MLP (4 → 64 → 128) ───────────► 128-d ─┼─► concat (384) ─► MLP trunk (384 → 256 → 256) ─┬─► final position (3)
action                  ──► Action encoder (differs) ─────► 128-d ─┘                                                ├─► goal / collision logits (2)
                                                                                                                    ├─► first contact point (3)
                                                                                                                    └─► mask decoder ─► final mask (128×128)
```

| Block | Layers |
|---|---|
| Image CNN (scene) | 5 × [3×3 conv, stride 2 → GroupNorm → SiLU], channels 32-64-96-128-128, then global average pooling |
| Goal MLP | Linear 4 → 64, SiLU, Linear 64 → 128 |
| **Action encoder — numeric** (EEF-Num, Full-Num) | Normalise with training mean/std → 3 × [1D conv, kernel 5] (channels 64-128-128, strides 1-2-2, SiLU) → average over the valid time steps (padding masked out) |
| **Action encoder — image** (EEF-Img, Full-Img) | 25 frames × 2 channels stacked as 50 input channels → same 5-layer CNN as the scene encoder |
| Trunk | Linear 384 → 256, SiLU, Linear 256 → 256, SiLU |
| Heads | Linear layers for position (3), goal + collision logits (2), contact point (3); position outputs are offset by the training-set mean |
| Mask decoder | Linear 256 → 64 × 8 × 8 → 3 × [upsample ×2, 3×3 conv, SiLU] → 3×3 conv → bilinear upsample to 128 × 128 |

### 3.4 Size

| Model | Parameters |
|---|---|
| EEF-Numeric | 1,747,593 |
| Full-Numeric | 1,767,433 |
| EEF-Image | 1,969,673 |
| Full-Image | 1,969,673 |

- **Scope comparison (EEF vs. Full) is tightly controlled**: the two image models are identical; the two numeric models differ only in the number of input channels of the first layer (7 vs. 69).
- **Format comparison (numeric vs. image) is not**: 1D and 2D encoders are different networks (~0.2 M parameters apart). This is discussed in Section 3.8.

### 3.5 Training ("training mode")

| Item | Setting |
|---|---|
| Learning type | **Offline supervised learning** on recorded simulation episodes (no interaction with the simulator during training) |
| Training data | Task A and Task B episodes together, one model per representation |
| Loss (multi-task, summed) | Final position: smooth L1 (β = 0.01) × 10 · Goal: binary cross-entropy · Collision: binary cross-entropy, Task B episodes only · Final mask: per-pixel binary cross-entropy · Contact point: smooth L1 × 10, episodes with contact only |
| Optimiser | AdamW, learning rate 1e-3, weight decay 1e-4 |
| Batch size | 16 |
| Epochs | 30 (test model); **100 with cosine decay from now on** (see 3.8) |
| Seeds | 3 per model (the test model used 1) |
| Split | By scene: 70 / 15 / 15 (train / val / test), matched pairs never split |
| Data loading | All episodes packed into one HDF5 file (128 px, 25 frames); about 9× faster training than reading the episode folders |
| Hardware | Test model: RTX 4060 Laptop GPU, 15–36 s per model. Main run: RTX 5090 (32 GB) |

### 3.6 How the model output is used

1. For a new scene, predict all 16 candidates.
2. Score each candidate: P(goal) in Task A, P(goal) × (1 − P(collision)) in Task B.
3. Execute the highest-scoring candidate in simulation and check the true outcome → success rate, collision rate, regret (Section 2.2).

### 3.7 Current results of the test model (pipeline check only)

Data: 20 scenes (640 episodes); 14 scenes for training, 3 for testing. These numbers only show that the pipeline works; they are **not** results for the hypotheses.

| | EEF-Num | Full-Num | EEF-Img | Full-Img |
|---|---|---|---|---|
| Loss, first → last epoch | 2.38 → 1.16 | 2.40 → 1.07 | 2.56 → 0.30 | 2.56 → 0.23 |
| Training scenes: goal AUPRC (Task A) | 0.76 | 0.83 | 1.00 | 1.00 |
| Training scenes: success rate (Task A, 14 scenes) | 0.50 | 0.79 | 1.00 | 1.00 |
| Training scenes: matched-pair accuracy (Task B) | 0.50 | 0.95 | 0.50 | 1.00 |
| Test scenes: success rate (Task A, 3 scenes) | 0.00 | 0.33 | 0.00 | 0.00 |

- Everything runs end to end: gradients reach every block, the loss decreases, save → reload gives identical outputs, and the evaluation produces all metrics and H1–H5 verdicts.
- The EEF models score exactly 0.50 on matched pairs, as designed (identical input for both trajectories).
- On the 3 unseen scenes the models are worse than the "candidate type only" baseline. This is expected with only 14 training scenes (memorisation). Meaningful results need the main dataset: **200–320 scenes (~6,400–10,000 episodes)**.

### 3.8 Open issues before the main training run

1. **The numeric encoder looked weak — now resolved (2026-10-24).** The main cause was training that was too short (30 epochs, constant learning rate): with 100 epochs and cosine decay, the numeric models fit the training data (3–6 mm instead of 41–47 mm). The information is present in the numeric action (hand-crafted features alone reach 3.7 mm). The encoders were then chosen with the same budget and rules for both formats, fixed before the runs: **numeric → GRU, image → current CNN, both with learning rate 3e-4** (1.88 M vs. 1.97 M parameters). Details: `Week9_数値エンコーダ問題の原因と解決.md` (English version at the end). This choice will be re-checked on the first 50–100 scenes of the main dataset.
2. **Contact heatmap.** The proposal lists a contact heatmap as the fifth target. It is stored in the dataset, but the test model predicts the **first contact point** instead. To decide: predict the heatmap, or replace it with the contact point.
3. The `toppled` label is stored but not used as a target (no toppled cubes in the current 20 scenes).
