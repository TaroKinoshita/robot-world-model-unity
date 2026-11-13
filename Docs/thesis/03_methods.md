# 3 Methods

*Draft, 2026-11-13. Numbers are taken from `Docs/encoder_fix/protocol.md`, `Docs/encoder_fix/cnn_design.md`, `Docs/THESIS_HANDOFF.md`, `Docs/task2/` and `Docs/task3/`.*

## 3.1 Overview

We compare four action representations for a learned world model that predicts the outcome of a planned robot trajectory. The four representations form a 2 × 2 design:

- **scope:** end-effector only (EEF) vs. full body (Full);
- **format:** numeric vs. a camera-aligned image (Img).

All four models receive the same initial scene and goal; only the action input differs. We test the representations on three simulated tasks that differ in how strongly the outcome depends on the arm configuration (Table 3.1). Every task contains *matched pairs*: two candidate trajectories with the same end-effector path but different inverse-kinematics (IK) solutions, so that only the arm configuration differs.

**Table 3.1 – The three tasks.**

| | Task 1: push with obstacle | Task 2: shelf placement | Task 3: forearm push |
|---|---|---|---|
| What the robot does | pushes a cube towards a goal | carries a held apple into a basket inside a shelf compartment | moves the gripper past a tall box |
| Arm–object contact | unintended (arm or cube hits a cylinder) | unintended (arm hits the shelf → protective stop) | **intended** (forearm/wrist pushes the box) |
| What the arm configuration decides | whether a collision happens | whether the arm stops | where the box ends up |
| Matched pair | wrist flip (wrist-down / wrist-up) | shoulder A / B | shoulder A / B |
| Scenes (pairs of Task A/B rows) | 511 used (train 358 / val 77 / test 76) | 300 (210 / 45 / 45) | 300 (210 / 45 / 45) |

## 3.2 Simulation

All data are generated in Unity 6 (6000.0.84f1, HDRP 17) with PhysX.

- **Robot:** UR5e with a Robotiq 2F-140 gripper, imported from URDF. The collision meshes of the arm links are convex-decomposed.
- **Physics:**
  - fixed time step of 0.02 s (50 Hz);
  - TGS solver;
  - joints driven by PD drives along the planned joint trajectory.
- **Contact:** a contact counts only if the separation is at most 1 mm. PhysX reports contacts already within its contact offset, so this filter removes reports of near-misses.
- **Camera:** one fixed camera with a 60° vertical field of view, rendering 256 × 256 images.
  - Tasks 2 and 3 use the task-1 view pulled back by 0.2 m, so that the shelf or box and the whole arm are visible.
- **Initial image:** taken *robot-free* (the robot is hidden). The robot's configuration therefore reaches the model only through the action representation.
- **Determinism:** every task was checked to give identical results when run twice.
  - Task 1 runs each candidate in a fresh physics scene, so repeated runs are bit-identical.
  - Tasks 2 and 3 run at an 8× time scale. Their binary labels match a 1× run on all checked candidates, and final positions agree within 0.07 mm.

## 3.3 Tasks

### Task 1: push with an obstacle (main task)

**Setup.** A cube (8 cm, 0.3 kg) on a table is pushed towards a goal of radius 3 cm. Each scene has 16 candidate pushes:

- 2 matched pairs;
- 3 towards the goal;
- 3 at the edge of the goal;
- 3 random;
- 3 towards the cylinder.

**Task A and Task B.** Each scene is recorded twice:

- **Task A:** without an obstacle.
- **Task B:** with a capsule (height 32–40 cm) placed so that, in one matched pair, only one of the two arm configurations hits it.

**Labels.**

- **Goal reached:** the cube stops within 3 cm of the goal, upright and at rest.
- **Collision (Task B only):** the robot or the cube touches the obstacle.

**Matched pairs.** The two candidates share the same tool-centre-point (TCP) path, with a maximum difference below 0.15 mm, but use the wrist-down vs. wrist-up IK solution. Their forearm and wrist differ by about 20 cm.

### Task 2: placing a held apple into a basket in a shelf compartment

**Motivation.** Fishman et al. (2022) treat cubbies as a harder environment class than an open tabletop.

**Setup.**

- An apple (8 cm) is attached to the gripper with a fixed joint, so grasping is not modelled.
- The gripper moves in a straight line from 0.30 m in front of the compartment to above a basket and releases the apple.
- The compartment's width, height and lateral offset are random. The back board is removed.

**Protective stop.** If any arm link or the gripper touches the shelf or the basket rim, the arm freezes and the apple is not placed.

**Candidates.** The gripper enters the compartment at four lateral offsets, dx ∈ {−0.10, −0.05, 0, +0.05} m. Each is executed with the two shoulder solutions of the same TCP path (A: base joint ≈ −13°, B: ≈ −167°).

**Labels.**

- **goal:** the apple comes to rest in the basket;
- **coll:** protective stop.

### Task 3: forearm pushing

**Motivation.** Tasks 1 and 2 only involve unintended contact. Task 3 makes the arm the tool, following the "pushing" class of whole-arm manipulation (Salisbury 1987; Eberman 1989) and the elbow-contact task of Ganapathi et al. (2022).

**Setup.**

- A tall box (6–10 × 5–8 × 28–42 cm, 0.4 kg) stands 0.10–0.20 m beside the gripper path.
- The gripper (pointing down, 6 cm above the table) moves 0.30 m forward with the same shoulder A/B solutions as in Task 2.
- The arm is not stopped. If the forearm or wrist touches the box, the box is pushed and usually tips over (about 0.2 m displacement).
- Layouts in which the gripper comes within 10 cm of the box are rejected, so contact can only come from the arm. No gripper contact occurred in the data.

**Labels.**

- **coll:** an arm link (not the gripper) touches the box;
- **goal:** the box tips over (tilt > 30°);
- the box's final position.

## 3.4 Action representations

All representations are computed from the **planned** trajectory, never from the executed one, so the outcome cannot leak into the input. Each trajectory is sampled at the same 25 time steps.

| Representation | Content |
|---|---|
| EEF-Num | TCP position, tool orientation (quaternion), gripper opening: 7 values × 25 steps |
| Full-Num | 6 joint angles + position and orientation of 9 points along the arm: 69 values × 25 steps |
| EEF-Img | mask + depth of the gripper mesh, rendered from the scene camera: 2 channels × 25 steps |
| Full-Img | mask + depth of the whole robot mesh: 2 channels × 25 steps |

- **Rendering:** the images are rendered on the CPU with a z-buffer from the planned joint angles, at 256 px, and downsampled to 128 px for training.
  - Step 0 of the full-body rendering matches the actual robot in the scene image with IoU 0.997.
- **Matched pairs:** in each pair, the EEF input of the second candidate is replaced by that of the first (the paths differ by < 0.15 mm). The EEF models therefore cannot tell the two candidates apart, by construction.

## 3.5 Model

The four models share every part except the action encoder (Table 3.2). The design follows prior work:

- 1D temporal convolution for the numeric sequences (Bai et al. 2018);
- camera-aligned action images read by a CNN (Khansari et al. 2020);
- concatenation of image and action features (Agrawal et al. 2016).

**Table 3.2 – Architecture.**

| Part | Input | Structure | Output |
|---|---|---|---|
| Scene encoder | robot-free RGB + depth (4 × 128 × 128) | 5-layer 2D CNN, GroupNorm, SiLU, average pooling | 128-d |
| Goal encoder | goal (x, z) and its image position (u, v) | 2-layer MLP | 128-d |
| Action encoder, numeric | 25 × 7 or 25 × 69 (normalised) | 3-layer 1D CNN over time, masked mean | 128-d |
| Action encoder, image | 25 steps × (mask, depth) stacked as 50 channels | 5-layer 2D CNN (same shape as the scene encoder) | 128-d |
| Fusion | concatenation (3 × 128) | 2-layer MLP | 256-d |
| Heads | – | linear: final position, goal, collision, contact point; CNN decoder: final object mask | – |

- **Size:** about 1.7 M parameters for the numeric models and 1.9 M for the image models.
- **Optimiser:** Adam, learning rate 1e-3 (numeric) or 3e-4 (image), at most 40 epochs, early stopping with patience 8 on the validation loss.
- **Seeds:** 5 per model. All results are averaged over the 5 seeds.

## 3.6 Evaluation

**Prediction metrics.**

- final position error (mm);
- goal and collision AUPRC;
- final mask IoU;
- contact-point error.

**Decision metrics (Task 1).** In each scene, the candidate with the highest predicted probability of "goal and no collision" is selected. We report its success rate, the collision rate, the regret, and the pairwise ranking accuracy.

**Matched pairs (H5).** On pairs where exactly one candidate collides, a model is correct if it gives the colliding candidate the higher collision probability. Because collisions are unbalanced between the two branches (in Task 1, 73% of colliding pairs collide on the wrist-up side), we use **balanced accuracy**: the mean of the accuracy on pairs where the branch-0 side collides and on pairs where the branch-1 side collides.

- A predictor that uses only the branch label scores exactly 0.5 on this metric.
- So does any EEF model.

**Statistics.**

- 95% confidence intervals come from a paired bootstrap over scene pairs (2000 resamples). All models, seeds and tasks are resampled with the same draws.
- When a hypothesis is claimed for both formats, the p-values are Holm-corrected.

## 3.7 Hypotheses and pre-registration

The hypotheses, data, splits and decision rules were written into a protocol file and committed to version control **before** the data they concern were generated or evaluated. The test split of each task was used once.

| Task | Hypothesis | Rule |
|---|---|---|
| 1 | **H1** (Task A): EEF is not worse than Full | CI upper bound of the EEF degradation below the margin on all three metrics: position 3 mm, goal AUPRC 0.05, ranking accuracy 5 pt |
| 1, 2 | **H2**: Full has higher collision AUPRC | CI lower bound of Full − EEF > 0 |
| 1 | **H3**: Image improves mask IoU without > 3 mm position loss | as in the proposal |
| 1 | **H4**: Full selects trajectories with lower collision rate and regret | both CI lower bounds > 0 |
| 1, 2, 3 | **H5**: Full distinguishes matched pairs | CI lower bound of the 5-seed mean balanced accuracy > 0.5 |
| 3 | **H1′**: Full predicts the box's final position better | CI lower bound of EEF − Full position error > 0 |
| 3 | H2 (reported) | as above |

**Changes from the proposal**, all recorded in the protocol before the affected data were used:

- the wrist flip replaces elbow-up/down, because the elbow-down solution passes below the table;
- 5 seeds instead of 3;
- balanced accuracy for H5;
- Tasks 2 and 3 were added after Task 1 had been evaluated.

H3 and H4 are not tested in Tasks 2 and 3: those tasks have only 4–8 candidates per scene, so selecting among them is not a meaningful decision problem.
