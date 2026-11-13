# 5 Discussion

*Draft, 2026-11-13.*

## 5.1 The arm matters where the arm decides the outcome

**What every task shows.** Across all three tasks, the full-body representation distinguishes matched pairs, while the end-effector representation cannot (H5 supported in all six task × format cells).

- Both candidates of a pair share the same gripper path. Any information that separates them must come from the arm.
- The result therefore shows that a world model can use arm-configuration information when it is given, and that an end-effector-only interface removes it.

**Strength of the evidence differs per task.**

- **Task 1:** both branches are reachable for collisions, and balanced accuracy controls for the branch prior.
- **Task 2:** the result replicates under a different robot motion, environment and pairing (shoulder rather than wrist).
- **Task 3:** the discrimination is trivially separable (box side → shoulder). It is consistent with the other tasks but adds little on its own.

## 5.2 How large the gap is depends on the task

The tasks span a range of how much the outcome depends on the arm configuration:

- **Task 1:** the arm only matters through a rare, unintended collision.
- **Task 2:** the arm matters through a protective stop, but the gripper path predicts most stops on its own.
- **Task 3:** the arm is the instrument that moves the object.

The gap between full-body and end-effector grows along this range (Figure 4.1).

- **Object position:**
  - Tasks 1 and 2: the EEF model is as good or better (H1 supported in Task 1; n.s. in Task 2).
  - Task 3: it is clearly worse (H1′, 8–11 mm).
- **Collision or contact prediction:**
  - Tasks 1 and 2: the gap is small (+0.03 to +0.05, numeric only in Task 2).
  - Task 3: the gap is large (+0.43 to +0.45).

**Link to Ganapathi et al. (2022).** This is the world-model counterpart of the policy-learning result of Ganapathi et al.: joint-space actions win when the task relies on elbow contact (box flipping, 98.4% vs. 38.6%), and Cartesian actions win when it relies on the tool path (sweeping, 79.4% vs. 44.3%).

**Practical consequence.** The choice of action representation is not universal. It should follow how much the predicted quantity depends on the arm.

**Why the EEF model wins when the arm is irrelevant.** When the arm is irrelevant, the EEF model is *better* (Task 1 position error, about 4 mm).

- A plausible explanation is input dimensionality. The EEF numeric input has 7 values per step, against 69 for full-body.
- With 358 training scenes, the smaller input is easier to fit.
- Removing information that is irrelevant to the task helps.

## 5.3 Why the image models are weaker

**Observation.** In Tasks 1 and 2, the full-body image model is weaker than the numeric one on H5:

- Task 1: 0.72 vs. 0.84;
- Task 2: 0.62 vs. 0.90.

The image model also has higher position error in Task 1.

**The information is in the images.** In Task 1, a rule without learning, using only the action image and the initial depth, reaches a balanced accuracy of 0.89 on test. So the weakness lies in how the model uses the images.

**Two exploratory findings** point to the fusion and the decision bias rather than the representation itself:

1. **Late fusion.**
   - The scene and the action image are each compressed to a vector before they are compared, so the pixel-wise overlap of arm and obstacle is lost.
   - Stacking them as channels of one encoder (early fusion) raises Task-1 H5 balanced accuracy by +0.08 on both validation and the H5 set.
   - This partly reproduces the benefit Khansari et al. (2020) report for action images.
2. **Branch bias.**
   - Both full-body models, and the image model in particular, favour the branch that collides more often in training.
   - In Task 2, the image model ranks the two candidates correctly (AUROC 0.86) but with an offset. Re-centring the threshold on validation raises balanced accuracy from 0.62 to 0.79.

**When the gap disappears.** In Task 3, where the relevant cue (box side) is large and visible, the image model reaches the same H5 and a similar position gain as the numeric model.

**Interpretation.**

- Numeric joint information is the more reliable input for a small model trained from scratch with a fixed camera and a single robot.
- Image actions may become competitive with early fusion and debiasing.
- Large pretrained video models that output pixels and transfer across robots (Wang et al. 2025; Kim et al. 2026; Alzayer et al. 2026) operate under different conditions, so our result does not contradict theirs.

## 5.4 Better prediction does not imply better decisions

**H4.** In Task 1, the full-body models predict collisions better (H2), but the trajectories they select are not significantly safer (H4).

- The main reason is a floor effect: only 5.4% of goal-reaching candidates collide.
- This agrees with Yu et al. (2026): improvements in world-model prediction do not automatically yield better decisions.

**What would test the decision benefit.** A decision-level comparison would need tasks in which the arm configuration often decides between otherwise good candidates, for example Task 3 with a goal for the box.

## 5.5 Limitations

- **Simulation only, one robot, one camera.**
  - Real sensor noise, calibration error and other morphologies are not tested.
  - Sim-to-real transfer and a real UR5e check are left to future work.
- **Small models trained from scratch.** The conclusions concern this regime, not large pretrained world models.
- **Task design.**
  - The three tasks were designed by us. Tasks 2 and 3 were added after Task 1 had been evaluated.
  - To limit researcher degrees of freedom, each task's hypotheses and decision rules were committed before its data were generated, and each test split was used once.
  - Even so, the choice of tasks is a design decision.
- **Matched pairs are one specific contrast.**
  - Task 3's pairs are trivially separable.
  - Task 2's discordant pairs are unbalanced (60 vs. 188).
- **H5 metric chosen late.** The balanced-accuracy rule for H5 was chosen after seeing Task 1 validation results (but before test). This is recorded in the protocol.
- **Task 2 duplicates rows.** Tasks 2 and 3 reuse the Task 1 data format by writing each scene twice (with and without the collision label), so every episode appears twice in training.

## 5.6 Conclusion

How much of the robot should a world model see? Enough to capture what the outcome depends on.

- **When only the gripper touches the world:** an end-effector representation is sufficient and even slightly better.
- **When the arm can collide:** the full-body representation is needed to tell apart trajectories that share the same gripper path.
- **When the arm itself manipulates the object:** it is needed to predict the outcome at all.
- **Format:** numeric full-body actions were the most reliable choice for a small model. Image actions were held back by late fusion and bias rather than by missing information.
