# 1 Introduction

*Draft, 2026-11-15.*

## 1.1 Motivation

**World models and their action interface.** A world model predicts how the scene will change when the robot executes an action. Such models are used to evaluate candidate trajectories before execution, to plan, and to train policies (Finn et al. 2016; Yin et al. 2026; Zhang & Du 2026). Every world model needs an *action interface*: a description of what the robot is going to do.

**The common choice.** In manipulation, the default interface is the end-effector path: the position and orientation of the gripper over time. It is compact, robot-independent and matches how most tasks are specified.

**What it leaves out.** A manipulator is more than its gripper. The same gripper path can be executed with different arm configurations, and the rest of the arm can hit an obstacle, be stopped by a safety system, or push an object on purpose. An end-effector-only model cannot tell these cases apart.

**The alternative and its costs.** One can give the model the whole robot: joint angles and link poses, or an image of the robot rendered from the camera's viewpoint (Khansari et al. 2020; Wang et al. 2025). This adds input dimensions and may make learning harder when the arm does not matter.

**Evidence so far.** In policy learning, Ganapathi et al. (2022) found that neither action space is best everywhere:

- joint-space actions win when the task relies on elbow contact;
- Cartesian actions win when it relies on the tool path.

For world models, this question has not been tested with a controlled contrast.

## 1.2 Research question

**Main question.** *How much of the robot should a world model see?*

We compare four action representations:

- **scope:** end-effector only vs. full body;
- **format:** numeric vs. camera-aligned image.

All four models are identical except for the action encoder. The sub-questions are:

- **RQ1 (scope):** When does a full-body representation improve predictions of the outcome, and when is the end-effector path sufficient?
- **RQ2 (format):** For the same scope, do numeric or image representations predict better?
- **RQ3 (decisions):** Do better predictions lead to better trajectory selection?
- **RQ4 (matched pairs):** Can a model distinguish two trajectories that share the gripper path but differ in arm configuration?

## 1.3 Approach

**Tasks.** We generate data in simulation with a UR5e arm in three tasks. They are ordered by how much the outcome depends on the arm configuration:

1. **Push with an obstacle.** The robot pushes a cube to a goal. In a second variant, a cylinder stands where the arm may hit it.
2. **Shelf placement.** The robot carries an apple into a basket inside a shelf compartment, and stops if the arm touches the shelf.
3. **Forearm pushing.** The gripper passes a tall box, and the forearm may push it over.

**Matched pairs.** Every task contains *matched pairs*: two candidates with the same end-effector path, executed with two different inverse-kinematics solutions. This isolates the contribution of the arm configuration.

**Pre-registration.** Hypotheses and decision rules were written and committed to version control before each task's data were generated, and each test split was evaluated once.

## 1.4 Contributions

1. **A controlled comparison** of end-effector and full-body action representations for a learned world model, crossed with numeric and image formats, using matched pairs that differ only in arm configuration.
2. **Evidence across three tasks** that the benefit of the full-body representation depends on how much the outcome depends on the arm:
   - it is absent for object position when the arm does not touch the object;
   - it is present for collisions;
   - it is large when the arm is the tool.
3. **An analysis of why image actions underperform** in a small model: the information is present, but late fusion and a learned branch bias limit how it is used.
4. **A reproducible simulation pipeline and a pre-registered protocol** for generating matched-pair data and evaluating action representations.

## 1.5 Outline

- **Chapter 2** reviews world models, action representations and whole-arm manipulation.
- **Chapter 3** describes the tasks, representations, model and evaluation.
- **Chapter 4** reports the results per task and across tasks.
- **Chapter 5** discusses what they mean, their limitations, and the answer to the research question.
