# Abstract

*Draft, 2026-11-15. About 250 words.*

Learned world models predict what a robot's action will do to the scene, but most of them describe the action only by the end-effector's path. This thesis asks how much of the robot a world model should see.

**Design.** We compare four action representations in a 2 × 2 design:

- scope: end-effector only vs. full body;
- format: numeric vs. camera-aligned image.

All four models are otherwise identical and trained from scratch. We test them on three simulated UR5e tasks that differ in how much the outcome depends on the arm configuration:

- pushing a cube past an obstacle;
- placing a held apple into a basket inside a shelf compartment, with a protective stop on contact;
- pushing a tall box with the forearm.

**Matched pairs.** Each task contains matched pairs: trajectories with the same end-effector path but different inverse-kinematics solutions.

**Protocol.** Hypotheses and decision rules were committed before each task's data were generated, and each test split was used once.

**Results.**

- **Matched pairs:** the full-body models distinguish matched pairs in all three tasks and both formats (balanced accuracy 0.62–1.00 against 0.5 for end-effector models).
- **Arm irrelevant:** where the arm does not touch the object, the end-effector models predict the object's final position as well or better (about 4 mm lower error in the main task).
- **Arm as the tool:** where the arm pushes the object, the full-body models predict the final position 8–11 mm more accurately.

The advantage of seeing the whole robot therefore grows with how much the outcome depends on the arm.

**Format.** Image actions were weaker than numeric ones in two of the three tasks. Exploratory analyses trace this to late fusion and a learned branch bias rather than to missing information.

**Conclusion.** A world model should see as much of the robot as the outcome depends on.
