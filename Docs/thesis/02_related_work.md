# 2 Related Work

*Draft, 2026-11-13. Check each reference's details against the paper before submission.*

## 2.1 Action-conditioned world models

**Video prediction.** Action-conditioned prediction of physical interaction was established by Finn et al. (2016), who condition a video-prediction model on robot actions. Agrawal et al. (2016) predict the outcome of pokes from image features concatenated with the action, the fusion scheme we use.

**Recent large models.** Recent work uses large world models for robot learning and evaluation:

- PlayWorld (Yin et al. 2026);
- World Action Planner (Zhang & Du 2026).

**Evaluation for decision-making.** Yu et al. (2026) show that a world model's prediction quality does not guarantee good decisions, which motivates our separate decision metrics (H4).

**Our focus.** Most of this work fixes the action interface: typically end-effector poses or joint commands. It studies the model rather than the interface. We hold the model fixed and vary only the action representation.

## 2.2 Action representations

**Action spaces in policy learning.** Ganapathi et al. (2022) compare joint and Cartesian action spaces with Implicit Kinematic Policies.

- **Result:** neither is best everywhere.
  - Joint actions win on bimanual box flipping, which relies on elbow contact (98.4% vs. 38.6%).
  - Cartesian actions win on sweeping (79.4% vs. 44.3%).
- **Their solution:** they combine the two through a differentiable kinematic chain.
- **Relation to our work:** their result motivates our hypotheses. Our three tasks transfer the question from policies to world models.

**Actions as images.** Several works render actions as images aligned with the camera:

- Khansari et al. (2020): action images stacked with the observation in one CNN improve grasp success.
- Wang et al. (2025): visual action prompts for video generation.
- Kim et al. (2026): robot-factored world models.
- Alzayer et al. (2026): masked visual actions.

**Recent kinematic conditioning.** Video world models now condition on kinematics rendered into the image:

- iMaC renders robot motion and contact images from the URDF and forward kinematics;
- EA-WM lifts actions and kinematic states into the camera view;
- OSCAR conditions on rendered skeletons across embodiments.

These works show that giving the model the whole robot in image form helps large generative models. None of them isolates the effect of scope (end-effector vs. full body) with matched trajectories that share the same end-effector path. That is the contrast this thesis tests.

## 2.3 Whole-body and whole-arm manipulation

**The arm as a manipulator.** Whole-arm manipulation treats the whole arm, not only the gripper, as a manipulator:

- Salisbury (1987) introduced the idea.
- Eberman (1989) classified it into pushing, searching, enclosure and exclusion.
- More recent examples include forearm pushing for sorting objects (Cosgun et al.) and contact-rich whole-arm control (e.g. TACTIC).

Our Task 3 is a pushing task in this sense.

**The arm as an obstacle.** In collision-aware motion generation, the whole arm matters as an obstacle:

- Fishman et al. (2022) train neural motion planners on procedurally generated environments, including cubbies, which they treat as harder than open tabletops. Our Task 2 follows this environment class.
- Redundancy-resolution and null-space methods (e.g. whole-body obstacle avoidance for 7-DoF arms) exploit the fact that one end-effector pose admits several arm configurations. Our matched pairs build on the same fact.

## 2.4 Model components

**Numeric encoder.** Numeric action sequences are encoded with a simple 1D temporal convolution, following the evidence that convolutional sequence models are competitive with recurrent ones (Bai et al. 2018).

**Image encoder.** Action images are read by a 2D CNN that keeps their spatial layout, in the spirit of Transporter Networks (Zeng et al. 2021).

## References (to complete)

- Agrawal, Nair, Abbeel, Malik, Levine (2016). Learning to poke by poking. arXiv:1606.07419.
- Alzayer et al. (2026). Masked visual actions. arXiv:2607.19343.
- Bai, Kolter, Koltun (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv:1803.01271.
- Borkman et al. (2021). Unity Perception.
- Cosgun et al. Forearm-based multi-object pushing (Monash). *Full citation to check.*
- Eberman (1989). Whole-arm manipulation: kinematics and control. MSc thesis, MIT.
- EA-WM (2026). arXiv:2605.06192.
- Finn, Goodfellow, Levine (2016). Unsupervised learning for physical interaction through video prediction. NeurIPS.
- Fishman et al. (2022). Motion Policy Networks. CoRL.
- Ganapathi, Florence, Varley, Burns, Goldberg, Zeng (2022). Implicit kinematic policies. ICRA. arXiv:2203.01983.
- iMaC (2026). arXiv:2606.09813.
- Khansari et al. (2020). Action image representation. arXiv:2005.06594.
- Kim et al. (2026). Robot-factored world models. arXiv:2607.22535.
- OSCAR (2026). arXiv:2606.04463.
- Salisbury (1987). Whole-arm manipulation. Proc. 4th ISRR.
- Wang et al. (2025). Visual action prompts. ICCV.
- Yin et al. (2026). PlayWorld. arXiv:2603.09030.
- Yu et al. (2026). World model evaluation for decision-making. arXiv:2606.15032.
- Zeng et al. (2021). Transporter Networks. CoRL 2020.
- Zhang & Du (2026). World Action Planner. arXiv:2607.27599.
