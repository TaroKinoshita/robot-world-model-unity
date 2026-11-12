# Meeting prep (private, not for Rekha) — if asked to redesign the image model

**Stance:** do not redesign. Report the cause and the early-fusion result in the discussion, and put the redesign in future work.

## Three reasons

1. **Scope of the question.** The thesis asks how much of the robot the model should see (EEF vs full body), not which input format is strongest. The full-body effect (H2, H5) holds for both formats, so a stronger image model would not change the conclusion.
2. **The test set is spent.** The design was pre-registered and test was used once. Redesigning after seeing the results and re-testing on the same set is no longer a clean evaluation. A redesign needs a new test set, i.e. new data generation.
3. **The cause is already explained.** The images contain the information (no-learning rule ≈ 0.9). Early fusion improves H5 by +0.08 (exploratory). The remaining directions (resolution, skip connection to the mask decoder) are concrete future work.

## Lines to use

> "The research question is how much of the robot the model should see, not which input format is strongest. The full-body effect holds for both formats, so improving the image model wouldn't change that conclusion."

> "The design was pre-registered and the test set has already been used once. Re-designing now and re-testing on the same set would no longer be a clean evaluation."

> "I checked why the image model was weaker: the information is in the images, and early fusion improves H5 by 0.08 in an exploratory run. I'll report the cause and the fix direction in the discussion and future work."

## Likely follow-ups

- **"Early fusion is better, so why not make it the main model?"** — Because it was chosen after seeing test-era results; it is reported honestly as exploratory.
- **"Isn't the image model just buggy?"** — Camera alignment and information content were verified in Unity (1.45 px median offset; no-learning rule ≈ 0.9). It is a design limit, not a bug. Wang et al. (2025) also note numeric states are nearly lossless with a fixed camera.
- **"You still have time."** — The ~9 weeks left go to writing. A redesign means data generation, training and evaluation again.

## Fallback if pushed hard

Offer a bounded version instead of Phase 3: generate a **new** test set and evaluate the already-trained early-fusion models **once**. About 1–2 weeks. Fix the scope myself so it does not grow.
