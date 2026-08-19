# Research notes and implementation choices

投稿前完整方法调研、相关工作矩阵、新颖性审计与证伪条件见
[`ieee-method-research-review.md`](ieee-method-research-review.md)。该文档优先于本页较早的
MVP 研究备注；任何论文主张都必须先通过其中定义的 Phase A 小规模证据实验。

## Direct precedents

### DROID

DROID's post-hoc camera calibration uses rendered robot masks, IoU,
reprojection errors and reciprocal 3D match counts to assess geometry before
publishing curated calibration subsets. EgoQC-Lite adopts the same principle:
prefer measurable projection and transformation consistency over a generic
vision-language quality score.

- Paper: https://arxiv.org/abs/2403.12945
- Project: https://droid-dataset.github.io/

### HOT3D

HOT3D retains the complete capture, releases validity information, and creates
curated clips only after annotation presence, visibility, exposure and visual
alignment checks. EgoQC-Lite similarly never deletes raw captures and emits
purpose-specific validity through issues and tiers.

- Paper: https://arxiv.org/abs/2411.19167
- Project: https://facebookresearch.github.io/hot3d/

### Ego-Exo4D

Ego-Exo4D separates automatic/pseudo ground truth from manual ground truth and
retains confidence evidence such as the number of triangulation views.
EgoQC-Lite's planned visual stage follows this provenance-first design:
observed, tracked and interpolated pose must not collapse into one boolean.

- Paper: https://arxiv.org/abs/2311.18259
- Pose docs: https://docs.ego-exo4d-data.org/annotations/ego_pose/

### HaWoR

HaWoR separates camera-frame hand reconstruction, camera trajectory, metric
scale and motion infilling. This motivates distinct EgoQC issue families for
hand pose, camera pose, scale, interpolation and representation conversion.

- Paper/code: https://hawor-project.github.io/

## Useful later, not required by the MVP

### EgoVid-5M

CLIP consistency, optical flow, motion smoothness and DOVER clarity are useful
video-level metadata, but too costly to apply blindly to hundreds of TB. They
should run only on sampled or suspicious clips.

- Paper: https://arxiv.org/abs/2411.08380

### DemInf

VAE embeddings and k-nearest-neighbor mutual-information estimates rank robot
trajectories by action diversity and predictability. This evaluates downstream
imitation-learning value after structural/geometry gates, not annotation
correctness.

- Paper/code: https://joeyhejna.com/demonstration-info/

### SCIZOR

SCIZOR combines a self-supervised suboptimal classifier with video semantic
deduplication. Its public reproduction stack requires GPU FAISS, video
encoders, OXE/Octo components and is intentionally not an MVP dependency.

- Code: https://github.com/UT-Austin-RPL/SCIZOR

### SemDeDup and DataComp

SemDeDup motivates embedding-based semantic deduplication; DataComp motivates
validating any filtering policy with a fixed downstream probe. At this scale,
exact hashes and perceptual hashes should precede embeddings.

- SemDeDup: https://arxiv.org/abs/2303.09540
- DataComp: https://arxiv.org/abs/2304.14108

## Teacher/student decision

A general VLM is optional and offline. The planned student is a small temporal
overlay-alignment classifier trained mostly from programmatic corruptions:

- video/pose offsets;
- intrinsic and extrinsic perturbations;
- w2c/c2w inversion;
- left/right swaps and mirror mistakes;
- MANO joint permutation and noise;
- pose freeze and missing spans.

These corruptions provide exact labels without API calls. A small human gold set
calibrates real-world error rates. A local VLM can be added only for uncertain
semantic cases.
