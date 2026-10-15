# HomeVision — Indoor Scene Recognition

Classifying indoor scenes into five room types to drive smart-home automation, and measuring
**exactly what ImageNet pre-training is worth** on a small dataset.

Two models are trained under a protocol designed so that the only variable between them is
pre-training: same split, same input pipeline, same augmentation, same epoch budget, same stopping
rule, same test set evaluated once.

| | Custom CNN (scratch) | MobileNetV2 (transfer) |
|---|---|---|
| Test accuracy | 0.6522 | **0.8012** |
| Macro F1 | 0.6472 | **0.7986** |
| Top-2 accuracy | 0.8696 | **0.9524** |
| Parameters | 1,208,677 | 2,264,389 |
| Epochs to converge | 45 | **28** |
| Training time | 18.6 min | **3.4 min** |

**Pre-training is worth +0.151 macro F1 — a 23.4% relative gain — in under a fifth of the training
time.**

---

## The problem

A ceiling- or wall-mounted camera has to recognise which room it is looking at, so the building can
act on that context: lighting presets, HVAC set-points, occupancy-aware automation.

Real rooms make this hard. Lighting runs from daylight to a single lamp, furniture moves, cameras
sit at odd angles, scenes are cluttered, and several room types genuinely look alike.

**Classes:** `bedroom` · `kitchen` · `livingroom` · `restaurant` · `bar`
**Data:** [MIT Indoor Scenes](https://web.mit.edu/torralba/www/indoor.html) — 3,219 usable images
after an integrity pass.

The five classes were chosen under two constraints set before any training: at least 400 original
images per class, and at most 1.5× duplication when balancing. That ruled out `bathroom` (197
images) and `dining_room` (274), which would have needed 3.65× and 2.62× duplication.

---

## Method

**Data pipeline.** Every candidate file is decoded once up front, so truncated JPEGs and mislabelled
PNGs fail at setup rather than mid-epoch inside `tf.data`. Stratified 70/15/15 split made at file
level *before* any image is loaded, with assertions that no file appears in two splits. The training
set is then resampled to 500 images per class — **after** the split, so no duplicate can reach
validation or test.

**Preprocessing lives inside each model, not in the pipeline.** The shared pipeline stops at raw
0–255 pixels; the from-scratch CNN starts with `Rescaling(1/255)`, MobileNetV2 with its own
`preprocess_input`. Feeding a pre-trained backbone the wrong input range collapses accuracy
*silently, with no error* — binding each model to its own scaling makes that mistake structurally
impossible while both still share one identical pipeline.

**Augmentation as in-model layers**, so Keras applies it in training only by construction rather
than by remembering a flag. Each transform maps to a real deployment failure mode — variable
lighting, camera angle, mounting distance, clutter. No vertical flips: a ceiling camera never
produces an inverted room.

**Two-stage transfer learning.** Feature extraction with a frozen base (6,405 trainable parameters
of 2.26M — too few to memorise 2,253 images), then fine-tuning the top 30 layers at 1e-5, with
BatchNorm deliberately left frozen so ImageNet's running statistics aren't corrupted by batches of
32.

---

## What was measured, not assumed

**Learning-rate sweep.** Three rates, same seed, fixed budget, early stopping disabled so the
comparison is controlled.

| rate | best val accuracy | |
|---|---|---|
| 1e-2 | 0.7867 | overshoots — val loss climbed while accuracy stalled |
| **1e-3** | **0.8178** | **winner, margin 0.031** |
| 1e-4 | 0.7164 | still improving when the budget expired |

**Augmentation ablation.** Two identical feature-extraction models, 12-epoch fixed budget.

| arm | best val acc. | final train acc. | train−val gap |
|---|---|---|---|
| no augmentation | 0.8199 | 0.9036 | **+0.084** |
| with augmentation | 0.8095 | 0.7628 | **−0.014** |

Augmentation costs 0.0104 accuracy on a short budget and removes the generalisation gap entirely.
The negative gap is the point: the augmented model is taught on distorted images and graded on clean
ones, which is the behaviour that survives a real camera.

---

## Findings worth reading

**The hard pair wasn't the one I predicted.** I selected `restaurant` and `bar` as the deliberately
confusable pair. They do confuse — 14.4% of bars read as restaurants — but both still score ~0.8 F1.
The weakest class is **livingroom at 0.604 recall**, with 19.8% misread as bedroom. In hindsight the
difficulty is which diagnostic object survives the frame: a bar's bottle wall is large and usually
visible, while a livingroom is partly defined by the *absence* of a bed or a hob.

**Confidence is informative, so abstention works.** Mean confidence is 0.879 when the model is right
and 0.650 when wrong. Withholding predictions below τ = 0.70 answers 76% of frames at 88.8%
accuracy. A camera sees a new frame every few seconds, so "not sure, change nothing" beats
confidently switching a bedroom into kitchen ventilation.

**The efficient model measured slower.** MobileNetV2: 149.9 ms vs the CNN's 27.3 ms — on a T4 GPU.
MobileNetV2 optimises multiply-accumulates, not GPU throughput; its depthwise convolutions are
memory-bandwidth-bound and its 150+ layers incur per-layer launch overhead at batch size 1. On the
CPU/NPU hardware this would actually deploy to, that ordering typically reverses. The table ranks
the models on hardware the product would never use, so the deployment decision needs re-measuring on
target.

**Grad-CAM confirms the model looks at the right things** — heat on beds, counters, sofas, bottle
shelves rather than on watermarks or borders. Test accuracy alone would never reveal the difference.

---

## Repository

```
notebooks/indoor_scene_recognition.ipynb   Full pipeline: acquisition → deployment analysis
report/technical_report.pdf                Written analysis, Harvard-referenced
```

The notebook is self-contained and runs top-to-bottom in Google Colab on a T4. It downloads the
dataset itself (Kaggle API token required) and takes roughly 25 minutes end to end. All outputs are
committed, so it is readable without running anything.

---

## Honest limitations

- **No cross-validation.** Single split, so these are point estimates without confidence intervals.
  Five-fold across two models would be ~5× the runtime.
- **Latency measured on the wrong hardware**, as above.
- **Domain shift.** MIT Indoor images are human-composed photographs — well-lit, well-framed. A real
  ceiling camera produces wide-angle, badly-lit, cluttered frames. The reported accuracy is an
  optimistic upper bound.
- **Geographic bias.** The dataset skews Western and affluent. Deploying elsewhere would need local
  data first — a fairness problem, not just an accuracy one.
- **Single-frame prediction**, discarding the strongest prior available: a fixed camera is in the
  same room it was in a second ago. Temporal smoothing is the highest value-per-effort improvement
  and needs no retraining.

---

## Built with

TensorFlow 2.20 · Keras 3.13 · scikit-learn · pandas · matplotlib · seaborn

Dataset: Quattoni & Torralba, *Recognizing Indoor Scenes*, CVPR 2009. Not redistributed here — the
notebook fetches it.
