# OSNet-x1.0 re-identification: evaluate, fine-tune, test on harder data

RSKD Labs internship take-home. All numbers below are reproduced by `evaluate.py`; fine-tuning is done by `train.py`.

## Results (assignment steps)

| Step | Model | Test data | mAP | Rank-1 | Split | Image size | Epochs | LR |
|---|---|---|---|---|---|---|---|---|
| 1 | OSNet-x1.0, ImageNet weights only | Market-1501 | 4.0 | 13.6 | standard test split (750 IDs; 3,368 query / 15,913 gallery) | 256×128 | 0 | – |
| 2 | OSNet-x1.0, fine-tuned on Market-1501 | Market-1501 | 71.6 | 89.9 | same | 256×128 | 20 | 0.0015 |
| 3 | same model as step 2, no further training | Market-1501 + 500k distractors | 61.7 | 83.6 | standard test split, 500,000 distractors added to the gallery (515,913 gallery) | 256×128 | 20 | 0.0015 |

**Drop from step 2 to step 3:** mAP −9.9 points (14% relative), Rank-1 −6.3 points (7% relative).

**Step 3 uses the Market-1501 500k distractor set, not MSMT17.** MSMT17 needs a signed release agreement, and it did not arrive before the deadline. Note that this measures robustness to a much larger gallery of strangers on the same cameras. It is not a test on new cameras or a new dataset.

## Low-compute results (2 GB GPU budget)

Vendor hardware will not have a large GPU, so I re-ran the pipeline under a hard 2 GB cap on GPU memory (`torch.cuda.set_per_process_memory_fraction`; a run that needs more crashes). Both extra runs were planned before seeing their results, and the baseline above remains the headline model.

| Model | Market mAP / R1 | +500k mAP / R1 | Drop (mAP) | Train peak GPU | Eval peak GPU | Train time |
|---|---|---|---|---|---|---|
| A. Baseline: CE, fp32, batch 64, no cap | 71.6 / 89.9 | 61.7 / 83.6 | −9.9 | >2 GB (batch 64 OOMs under the cap even with AMP) | 3.3 GB (GPU ranking) | 5.8 min |
| **C. Same recipe, AMP, batch 32 × 2 accumulation, 2 GB cap** | **71.3 / 89.7** | **61.3 / 83.6** | −10.0 | **1.40 GB** | **0.41 GB** | **3.8 min** |
| B. C + triplet loss (weight 0.5, 8 IDs × 4 imgs per batch), 2 GB cap | 72.2 / 89.6 | 59.6 / 81.9 | −12.6 | 1.40 GB | 0.41 GB | 5.0 min |

- **The whole pipeline fits in 2 GB with no meaningful accuracy loss (A vs C, ≤0.4 points, single seed).** It is also faster with mixed precision. Training peaks at 1.4 GB, which leaves headroom for the driver overhead a real 2 GB card has. Batch 16 × 4 also works and peaks at 0.73 GB (measured in a 1-epoch probe).
- **Evaluation needs 411 MB of GPU memory, including searching the 515,913-image gallery.** This comes from fp16 feature extraction plus ranking on the CPU in chunks of 32 queries. fp16 features gave identical scores to fp32 (B: 59.6 / 81.9 both ways).
- **Triplet loss (OSNet paper, Appendix B) did not help in this setting (B vs C).** The likely reasons:
  - under the memory cap each batch holds only 8 identities, so hard-example mining is weak;
  - the triplet term fell to about 0.002 by epoch 15, so it stopped providing signal;
  - the paper's gain was measured on a 150-epoch schedule.
- Timings are from an RTX 5080 Laptop GPU. The memory cap simulates a small GPU's memory, not its speed.

### Training configuration (step 2 baseline)
- Start: torchreid's ImageNet-pretrained OSNet-x1.0. Training: a new 751-way classifier on the Market-1501 train split (751 IDs, 12,936 images).
- Softmax cross-entropy with label smoothing. AMSGrad, lr 0.0015, weight decay 5e-4, batch 64, cosine LR decay.
- 20 epochs: the first 2 train the classifier only, the backbone is frozen. The epoch count came from a 1-epoch timing run (`logs/train_test.txt`).
- Augmentation: random horizontal flip only. Random erasing is off, because the OSNet paper (Appendix A) reports it hurts cross-domain results.
- Last epoch is used. No evaluation on the test set during training and no checkpoint selection.
- Seed 1, single run per model.
- Training time: **5.8 min on an RTX 5080 Laptop GPU**. I estimate about 30 min on a free Colab/Kaggle T4, but I did not measure it.

### Evaluation protocol
- 512-d features from OSNet's final layer, ranked by Euclidean distance. No re-ranking and no test-time augmentation.
- For each query, gallery images of the same person from the same camera are removed (standard Market-1501 protocol). Images labelled junk (`-1`) are excluded.
- Ranking runs in chunks of queries, so the 3,368 × 515,913 distance matrix never has to fit in memory. On the standard gallery, `--check` confirms the numbers match torchreid's own `evaluate_rank` exactly.

## Reproduce

```bash
# environment: install PyTorch for your GPU, then torchreid from source (see requirements.txt)
# data: Market-1501 (and the 500k distractors in Market-1501-v15.09.15/images/) under
#       <data-root>/market1501/Market-1501-v15.09.15/
# all checkpoints are included in checkpoints/, so training is optional

# A. baseline (assignment steps 1-3)
python train.py --epochs 20 --out checkpoints/osnet_x1_0_market.pth --data-root <data-root>
python evaluate.py --step all --weights checkpoints/osnet_x1_0_market.pth --target market500k --check --data-root <data-root>

# C. same recipe within a 2 GB GPU budget
python train.py --epochs 20 --amp --max-vram-gb 2 --batch-size 32 --accum-steps 2 --out checkpoints/osnet_x1_0_market_ce_2gb.pth --data-root <data-root>
python evaluate.py --step 3 --weights checkpoints/osnet_x1_0_market_ce_2gb.pth --max-vram-gb 2 --amp --batch-size 64 --rank-device cpu --chunk 32 --no-cache --data-root <data-root>
#   (use --step 2 for the standard gallery)

# B. C + triplet loss
python train.py --epochs 20 --loss softmax_triplet --amp --max-vram-gb 2 --batch-size 32 --accum-steps 2 --out checkpoints/osnet_x1_0_market_triplet_2gb.pth --data-root <data-root>
```

Every evaluation run appends a row to `results.csv` with all settings, including peak GPU memory. `results_v1.csv` holds the first baseline run, in an older column layout. Console output for every run is in `logs/`.

## Questions

### 1. Why does the cross-dataset score fall, and what would I try first for unlabelled maritime cameras?

**Why it falls.**
- My step-3 drop (−9.9 mAP) comes only from gallery size. It uses the same cameras with 500k extra strangers: more look-alikes get ranked above the harder true matches, so mAP falls faster than Rank-1.
- A real cross-dataset test adds *domain shift*, which is much larger:
  - the new cameras differ in colour response, resolution, viewpoint, lighting and weather;
  - backgrounds and clothing styles differ;
  - the network has partly learned cues specific to Market-1501, including its BatchNorm statistics, and those do not transfer.
- The OSNet paper shows the scale of the drop:
  - OSNet trained *on* MSMT17 reaches only 52.9 mAP there (Table 3);
  - a model trained on Market and tested on Duke gets 48.5 Rank-1 (Table 11), against about 88.6 when trained on Duke itself.

**What I would try first for maritime cameras with no labels, in order of cost:**
1. **Start from a model built to generalise.** That means OSNet-IBN/AIN, which adds instance normalisation to the early layers to remove camera colour and contrast cues. Train it on several labelled public datasets, not just Market. Appendix A shows this alone is competitive with unsupervised adaptation methods, and it needs zero maritime labels.
2. **Re-estimate BatchNorm statistics on unlabelled maritime crops (AdaBN).** It takes minutes, needs no labels and no gradient training, and often recovers part of the gap.
3. **Unsupervised domain adaptation with pseudo-labels** (SpCL, Cluster Contrast):
   - cluster features of unlabelled maritime detections, treat each cluster as an identity, retrain, and repeat;
   - per-camera tracking gives free positive pairs, since detections in one track are the same person.
4. **Maritime-specific risks to design for:**
   - crew in identical coveralls or hi-vis gear, which makes clothing colour a weak cue;
   - glare, spray and night-time IR;
   - long distances, which mean low-resolution crops (Table 6 shows resolution hurts most);
   - heavy occlusion by equipment.

Even with no *training* labels, I would hand-label a small maritime *test* set (a few dozen people across cameras). Without one, none of these steps can be measured.

### 2. Which evaluation mistake would make these numbers look better than they are, and how did I avoid it?

**Counting same-camera matches.** Market-1501 contains many consecutive frames of the same person from the same camera. If those stay in the gallery, the query's nearest neighbour is often a near-duplicate frame, so Rank-1 and mAP measure frame matching, not re-identification across cameras.
- `evaluate.py` removes, for each query, every gallery image with the same identity **and** the same camera. This is the standard Market-1501 protocol.
- On the standard gallery, `--check` confirms my numbers match torchreid's reference `evaluate_rank` exactly (71.60 / 89.88 for step 2).

**The second trap: tuning on the test set.** Market-1501 has no validation split, so it is tempting to evaluate after every epoch and keep the best checkpoint, or to try variants until one wins on the test set.
- `train.py` never touches query or gallery. It trains a fixed schedule and saves the last epoch.
- The two extra 2 GB runs were planned before I saw their results. Both are reported, including the triplet-loss run that did worse.
- The headline model stayed the original baseline.
- Train and test identities are disjoint (standard split). Step 1 uses ImageNet-only weights, not torchreid's ReID-trained model-zoo weights.

### 3. AI tools and web resources used

| Tool / resource | What for |
|---|---|
| Claude (claude.ai chat) | Explaining the assignment; a section-by-section walkthrough of the OSNet paper; planning the pipeline; drafting `train.py` and `evaluate.py`; debugging (checkpoint loading with PyTorch's `weights_only` default, a broken multi-line shell command); planning the 2 GB experiments; drafting this README. |
| Claude Code | Local environment setup: venv, CUDA PyTorch build for the RTX 5080, torchreid install, dataset and weight downloads, an offline verification script (`verify_offline.py`, `SETUP_LOG.md`). |
| [CHECK: any other tools, e.g. ChatGPT, Copilot, Google, Stack Overflow] | |
| torchreid repo and docs | API, dataset layout, `market1501_500k` option |
| Market-1501 project page | Dataset and distractor downloads, filename convention |
| MSMT17 page | Requested access; not received in time |
| OSNet paper (arXiv 1905.00953), incl. Appendices A–B | Training recipe, cross-domain discussion, triplet-loss rule of thumb |
| Bag of Tricks (1903.07071), IBN-Net, SpCL, Cluster Contrast | Background for question 1 |

**What I did and checked myself:**
- ran every experiment and read the logs;
- chose the 20-epoch schedule from a timed 1-epoch run;
- verified the evaluation against torchreid (`--check`);
- caught that a broken line continuation had nearly overwritten the baseline checkpoint;
- chose batch 32 × 2 over 16 × 4 before evaluating;
- decided to report the triplet run even though it was worse.

[CHECK: edit this to match what you actually decided yourself]

## What I would do next (not done for lack of time)
- Hold out ~100 training identities as a validation split for choosing hyperparameters.
- Run 3 seeds per configuration (the paper does), since single-seed differences under ~0.5 points are noise.
- Evaluate on MSMT17 for a true cross-dataset number, and compare OSNet-IBN.
- Time the 20-epoch schedule on an actual free T4.
- **Bandwidth for deployment:**
  - extract features at the camera and send them instead of video;
  - compress them (fp16 or int8; PCA fitted on training features) and measure the accuracy cost;
  - send one averaged feature per track instead of one per frame.
