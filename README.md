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
TODO

### 2. Which evaluation mistake would make these numbers look better than they are, and how did I avoid it?
TODO

### 3. AI tools and web resources used
TODO
