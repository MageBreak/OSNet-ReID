# OSNet-x1.0 re-identification: evaluate, fine-tune, test on harder data

## Results (assignment steps)

| Step | Model | Test data | mAP | Rank-1 | Split | Image size | Epochs | LR |
|---|---|---|---|---|---|---|---|---|
| 1 | OSNet-x1.0, ImageNet weights only | Market-1501 | 4.0 | 13.6 | standard test split (750 IDs; 3,368 query / 15,913 gallery) | 256×128 | 0 | – |
| 2 | OSNet-x1.0, fine-tuned on Market-1501 | Market-1501 | 71.6 | 89.9 | same | 256×128 | 20 | 0.0015 |
| 3 | same model as step 2, no further training | Market-1501 + 500k distractors | 61.7 | 83.6 | standard test split + 500,000 distractors in the gallery (515,913 gallery) | 256×128 | 20 | 0.0015 |

**Drop from step 2 to step 3:** mAP −9.9 points (14% relative), Rank-1 −6.3 points (7% relative).

**Step 3 uses the Market-1501 500k distractor set, not MSMT17** (MSMT17 access did not arrive in time).

## Low-compute results (2 GB VRAM constraint)

| Model | Market mAP / R1 | +500k mAP / R1 | Drop (mAP) | Train peak GPU | Eval peak GPU | Train time |
|---|---|---|---|---|---|---|
| A. Baseline: CE, fp32, batch 64, no cap | 71.6 / 89.9 | 61.7 / 83.6 | −9.9 | >2 GB (OOMs under the cap) | 3.3 GB | 5.8 min |
| **C. Same recipe, AMP, batch 32 × 2 accumulation, 2 GB cap** | **71.3 / 89.7** | **61.3 / 83.6** | −10.0 | **1.40 GB** | **0.41 GB** | **3.8 min** |
| B. C + triplet loss (weight 0.5), 2 GB cap | 72.2 / 89.6 | 59.6 / 81.9 | −12.6 | 1.40 GB | 0.41 GB | 5.0 min |

- **The pipeline fits in 2 GB with no meaningful accuracy loss** (A vs C: ≤0.4 points, single seed), and it is faster with mixed precision.
- **Evaluation needs 411 MB of GPU memory, including the 515,913-image gallery** (fp16 features, ranking on the CPU).
- Triplet loss did not help in this short, memory-limited run. With batch 32 × 2, each micro-batch holds only 8 identities, so hard-example mining is weak.
- Timings are from an RTX 5080 Laptop GPU. The memory cap simulates a small GPU's memory, not its speed.

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

## Questions

**1. Why does the cross-dataset score fall, and what would I try first for unlabelled maritime cameras?**

My step-3 drop comes only from a larger gallery on the same cameras. A real cross-dataset test adds domain shift: new cameras, lighting, viewpoints, backgrounds and clothing, plus Market-specific cues the network has learned, including its BatchNorm statistics.

For maritime cameras with no labels, I would try, in order:
1. OSNet-IBN, trained on several public datasets. Its instance normalisation makes it more robust to new cameras (OSNet paper, Appendix A).
2. Re-estimating BatchNorm statistics on unlabelled maritime footage.
3. Pseudo-label adaptation: cluster unlabelled detections, treat clusters as identities, retrain and repeat (SpCL / Cluster Contrast).

I would also hand-label a small maritime test set, so that progress can be measured.

**2. Which evaluation mistake would make these numbers look better than they are, and how did I avoid it?**

Counting gallery images of the same person from the same camera, which are near-duplicate frames, as correct matches. `evaluate.py` removes them, and `--check` confirms the results match torchreid's own evaluation. I also did not tune on the test set:
- fixed 20-epoch schedule, last epoch saved;
- extra runs planned in advance, all reported;
- the baseline stays the headline model.

**3. AI tools and web resources used**
- **Claude (chat):** explaining the OSNet paper, writing the first drafts and debugging `train.py` / `evaluate.py`, planning the 2 GB experiments, formatting this README.
- **Claude Code:** environment setup.
- **Web resources:** torchreid repo and docs, Market-1501 and MSMT17 pages, the OSNet paper (arXiv 1905.00953).
- **I ran every experiment and checked the outputs myself.**
