# OSNet-x1.0 re-identification: evaluate, fine-tune, test on harder data

RSKD Labs internship take-home. All numbers below are reproduced by `evaluate.py`; fine-tuning is done by `train.py`.

## Results

| Step | Model | Test data | mAP | Rank-1 | Split | Image size | Epochs | LR |
|---|---|---|---|---|---|---|---|---|
| 1 | OSNet-x1.0, ImageNet weights only | Market-1501 | 4.0 | 13.6 | standard test split (750 IDs; 3,368 query / 15,913 gallery) | 256×128 | 0 | – |
| 2 | OSNet-x1.0, fine-tuned on Market-1501 | Market-1501 | 71.6 | 89.9 | same | 256×128 | 20 | 0.0015 |
| 3 | same model as step 2, no further training | Market-1501 + 500k distractors | 61.7 | 83.6 | standard test split, 500,000 distractors added to the gallery (515,913 gallery) | 256×128 | 20 | 0.0015 |

**Drop from step 2 to step 3:** mAP −9.9 points (14% relative), Rank-1 −6.3 points (7% relative).

**Step 3 uses the Market-1501 500k distractor set, not MSMT17.** MSMT17 needs a signed release agreement, and it did not arrive before the deadline. Note that this measures robustness to a much larger gallery of strangers on the same cameras. It is not a test on new cameras or a new dataset.

Full output: `results.csv`, `logs/`.

### Training configuration (step 2)
- Start: torchreid's ImageNet-pretrained OSNet-x1.0. Training: a new 751-way classifier on the Market-1501 train split (751 IDs, 12,936 images).
- Softmax cross-entropy with label smoothing. AMSGrad, lr 0.0015, weight decay 5e-4, batch 64, cosine LR decay.
- 20 epochs: the first 2 train the classifier only, the backbone is frozen.
- Augmentation: random horizontal flip only. Random erasing is off, because the OSNet paper (Appendix A) reports it hurts cross-domain results.
- Last epoch is used. No evaluation on the test set during training and no checkpoint selection.
- Seed 1, single run.
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

# train (or use the included checkpoints/osnet_x1_0_market.pth)
python train.py --epochs 20 --out checkpoints/osnet_x1_0_market.pth --data-root <data-root>

# every number in the table
python evaluate.py --step all --weights checkpoints/osnet_x1_0_market.pth \
    --target market500k --check --data-root <data-root>
```

## Questions

### 1. Why does the cross-dataset score fall, and what would I try first for unlabelled maritime cameras?
TODO

### 2. Which evaluation mistake would make these numbers look better than they are, and how did I avoid it?
TODO

### 3. AI tools and web resources used
TODO
