# SETUP_LOG (2026-11-07)  — set up by Claude Code (AI tool), supervised by user

## Versions
Python 3.10.22 (uv-managed; system Python is 3.14.4 without python3-venv) | torch 2.11.0+cu128 | torchvision 0.26.0+cu128
torchreid 1.4.0 (editable, Cython rank eval extension built OK) | numpy 2.2.6 | scipy 1.15.3 | Cython 3.3.1 | gdown 6.5.1
GPU: RTX 5080 Laptop (sm_120, Blackwell), driver 615.71.09, CUDA build 12.8 -> torch.cuda.is_available() True.

## Decisions / commands that mattered
1. nvidia-smi found GPU -> CUDA build: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` (cu128 needed for Blackwell).
2. `python3 -m venv` FAILED (no ensurepip, apt python3-venv missing). Fix without sudo: `uv venv --python 3.10 --seed env`.
3. `git clone https://github.com/KaiyangZhou/deep-person-reid.git`; `pip install -r requirements.txt`; `pip install cython jupyter matplotlib pandas gdown`; `python setup.py develop` -> OK first try.
   numpy<1.24 pin was NOT needed (numpy 2.2.6 imports and runs fine; verify script passes).
4. gdown 6.5.1 has no `--fuzzy` flag (first attempt failed); passing the URL directly works.
5. Market-1501: official Drive link OK, 152,733,771 B zip, `unzip -t` clean. Extracted to data/market1501/Market-1501-v15.09.15/.
   Counts: bounding_box_train 12936, bounding_box_test 19732 (gallery incl. junk), query 3368 (torchreid: 751 ids train, 750 query ids, gallery 15913 after junk removal).
6. **Market-1501 500k distractors: FAILED.** Official Drive link -> gdown "Cannot retrieve the public link of the file ... or have had many accesses".
   Per instructions I stopped, did not retry or use another source. NEEDS MANUAL BROWSER DOWNLOAD.
   Expected location after extracting: data/market1501/Market-1501-v15.09.15/images/ (~500k jpgs); use `market1501_500k=True`.
7. Weights (ImageNet only, via pretrained=True): osnet_x1_0_imagenet.pth (10,910,553 B), osnet_ibn_x1_0_imagenet.pth (10,914,272 B).
   torchreid cache: ~/.cache/torch/checkpoints/ (copies in weights/). pretrained=True ONLY looks in that cache, not weights/;
   if the cache is lost: `cp weights/*.pth ~/.cache/torch/checkpoints/`. No MSMT17 / model-zoo weights downloaded.
8. Papers (arxiv.org/pdf/<id>, first-page titles verified with pypdf): 1905.00953 OSNet, 1910.06827 OSNet-AIN, 1807.09441 IBN-Net,
   1903.07071 Bag of Tricks, 2006.02713 SpCL, 2103.11568 Cluster Contrast. (8.3 MB)
9. Docs: `wget -r -np -k -p -E` mirror of kaiyangzhou.github.io/deep-person-reid -> docs/kaiyangzhou.github.io/deep-person-reid/index.html
   (user_guide, datasets, MODEL_ZOO, evaluation, pkg/*, source modules). Also repo's own docs/ and configs/ are in deep-person-reid/.
10. Offline verification: `unshare -rn` NOT permitted (user namespaces restricted; didn't use sudo). Instead verify_offline.py, with OFFLINE_BLOCK=1,
    patches socket (DNS + non-loopback connect raise) and I set HF_HUB_OFFLINE=1 + dead proxies. This is process-level, not kernel-level, isolation.
    Run: `OFFLINE_BLOCK=1 HF_HUB_OFFLINE=1 env/bin/python -I verify_offline.py`
    Result: DNS blocked; num_train_pids 751; query 3368; gallery 15913; train 12936; osnet_x1_0 pretrained loaded from cache;
    batch (32,3,256,128) -> logits (32,751) on cuda; VERIFY_OK.
    Distractor check: 0 images found -> MISSING (script reports this and runs the 500k test only when present).

## Sizes
env 7.5G | market1501 270M | weights 21M | papers 8.3M | docs 7.2M | deep-person-reid 20M

## Notes
- The ~/.cache/torch/checkpoints dir was briefly moved and restored during testing; contents intact.
- Data placed under data/ must stay at root=~/Documents/RSKDLabs/data.
