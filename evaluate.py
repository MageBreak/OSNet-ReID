"""
evaluate.py - reproduces every number in the results table.

  Step 1: OSNet-x1.0 with ImageNet weights only        -> Market-1501 test set
  Step 2: OSNet-x1.0 fine-tuned on Market (train.py)   -> Market-1501 test set
  Step 3: the same fine-tuned model, no more training  -> MSMT17 test set
          or Market-1501 test set with 500k distractors added to the gallery

Usage:
  python evaluate.py --step 1
  python evaluate.py --step 2 --weights checkpoints/osnet_x1_0_market.pth
  python evaluate.py --step 3 --weights checkpoints/osnet_x1_0_market.pth --target market500k
  python evaluate.py --step all --weights checkpoints/osnet_x1_0_market.pth --target market500k
  add --check to also run torchreid's own evaluation and confirm the numbers match

Evaluation protocol (standard Market-1501 / MSMT17 protocol):
  - a query is ranked against the whole gallery by feature distance
  - gallery images of the SAME person from the SAME camera as the query are removed
    (otherwise near-duplicate frames make the task trivially easy)
  - Rank-k = % of queries whose first correct match is in the top k
  - mAP    = mean over queries of average precision over all correct matches
The ranking is done in chunks of queries so the 3,368 x 515,913 distance matrix
for the 500k-distractor gallery never has to fit in memory at once.
"""
import argparse
import csv
import datetime
import hashlib
import os

import torch
import torchreid

TARGETS = {
    # name: (torchreid dataset name, extra dataset kwargs, split description)
    "market1501": ("market1501", {}, "standard test split (750 IDs; 3,368 query / 15,913 gallery)"),
    "market500k": ("market1501", {"market1501_500k": True},
                   "standard test split + 500k distractors in gallery (515,913 gallery)"),
    "msmt17": ("msmt17", {}, "standard test split (3,060 IDs; 11,659 query / 82,161 gallery)"),
}


def unpack(batch):
    if isinstance(batch, dict):
        return batch["img"], batch["pid"], batch["camid"]
    return batch[0], batch[1], batch[2]


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:12]


def build_model(weights, device):
    """weights=None -> ImageNet weights (step 1); otherwise a train.py checkpoint."""
    model = torchreid.models.build_model(
        name="osnet_x1_0", num_classes=751, loss="softmax", pretrained=(weights is None)
    )
    config = {}
    if weights is not None:
        # weights_only loading (safe); TorchVersion is allowed because early train.py
        # versions stored torch.__version__ in the checkpoint config
        from torch.torch_version import TorchVersion
        with torch.serialization.safe_globals([TorchVersion]):
            ckpt = torch.load(weights, map_location="cpu")
        state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        model.load_state_dict(state)  # strict: fails loudly if any weight is missing
        config = ckpt.get("config", {})
    return model.to(device).eval(), config


@torch.no_grad()
def extract(model, loader, device):
    """Run images through the network; in eval mode OSNet returns the 512-d feature."""
    feats, pids, camids = [], [], []
    for batch in loader:
        imgs, pid, camid = unpack(batch)
        feats.append(model(imgs.to(device, non_blocking=True)).float().cpu())
        pids.append(pid)
        camids.append(camid)
    return torch.cat(feats), torch.cat(pids), torch.cat(camids)


def get_features(model, loader, device, cache_path):
    if cache_path and os.path.exists(cache_path):
        print(f"  using cached features {cache_path}")
        return torch.load(cache_path)
    out = extract(model, loader, device)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(out, cache_path)
    return out


def distance(qf, gf, metric):
    if metric == "cosine":
        qf = torch.nn.functional.normalize(qf, dim=1)
        gf = torch.nn.functional.normalize(gf, dim=1)
        return 1 - qf @ gf.t()
    # squared euclidean (same ranking as euclidean, as in torchreid)
    return (qf.pow(2).sum(1, keepdim=True) + gf.pow(2).sum(1)[None, :] - 2 * qf @ gf.t())


@torch.no_grad()
def rank_metrics(qf, q_pids, q_camids, gf, g_pids, g_camids, metric, device,
                 chunk=128, ranks=(1, 5, 10)):
    """Chunked Market-style evaluation. Returns (mAP, {k: rank-k}) as percentages."""
    gf, g_pids, g_camids = gf.to(device), g_pids.to(device), g_camids.to(device)
    G = gf.size(0)
    ap_sum, n_valid, hits = 0.0, 0, {k: 0 for k in ranks}

    for s in range(0, qf.size(0), chunk):
        q = qf[s:s + chunk].to(device)
        qp = q_pids[s:s + chunk].to(device)[:, None]
        qc = q_camids[s:s + chunk].to(device)[:, None]

        order = distance(q, gf, metric).argsort(dim=1)      # gallery sorted by distance
        same_id = g_pids[order] == qp
        junk = same_id & (g_camids[order] == qc)            # same person, same camera
        keep = ~junk
        match = same_id & keep                              # the true matches we count

        pos = keep.cumsum(1)                                # rank after removing junk
        n_rel = match.sum(1)
        valid = n_rel > 0                                   # queries with >=1 true match

        first = torch.where(match, pos, torch.full_like(pos, G + 1)).min(1).values
        for k in ranks:
            hits[k] += ((first <= k) & valid).sum().item()

        precision = match.cumsum(1).float() / pos.clamp(min=1).float()
        ap = (precision * match).sum(1) / n_rel.clamp(min=1)
        ap_sum += ap[valid].sum().item()
        n_valid += valid.sum().item()
        del order, same_id, junk, keep, match, pos, precision

    return 100 * ap_sum / n_valid, {k: 100 * hits[k] / n_valid for k in ranks}


def torchreid_check(qf, q_pids, q_camids, gf, g_pids, g_camids, metric):
    """Cross-check against torchreid's built-in evaluation (only for small galleries)."""
    from torchreid.metrics import evaluate_rank
    dist = distance(qf, gf, metric).numpy()
    cmc, mAP = evaluate_rank(dist, q_pids.numpy(), g_pids.numpy(),
                             q_camids.numpy(), g_camids.numpy())
    return 100 * mAP, 100 * cmc[0]


def run_step(step, args, device):
    weights = None if step == 1 else args.weights
    if step != 1 and not weights:
        raise SystemExit(f"step {step} needs --weights (a checkpoint from train.py)")
    target = "market1501" if step in (1, 2) else args.target
    ds_name, ds_kwargs, split_desc = TARGETS[target]

    print(f"\n=== Step {step}: weights={weights or 'ImageNet'} | target={target} ===")
    model, cfg = build_model(weights, device)

    dm = torchreid.data.ImageDataManager(
        root=os.path.expanduser(args.data_root), sources="market1501", targets=ds_name,
        height=args.height, width=args.width, batch_size_test=args.batch_size,
        workers=args.workers, **ds_kwargs,
    )
    loaders = dm.test_loader[ds_name]

    tag = "imagenet" if weights is None else file_hash(weights)
    cache = None if args.no_cache else os.path.join(
        args.cache_dir, f"{tag}_{target}_{args.height}x{args.width}")
    qf, q_pids, q_camids = get_features(model, loaders["query"], device,
                                        cache and cache + "_query.pt")
    gf, g_pids, g_camids = get_features(model, loaders["gallery"], device,
                                        cache and cache + "_gallery.pt")
    print(f"  query {qf.size(0)} | gallery {gf.size(0)} | feature dim {qf.size(1)}")

    mAP, cmc = rank_metrics(qf, q_pids, q_camids, gf, g_pids, g_camids,
                            args.dist, device, chunk=args.chunk)
    print(f"  mAP {mAP:.1f} | Rank-1 {cmc[1]:.1f} | Rank-5 {cmc[5]:.1f} | Rank-10 {cmc[10]:.1f}")

    if args.check:
        if gf.size(0) > 100_000:
            print("  --check skipped: gallery too large for torchreid's in-memory evaluation")
        else:
            t_map, t_r1 = torchreid_check(qf, q_pids, q_camids, gf, g_pids, g_camids, args.dist)
            ok = abs(t_map - mAP) < 0.05 and abs(t_r1 - cmc[1]) < 0.05
            print(f"  torchreid check: mAP {t_map:.2f} | Rank-1 {t_r1:.2f} -> "
                  f"{'MATCH' if ok else 'MISMATCH'}")

    row = {
        "step": step,
        "model": "OSNet-x1.0 (ImageNet weights)" if weights is None
                 else "OSNet-x1.0 (fine-tuned on Market-1501)",
        "train_data": "ImageNet only" if weights is None else "Market-1501 train (751 IDs)",
        "test_data": target,
        "mAP": round(mAP, 1),
        "rank1": round(cmc[1], 1),
        "rank5": round(cmc[5], 1),
        "split": split_desc,
        "image_size": f"{args.height}x{args.width}",
        "epochs": cfg.get("epochs", 0),
        "lr": cfg.get("lr", "-"),
        "distance": args.dist,
        "weights_sha": tag,
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if f.tell() == 0:
            w.writeheader()
        w.writerow(row)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--step", choices=["1", "2", "3", "all"], required=True)
    p.add_argument("--weights", help="checkpoint from train.py (steps 2 and 3)")
    p.add_argument("--target", choices=["market500k", "msmt17"], default="market500k",
                   help="unseen-data test for step 3")
    p.add_argument("--data-root", default="~/Documents/RSKDLabs/data")
    p.add_argument("--dist", choices=["euclidean", "cosine"], default="euclidean")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk", type=int, default=128, help="queries ranked at once")
    p.add_argument("--cache-dir", default="features")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--check", action="store_true",
                   help="cross-check against torchreid's own evaluation")
    p.add_argument("--out", default="results.csv")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    steps = [1, 2, 3] if args.step == "all" else [int(args.step)]
    rows = {s: run_step(s, args, device) for s in steps}

    print("\n| Step | Model | Test data | mAP | Rank-1 | Split | Image size | Epochs | LR |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s, r in rows.items():
        print(f"| {s} | {r['model']} | {r['test_data']} | {r['mAP']} | {r['rank1']} | "
              f"{r['split']} | {r['image_size']} | {r['epochs']} | {r['lr']} |")
    if 2 in rows and 3 in rows:
        d_map = rows[2]["mAP"] - rows[3]["mAP"]
        d_r1 = rows[2]["rank1"] - rows[3]["rank1"]
        print(f"\nDrop from step 2 to step 3: mAP -{d_map:.1f} points "
              f"({100 * d_map / rows[2]['mAP']:.0f}% relative), "
              f"Rank-1 -{d_r1:.1f} points ({100 * d_r1 / rows[2]['rank1']:.0f}% relative)")


if __name__ == "__main__":
    main()
