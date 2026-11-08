"""
train.py - fine-tune OSNet-x1.0 (ImageNet weights) on Market-1501 (assignment step 2).

Baseline recipe: a shortened version of the paper's fine-tuning setup (Sec. 4.1):
  softmax cross-entropy + label smoothing, AMSGrad, weight decay 5e-4, batch 64,
  256x128 images, classifier-only warm-up ("fixbase"), cosine learning-rate decay.

Optional (chosen in advance from the OSNet paper, Appendix B - not tuned on the test set):
  --loss softmax_triplet   adds a hard-mining triplet loss (weight 0.5), with batches of
                           P identities x K images (RandomIdentitySampler)
  --amp                    mixed precision (fp16): less GPU memory, faster
  --max-vram-gb 2          hard cap on GPU memory; the run fails if it needs more
  --accum-steps N          gradient accumulation (smaller batches in memory, same update size)

The test set (query/gallery) is never touched here: no evaluation during training,
no picking the "best" epoch. We train for a fixed schedule and save the last epoch.

Usage:
  python train.py --epochs 20 --out checkpoints/osnet_x1_0_market.pth
  python train.py --epochs 20 --loss softmax_triplet --amp --max-vram-gb 2 \
      --out checkpoints/osnet_x1_0_market_triplet_2gb.pth
"""
import argparse
import json
import os
import time

import torch
import torchreid
from torchreid.utils import open_all_layers, open_specified_layers, set_random_seed


def unpack(batch):
    """torchreid batches are dicts in recent versions, tuples in older ones."""
    if isinstance(batch, dict):
        return batch["img"], batch["pid"]
    return batch[0], batch[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="~/Documents/RSKDLabs/data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--fixbase-epochs", type=int, default=2,
                   help="epochs at the start where only the new classifier trains")
    p.add_argument("--lr", type=float, default=0.0015)
    p.add_argument("--batch-size", type=int, default=64, help="images per forward pass")
    p.add_argument("--accum-steps", type=int, default=1,
                   help="forward passes per weight update (effective batch = batch-size x this)")
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--transforms", nargs="+", default=["random_flip"],
                   help="torchreid augmentations, e.g. random_flip random_erase color_jitter")
    p.add_argument("--loss", choices=["softmax", "softmax_triplet"], default="softmax")
    p.add_argument("--triplet-weight", type=float, default=0.5)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--num-instances", type=int, default=4,
                   help="images per identity in a batch (triplet loss only)")
    p.add_argument("--amp", action="store_true", help="mixed precision (fp16)")
    p.add_argument("--max-vram-gb", type=float, default=None,
                   help="cap on GPU memory PyTorch may allocate")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="checkpoints/osnet_x1_0_market.pth")
    args = p.parse_args()

    set_random_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = args.amp and device == "cuda"
    if device == "cuda":
        if args.max_vram_gb:
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1.0, args.max_vram_gb * 1024**3 / total))
            print(f"GPU memory capped at {args.max_vram_gb} GB")
        else:
            torch.backends.cudnn.benchmark = True  # autotuning can need extra memory, so only uncapped
        torch.cuda.reset_peak_memory_stats()

    use_triplet = args.loss == "softmax_triplet"
    if use_triplet and args.batch_size % args.num_instances != 0:
        raise SystemExit("--batch-size must be divisible by --num-instances")

    # ---- data: Market-1501 training split only (751 identities, 12,936 images) ----
    sampler = (dict(train_sampler="RandomIdentitySampler", num_instances=args.num_instances)
               if use_triplet else {})
    dm = torchreid.data.ImageDataManager(
        root=os.path.expanduser(args.data_root),
        sources="market1501",
        targets="market1501",
        height=args.height,
        width=args.width,
        transforms=args.transforms,
        batch_size_train=args.batch_size,
        batch_size_test=256,
        workers=args.workers,
        **sampler,
    )
    num_classes = dm.num_train_pids  # 751

    # ---- model: OSNet-x1.0 starting from ImageNet weights, new 751-way classifier ----
    # loss="triplet" makes OSNet return (class scores, 512-d feature) in train mode
    model = torchreid.models.build_model(
        name="osnet_x1_0", num_classes=num_classes,
        loss="triplet" if use_triplet else "softmax", pretrained=True,
    ).to(device)

    optimizer = torchreid.optim.build_optimizer(
        model, optim="amsgrad", lr=args.lr, weight_decay=5e-4
    )
    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer, lr_scheduler="cosine", max_epoch=args.epochs
    )
    criterion = torchreid.losses.CrossEntropyLoss(
        num_classes=num_classes, use_gpu=(device == "cuda"), label_smooth=True
    )
    triplet = torchreid.losses.TripletLoss(margin=args.margin) if use_triplet else None
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"device={device} | classes={num_classes} | train images={len(dm.train_loader.dataset)}")
    print(f"effective batch = {args.batch_size} x {args.accum_steps} = "
          f"{args.batch_size * args.accum_steps}")
    print(f"config: {vars(args)}")

    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        if epoch < args.fixbase_epochs:
            open_specified_layers(model, ["classifier"])  # freeze backbone, train classifier
        else:
            open_all_layers(model)

        t_epoch, loss_sum, trip_sum, correct, seen = time.time(), 0.0, 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(dm.train_loader):
            imgs, pids = unpack(batch)
            imgs, pids = imgs.to(device, non_blocking=True), pids.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(imgs)
            logits, feats = out if use_triplet else (out, None)

            loss = criterion(logits.float(), pids)            # losses computed in fp32
            if use_triplet:
                loss_t = triplet(feats.float(), pids)
                loss = loss + args.triplet_weight * loss_t
                trip_sum += loss_t.item() * pids.size(0)

            scaler.scale(loss / args.accum_steps).backward()
            if (i + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            loss_sum += loss.item() * pids.size(0)
            correct += (logits.argmax(1) == pids).sum().item()
            seen += pids.size(0)

        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()
        phase = "classifier-only" if epoch < args.fixbase_epochs else "all layers"
        trip_txt = f" (triplet {trip_sum / seen:.4f})" if use_triplet else ""
        print(f"epoch {epoch + 1}/{args.epochs} [{phase}] "
              f"loss {loss_sum / seen:.4f}{trip_txt} | train acc {100 * correct / seen:.1f}% | "
              f"lr {lr_now:.6f} | {time.time() - t_epoch:.0f}s")

    train_minutes = (time.time() - t_start) / 60
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    peak = {}
    if device == "cuda":
        peak = {"peak_vram_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20),
                "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20)}
        print(f"peak GPU memory: {peak['peak_vram_allocated_mb']} MB allocated, "
              f"{peak['peak_vram_reserved_mb']} MB reserved")
    print(f"total training time: {train_minutes:.1f} min on {gpu}")

    # ---- save the LAST epoch (no checkpoint selection on the test set) ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    config = {**vars(args), "num_classes": num_classes, "train_minutes": round(train_minutes, 1),
              "gpu": gpu, **peak,
              "torch": str(torch.__version__), "torchreid": str(torchreid.__version__)}
    torch.save({"state_dict": model.state_dict(), "config": config}, args.out)
    with open(os.path.splitext(args.out)[0] + "_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
