"""
train.py - fine-tune OSNet-x1.0 (ImageNet weights) on Market-1501 (assignment step 2).

Recipe: a shortened version of the paper's fine-tuning setup (Sec. 4.1):
  softmax cross-entropy + label smoothing, AMSGrad, weight decay 5e-4, batch 64,
  256x128 images, classifier-only warm-up ("fixbase"), cosine learning-rate decay.

The test set (query/gallery) is never touched here: no evaluation during training,
no picking the "best" epoch. We train for a fixed schedule and save the last epoch.

Usage:
  python train.py --epochs 1                # time one epoch first
  python train.py --epochs 20 --out checkpoints/osnet_x1_0_market.pth
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
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--transforms", nargs="+", default=["random_flip"],
                   help="torchreid augmentations, e.g. random_flip random_erase color_jitter")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="checkpoints/osnet_x1_0_market.pth")
    args = p.parse_args()

    set_random_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data: Market-1501 training split only (751 identities, 12,936 images) ----
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
    )
    num_classes = dm.num_train_pids  # 751

    # ---- model: OSNet-x1.0 starting from ImageNet weights, new 751-way classifier ----
    model = torchreid.models.build_model(
        name="osnet_x1_0", num_classes=num_classes, loss="softmax", pretrained=True
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

    print(f"device={device} | classes={num_classes} | train images={len(dm.train_loader.dataset)}")
    print(f"config: {vars(args)}")

    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        if epoch < args.fixbase_epochs:
            open_specified_layers(model, ["classifier"])  # freeze backbone, train classifier
        else:
            open_all_layers(model)

        t_epoch, loss_sum, correct, seen = time.time(), 0.0, 0, 0
        for batch in dm.train_loader:
            imgs, pids = unpack(batch)
            imgs, pids = imgs.to(device, non_blocking=True), pids.to(device, non_blocking=True)

            logits = model(imgs)  # in train mode OSNet returns class scores
            loss = criterion(logits, pids)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * pids.size(0)
            correct += (logits.argmax(1) == pids).sum().item()
            seen += pids.size(0)

        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()
        phase = "classifier-only" if epoch < args.fixbase_epochs else "all layers"
        print(f"epoch {epoch + 1}/{args.epochs} [{phase}] "
              f"loss {loss_sum / seen:.4f} | train acc {100 * correct / seen:.1f}% | "
              f"lr {lr_now:.6f} | {time.time() - t_epoch:.0f}s")

    train_minutes = (time.time() - t_start) / 60
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"total training time: {train_minutes:.1f} min on {gpu}")

    # ---- save the LAST epoch (no checkpoint selection on the test set) ----
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    config = {**vars(args), "num_classes": num_classes, "train_minutes": round(train_minutes, 1),
              "gpu": gpu, "torch": torch.__version__, "torchreid": torchreid.__version__}
    torch.save({"state_dict": model.state_dict(), "config": config}, args.out)
    with open(os.path.splitext(args.out)[0] + "_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
