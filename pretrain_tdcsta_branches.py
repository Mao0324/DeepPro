import argparse
import os
import sys
import importlib
import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from data_utils.TrainDataLoader import TrainIRSeqDataLoader
from runtime_utils import atomic_torch_save, parse_visible_devices

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "networks/models"))


class SoftIoULoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=tuple(range(1, pred.ndim)))
        pred_sum = torch.sum(pred, dim=tuple(range(1, pred.ndim)))
        target_sum = torch.sum(target, dim=tuple(range(1, target.ndim)))
        iou = intersection_sum / (pred_sum + target_sum - intersection_sum + 1e-6)
        return 1 - torch.mean(iou)


class SpatialBranchPretrainNet(nn.Module):
    def __init__(self, branch, dim=32):
        super().__init__()
        self.branch = branch
        self.head = nn.Conv2d(dim, 1, kernel_size=1)

    def forward(self, images):
        # images: [B, 1, T, H, W]
        x = images[:, :, -1, :, :]
        feat = self.branch(x)
        pred = self.head(feat).squeeze(1)
        return pred


class STBranchPretrainNet(nn.Module):
    def __init__(self, branch, dim=32):
        super().__init__()
        self.branch = branch
        self.head = nn.Conv3d(dim, 1, kernel_size=1)

    def forward(self, images):
        # images: [B, 1, T, H, W]
        feat = self.branch(images)
        pred = self.head(feat).squeeze(1)
        return pred


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["2d", "3d"], required=True)
    parser.add_argument("--model", type=str, default="DeepPro-Plus_TDCSTA")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./pretrained_tdcsta")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seqlen", type=int, default=40)
    parser.add_argument("--sample_rate", type=float, default=0.04)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--overwrite", type=int, default=0, choices=[0, 1])
    return parser.parse_args()


def main(args):
    visible_devices = parse_visible_devices(args.gpu)
    if len(visible_devices) != 1:
        raise ValueError('Branch pretraining requires exactly one GPU.')
    if args.batch_size <= 0 or args.epoch <= 0 or args.num_workers < 0:
        raise ValueError('batch_size/epoch must be positive and workers non-negative.')
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices[0]
    save_dir = Path(args.save_dir).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = TrainIRSeqDataLoader(
        args.dataset,
        data_root=args.datapath,
        seq_len=args.seqlen,
        sample_rate=args.sample_rate,
        patch_size=args.patch_size,
        transform=None,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed(x),
    )

    models_dir = (Path(BASE_DIR) / 'networks' / 'models').resolve()
    model_path = (models_dir / ('%s.py' % args.model)).resolve()
    if model_path.parent != models_dir or not model_path.is_file():
        raise ValueError('Unknown or unsafe model name: %s' % args.model)
    MODEL = importlib.import_module(args.model)
    front = MODEL.TDCSTAFront(dim=32)

    if args.stage == "2d":
        net = SpatialBranchPretrainNet(front.spatial_branch, dim=32)
        default_save_name = "spatial_branch.pth"
    else:
        net = STBranchPretrainNet(front.st_branch, dim=32)
        default_save_name = "st_branch.pth"

    save_name = args.save_name if args.save_name is not None else default_save_name
    save_path = (save_dir / save_name).resolve()
    try:
        save_path.relative_to(save_dir)
    except ValueError as error:
        raise ValueError('--save_name must stay inside %s.' % save_dir) from error
    if save_path.exists() and not args.overwrite:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        save_path = save_path.with_name(f"{save_path.stem}_{timestamp}{save_path.suffix}")

    net = net.cuda()
    criterion = SoftIoULoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    best_loss = float("inf")
    best_epoch = -1

    for epoch in range(args.epoch):
        net.train()
        loss_sum = 0.0

        for images, targets in tqdm(loader, desc=f"{args.stage} epoch {epoch + 1}/{args.epoch}"):
            images = images.float().cuda()
            targets = targets.float().cuda()

            optimizer.zero_grad()
            pred = net(images)

            if args.stage == "2d":
                target = targets[:, -1, :, :]
            else:
                target = targets

            if pred.shape[-2:] != target.shape[-2:]:
                if pred.ndim == 3:
                    pred = F.interpolate(pred.unsqueeze(1), size=target.shape[-2:], mode="bilinear", align_corners=True).squeeze(1)
                else:
                    b, t, h, w = pred.shape
                    pred = F.interpolate(
                        pred.reshape(b * t, 1, h, w),
                        size=target.shape[-2:],
                        mode="bilinear",
                        align_corners=True,
                    ).reshape(b, t, target.shape[-2], target.shape[-1])

            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        avg_loss = loss_sum / len(loader)
        print(f"Epoch {epoch + 1}: loss={avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            atomic_torch_save(
                {
                    "epoch": epoch,
                    "stage": args.stage,
                    "branch_state_dict": net.branch.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "args": vars(args),
                },
                save_path,
            )
            print(f"Saved best pretrained branch to {save_path}")

    print(f"Best epoch: {best_epoch + 1}, best loss: {best_loss:.6f}")


if __name__ == "__main__":
    main(parse_args())
