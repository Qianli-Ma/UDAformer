import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import yaml
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from my_dataset import MyDataSet as data
from model import UFUformer as net


REPO_ROOT = Path(__file__).resolve().parents[4]


def resolve_path(value):
    path_value = Path(value)
    if path_value.is_absolute():
        return path_value
    return (REPO_ROOT / path_value).resolve()


def load_config(config_path):
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config structure in {config_path}")
    return config


def require_value(config, keys, name):
    cursor = config
    for key in keys:
        if not isinstance(cursor, dict) or key not in cursor:
            raise ValueError(f"Missing required config value: {name}")
        cursor = cursor[key]
    return cursor


def training_epochs(training_cfg):
    if "epochs" in training_cfg:
        return int(training_cfg["epochs"])
    if "epchos" in training_cfg:
        return int(training_cfg["epchos"])
    raise ValueError("Missing required config value: training.epochs (or legacy training.epchos)")


def parse_args():
    parser = argparse.ArgumentParser(description="Train UDAformer with YAML-driven paths/config.")
    parser.add_argument(
        "--config",
        default="experiments/configs/UDAformer/UDAformer.yaml",
        help="Path to UDAformer YAML config (absolute or repo-relative).",
    )
    parser.add_argument("-b", "--batch", type=int, default=None, help="Override training.batch_size from config.")
    parser.add_argument("-e", "--epoch", type=int, default=None, help="Override training.epochs from config.")
    parser.add_argument("-r", "--resume", action="store_true", help="Resume from the latest checkpoint.")
    parser.add_argument("--lr", type=float, default=None, help="Override training.lr from config.")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers.")
    return parser.parse_args()


def log_images(writer, img, out, ll256, gt, iteration):
    images_array = vutils.make_grid(img).to("cpu")
    out_array = vutils.make_grid(out * 255).to("cpu").detach()
    ll256_array = vutils.make_grid(ll256 * 255).to("cpu").detach()
    gt_array = vutils.make_grid(gt).to("cpu")

    writer.add_image("input", images_array, iteration)
    writer.add_image("out", out_array, iteration)
    writer.add_image("ll256", ll256_array, iteration)
    writer.add_image("gt", gt_array, iteration)


def main():
    args = parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        cwd_candidate = (Path.cwd() / config_path).resolve()
        repo_candidate = (REPO_ROOT / config_path).resolve()
        config_path = cwd_candidate if cwd_candidate.exists() else repo_candidate
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cfg = load_config(config_path)
    train_data_cfg = require_value(cfg, ["paths", "train_data"], "paths.train_data")
    save_path_cfg = require_value(cfg, ["paths", "save_path"], "paths.save_path")
    raw_folder = cfg.get("paths", {}).get("raw_folder", "raw-890")
    reference_folder = cfg.get("paths", {}).get("reference_folder", "reference-890")
    training_cfg = cfg.get("training", {})
    if not isinstance(training_cfg, dict):
        raise ValueError("Invalid config structure: training must be a mapping")

    train_data_path = resolve_path(str(train_data_cfg))
    save_path = resolve_path(str(save_path_cfg))
    checkpoint_dir = save_path / "checkpoint"
    sample_dir = save_path / "sample"
    tb_dir = save_path / "tensorboard"

    epochs = args.epoch if args.epoch is not None else training_epochs(training_cfg)
    batch_size = args.batch if args.batch is not None else int(training_cfg.get("batch_size", 4))
    lr = args.lr if args.lr is not None else float(training_cfg.get("lr", 1e-4))
    step_size = int(training_cfg.get("lr_step_size", 150))
    gamma = float(training_cfg.get("lr_gamma", 0.5))

    train_data_path.mkdir(parents=True, exist_ok=True)
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    dataset = data(str(train_data_path), raw_folder=raw_folder, reference_folder=reference_folder)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
    )
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    def proimage(im, image_idx):
        images = im[image_idx, :, :, :].clone().detach().requires_grad_(False)
        image = torch.transpose(images, 0, 1)
        image = torch.transpose(image, 1, 2).cpu().numpy() * 255
        return image

    start_epoch = 0
    if args.resume:
        checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_*_epoch.pkl"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        latest_ckpt = checkpoint_files[-1]
        state = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = int(state["epoch"]) + 1
        print(f"Checkpoint loaded: {latest_ckpt} | resume from epoch {start_epoch}")

    writer = SummaryWriter(log_dir=str(tb_dir), comment="UDAformer", filename_suffix="train")
    global_step = 0

    print(f"Using config: {config_path}")
    print(f"Train data: {train_data_path}")
    print(f"Save path: {save_path}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, epochs: {epochs}, lr: {lr}")
    print(f"Raw folder: {raw_folder}")
    print(f"Reference folder: {reference_folder}")

    for epoch in range(start_epoch, epochs):
        print(epoch)
        progress = tqdm(enumerate(dataloader), total=len(dataloader))
        if epoch < 1000:
            l1_loss_fn = nn.MSELoss()
        else:
            l1_loss_fn = nn.L1Loss()

        for step, batch in progress:
            raw = batch[0].to(device, non_blocking=True)
            gt = batch[1].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(raw)
                l1loss = l1_loss_fn(output, gt)
                loss = l1loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress.set_postfix(Loss1=l1loss.item())
            global_step += 1
            writer.add_scalar("loss", loss.item(), global_step)

            if step % 100 == 0:
                image_idx = 0
                predi = output[image_idx, :, :, :].clone().detach().requires_grad_(False)
                predi = torch.transpose(predi, 0, 1)
                predi = torch.transpose(predi, 1, 2).cpu().numpy() * 255
                gti = proimage(gt, image_idx)
                rawi = proimage(raw, image_idx)
                image = np.concatenate((rawi, predi, gti), axis=1)
                image_name = sample_dir / f"out{epoch}_{step}.png"
                cv2.imwrite(str(image_name), image)
                log_images(writer, raw, output, output, gt, global_step)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        path_checkpoint = checkpoint_dir / f"checkpoint_{epoch}_epoch.pkl"
        torch.save(checkpoint, path_checkpoint)

        scheduler.step()
        print(optimizer.param_groups[0]["lr"])

    writer.close()


if __name__ == "__main__":
    main()
