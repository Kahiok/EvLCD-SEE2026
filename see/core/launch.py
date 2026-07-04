#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import random
import shutil
import time
from collections import OrderedDict
from os.path import isfile, join

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from absl.logging import debug, flags, info
from pudb import set_trace
class AverageMeter:
    def __init__(self, name=""):
        self.name = name
        self.reset()
    def reset(self):
        self.val = self.avg = self.sum = self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from see.core.optimizer import Optimizer
from see.datasets import get_dataset
from see.datasets.basic_batch import EVENT_LOW_LIGHT_BATCH as ELB
from see.losses import get_loss, get_metric
from see.models import get_model
from see.visualize import get_visulization

FLAGS = flags.FLAGS


def _tile_forward_2x2(model, batch):
    """2×2 non-overlapping patch inference: split image/event spatially, stitch predictions.

    NOTE: This model uses FiLM brightness conditioning derived from NL.mean(). Each tile
    uses its local NL tile mean as conditioning, which is closer to the training-time
    crop-mean conditioning but introduces brightness inconsistency at tile boundaries.
    In practice this hurts PSNR (~1 dB) vs full-image inference for this architecture.
    The flag PATCH_INFERENCE is kept for experimentation purposes.
    """
    H, W = batch[ELB.LL].shape[2:]
    hH, hW = H // 2, W // 2
    pred_tiles = []
    for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        rs = slice(r * hH, (r + 1) * hH)
        cs = slice(c * hW, (c + 1) * hW)
        tile_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.dim() == 4 and v.shape[2] == H and v.shape[3] == W:
                tile_batch[k] = v[:, :, rs, cs].contiguous()
            else:
                tile_batch[k] = v
        out = model(tile_batch)
        pred_tiles.append(out[ELB.PRD])
    top = torch.cat([pred_tiles[0], pred_tiles[1]], dim=3)
    bot = torch.cat([pred_tiles[2], pred_tiles[3]], dim=3)
    batch[ELB.PRD] = torch.cat([top, bot], dim=2)
    return batch


def _tta_flip_forward(model, batch):
    """4-flip TTA: average predictions over {original, H-flip, V-flip, HV-flip}.

    Safe for this model because FiLM conditioning uses NL.mean() which is
    flip-invariant — all 4 passes get identical brightness conditioning.
    """
    augments = [(False, False), (True, False), (False, True), (True, True)]
    preds = []
    for flip_h, flip_v in augments:
        aug_batch = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.dim() == 4:
                t = torch.flip(v, dims=[3]) if flip_h else v
                t = torch.flip(t, dims=[2]) if flip_v else t
                aug_batch[k] = t.contiguous()
            else:
                aug_batch[k] = v
        out = model(aug_batch)
        pred = out[ELB.PRD]
        if flip_v:
            pred = torch.flip(pred, dims=[2])
        if flip_h:
            pred = torch.flip(pred, dims=[3])
        preds.append(pred)
    batch[ELB.PRD] = torch.stack(preds, dim=0).mean(dim=0)
    return batch


def move_tensors_to_cuda(dictionary_of_tensors):
    if isinstance(dictionary_of_tensors, dict):
        return {key: move_tensors_to_cuda(value) for key, value in dictionary_of_tensors.items()}
    if isinstance(dictionary_of_tensors, torch.Tensor):
        return dictionary_of_tensors.cuda(non_blocking=True)
    else:
        return dictionary_of_tensors


class ParallelLaunch:
    def __init__(self, config):
        """The main class for parallel training. The entry point is the `run` method.

        Args:
            config (EasyDict): The config of an training experiment.
        """
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6666"
        info(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        info(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
        # 0. config
        self.config = config
        # # 1. init environment
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True
        # 1.1 init global random seed
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed(config.SEED)
        random.seed(config.SEED)
        np.random.seed(config.SEED)
        # 1.2 init the tensorboard log dir
        self.tb_recoder = SummaryWriter(FLAGS.log_dir)
        # 2. device
        self.visualizer = None
        if config.VISUALIZE:
            self.visualizer = get_visulization(config.VISUALIZATION)

    def run(self):
        # 0. Init
        train_dataset, val_dataset = get_dataset(self.config.DATASET)
        model = get_model(self.config.MODEL)
        task_weights = getattr(self.config, "TASK_LOSS_WEIGHTS", None)
        criterion = get_loss(self.config.LOSS, task_weights=task_weights)
        metrics = get_metric(self.config.METRICS)
        opt = Optimizer(self.config.OPTIMIZER, model)
        # 1. Build model
        if self.config.IS_CUDA:
            model = nn.DataParallel(model)
            model = model.cuda()
            criterion = criterion.cuda()
            metrics = metrics.cuda()

        if self.config.RESUME.PATH:
            if not isfile(self.config.RESUME.PATH):
                raise ValueError(f"File not found, {self.config.RESUME.PATH}")
            if self.config.IS_CUDA:
                checkpoint = torch.load(
                    self.config.RESUME.PATH,
                    map_location=lambda storage, loc: storage.cuda(0),
                )
            else:
                checkpoint = torch.load(self.config.RESUME.PATH, map_location=torch.device("cpu"))
                new_state_dict = OrderedDict()
                for k, v in checkpoint["state_dict"].items():
                    name = k[7:]
                    new_state_dict[name] = v
                checkpoint["state_dict"] = new_state_dict

            if self.config.RESUME.SET_EPOCH:
                self.config.START_EPOCH = checkpoint["epoch"]
                opt.optimizer.load_state_dict(checkpoint["optimizer"])
                opt.scheduler.load_state_dict(checkpoint["scheduler"])

            if self.config.RESUME_STRICT:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model_dict = model.state_dict()
                pretrained_dict = {
                    k: v
                    for k, v in checkpoint["state_dict"].items()
                    if k in model_dict and model_dict[k].shape == v.shape
                }
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False)

        # 2. Build Dataloader
        train_loader = None
        if not self.config.TEST_ONLY and train_dataset:
            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.config.TRAIN_BATCH_SIZE,
                shuffle=True,
                num_workers=self.config.JOBS,
                pin_memory=True,
                drop_last=True,
            )
        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.JOBS,
            pin_memory=True,
            drop_last=True,
        )
        # 3. if test only
        if self.config.TEST_ONLY:
            self.valid(val_loader, model, criterion, metrics, 0)
            return
        # 4. train
        min_loss = 123456789.0
        for epoch in range(self.config.START_EPOCH, self.config.END_EPOCH):
            self.train(train_loader, model, criterion, metrics, opt, epoch)
            # save checkpoint
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "optimizer": opt.optimizer.state_dict(),
                "scheduler": opt.scheduler.state_dict(),
            }
            path = join(self.config.SAVE_DIR, "checkpoint.pth.tar")
            time.sleep(1)
            # valid
            if epoch % self.config.VAL_INTERVAL == 0:
                torch.save(checkpoint, path)
                val_loss = self.valid(val_loader, model, criterion, metrics, epoch)
                if val_loss < min_loss:
                    min_loss = val_loss
                    copy_path = join(self.config.SAVE_DIR, "model_best.pth.tar")
                    shutil.copy(path, copy_path)
            # train
            if epoch % self.config.MODEL_SANING_INTERVAL == 0:
                path = join(
                    self.config.SAVE_DIR,
                    f"checkpoint-{str(epoch).zfill(3)}.pth.tar",
                )
                torch.save(checkpoint, path)

    def train(self, train_loader, model, criterion, metrics, opt, epoch):
        model = model.train()
        info(f"Train Epoch[{epoch}/{self.config.END_EPOCH}]:len({len(train_loader)})")
        length = len(train_loader)
        # 1. init meter
        losses_meter = {"TotalLoss": AverageMeter(f"Valid/TotalLoss")}
        for config in self.config.LOSS:
            losses_meter[config.NAME] = AverageMeter(f"Train/{config.NAME}")
        metric_meter = {}
        for config in self.config.METRICS:
            metric_meter[config.NAME] = AverageMeter(f"Train/{config.NAME}")
        batch_time_meter = AverageMeter("Train/BatchTime")
        # Exposure head scalars (only populated when model outputs e_scale / e_gamma)
        exp_meter = {
            "e_scale_mean": AverageMeter("Train/e_scale_mean"),
            "e_scale_std":  AverageMeter("Train/e_scale_std"),
            "e_gamma_mean": AverageMeter("Train/e_gamma_mean"),
            "e_gamma_std":  AverageMeter("Train/e_gamma_std"),
        }
        # For corr(pred_mean, gt_mean) — brightness tracking indicator
        _train_pred_means = []
        _train_gt_means   = []
        # 2. start a training epoch
        start_time = time.time()
        time_recoder = time.time()
        scaler = torch.amp.GradScaler("cuda")
        for index, batch in enumerate(train_loader):
            if self.config.IS_CUDA:
                batch = move_tensors_to_cuda(batch)
            if self.config.MIX_PRECISION:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(batch)
                    losses, name_to_loss = criterion(outputs)
                    # 2.1 forward
                    name_to_measure = metrics(outputs)
                    scaler.scale(losses).backward()
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()
            else:
                outputs = model(batch)
                losses, name_to_loss = criterion(outputs)
                # 2.1 forward
                name_to_measure = metrics(outputs)
                # 2.2 backward
                opt.zero_grad()
                losses.backward()
                # 2.3 update weights
                # clip the grad
                clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
                opt.step()
            # 2.4 update measure
            # 2.4.1 time update
            now = time.time()
            batch_time_meter.update(now - time_recoder)
            time_recoder = now
            # 2.4.2 loss update
            losses_meter["TotalLoss"].update(losses.detach().item())
            for name, loss_item in name_to_loss:
                loss_item = loss_item.detach().item()
                losses_meter[name].update(loss_item)
            # 2.4.3 measure update
            for name, measure_item in name_to_measure:
                measure_item = measure_item.detach().item()
                metric_meter[name].update(measure_item)
            # 2.4.4 brightness tracking: accumulate pred/gt means for corr at epoch end
            if ELB.PRD in outputs and ELB.NL in outputs:
                _pred_m = outputs[ELB.PRD].detach().mean(dim=[1,2,3]).cpu()
                _gt_m   = outputs[ELB.NL].detach().mean(dim=[1,2,3]).cpu()
                _train_pred_means.append(_pred_m)
                _train_gt_means.append(_gt_m)
            # 2.4.5 exposure head scalars (optional — only present for v9+)
            if 'e_scale' in outputs:
                e_s = outputs['e_scale'].float()
                e_g = outputs['e_gamma'].float()
                exp_meter["e_scale_mean"].update(e_s.mean().item())
                exp_meter["e_scale_std"].update(e_s.std().item())
                exp_meter["e_gamma_mean"].update(e_g.mean().item())
                exp_meter["e_gamma_std"].update(e_g.std().item())
            # 2.5 log
            if index % self.config.LOG_INTERVAL == 0:
                info(f"ConfigFile: {FLAGS.yaml_file}")
                info(f"Train Epoch[{epoch}/{self.config.END_EPOCH}, {index}/{length}]:")
                for name, meter in losses_meter.items():
                    info(f"    loss:    {name}: {meter.avg}")
                for name, measure in metric_meter.items():
                    info(f"    measure: {name}: {measure.avg}")
            if index >= 100000 and index % 100000 == 0:
                # save checkpoint
                checkpoint = {
                    "epoch": epoch,
                    "index": index,
                    "state_dict": model.state_dict(),
                    "optimizer": opt.optimizer.state_dict(),
                    "scheduler": opt.scheduler.state_dict(),
                }
                path = join(self.config.SAVE_DIR, f"Echeckpoint-E{epoch}-S{index}.pth.tar")
                torch.save(checkpoint, path)

        # 3. record a training epoch
        # 3.1 record epoch time
        epoch_time = time.time() - start_time
        batch_time = batch_time_meter.avg
        info(
            f"Train Epoch[{epoch}/{self.config.END_EPOCH}]:time:epoch({epoch_time}),batch({batch_time})"
            f"lr({opt.get_lr()})"
        )
        self.tb_recoder.add_scalar(f"Train/EpochTime", epoch_time, epoch)
        self.tb_recoder.add_scalar(f"Train/BatchTime", batch_time, epoch)
        self.tb_recoder.add_scalar(f"Train/LR", opt.get_lr(), epoch)
        for name, meter in losses_meter.items():
            info(f"    loss:    {name}: {meter.avg}")
            self.tb_recoder.add_scalar(f"Train/{name}", meter.avg, epoch)
        for name, measure in metric_meter.items():
            info(f"    measure: {name}: {measure.avg}")
            self.tb_recoder.add_scalar(f"Train/{name}", measure.avg, epoch)
        # Log exposure head scalars if they were collected this epoch
        if exp_meter["e_scale_mean"].count > 0:
            for name, meter in exp_meter.items():
                info(f"    exposure: {name}: {meter.avg:.4f}")
                self.tb_recoder.add_scalar(f"Train/{name}", meter.avg, epoch)
        # Log corr(pred_mean, gt_mean) — brightness tracking indicator (EXP-011)
        if _train_pred_means:
            import numpy as _np
            _pm = torch.cat(_train_pred_means).numpy()
            _gm = torch.cat(_train_gt_means).numpy()
            _corr = float(_np.corrcoef(_pm, _gm)[0, 1])
            info(f"    brightness: corr(pred_mean, gt_mean): {_corr:.4f}")
            self.tb_recoder.add_scalar("Train/corr_pred_gt_mean", _corr, epoch)
        # adjust learning rate
        opt.lr_schedule()

    def valid(self, valid_loader, model, criterion, metrics, epoch):
        model = model.eval()
        length = len(valid_loader)
        info(f"Valid Epoch[{epoch}/{self.config.END_EPOCH}] starting: length({length})")
        # 1. init meter
        losses_meter = {"total": AverageMeter(f"Valid/TotalLoss")}
        for config in self.config.LOSS:
            losses_meter[config.NAME] = AverageMeter(f"Valid/{config.NAME}")
        metric_meter = {}
        for config in self.config.METRICS:
            metric_meter[config.NAME] = AverageMeter(f"Valid/{config.NAME}")
        batch_time_meter = AverageMeter("Valid/BatchTime")
        # 2. start a validating epoch
        time_recoder = time.time()
        start_time = time_recoder
        for index, batch in enumerate(valid_loader):
            if self.config.IS_CUDA:
                batch = move_tensors_to_cuda(batch)
            with torch.no_grad():
                patch_infer = getattr(self.config, "PATCH_INFERENCE", False)
                tta = getattr(self.config, "TTA", False)
                if self.config.MIX_PRECISION:
                    with torch.amp.autocast(device_type="cuda"):
                        if patch_infer:
                            outputs = _tile_forward_2x2(model, batch)
                        elif tta:
                            outputs = _tta_flip_forward(model, batch)
                        else:
                            outputs = model(batch)
                        losses, name_to_loss = criterion(outputs)
                        # 2.2. recorder
                        name_to_measure = metrics(outputs)
                else:
                    if patch_infer:
                        outputs = _tile_forward_2x2(model, batch)
                    elif tta:
                        outputs = _tta_flip_forward(model, batch)
                    else:
                        outputs = model(batch)
                    losses, name_to_loss = criterion(outputs)
                    # 2.2. recorder
                    name_to_measure = metrics(outputs)
            # 2.3 visualization
            if self.visualizer:
                self.visualizer(outputs)
                if self.config.VISUALIZATION.ONLY_VIS:
                    continue
            # 2.4. update measure
            now = time.time()
            batch_time_meter.update(now - time_recoder)
            time_recoder = now
            loss = losses.detach().item() if isinstance(losses, torch.Tensor) else losses
            losses_meter["total"].update(loss)
            for name, loss_item in name_to_loss:
                loss_item = loss_item.detach().item() if isinstance(loss_item, torch.Tensor) else loss_item
                losses_meter[name].update(loss_item)
            for name, measure_item in name_to_measure:
                measure_item = measure_item.detach().item() if isinstance(measure_item, torch.Tensor) else measure_item
                metric_meter[name].update(measure_item)
            if index % self.config.LOG_INTERVAL == 0:
                info(f"ConfigFile: {FLAGS.yaml_file}")
                info(f"Valid Epoch[{epoch}/{self.config.END_EPOCH}, {index}/{length}]:")
                info(f"    batch-time: {batch_time_meter.avg}")
                for name, meter in losses_meter.items():
                    info(f"    loss:    {name}: {meter.avg}")
                for name, measure in metric_meter.items():
                    info(f"    measure: {name}: {measure.avg}")
        # 3. record a training epoch
        # 3.1 record epoch time
        epoch_time = time.time() - start_time
        batch_time = batch_time_meter.avg
        info(f"Valid Epoch[{epoch}/{self.config.END_EPOCH}]:" f"time:epoch({epoch_time}),batch({batch_time})")
        self.tb_recoder.add_scalar(f"Valid/EpochTime", epoch_time, epoch)
        self.tb_recoder.add_scalar(f"Valid/BatchTime", batch_time, epoch)
        for name, meter in losses_meter.items():
            info(f"    loss:    {name}: {meter.avg}")
            self.tb_recoder.add_scalar(f"Valid/{name}", meter.avg, epoch)
        for name, measure in metric_meter.items():
            info(f"    measure: {name}: {measure.avg}")
            self.tb_recoder.add_scalar(f"Valid/{name}", measure.avg, epoch)
        # Gap = PSNR-Linear_N - PSNR  (key indicator for EXP-011)
        if "PSNR" in metric_meter and "PSNR-Linear_N" in metric_meter:
            gap = metric_meter["PSNR-Linear_N"].avg - metric_meter["PSNR"].avg
            info(f"    measure: Gap(PSNR-L_N - PSNR): {gap:.4f}")
            self.tb_recoder.add_scalar("Valid/Gap_PSNR_LN_minus_PSNR", gap, epoch)

        # ── Dual val: second pass with NL zeroed (NL_FORCE_ZERO) ─────────────
        if getattr(self.config, "DUAL_VAL", False):
            info(f"Valid Epoch[{epoch}] DUAL_VAL: running NL-free pass (NL_FORCE_ZERO)")
            nlfree_meter = {n: AverageMeter(f"ValidNLFree/{n}") for n in metric_meter}
            for index, batch in enumerate(valid_loader):
                if self.config.IS_CUDA:
                    batch = move_tensors_to_cuda(batch)
                batch["NL_FORCE_ZERO"] = True
                with torch.no_grad():
                    if self.config.MIX_PRECISION:
                        with torch.amp.autocast(device_type="cuda"):
                            outputs = model(batch)
                            name_to_measure_nf = metrics(outputs)
                    else:
                        outputs = model(batch)
                        name_to_measure_nf = metrics(outputs)
                for name, measure_item in name_to_measure_nf:
                    measure_item = measure_item.detach().item() if isinstance(measure_item, torch.Tensor) else measure_item
                    nlfree_meter[name].update(measure_item)
            info(f"Valid Epoch[{epoch}] NL-free results:")
            for name, meter in nlfree_meter.items():
                info(f"    NL-free measure: {name}: {meter.avg}")
                self.tb_recoder.add_scalar(f"ValidNLFree/{name}", meter.avg, epoch)
            if "PSNR" in nlfree_meter and "PSNR-Linear_N" in nlfree_meter:
                gap_nf = nlfree_meter["PSNR-Linear_N"].avg - nlfree_meter["PSNR"].avg
                info(f"    NL-free measure: Gap(PSNR-L_N - PSNR): {gap_nf:.4f}")
                self.tb_recoder.add_scalar("ValidNLFree/Gap_PSNR_LN_minus_PSNR", gap_nf, epoch)
        return losses_meter["total"].avg
