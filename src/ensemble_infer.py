"""Training / validation loops.

Kept deliberately small and readable: one epoch of training, one pass of
inference, and a checkpointing loop that monitors validation QWK. Mixed precision
(AMP) is used on CUDA for speed and memory headroom on the 16 GB GPU.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import EvalResult, evaluate


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                    optimizer: torch.optim.Optimizer, device: str,
                    scaler: torch.amp.GradScaler | None, epoch: int) -> float:
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc=f"train epoch {epoch}", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        running += loss.item() * images.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    return running / len(loader.dataset)


@torch.no_grad()
def infer(model: nn.Module, loader: DataLoader, device: str
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_true, y_pred, y_prob) over a loader."""
    model.eval()
    ys, preds, probs = [], [], []
    for images, labels in tqdm(loader, desc="infer", leave=False):
        images = images.to(device, non_blocking=True)
        logits = model(images)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs.append(p)
        preds.append(p.argmax(1))
        ys.append(labels.numpy())
    return np.concatenate(ys), np.concatenate(preds), np.concatenate(probs)


def evaluate_loader(model: nn.Module, loader: DataLoader, device: str) -> EvalResult:
    y_true, y_pred, y_prob = infer(model, loader, device)
    return evaluate(y_true, y_pred, y_prob)
