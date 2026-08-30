import os

import numpy as np
import torch


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> "
                f"{val_loss:.6f}). Saving model."
            )
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))
        self.val_loss_min = val_loss


def adaptive_adjust_lr(optimizer, epoch, config, val_losses=None):
    def set_lr(value):
        for param_group in optimizer.param_groups:
            param_group["lr"] = value

    initial_lr = config.learning_rate
    total_epochs = config.train_epochs
    warmup_epochs = max(3, min(10, int(total_epochs * 0.05)))
    if epoch <= warmup_epochs:
        set_lr(initial_lr * (epoch / warmup_epochs) ** 0.8)
        return
    if val_losses is None or len(val_losses) < 4:
        return
    if not hasattr(optimizer, "smooth_val_loss"):
        optimizer.smooth_val_loss = np.mean(val_losses[-3:])
    else:
        optimizer.smooth_val_loss = 0.7 * optimizer.smooth_val_loss + 0.3 * val_losses[-1]
    stable = optimizer.smooth_val_loss * 1.005 < np.min(val_losses[:-3])
    rising = any(
        val_losses[-index] > val_losses[-index - 1] * 1.005
        for index in range(1, min(4, len(val_losses)))
    )
    if epoch < total_epochs * 0.7 and stable:
        new_lr = optimizer.param_groups[0]["lr"]
    elif rising:
        new_lr = optimizer.param_groups[0]["lr"] * 0.9
    else:
        new_lr = initial_lr * (0.8 ** (epoch // 10))
    set_lr(max(new_lr, initial_lr * 0.001))
