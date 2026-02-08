import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import DetectionLoss, load_model, save_model
from .datasets.road_dataset import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    # Use augmentation for training data
    train_data = load_data("drive_data/train", transform_pipeline="aug", shuffle=True, batch_size=batch_size, num_workers=0)
    val_data = load_data("drive_data/val", transform_pipeline="default", shuffle=False, batch_size=batch_size, num_workers=0)

    # create loss function and optimizer
    loss_func = DetectionLoss(lambda_depth=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0

    # training loop
    for epoch in range(num_epoch):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in train_data:
            img = batch["image"].to(device)
            target_logits = batch["track"].to(device).long()  # (b, h, w)
            target_depth = batch["depth"].to(device)  # (b, h, w)

            # Forward pass
            logits, depth = model(img)
            loss = loss_func(logits, depth, target_logits, target_depth)

            # Backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            global_step += 1

        avg_train_loss = train_loss / len(train_data)

        # Validation phase
        with torch.inference_mode():
            model.eval()
            val_loss = 0

            for batch in val_data:
                img = batch["image"].to(device)
                target_logits = batch["track"].to(device).long()
                target_depth = batch["depth"].to(device)

                logits, depth = model(img)
                loss = loss_func(logits, depth, target_logits, target_depth)
                val_loss += loss.item()

            avg_val_loss = val_loss / len(val_data)

        # Log metrics
        logger.add_scalar("train_loss", avg_train_loss, epoch)
        logger.add_scalar("val_loss", avg_val_loss, epoch)

        # Print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_loss={avg_train_loss:.4f} "
                f"val_loss={avg_val_loss:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
