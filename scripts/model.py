from pathlib import Path

import pandas as pd
import timm
import torch
import torchmetrics
from torch import nn
from torch.optim import AdamW
from transformers import AutoModel

from scripts.config import Config


class DishCaloriesRegressor(nn.Module):
    """
    Мультимодальная модель для предсказания калорийности по фото блюда,
    его ингредиентам и массе порции.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        text_encoder: nn.Module,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        image_dim = image_encoder.num_features
        self.image_encoder = image_encoder
        self.image_projection = nn.Linear(image_dim, hidden_dim)

        text_dim = text_encoder.config.hidden_size
        self.text_encoder = text_encoder
        self.text_projection = nn.Linear(text_dim, hidden_dim)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image, input_ids, attention_mask, total_mass):
        image_features = self.image_encoder(image)
        if isinstance(image_features, (list, tuple)):
            image_features = image_features[-1]
            if len(image_features.shape) == 4:
                image_features = image_features.mean(dim=[2, 3])
        image_emb = self.image_projection(image_features)

        text_features = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        text_emb = self.text_projection(text_features)

        fused = torch.cat([image_emb, text_emb, total_mass], dim=1)

        return self.regressor(fused)

    @classmethod
    def from_config(cls, config: Config):
        image_encoder = timm.create_model(
            config.IMG_MODEL,
            pretrained=True,
            num_classes=0,
        )
        text_encoder = AutoModel.from_pretrained(config.TEXT_MODEL)

        return DishCaloriesRegressor(
            image_encoder, text_encoder, config.HIDDEN_DIM, config.DROPOUT
        )


def train_epoch(model, data_loader, optimizer, criterion, device):

    model.train()
    total_loss = 0.0

    for batch in data_loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        masses = batch["mass"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        preds = model(images, input_ids, attention_mask, masses)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(data_loader.dataset)


@torch.no_grad()
def validate(model, data_loader, device, report_all_results: bool = False):

    model.eval()
    mae = torchmetrics.MeanAbsoluteError().to(device)

    all_dish_ids = []
    all_preds = []
    all_targets = []

    for batch in data_loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        masses = batch["mass"].to(device)
        targets = batch["target"].to(device)

        preds = model(images, input_ids, attention_mask, masses)
        mae.update(preds, targets)

        if report_all_results:
            all_dish_ids.extend(batch["dish_id"])
            all_preds.extend(preds.squeeze().cpu().tolist())
            all_targets.extend(targets.squeeze().cpu().tolist())

    total_mae = mae.compute().item()

    if report_all_results:
        return total_mae, pd.DataFrame({
            "dish_id": all_dish_ids,
            "pred": all_preds,
            "target": all_targets,
        })

    return total_mae


def train(train_loader, val_loader, config: Config, device):

    torch.manual_seed(config.SEED)
    model = DishCaloriesRegressor.from_config(config).to(device)

    optimizer = AdamW(
        [
            {"params": model.text_encoder.parameters(), "lr": config.TEXT_LR},
            {"params": model.image_encoder.parameters(), "lr": config.IMG_LR},
            {"params": model.text_projection.parameters(), "lr": config.REGRESSOR_LR},
            {"params": model.image_projection.parameters(), "lr": config.REGRESSOR_LR},
            {"params": model.regressor.parameters(), "lr": config.REGRESSOR_LR},
        ],
        weight_decay=config.WEIGHT_DECAY,
    )

    criterion = nn.L1Loss()

    checkpoints_dir = Path(config.CHECKPOINTS_DIR)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoints_dir / "best_model.pt"

    best_mae = float("inf")
    patience = config.PATIENTCE
    epochs_no_improve = 0

    history = {"train_loss": [], "val_mae": []}

    for epoch in range(1, config.EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_mae = validate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)

        print(
            f"Эпоха {epoch}/{config.EPOCHS} | train_loss: {train_loss:.4f} | val_mae: {val_mae:.4f}"
        )

        if val_mae < best_mae:
            best_mae = val_mae
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)

            print(f"Сохраняем модель с MAE {best_mae:.4f} в {checkpoint_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Нет улучшения MAE после {patience} последних эпох - останавлием обучение."
                )
                break

    return model, history
