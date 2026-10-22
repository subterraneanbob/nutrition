from functools import partial
from pathlib import Path
from typing import NamedTuple

import albumentations as A
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from scripts.config import Config
from scripts.utils import load_dish_image


class DishesData(NamedTuple):
    train: DataLoader
    val: DataLoader
    test: DataLoader


class DishesDataset(Dataset):
    """
    Датасет блюд: идентификатор, изображение,
    токенизированный список ингредиентов и маска,
    вес порции, калорийность (целевая переменная).
    """

    def __init__(
        self,
        dishes_df: pd.DataFrame,
        image_transform,
        data_dir: str | Path,
        mass_scaler: StandardScaler,
        tokenizer,
        max_length: int = 128,
    ):
        self.data_dir = Path(data_dir)
        self.image_transform = image_transform

        self.dish_ids = list(dishes_df.index)
        self.targets = dishes_df["total_calories"].to_numpy()
        self.masses = mass_scaler.transform(dishes_df[["total_mass"]])

        encoded_text = tokenizer(
            dishes_df["ingredients"].tolist(),
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        self.input_ids = encoded_text["input_ids"]
        self.attention_mask = encoded_text["attention_mask"]

    def __len__(self) -> int:
        return len(self.dish_ids)

    def __getitem__(self, index: int) -> dict:
        dish_id = self.dish_ids[index]

        image = load_dish_image(dish_id, data_dir=self.data_dir)
        image = np.asarray(image, dtype=np.uint8)
        image = self.image_transform(image=image)["image"]

        return {
            "dish_id": dish_id,
            "image": image,
            "input_ids": torch.tensor(self.input_ids[index], dtype=torch.long),
            "attention_mask": torch.tensor(
                self.attention_mask[index], dtype=torch.long
            ),
            "mass": torch.tensor(self.masses[index], dtype=torch.float32),
            "target": torch.tensor([self.targets[index]], dtype=torch.float32),
        }


def collate_ingredients(batch: list[dict], pad_token_id: int) -> dict:
    """Дополняет последовательности токенов до максимальной в батче."""

    input_ids = pad_sequence(
        [item["input_ids"] for item in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_mask = pad_sequence(
        [item["attention_mask"] for item in batch],
        batch_first=True,
        padding_value=0,
    )

    return {
        "dish_id": [item["dish_id"] for item in batch],
        "image": torch.stack([item["image"] for item in batch]),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "mass": torch.stack([item["mass"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
    }


def build_image_transform(
    is_train: bool,
    img_size: int = 384,
    img_mean: float | tuple[float] = 0.5,
    img_std: float | tuple[float] = 0.5,
) -> A.Compose:
    if is_train:
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02, p=0.5
            ),
            A.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(16, 32),
                hole_width_range=(16, 32),
                p=0.3,
            ),
            A.Normalize(img_mean, img_std),
            A.ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(img_mean, img_std),
            A.ToTensorV2(),
        ])


def build_dataloaders(dishes_df: pd.DataFrame, config: Config) -> DishesData:

    train_df = dishes_df.query("split == 'train'").copy()
    test_df = dishes_df.query("split == 'test'").copy()

    # Выделяем валидационную часть из тренировочного набора со стратификацией по калорийности
    strata = pd.qcut(
        train_df["total_calories"],
        q=10,
        labels=False,
        duplicates="drop",
    ).fillna(0)

    train_df, val_df = train_test_split(
        train_df,
        test_size=config.VAL_FRACTION,
        random_state=config.SEED,
        stratify=strata,
    )

    train_image_transform = build_image_transform(is_train=True)
    test_image_transform = build_image_transform(is_train=False)

    tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL)

    mass_scaler = StandardScaler()
    mass_scaler.fit(train_df[["total_mass"]])

    collate_fn = partial(collate_ingredients, pad_token_id=tokenizer.pad_token_id)

    dataset_kwargs = {
        "data_dir": config.DATA_DIR,
        "mass_scaler": mass_scaler,
        "tokenizer": tokenizer,
        "max_length": config.MAX_TEXT_LEN,
    }

    loader_kwargs = {
        "batch_size": config.BATCH_SIZE,
        "collate_fn": collate_fn,
    }

    train_loader = DataLoader(
        DishesDataset(train_df, train_image_transform, **dataset_kwargs),
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        DishesDataset(val_df, test_image_transform, **dataset_kwargs),
        shuffle=False,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        DishesDataset(test_df, test_image_transform, **dataset_kwargs),
        shuffle=False,
        **loader_kwargs,
    )

    return DishesData(train_loader, val_loader, test_loader)
