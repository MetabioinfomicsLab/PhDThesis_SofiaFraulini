#!/usr/bin/env python

import os
import json
import math
import random
import argparse
import itertools
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.utils.class_weight import compute_class_weight

# =========================================================
# SEED
# =========================================================

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =========================================================
# LOAD EMBEDDINGS
# =========================================================

def load_embeddings(folder, comp=None):

    EMB_DIRS = [Path(folder)]

    emb_files = []

    for d in EMB_DIRS:
        if d.exists():
            emb_files.extend(sorted(d.glob("*.npy")))

    if not emb_files:
        raise SystemExit("No .npy embeddings found.")

    vectors = []
    records = []

    for p in emb_files:

        v = np.load(p).astype(np.float32, copy=False)

        meta = {}

        mp = p.with_suffix(".meta.json")

        if mp.exists():
            try:
                meta = json.loads(mp.read_text())
            except Exception:
                pass

        vectors.append(v)

        records.append({
            "file": p.name,
            "stem": p.stem,
            "vector_dim": int(v.shape[-1]),
            "length_tokens": meta.get("length_tokens"),
            "input_file": meta.get("input_file"),
            "components": ",".join(meta.get("components", []))
        })

    hidden = 1920

    components = {
        'mean': slice(0, hidden),
        'std': slice(hidden, hidden * 2),
        'max': slice(hidden * 2, hidden * 3),
        'wmean': slice(hidden * 3, hidden * 4)
    }

    if comp:
        X = np.vstack([v[components[comp]] for v in vectors])
    else:
        X = np.vstack([v for v in vectors])

    names = [r['stem'] for r in records]

    X_df = pd.DataFrame(X, index=names)

    return X_df

# =========================================================
# EARLY STOPPING
# =========================================================

class EarlyStopping:

    def __init__(
        self,
        patience=15,
        delta=1e-3,
        monitor='val_loss',
        mode='min'
    ):

        self.monitor = monitor
        self.patience = patience
        self.delta = delta
        self.mode = mode

        self.best_score = None
        self.best_epoch = 0

        self.no_improvement_count = 0
        self.stop_training = False

    def check(self, metrics, epoch):

        metric_now = metrics[self.monitor]

        if self.best_score is None:

            self.best_score = metric_now
            self.best_epoch = epoch

            return True

        if self.mode == 'min':

            improved = metric_now < (self.best_score - self.delta)

        else:

            improved = metric_now > (self.best_score + self.delta)

        if improved:

            self.best_score = metric_now
            self.best_epoch = epoch
            self.no_improvement_count = 0

            return True

        else:

            self.no_improvement_count += 1

            if self.no_improvement_count >= self.patience:
                self.stop_training = True

            return False

# =========================================================
# MODEL
# =========================================================

class TwoTowersClassifier(nn.Module):

    def __init__(
        self,
        emb_dim,
        ko_dim,
        proj_dim=128,
        proj_ko_dim=64,
        dropout_emb=0.2,
        dropout_ko=0.2,
        num_classes=3
    ):

        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.TowerEmb = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_emb),

            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, proj_dim)
        )

        self.TowerKO = nn.Sequential(
            nn.Linear(ko_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout_ko),

            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, proj_ko_dim)
        )

        self.classifier = nn.Linear(
            proj_dim + proj_ko_dim,
            num_classes
        )

    def forward(self, emb, ko, detach_ko=False):

        z_emb = self.TowerEmb(emb)
        z_ko = self.TowerKO(ko)

        if detach_ko:
            z_ko = z_ko.detach()

        z_emb = F.normalize(z_emb, dim=1)
        z_ko = F.normalize(z_ko, dim=1)
        alpha = torch.sigmoid(self.alpha) #keep between 0 and 1
        z_concat = torch.cat((alpha * z_emb, (1 - alpha) * z_ko), dim=1)
        #z_concat = torch.cat((z_emb, z_ko), dim=1)

        logits = self.classifier(z_concat)

        return logits

# =========================================================
# TRAIN
# =========================================================

def train_and_evaluate_model(
    model,
    train_loader,
    val_loader,
    class_weights,
    device,
    config
):

    ce_loss = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config["label_smoothing"]
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["wd"]
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    early_stopping = EarlyStopping(
        patience=config["patience"],
        delta=config["delta"],
        monitor="val_loss",
        mode="min"
    )

    best_state_dict = None

    best_val_f1 = -1
    best_epoch = -1

    for epoch in range(200):

        detach_ko = epoch < config["detach_ko_epochs"]

        # ==========================
        # TRAIN
        # ==========================

        model.train()

        y_true_train = []
        y_pred_train = []

        train_loss = 0.0

        for emb_batch, ko_batch, y_batch in train_loader:

            emb_batch = emb_batch.to(device)
            ko_batch = ko_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            logits = model(
                emb_batch,
                ko_batch,
                detach_ko=detach_ko
            )

            loss = ce_loss(logits, y_batch)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            y_true_train.extend(y_batch.cpu().numpy())
            y_pred_train.extend(
                logits.argmax(dim=1).cpu().numpy()
            )

        train_loss /= len(train_loader)

        # ==========================
        # VALIDATION
        # ==========================

        model.eval()

        val_loss = 0.0

        y_true = []
        y_pred = []

        with torch.no_grad():

            for emb_batch, ko_batch, y_batch in val_loader:

                emb_batch = emb_batch.to(device)
                ko_batch = ko_batch.to(device)
                y_batch = y_batch.to(device)

                logits = model(emb_batch, ko_batch)

                loss = ce_loss(logits, y_batch)

                val_loss += loss.item()

                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(
                    logits.argmax(dim=1).cpu().numpy()
                )

        val_loss /= len(val_loader)

        val_acc = accuracy_score(y_true, y_pred)

        val_f1 = f1_score(
            y_true,
            y_pred,
            average="macro"
        )

        val_prec = precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0
        )

        scheduler.step(val_loss)

        print(
            f"[Epoch {epoch+1}] "
            f"Val Loss={val_loss:.4f} "
            f"Val Acc={val_acc:.4f} "
            f"Val F1={val_f1:.4f}"
        )

        metrics = {
            "val_loss": val_loss
        }

        improved = early_stopping.check(metrics, epoch)

        if improved:

            best_val_f1 = val_f1
            best_epoch = epoch

            best_state_dict = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

        if early_stopping.stop_training:

            print(
                f"EARLY STOPPING "
                f"(best epoch={best_epoch+1})"
            )

            break

    model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch
    }

# =========================================================
# TEST
# =========================================================

def evaluate_test(
    model,
    test_loader,
    class_weights,
    device
):

    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    model.eval()

    test_loss = 0.0

    y_true = []
    y_pred = []

    with torch.no_grad():

        for emb_batch, ko_batch, y_batch in test_loader:

            emb_batch = emb_batch.to(device)
            ko_batch = ko_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(emb_batch, ko_batch)

            loss = ce_loss(logits, y_batch)

            test_loss += loss.item()

            y_true.extend(y_batch.cpu().numpy())

            y_pred.extend(
                logits.argmax(dim=1).cpu().numpy()
            )

    test_loss /= len(test_loader)

    test_acc = accuracy_score(y_true, y_pred)

    test_f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    test_prec = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    return {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "test_prec": test_prec
    }

# =========================================================
# GRID SEARCH
# =========================================================

def generate_search_space():

    grid = {

        "patience": [15],

        "delta": [1e-3],

        "proj_dim": [256, 128, 64,32],

        "proj_ko_dim": [256,128,64, 32],

        "dropout_emb": [0.1],

        "dropout_ko": [0.2],

        "lr": [3e-5, 1e-4, 3e-4],

        "wd": [1e-3],

        "label_smoothing": [0.01, 0.05],

        "detach_ko_epochs": [3, 5, 10, 15, 25,50],

        "batch_size": [144]
    }

    keys = list(grid.keys())

    values = list(grid.values())

    all_combinations = list(itertools.product(*values))

    configs = []

    for combo in all_combinations:

        config = dict(zip(keys, combo))

        configs.append(config)

    return configs

# =========================================================
# MAIN
# =========================================================

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input_folder",
        required=True
    )

    parser.add_argument(
        "-o",
        "--output_folder",
        required=True
    )

    parser.add_argument(
        "-c",
        "--component",
        choices={"mean", "std", "max", "wmean"},
        default=None
    )

    return parser.parse_args()

# =========================================================

if __name__ == "__main__":

    args = parse_arguments()

    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(args.output_folder, exist_ok=True)

    # =====================================================
    # LOAD DATA
    # =====================================================

    print("Loading embeddings...")

    X = load_embeddings(
        args.input_folder,
        args.component
    )

    print("Loaded:", X.shape)

    Ko = pd.read_csv(
        '/data/users/sofia/england/ko_kept_with_global.csv'
    ).set_index('Genomes')

    Ko.index = [i[:i.index('.')] for i in Ko.index]

    Ko_X = X.merge(
        Ko,
        left_index=True,
        right_index=True
    )

    labels_df = pd.read_csv(
        "/data/users/sofia/england/labels_ok_environmental_together.csv"
    ).set_index("Unnamed: 0")["Macro macro environment"]

    labels_df = labels_df.drop("3300025899_10")

    X_sorted = Ko_X.loc[labels_df.index]

    label_mapping = {
        label: idx
        for idx, label in enumerate(labels_df.unique())
    }

    #y = labels_df.map(label_mapping).values.astype(np.int64)
    y = pd.Series(labels_df.map(label_mapping).values.astype(np.int64), index = labels_df.index)

    names = list(X_sorted.index)

    emb_sorted = X_sorted[
        list(X_sorted.columns)[:X.shape[1]]
    ]

    ko_sorted = X_sorted[
        list(X_sorted.columns)[X.shape[1]:]
    ]

    # =====================================================
    # SPLIT
    # =====================================================

    train_labels = pd.read_csv('/data/users/sofia/england/MikaelSplits/y_train_metagenomesplits.csv').set_index('mag_id').drop('3300025899_10')
    validation_labels = pd.read_csv('/data/users/sofia/england/MikaelSplits/y_validation_metagenomesplits.csv').set_index('Unnamed: 0')
    test_labels = pd.read_csv('/data/users/sofia/england/MikaelSplits/y_test_metagenomesplits.csv').set_index('Unnamed: 0')
    
    emb_train, ko_train, y_train = emb_sorted.loc[train_labels.index].to_numpy(dtype=np.float32), ko_sorted.loc[train_labels.index].to_numpy(dtype=np.float32), y.loc[train_labels.index].to_numpy(dtype=np.int64)
    emb_val, ko_val, y_val = emb_sorted.loc[validation_labels.index].to_numpy(dtype=np.float32), ko_sorted.loc[validation_labels.index].to_numpy(dtype=np.float32), y.loc[validation_labels.index].to_numpy(dtype=np.int64)
    emb_test, ko_test, y_test = emb_sorted.loc[test_labels.index].to_numpy(dtype=np.float32), ko_sorted.loc[test_labels.index].to_numpy(dtype=np.float32), y.loc[test_labels.index].to_numpy(dtype=np.int64)
    '''emb_train, emb_temp, ko_train, ko_temp, y_train, y_temp = train_test_split(
        emb_sorted.values,
        ko_sorted.values,
        y,
        test_size=0.3,
        random_state=0,
        stratify=y
    )

    emb_val, emb_test, ko_val, ko_test, y_val, y_test = train_test_split(
        emb_temp,
        ko_temp,
        y_temp,
        test_size=0.5,
        random_state=0,
        stratify=y_temp
    )'''

    # =====================================================
    # SCALE
    # =====================================================

    scaler_emb = StandardScaler()

    emb_train = scaler_emb.fit_transform(emb_train)
    emb_val = scaler_emb.transform(emb_val)
    emb_test = scaler_emb.transform(emb_test)

    ko_train = np.log1p(ko_train)
    ko_val = np.log1p(ko_val)
    ko_test = np.log1p(ko_test)

    # =====================================================
    # CLASS WEIGHTS
    # =====================================================

    classes = np.unique(y_train)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    ).to(device)

    # =====================================================
    # GRID SEARCH
    # =====================================================

    configs = generate_search_space()

    print(f"\nTOTAL TRIALS: {len(configs)}\n")

    results = []

    for trial_idx, config in enumerate(configs):

        print("\n" + "="*80)
        print(f"TRIAL {trial_idx}")
        print(config)
        print("="*80)

        trial_dir = os.path.join(
            args.output_folder,
            f"trial_{trial_idx}"
        )

        os.makedirs(trial_dir, exist_ok=True)

        # =============================================
        # DATALOADERS
        # =============================================

        batch_size = config["batch_size"]

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(emb_train, dtype=torch.float32),
                torch.tensor(ko_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.long)
            ),
            batch_size=batch_size,
            shuffle=True
        )

        val_loader = DataLoader(
            TensorDataset(
                torch.tensor(emb_val, dtype=torch.float32),
                torch.tensor(ko_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long)
            ),
            batch_size=batch_size,
            shuffle=False
        )

        test_loader = DataLoader(
            TensorDataset(
                torch.tensor(emb_test, dtype=torch.float32),
                torch.tensor(ko_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long)
            ),
            batch_size=batch_size,
            shuffle=False
        )

        # =============================================
        # MODEL
        # =============================================

        model = TwoTowersClassifier(
            emb_dim=emb_train.shape[1],
            ko_dim=ko_train.shape[1],
            proj_dim=config["proj_dim"],
            proj_ko_dim=config["proj_ko_dim"],
            dropout_emb=config["dropout_emb"],
            dropout_ko=config["dropout_ko"],
            num_classes=3
        ).to(device)

        # =============================================
        # TRAIN
        # =============================================

        train_results = train_and_evaluate_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            class_weights=class_weights,
            device=device,
            config=config
        )

        model = train_results["model"]

        # =============================================
        # TEST
        # =============================================

        test_results = evaluate_test(
            model=model,
            test_loader=test_loader,
            class_weights=class_weights,
            device=device
        )

        # =============================================
        # SAVE MODEL
        # =============================================

        torch.save(
            model.state_dict(),
            os.path.join(trial_dir, "best_model.pt")
        )

        # =============================================
        # STORE RESULTS
        # =============================================

        row = {
            "trial": trial_idx,
            **config,
            "best_epoch": train_results["best_epoch"] + 1,
            "best_val_f1": train_results["best_val_f1"],
            **test_results
        }

        results.append(row)

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values(
            by="test_f1",
            ascending=False
        )

        results_df.to_csv(
            os.path.join(
                args.output_folder,
                "gridsearch_results.csv"
            ),
            index=False
        )

    print("\nDONE.")