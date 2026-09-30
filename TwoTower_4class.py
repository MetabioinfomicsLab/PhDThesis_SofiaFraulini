import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score
import numpy as np
import pandas as pd
from collections import Counter
import argparse
import random
import os
import matplotlib.pyplot as plt
import json
from pathlib import Path
import copy
from sklearn.metrics import confusion_matrix
import seaborn as sns
import umap
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
from torch.utils.data import Sampler
import math
from collections import defaultdict


# ----------------------------
# Seed
# ----------------------------
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------
# Your embedding loader (kept)
# ----------------------------
def load_embeddings(folder, comp=None):
    EMB_DIRS = [Path(folder)]
    emb_files = []
    for d in EMB_DIRS:
        if d.exists():
            emb_files.extend(sorted(d.glob("*.npy")))
    if not emb_files:
        raise SystemExit("No .npy embeddings found.")

    vectors, records = [], []
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

    # This part only matters if you use concatenated components
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

def plot_classifier_weight_blocks(model, emb_dim, ko_dim, class_names, save_path):
    """
    Plots how much the final classifier relies on embedding dims vs KO dims.

    Assumes model.classifier is a Linear layer with input dim = emb_dim + ko_dim.
    """

    W = model.classifier.weight.detach().cpu().numpy()  # (num_classes, in_dim)
    num_classes, in_dim = W.shape
    assert in_dim == emb_dim + ko_dim, (
        f"Classifier input dim mismatch. Got {in_dim}, expected {emb_dim + ko_dim}."
    )
    weights = pd.DataFrame(W, index=class_names).T
    weights['Mode'] = ['Embedding' if int(i) < 64 else 'KO' for i in weights.index]
    
    f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize = (8, 8), sharex= True)
    sns.barplot(data = weights, x = weights.index, y = 'Aquatic', hue = 'Mode', ax = ax1, legend= None)
    sns.barplot(data = weights, x = weights.index, y = 'Terrestrial', hue = 'Mode', ax = ax4, legend= None)
    sns.barplot(data = weights, x = weights.index, y = 'Engineered', hue = 'Mode', ax = ax2, legend= None)
    sns.barplot(data = weights, x = weights.index, y = 'Host-associated', hue = 'Mode', ax = ax3, legend= None)
    ax1.tick_params(axis = 'x', which = 'both', bottom = False)
    ax3.tick_params(axis = 'x', which = 'both', bottom = False)
    ax2.tick_params(axis = 'x', which = 'both', bottom = False)
    ax4.tick_params(axis = 'x', which = 'both', bottom = False, labelbottom=False)
    ax1.set_ylabel('Aquatic', size = 12)
    ax2.set_ylabel('Engineered', size = 12)
    ax3.set_ylabel('Host-associated', size = 12)
    ax4.set_ylabel('Terrestrial', size = 12)
    ax4.set_xlabel('Features', size = 12)
    ax1.tick_params(axis = 'y', which = 'major', labelsize = 10)
    ax2.tick_params(axis = 'y', which = 'major', labelsize = 10)
    ax3.tick_params(axis = 'y', which = 'major', labelsize = 10)
    ax4.tick_params(axis = 'y', which = 'major', labelsize = 10)
    ax1.set_title('Features weights across classes, KO vs emb', size = 14)
    plt.tight_layout()  
    plt.savefig(os.path.join(args.output_folder, 'weights_barplot_acrossclasses.png'), dpi = 300)
    
    
    weights['dim'] = weights.index
    weights_melted = pd.melt(weights, id_vars = ['dim', 'Mode'])
    weights_melted['abs'] = abs(weights_melted['value'])
    plt.figure(figsize = (12,3))
    sns.barplot(data = weights_melted, x = 'dim', y = 'abs', hue = 'Mode', errorbar=None)
    plt.tick_params(axis = 'x', which = 'both', bottom = False, labelbottom=False)
    plt.tick_params(axis = 'y', which = 'major', labelsize = 10)
    plt.ylabel('Abs weight', size = 12)
    plt.xlabel('Feature', size = 12)
    plt.legend(title = 'Data')
    plt.title('Absolute weights averaged across classes', size = 14)
    plt.tight_layout()  
    plt.savefig(os.path.join(args.output_folder, 'weights_abs_barplot.png'), dpi = 300)
    
    plt.figure(figsize = (5,3))
    sns.violinplot(data = weights_melted, x = 'variable', y = 'abs', hue = 'Mode', width=.7)
    plt.tick_params(axis = 'x', which = 'major', labelsize = 10, labelrotation = 10)
    plt.tick_params(axis = 'y', which = 'major', labelsize = 10)
    plt.ylabel('Abs weight', size = 12)
    plt.xlabel('Class', size = 12)
    plt.legend(title = 'Data',bbox_to_anchor = (1,1))
    plt.title('Abs weight distribution across classes')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder, 'weights_abs_violin.png'), dpi = 300)

    weights['Avg'] = abs(weights.drop(['dim', 'Mode'], axis = 1)).T.mean() #Avg across absolute values
    plt.figure(figsize = (6,4))
    sns.histplot(data = weights, x = 'Avg', bins = 30, hue = 'Mode', stat = 'proportion', multiple='stack')
    plt.xlabel('Abs weights (avg across classes)', size = 12)
    plt.ylabel('Proportion', size = 12)
    plt.tick_params(axis = 'both', which = 'major', labelsize = 10)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder, 'weights_abs_hist.png'), dpi = 300)
    
    weights.sort_values(by = 'Avg').to_csv(os.path.join(args.output_folder, 'final_weights.csv'))
# ----------------------------
# Early stopping (fixed)
# ----------------------------
class EarlyStopping:
    def __init__(self, patience=15, delta=1e-3, monitor='val_loss', mode='min'):
        self.monitor = monitor
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.no_improvement_count = 0
        self.stop_training = False
        self.mode = mode
        self.best_epoch = 0

    def check(self, metrics, epoch):
        metric_now = metrics.get(self.monitor)
        if metric_now is None:
            raise ValueError(f"Metric '{self.monitor}' not found in metrics dict.")

        if self.best_score is None:
            self.best_score = metric_now
            self.no_improvement_count = 0
            self.best_epoch = epoch
            return True

        improved = False
        if self.mode == 'min':
            improved = metric_now < (self.best_score - self.delta)
        elif self.mode == 'max':
            improved = metric_now > (self.best_score + self.delta)
        else:
            raise ValueError("mode must be 'min' or 'max'")

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
        
class TwoTowersClassifier(nn.Module):
    def __init__(self, emb_dim, ko_dim, proj_dim=256, proj_ko_dim = 64, dropout_emb = 0.2, dropout_ko = 0.1, num_classes = 3):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.TowerEmb = nn.Sequential(nn.Linear(emb_dim, 512), nn.LayerNorm(512), 
                        nn.GELU(), nn.Dropout(dropout_emb), nn.Linear(512, 128),
                        nn.LayerNorm(128), nn.GELU(), nn.Linear(128, proj_dim))
        self.TowerKO = nn.Sequential(nn.Linear(ko_dim, 512), nn.LayerNorm(512), 
                        nn.GELU(), nn.Dropout(dropout_ko), nn.Linear(512, 128),
                        nn.LayerNorm(128), nn.GELU(), nn.Linear(128, proj_ko_dim))#,
                        #nn.LayerNorm(proj_ko_dim), nn.ReLU())
        self.classifier = nn.Linear(proj_ko_dim+proj_dim, num_classes)
        
    def forward(self, emb, ko, detach_ko = False):
        z_emb = self.TowerEmb(emb)
        z_ko = self.TowerKO(ko)

        if detach_ko:
            z_ko = z_ko.detach()
        
        z_emb = F.normalize(z_emb, dim=1)
        z_ko  = F.normalize(z_ko, dim=1)
        alpha = torch.sigmoid(self.alpha) #keep between 0 and 1
        z_concat = torch.cat((alpha * z_emb, (1 - alpha) * z_ko), dim=1)
        #z_concat = torch.cat((z_emb,z_ko), dim =1)
        logits = self.classifier(z_concat)
        return logits, z_emb, z_ko, z_concat
        
def plot_umap(features, labels, class_names, epoch, output_folder, emb_dim = 64, ko_dim = 64,min_dist = 0.1,n_neighbors=100):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric = 'cosine')
    X_umap = reducer.fit_transform(features)
    
    if features.shape[1] == emb_dim:
        mode = 'Embeddings'
    elif features.shape[1] == ko_dim: 
        mode = 'KO'
    else:
        mode = 'KO+Emb'
    if len(set(labels)) == 2:
        case = '_ENV'
    else:
        case = ''
    plt.figure(figsize=(6,5))
    for cls_idx, cls_name in enumerate(class_names):
        plt.scatter(
            X_umap[labels == cls_idx, 0],
            X_umap[labels == cls_idx, 1],
            label=cls_name,
            alpha=0.7,
            s=20
        )
    plt.legend()
    plt.title(f"UMAP Epoch {epoch+1} from {mode}")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"umap_epoch_{mode}_{epoch+1}{case}.png"))
    plt.close()
    
def plot_umap_at_beginning(X, y, class_names, title, save_path=None, n_neighbors=100, min_dist=0.1, random_state=0):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric = 'cosine')
    X_umap = reducer.fit_transform(X)

    plt.figure(figsize=(6,5))
    for cls_idx, cls_name in enumerate(class_names):
        plt.scatter(
            X_umap[y == cls_idx, 0],
            X_umap[y == cls_idx, 1],
            label=cls_name,
            alpha=0.7,
            s=20
        )
    plt.legend()
    plt.title(title)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()   

def hook_forward(module_name, grads, hook_backward):
    def hook(module, args, output):
        if output.requires_grad:
            output.register_hook(hook_backward(module_name, grads))
    return hook

def hook_backward(module_name, grads):
    def hook(grad):
        #Appends gradients
        grads.append((module_name, grad))
    return hook

def get_all_layers(model, hook_forward, hook_backward):
    #Register forward pass hook (which registers a backward hook) to model outputs
    layers = dict()
    grads = []
    for name, layer in model.named_modules():
        # skip Sequential and/or wrapper modules
        if any(layer.children()) is False:
            layers[layer] = name
            layer.register_forward_hook(hook_forward(name, grads, hook_backward))
    return layers, grads

#Collect activation zero percentage (ReLU check) + identity checks across linear layers
def register_diagnostic_hooks(model):
    activation_stats = {}
    identity_stats = {}

    def relu_hook(name):
        def hook(module, input, output):
            with torch.no_grad():
                zero_pct = (output == 0).float().mean().item()
                activation_stats[name] = zero_pct
        return hook
    
    def identity_hook(name):
        def hook(module, input, output):
            inp = input[0].detach()
            out = output.detach()
            if inp.shape == out.shape:
                diff = (out - inp).abs().mean().item()
                identity_stats[name] = diff
        return hook

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.GELU):
            layer.register_forward_hook(relu_hook(name))
        if isinstance(layer, torch.nn.Linear):
            layer.register_forward_hook(identity_hook(name))    

    return activation_stats, identity_stats     
        
def train_and_evaluate_model(model, train_loader, val_loader,grads_bn,epochs=200, lr=3e-4, wd=1e-4,patience=20, delta=1e-4,
                        device="cuda", class_names = None, class_weights = None):

    device = next(model.parameters()).device
    print(wd)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.01)
    num_classes=len(class_weights)

    
    # Important: centers need optimizer too
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3,
    min_lr=1e-6, threshold_mode = 'abs',
)
    early_stopping = EarlyStopping(
        patience=patience,
        delta=delta,
        monitor="val_loss",
        mode="min"
    )

    activation_history = defaultdict(lambda: defaultdict(list))
    identity_history = defaultdict(lambda: defaultdict(list))
    activation_grad_epoch_history = defaultdict(dict)
    grad_norms_per_par_epochs = defaultdict(dict)
    
    activation_stats, identity_stats = register_diagnostic_hooks(model)
    
    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_f1 = -1.0

    train_losses, val_losses = [], []
    alphas = []
    
    for epoch in range(epochs):
        
        detach_ko = epoch < 3
        # ----------------
        # Train
        # ----------------
        model.train()
        train_loss = 0.0
        epoch_activation_accumulator = defaultdict(list)
        y_true_train, y_pred_train = [], []
        alphas_per_epochs = 0
        
        for emb_batch, ko_batch, y_batch in train_loader:
            emb_batch, ko_batch, y_batch = emb_batch.to(device), ko_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            #optimizer_centers.zero_grad()
            logits, z_emb, z_ko, z_concat = model(emb_batch, ko_batch, detach_ko)
            alphas_per_epochs += torch.sigmoid(model.alpha).item()
            
            for k, v in activation_stats.items():
                activation_history[epoch][k].append(v)
            for k, v in identity_stats.items():
                identity_history[epoch][k].append(v)
                
            loss = ce_loss(logits, y_batch)
            loss.backward()
            
            #activation gradients
            for layer_name, grad in grads_bn:
                if grad is not None:
                    mean_grad = grad.abs().mean().item()
                    # accumulate for epoch averaging
                    epoch_activation_accumulator[layer_name].append(mean_grad)
            
            grads_bn.clear()
            
            #weight gradients norm
            for n,p in model.named_parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    if n in grad_norms_per_par_epochs[epoch].keys():
                        grad_norms_per_par_epochs[epoch][n].append(param_norm.item())
                    else:
                        grad_norms_per_par_epochs[epoch][n] = [param_norm.item()]
                        
            optimizer.step()

            train_loss += loss.item()
            y_true_train.extend(y_batch.cpu().numpy())
            y_pred_train.extend(logits.argmax(dim=1).cpu().numpy())

        for layer_name, values in epoch_activation_accumulator.items():
            activation_grad_epoch_history[epoch][layer_name] = np.mean(values)
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_f1 = f1_score(y_true_train, y_pred_train, average="macro")
        alphas_avg = alphas_per_epochs / len(train_loader)
        alphas.append(alphas_avg)
        
        # ----------------
        # Validation
        # ----------------
        model.eval()

        val_loss = 0.0
        y_true, y_pred = [], []

        with torch.no_grad():
            for emb_batch, ko_batch, y_batch in val_loader:
                emb_batch, ko_batch, y_batch = emb_batch.to(device), ko_batch.to(device), y_batch.to(device)
                logits, z_emb, z_ko, z_concat = model(emb_batch, ko_batch)

                loss = ce_loss(logits, y_batch)
                val_loss += loss.item()

                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(logits.argmax(dim=1).cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_acc = accuracy_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred, average="macro")
        val_f1_weighted = f1_score(y_true, y_pred, average="weighted")
        f1_per_class_val = f1_score(y_true, y_pred, average=None)
        val_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)

        print(
            f"[Epoch {epoch+1}] "
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
            f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Val Prec: {val_prec:.4f}"
        )

        print(
            f"[Epoch {epoch+1}] "
            f"Val MACRO F1: {val_f1:.6f} | Val weighted F1: {val_f1_weighted:.6f} | "
            f"Val F1 per class: {f1_per_class_val}")
        
        if (epoch+1) % 20 == 0:
            all_features_ko = []
            all_features_emb = []
            all_features_concat = []
            all_labels = []
            model.eval()
            with torch.no_grad():
                for emb_batch, ko_batch, y_batch in train_loader:
                    emb_batch = emb_batch.to(device)
                    ko_batch = ko_batch.to(device)
                    _, z_emb, z_ko, z_concat = model(emb_batch, ko_batch)
                    all_features_ko.append(z_ko.cpu().numpy())
                    all_features_emb.append(z_emb.cpu().numpy())
                    all_features_concat.append(z_concat.cpu().numpy())
                    all_labels.append(y_batch.cpu().numpy())
            all_features_ko = np.vstack(all_features_ko)
            all_features_emb = np.vstack(all_features_emb)
            all_features_concat = np.vstack(all_features_concat)
            all_labels = np.hstack(all_labels)
            plot_umap(all_features_ko, all_labels, class_names, epoch, args.output_folder)
            plot_umap(all_features_emb, all_labels, class_names, epoch, args.output_folder)
            plot_umap(all_features_concat, all_labels, class_names, epoch, args.output_folder)
            
            if num_classes == 2:
                case = '_ENV'
            else:
                case = ''
            train_cm = confusion_matrix(y_true_train, y_pred_train, normalize='true')
            train_cm_df = pd.DataFrame(train_cm, index = class_names, columns = class_names)
            train_cm_df.to_csv(os.path.join(args.output_folder, f"confusion_matrix_train_epoch_{epoch+1}{case}.csv"))
            plt.figure(figsize=(6,5))
            sns.heatmap(train_cm_df, annot=True, cmap="Blues")
            plt.title(f"Train Confusion Matrix Epoch {epoch+1}{case}")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_folder, f"confusion_matrix_train_epoch_{epoch+1}{case}.png"))
            plt.close()
            
            val_cm = confusion_matrix(y_true, y_pred, normalize='true')
            val_cm_df = pd.DataFrame(val_cm, index=class_names, columns=class_names)
            val_cm_df.to_csv(os.path.join(args.output_folder, f"confusion_matrix_val_epoch_{epoch+1}{case}.csv"))
            plt.figure(figsize=(6,5))
            sns.heatmap(val_cm_df, annot=True, cmap="Blues")
            plt.title(f"Validation Confusion Matrix Epoch {epoch+1}{case}")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_folder, f"confusion_matrix_val_epoch_{epoch+1}{case}.png"))
            plt.close()
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"LR={current_lr:.6g}")
        
        metrics = {"val_f1": val_f1, "val_loss": avg_val_loss}
        improved = early_stopping.check(metrics, epoch)

        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        if early_stopping.stop_training:
            print(
                f"Stopping early. Best epoch: {best_epoch+1} | Best Val F1: {best_val_f1:.6f}"
            )
            break

    model.load_state_dict(best_state_dict)
    
    plot_classifier_weight_blocks(
    model=model,
    emb_dim=64,
    ko_dim=64,
    class_names=class_names,
    save_path=os.path.join(args.output_folder, "classifier_weight_blocks.png")
)

    alphas_df = pd.Series(alphas, name = 'alpha')
    plt.figure(figsize = (4,6))
    sns.lineplot(alphas_df)
    plt.savefig(os.path.join(args.output_folder, 'alphas.png'))
    
    return model, train_losses, val_losses, best_epoch, best_val_f1,grad_norms_per_par_epochs, activation_history, identity_history,activation_grad_epoch_history

def extract_refined_embeddings(model, emb_np, ko_np, batch_size=128, device="cuda"):
    loader = DataLoader(TensorDataset(torch.tensor(emb_np, dtype=torch.float32), torch.tensor(ko_np, dtype=torch.float32)),
        batch_size=batch_size, shuffle=False)
    
    model.eval()
    refined = []

    with torch.no_grad():
        for emb_batch, ko_batch in loader:
            emb_batch, ko_batch = emb_batch.to(device), ko_batch.to(device)
            _, _, _, z = model(emb_batch, ko_batch)
            refined.append(z.cpu().numpy())

    return np.vstack(refined)

#Plots diagnostics
def plot_relu_zero(activation_history):
    plt.figure(figsize=(6, 4)) 
    for layer in activation_history[0].keys():
        epoch_means = []
        for epoch in activation_history.keys():
            vals = activation_history[epoch][layer]
            epoch_means.append(np.mean(vals))

        plt.plot(epoch_means, label=layer)

    plt.xlabel("Epoch", size = 14)
    plt.ylabel("Zero Activation prop", size = 14)
    plt.tick_params(which = 'major', axis = 'both', labelsize = 12)
    plt.legend(title = 'ReLU()', bbox_to_anchor = (1,1))
    plt.savefig(os.path.join(args.output_folder, 'zero_relu.png'), dpi = 300)
    plt.close()
    
def plot_identity(identity_history):
    plt.figure(figsize=(6, 4)) 
    all_layers = set()
    for epoch_dict in identity_history.values():
        all_layers.update(epoch_dict.keys())
    
    for layer in sorted(all_layers):
        epoch_means = []
        for epoch in identity_history.keys():
            vals = identity_history[epoch].get(layer, [])
            epoch_means.append(np.mean(vals))

        plt.plot(epoch_means, label=layer)

    plt.xlabel("Epoch", size = 14)
    plt.ylabel("Mean |Output - Input|", size = 14)
    plt.tick_params(which = 'major', axis = 'both', labelsize = 12)
    plt.legend(title = 'Layer')
    plt.savefig(os.path.join(args.output_folder, 'identity_checks.png'), dpi = 300)
    plt.close()
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Center-loss refiner + classifier for Evo2 embeddings")
    parser.add_argument("-i", "--input_folder", required=True, help="Folder with .npy embeddings")
    parser.add_argument("-o", "--output_folder", required=True, help="Output directory")
    parser.add_argument("-c", "--component", choices={"mean", "std", "max", "wmean"}, default=None)
    parser.add_argument("-s", "--seed", default=42, type = int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder, exist_ok=True)

    print("Loading embeddings...")
    X = load_embeddings(args.input_folder, args.component)
    print("Loaded:", X.shape)
    Ko = pd.read_csv('/data/users/sofia/england/ko_kept_with_global.csv').set_index('Genomes')
    Ko.index = [i[:i.index('.')] for i in Ko.index]
    Ko_X = X.merge(Ko, left_index= True, right_index= True)

    #labels_df = pd.read_csv("/data/users/sofia/england/labels_ok_environmental_together.csv").set_index("Unnamed: 0")["Macro macro environment"]
    labels_df_env = pd.read_csv("/data/users/sofia/england/labels_ok_environmental_together.csv").set_index("Unnamed: 0")['Macro environment']
    #labels_df = labels_df.drop("3300025899_10")
    labels_df_env = labels_df_env.drop("3300025899_10")
    
    X_sorted = Ko_X.loc[labels_df_env.index]
    print("Sorted X:", X_sorted.shape)

    label_mapping = {label: idx for idx, label in enumerate(labels_df_env.unique())}
    y = pd.Series(labels_df_env.map(label_mapping).values.astype(np.int64), index = labels_df_env.index)

    print("Label mapping:", label_mapping)
    print("Label counts:", Counter(y))
    
    names = list(X_sorted.index)
    
    emb_sorted, Ko_sorted = X_sorted[list(X_sorted.columns)[:X.shape[1]]], X_sorted[list(X_sorted.columns)[X.shape[1]:]]
    print(f"Sorted emb_X: {emb_sorted.shape}; Sorted Ko_X: {Ko_sorted.shape}")
    
    train_labels = pd.read_csv('/data/users/sofia/england/MikaelSplits/y_train_metagenomesplits.csv').set_index('mag_id').drop('3300025899_10')
    validation_labels = pd.read_csv('/data/users/sofia/england/MikaelSplits/y_validation_metagenomesplits.csv').set_index('Unnamed: 0')
    test_labels = pd.read_csv('/data/users/sofia/england/MikaelSplits/y_test_metagenomesplits.csv').set_index('Unnamed: 0')
    
    emb_train, ko_train, y_train, names_train = emb_sorted.loc[train_labels.index].to_numpy(dtype=np.float32), Ko_sorted.loc[train_labels.index].to_numpy(dtype=np.float32), y.loc[train_labels.index].to_numpy(dtype=np.int64), train_labels.index.tolist()
    emb_val, ko_val, y_val, names_val = emb_sorted.loc[validation_labels.index].to_numpy(dtype=np.float32), Ko_sorted.loc[validation_labels.index].to_numpy(dtype=np.float32), y.loc[validation_labels.index].to_numpy(dtype=np.int64), validation_labels.index.tolist()
    emb_test, ko_test, y_test, names_test = emb_sorted.loc[test_labels.index].to_numpy(dtype=np.float32), Ko_sorted.loc[test_labels.index].to_numpy(dtype=np.float32), y.loc[test_labels.index].to_numpy(dtype=np.int64), test_labels.index.tolist()
    #emb_val, emb_test, ko_val, ko_test, y_val, y_test, names_val, names_test = train_test_split(emb_temp, ko_temp, y_temp, names_temp, test_size=0.5, random_state=0, stratify=y_temp)

    print(f'Shape train set: {train_labels.shape[0]}; Shape validation set: {validation_labels.shape[0]}; Shape test set: {test_labels.shape[0]}')
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    print(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    # Standardize
    scaler_emb = StandardScaler()
    emb_train = scaler_emb.fit_transform(emb_train)
    emb_val = scaler_emb.transform(emb_val)
    emb_test = scaler_emb.transform(emb_test)   
    
    #scaler_ko = StandardScaler()
    ko_train = np.log1p(ko_train)
    ko_val = np.log1p(ko_val)
    ko_test = np.log1p(ko_test)
    #ko_train = scaler_ko.fit_transform(ko_train)
    #ko_val = scaler_ko.transform(ko_val)
    #ko_test = scaler_ko.transform(ko_test)   
    
    class_names = [label for label, _ in sorted(label_mapping.items(), key=lambda x: x[1])]
    #plot_umap_at_beginning(emb_train, y = y_train, class_names = class_names, title = 'Embeddings training set before training', save_path = os.path.join(args.output_folder, 'umap_before_training_train_set.png'), n_neighbors = 15)
    #plot_umap_at_beginning(emb_val, y = y_val, class_names = class_names, title = 'Embeddings validation set before training', save_path = os.path.join(args.output_folder, 'umap_before_training_val_set.png'), n_neighbors = 15)
    #plot_umap_at_beginning(ko_train, y = y_train, class_names = class_names, title = 'KO training set before training', save_path = os.path.join(args.output_folder, 'umap_before_training_train_set_KO.png'), n_neighbors = 15)
    #plot_umap_at_beginning(ko_val, y = y_val, class_names = class_names, title = 'KO validation set before training', save_path = os.path.join(args.output_folder, 'umap_before_training_val_set_KO.png'), n_neighbors = 15)
    
    # Loaders
    batch_size = 128  # IMPORTANT for center loss stability
    
    train_dataset = TensorDataset(torch.tensor(emb_train, dtype=torch.float32),torch.tensor(ko_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long) )
    
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(emb_val, dtype=torch.float32),torch.tensor(ko_val, dtype=torch.float32),torch.tensor(y_val, dtype=torch.long)),
                batch_size=batch_size,shuffle=False)
    
    model = TwoTowersClassifier(emb_dim=emb_train.shape[1], ko_dim=ko_train.shape[1],proj_dim=64,proj_ko_dim = 64, dropout_emb=0.1, dropout_ko=0.2, num_classes=4).to(device)
    
    layers_bn, grads_bn = get_all_layers(model, hook_forward, hook_backward)
    model, train_losses, val_losses, best_epoch, best_val_f1, grad_norms_per_par_epochs, activation_history, identity_history,activation_grad_epoch_history = train_and_evaluate_model(
        model, train_loader, val_loader, grads_bn, epochs=300, lr=1e-4, wd=1e-3, patience=15, delta=1e-3,
        device=device, class_names = class_names, class_weights = class_weights)
    
    # Plot loss
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder, "loss_curve.png"))
    plt.close()
    
    test_loader = DataLoader(TensorDataset(torch.tensor(emb_test, dtype=torch.float32), torch.tensor(ko_test, dtype=torch.float32),
                    torch.tensor(y_test, dtype=torch.long)),batch_size=128,shuffle=False)

    model.eval()
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)

    test_loss = 0.0
    y_true, y_pred = [], []
    all_probs = []

    with torch.no_grad():
        for emb_batch, ko_batch, y_batch in test_loader:
            emb_batch, ko_batch, y_batch = emb_batch.to(device), ko_batch.to(device), y_batch.to(device)
            logits, _, _, _ = model(emb_batch, ko_batch)
            
            loss = ce_loss(logits, y_batch)
            test_loss += loss.item()

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy())

    test_loss /= len(test_loader)
    test_acc = accuracy_score(y_true, y_pred)
    test_f1 = f1_score(y_true, y_pred, average="macro")
    test_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n=== TEST PERFORMANCE (Best Model) ===")
    print(f"Loss: {test_loss:.6f}")
    print(f"Accuracy: {test_acc:.6f}")
    print(f"Precision: {test_prec:.6f}")
    print(f"Macro F1: {test_f1:.6f}")

    # Save predictions
    probs_df = pd.DataFrame(all_probs, columns=[f"Prob_{c}" for c in class_names])
    probs_df["Predicted"] = [class_names[i] for i in y_pred]
    probs_df["True"] = [class_names[i] for i in y_true]
    probs_df.to_csv(os.path.join(args.output_folder, "test_predictions_with_probs.csv"), index=False)

    # Save model
    torch.save(model.state_dict(), os.path.join(args.output_folder, "best_model.pt"))

    print("\nSaved best model + centers.")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(args.output_folder, "confusion_matrix.csv"))

    plt.figure(figsize=(6,5))
    sns.heatmap(cm_df, annot=True, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True")  
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder, "confusion_matrix.png"))
    plt.close()
    print("Saved confusion matrix.")

    # --- Save refined embeddings for test set ---
    refined_train = extract_refined_embeddings(model, emb_train, ko_train, batch_size=batch_size, device=device)
    refined_val   = extract_refined_embeddings(model, emb_val, ko_val, batch_size=batch_size, device=device)
    refined_test  = extract_refined_embeddings(model, emb_test, ko_test, batch_size=batch_size, device=device)

    refined_embeddings_df = pd.DataFrame(refined_test, index=names_test)
    refined_embeddings_df.to_csv(os.path.join(args.output_folder, "refined_test_embeddings.csv"))
    refined_embeddings_train_df = pd.DataFrame(refined_train, index=names_train)
    refined_embeddings_train_df.to_csv(os.path.join(args.output_folder, "refined_train_embeddings.csv"))
    refined_embeddings_val_df = pd.DataFrame(refined_val, index=names_val)
    refined_embeddings_val_df.to_csv(os.path.join(args.output_folder, "refined_val_embeddings.csv"))
    print("Saved refined embeddings.")
    
        #Gradients analyses and extra checks
    weight_gradients_norms = {'Epoch': [], 'Layer': [], 'Batch': [],'Grad norm': []}
    for e in grad_norms_per_par_epochs:
        for l in grad_norms_per_par_epochs[e]:
            for n_b in range(len(grad_norms_per_par_epochs[e][l])):
                weight_gradients_norms['Epoch'].append(e)
                weight_gradients_norms['Layer'].append(l)
                weight_gradients_norms['Batch'].append(n_b)
                weight_gradients_norms['Grad norm'].append(grad_norms_per_par_epochs[e][l][n_b])
    weight_gradients_norms = pd.DataFrame(weight_gradients_norms)
    
    #Useful for weight gradients plots
    weight_gradients_norms.to_csv(os.path.join(args.output_folder, 'weight_gradients.csv'))
    plt.figure(figsize = (5,7))
    sns.pointplot(data = weight_gradients_norms.loc[weight_gradients_norms['Layer'].isin([i for i in weight_gradients_norms['Layer'] if 'weight' in i])], x= 'Layer', y = 'Grad norm', hue = 'Epoch')
    plt.tick_params(which = 'major', axis = 'x', labelrotation = 90, labelsize = 10)
    plt.yscale('log')
    plt.tick_params(which = 'major', axis = 'y', labelsize = 10)
    plt.ylabel('Weight grad L2 norm', size = 12)
    plt.xlabel('Layers', size = 12)
    plt.title('KO+embedding model', size = 14)
    plt.legend(bbox_to_anchor=(1,1), title = 'Epochs')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder, 'weight_gradients_acrosslayersepochs.png'), dpi = 300)
    plt.close()
    
    #Useful for activation gradients plots
    activation_grad_epoch_history = pd.DataFrame(activation_grad_epoch_history)
    activation_grad_epoch_history.to_csv(os.path.join(args.output_folder, 'activation_gradients.csv'))
    #Plots
    activation_grad_epoch_history['Layer'] = activation_grad_epoch_history.index
    activation_grad_epoch_history = pd.melt(activation_grad_epoch_history, id_vars = ['Layer'])
    activation_grad_epoch_history['Model'] = [i[:i.index('.')] if '.' in i else i for i in activation_grad_epoch_history['Layer']]
    plt.figure(figsize=(8, 6))
    sns.lineplot(data = activation_grad_epoch_history, y = 'value', x = 'variable', hue = 'Layer', style = 'Model')
    plt.yscale('log')
    plt.ylabel('Activation grad (mean abs)', size = 14)
    plt.xlabel('Epoch', size = 14)
    plt.tick_params(which = 'major', axis = 'both', labelsize = 12)
    plt.legend(bbox_to_anchor = (1,1))
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_folder, 'activation_gradients_acrosslayersepochs.png'), dpi = 300)
    plt.close()
    
    plot_relu_zero(activation_history)
    plot_identity(identity_history)