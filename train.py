import argparse
import os
import yaml
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

from dataset import build_dataloaders
from utils import set_seed, auto_device, save_confusion, print_classification_report, make_out_dir

def build_model(name: str, num_classes: int, freeze_backbone: bool = True):
    name = name.lower()
    if name == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_feats = net.fc.in_features
        net.fc = nn.Linear(in_feats, num_classes)
        if freeze_backbone:
            for p in net.parameters():
                p.requires_grad = False
            for p in net.fc.parameters():
                p.requires_grad = True
    elif name == "resnet18":
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_feats = net.fc.in_features
        net.fc = nn.Linear(in_feats, num_classes)
        if freeze_backbone:
            for p in net.parameters():
                p.requires_grad = False
            for p in net.fc.parameters():
                p.requires_grad = True
    elif name == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feats = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_feats, num_classes)
        if freeze_backbone:
            for p in net.parameters():
                p.requires_grad = False
            for p in net.classifier[-1].parameters():
                p.requires_grad = True
    elif name == "mobilenet_v3_large":
        net = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        in_feats = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_feats, num_classes)
        if freeze_backbone:
            for p in net.parameters():
                p.requires_grad = False
            for p in net.classifier[-1].parameters():
                p.requires_grad = True
    else:
        raise ValueError(f"Unknown model name: {name}")
    return net

def evaluate(model, loader, device, class_names):
    model.eval()
    y_true, y_pred = [], []
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / max(1, total)
    return acc, y_true, y_pred

def train(cfg):
    set_seed(cfg.get('seed', 42))
    device = auto_device(cfg['trainer'].get('device', 'auto'))
    out_dir = cfg['logging'].get('out_dir', 'outputs')
    make_out_dir(out_dir)

    train_loader, val_loader, class_names = build_dataloaders(
        cfg['data']['train_dir'],
        cfg['data']['val_dir'],
        cfg['data']['img_size'],
        cfg['data']['batch_size'],
        cfg['data']['num_workers']
    )

    num_classes = len(class_names) if cfg['model']['num_classes'] is None else cfg['model']['num_classes']
    model = build_model(cfg['model']['name'], num_classes, cfg['model']['freeze_backbone'])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg['trainer']['lr'], weight_decay=cfg['trainer']['weight_decay'])

    best_acc = 0.0
    patience = cfg['trainer']['early_stopping_patience']
    patience_cnt = 0
    max_epochs = cfg['trainer']['max_epochs']

    for epoch in range(1, max_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Eval
        val_acc, y_true, y_pred = evaluate(model, val_loader, device, class_names)
        print(f"Validation Accuracy: {val_acc:.4f}")

        # Early stopping & checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            patience_cnt = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'config': cfg
            }, os.path.join(out_dir, "best_model.pt"))
            print("✅ Saved best model.")
            if cfg['logging'].get('save_confusion_matrix', True):
                from utils import save_confusion, print_classification_report
                save_confusion(y_true, y_pred, class_names, os.path.join(out_dir, "confusion_matrix.png"))
                print_classification_report(y_true, y_pred, class_names)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print("⏹ Early stopping.")
                break

    print(f"Best val acc: {best_acc:.4f} (checkpoint saved in {out_dir}/best_model.pt)")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/config.yaml')
    # Lightweight overrides: key=value pairs, e.g., trainer.max_epochs=30 model.name=resnet18
    parser.add_argument('overrides', nargs='*', help="Optional key=value overrides")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    for kv in args.overrides:
        if '=' not in kv:
            continue
        key, val = kv.split('=', 1)
        # Nested set: e.g. trainer.max_epochs
        node = cfg
        keys = key.split('.')
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        # Try to parse numbers/bool
        if val.lower() in ['true', 'false']:
            val = val.lower() == 'true'
        else:
            try:
                if '.' in val:
                    val = float(val)
                else:
                    val = int(val)
            except ValueError:
                pass
        node[keys[-1]] = val

    # Derive num_classes from folders if left None
    if cfg['model'].get('num_classes', None) is None:
        cfg['model']['num_classes'] = None

    train(cfg)

if __name__ == '__main__':
    main()
