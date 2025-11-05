import argparse
import os
import glob
from typing import List
import torch
from torchvision import transforms, models
from PIL import Image
import yaml

def load_model(weights_path: str, device: str = None):
    ckpt = torch.load(weights_path, map_location='cpu')
    cfg = ckpt.get('config', None)
    class_names = ckpt.get('class_names', None)
    model_name = cfg['model']['name'] if cfg else 'resnet50'
    num_classes = len(class_names) if class_names else cfg['model']['num_classes']

    # Build same model arch
    if model_name == 'resnet50':
        net = models.resnet50(weights=None)
        in_feats = net.fc.in_features
        net.fc = torch.nn.Linear(in_feats, num_classes)
    elif model_name == 'resnet18':
        net = models.resnet18(weights=None)
        in_feats = net.fc.in_features
        net.fc = torch.nn.Linear(in_feats, num_classes)
    elif model_name == 'efficientnet_b0':
        net = models.efficientnet_b0(weights=None)
        in_feats = net.classifier[-1].in_features
        net.classifier[-1] = torch.nn.Linear(in_feats, num_classes)
    elif model_name == 'mobilenet_v3_large':
        net = models.mobilenet_v3_large(weights=None)
        in_feats = net.classifier[-1].in_features
        net.classifier[-1] = torch.nn.Linear(in_feats, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    net.load_state_dict(ckpt['model_state_dict'])
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    net.eval()
    return net, class_names, device

def collect_images(path: str) -> List[str]:
    if os.path.isdir(path):
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(path, e)))
        return sorted(files)
    if os.path.isfile(path):
        return [path]
    raise FileNotFoundError(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to best_model.pt')
    parser.add_argument('--images', type=str, required=True, help='Image file or folder')
    parser.add_argument('--class_names', type=str, default=None, help='Comma-separated class names if not in checkpoint')
    args = parser.parse_args()

    model, class_names, device = load_model(args.weights)
    if args.class_names and (not class_names or len(class_names) == 0):
        class_names = [c.strip() for c in args.class_names.split(',')]
    if not class_names:
        raise ValueError('Class names must be provided either in checkpoint or via --class_names')

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    files = collect_images(args.images)
    for fp in files:
        img = Image.open(fp).convert('RGB')
        x = tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1)
            conf, pred = prob.max(dim=1)
        label = class_names[pred.item()]
        print(f"{fp} -> {label} ({conf.item():.3f})")

if __name__ == '__main__':
    main()
