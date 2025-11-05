# Lychee Tree Disease Classification (Transfer Learning)

A clean, minimal PyTorch project to classify lychee tree leaf diseases using transfer learning.
Default backbone: `resnet50` (you can switch to `efficientnet_b0`, `mobilenet_v3_large`, etc.).

## 📂 Project Structure
```
lychee-disease-transfer-learning/
├─ src/
│  ├─ train.py           # Train a transfer learning classifier
│  ├─ infer.py           # Run inference on images/folder
│  ├─ dataset.py         # Dataset & transforms
│  ├─ utils.py           # Utilities (metrics, seeding, model factory)
│  └─ config.yaml        # Central config
├─ data/
│  ├─ train/CLASS_NAME/*.jpg
│  ├─ val/CLASS_NAME/*.jpg
│  └─ test/CLASS_NAME/*.jpg   (optional)
├─ outputs/              # Models, logs, metrics
├─ requirements.txt
└─ README.md
```

## ✅ Requirements
- Python 3.9+
- PyTorch + torchvision (CPU or CUDA)
- scikit-learn, PyYAML, tqdm, matplotlib

Install:
```bash
pip install -r requirements.txt
```

## 🧠 Prepare Data
Organize your dataset as:
```
data/
  train/
    healthy/
      img001.jpg
      ...
    disease_x/
      ...
  val/
    healthy/
    disease_x/
  test/  (optional)
```

If you only have one folder with images + labels CSV, adapt `dataset.py` or use `torchvision.datasets.ImageFolder` format above.

## 🚀 Train
```bash
python src/train.py --config src/config.yaml
```
Override any config on the CLI:
```bash
python src/train.py --config src/config.yaml trainer.max_epochs=30 data.batch_size=16 model.name=efficientnet_b0
```

## 🔍 Inference
```bash
python src/infer.py --weights outputs/best_model.pt --images /path/to/images_or_folder --class_names "healthy,disease_x,disease_y"
```

## 📝 Notes
- Uses ImageNet-pretrained weights and fine-tunes the final classification head.
- Logs accuracy, per-class precision/recall/F1, and saves a confusion matrix.
- Early stopping & best-model checkpointing by validation accuracy.

## 📄 License
MIT
