"""Fine-tune a ResNet-18 on a labeled ImageFolder dataset.
Expected structure: dataset/train/<class>/*.jpg and dataset/val/<class>/*.jpg
"""
import argparse
import copy
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from pathovision.metrics import classification_metrics
from pathovision.split_audit import validate_group_manifest


def _subset(dataset, limit: int | None, seed: int):
    """Select a repeatable subset for short experiments without changing labels."""
    if limit is None or limit >= len(dataset):
        return dataset
    if limit < 1:
        raise ValueError("sample limits must be positive")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


def _evaluate(model, loader, class_names, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device)).argmax(1).cpu()
            y_true.extend(y.reshape(-1).tolist())
            y_pred.extend(logits.tolist())
    return classification_metrics(y_true, y_pred, class_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path, help="Folder containing train/ and val/")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=3,
                        help="Stop after this many epochs without improving validation macro F1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=("imagefolder", "pathmnist"), default="imagefolder",
                        help="Use a class-folder dataset or real PathMNIST histology patches")
    parser.add_argument("--download", action="store_true",
                        help="Download PathMNIST from its official MedMNIST source when needed")
    parser.add_argument("--image-size", type=int, default=None,
                        help="Model input size (default: 28 for PathMNIST, 224 for ImageFolder)")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Optional repeatable training subset for quick experiments")
    parser.add_argument("--max-val-samples", type=int, default=None,
                        help="Optional repeatable validation subset; full validation is preferred")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Evaluate the selected checkpoint once on a separate test split")
    parser.add_argument("--group-manifest", type=Path, default=None,
                        help="CSV with split,group_id,relative_path columns; audits patient/slide leakage for ImageFolder")
    parser.add_argument("--output", type=Path, default=Path("models/resnet18.pt"))
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.patience < 1:
        parser.error("--patience must be at least 1")
    if args.image_size is not None and args.image_size < 16:
        parser.error("--image-size must be at least 16")
    for option, value in (("--max-train-samples", args.max_train_samples),
                          ("--max-val-samples", args.max_val_samples)):
        if value is not None and value < 1:
            parser.error(f"{option} must be positive")
    if args.download and args.dataset != "pathmnist":
        parser.error("--download is only available with --dataset pathmnist")
    if args.group_manifest is not None and args.dataset != "imagefolder":
        parser.error("--group-manifest is only available with --dataset imagefolder")
    image_size = args.image_size or (28 if args.dataset == "pathmnist" else 224)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    eval_tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    try:
        if args.dataset == "imagefolder":
            train_ds = datasets.ImageFolder(args.data / "train", transform=train_tf)
            val_ds = datasets.ImageFolder(args.data / "val", transform=eval_tf)
            class_names = train_ds.classes
            test_ds = datasets.ImageFolder(args.data / "test", transform=eval_tf) if args.evaluate_test else None
        else:
            try:
                from medmnist import PathMNIST
            except ImportError as exc:
                raise RuntimeError("PathMNIST support requires the optional dependency: pip install medmnist") from exc
            train_ds = PathMNIST(root=args.data, split="train", transform=train_tf, download=args.download)
            val_ds = PathMNIST(root=args.data, split="val", transform=eval_tf, download=args.download)
            class_names = [train_ds.info["label"][str(i)] for i in range(len(train_ds.info["label"]))]
            test_ds = (PathMNIST(root=args.data, split="test", transform=eval_tf, download=args.download)
                       if args.evaluate_test else None)
    except (FileNotFoundError, RuntimeError, KeyError) as exc:
        parser.error(f"could not load train/val ImageFolder dataset: {exc}")
    if args.dataset == "imagefolder" and train_ds.classes != val_ds.classes:
        raise ValueError("train and val class folders must match")
    if test_ds is not None and args.dataset == "imagefolder" and test_ds.classes != class_names:
        raise ValueError("train and test class folders must match")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("train and val folders must each contain labeled images")
    if test_ds is not None and len(test_ds) == 0:
        raise ValueError("test split must contain labeled images")
    group_counts = None
    if args.dataset == "imagefolder":
        if args.group_manifest is None:
            print("WARNING: no patient/slide group manifest supplied; split independence cannot be verified.")
        else:
            expected_paths = {
                "train": [Path(path) for path, _ in train_ds.samples],
                "val": [Path(path) for path, _ in val_ds.samples],
            }
            if test_ds is not None:
                expected_paths["test"] = [Path(path) for path, _ in test_ds.samples]
            try:
                group_counts = validate_group_manifest(args.data, args.group_manifest, expected_paths)
            except ValueError as exc:
                parser.error(str(exc))
            print("Patient/slide split audit passed; group counts: " + ", ".join(
                f"{split}={count}" for split, count in group_counts.items()))
    full_train_count, full_val_count = len(train_ds), len(val_ds)
    train_ds = _subset(train_ds, args.max_train_samples, args.seed)
    val_ds = _subset(val_ds, args.max_val_samples, args.seed + 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.to(device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_macro_f1 = -1.0
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        model.train(); loss_sum = correct = count = 0
        for x, y in train_loader:
            x, y = x.to(device), y.reshape(-1).to(device); optimizer.zero_grad()
            logits = model(x); loss = criterion(logits, y); loss.backward(); optimizer.step()
            loss_sum += loss.item() * y.size(0); correct += (logits.argmax(1) == y).sum().item(); count += y.size(0)
        model.eval(); y_true, y_pred = [], []
        with torch.inference_mode():
            for x, y in val_loader:
                logits = model(x.to(device)).argmax(1).cpu()
                y_true.extend(y.reshape(-1).tolist()); y_pred.extend(logits.tolist())
        metrics = classification_metrics(y_true, y_pred, class_names)
        print(f"epoch {epoch+1}/{args.epochs} train_loss={loss_sum/count:.4f} train_acc={correct/count:.3f} val_acc={metrics['accuracy']:.3f} val_macro_f1={metrics['macro_f1']:.3f}")
        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch+1} epochs; best epoch was {best_epoch}.")
                break
    model.load_state_dict(best_state)
    test_metrics = None
    if test_ds is not None:
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_metrics = _evaluate(model, test_loader, class_names, device)
        print(f"Held-out test accuracy={test_metrics['accuracy']:.3f} test_macro_f1={test_metrics['macro_f1']:.3f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "classes": class_names,
                "architecture": "resnet18", "image_size": image_size,
                "validation_metrics": best_metrics,
                "test_metrics": test_metrics,
                "training_config": {"seed": args.seed, "epochs_requested": args.epochs,
                                    "best_epoch": best_epoch, "batch_size": args.batch_size,
                                    "dataset": args.dataset,
                                    "train_samples": len(train_ds), "train_samples_available": full_train_count,
                                    "validation_samples": len(val_ds), "validation_samples_available": full_val_count,
                                    "test_samples": len(test_ds) if test_ds is not None else None,
                                    "test_samples_available": len(test_ds) if test_ds is not None else None,
                                    "selection_metric": "validation_macro_f1",
                                    "group_manifest_used": group_counts is not None,
                                    "group_counts": group_counts}}, args.output)
    metrics = best_metrics
    print(f"Selected best checkpoint from epoch {best_epoch} (validation macro F1={best_macro_f1:.3f}).")
    print("Validation per-class metrics (held-out split):")
    for row in metrics["per_class"]:
        print(f"  {row['class']}: precision={row['precision']:.3f} recall={row['recall']:.3f} f1={row['f1']:.3f} support={row['support']}")
    print("Confusion matrix (rows=actual, columns=predicted):")
    for row in metrics["confusion_matrix"]:
        print("  " + " ".join(str(value) for value in row))
    print(f"Saved checkpoint: {args.output}")

if __name__ == "__main__": main()
