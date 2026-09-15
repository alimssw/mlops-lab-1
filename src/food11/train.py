import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DATASETS = {
    "processed": DATA_DIR / "food11_processed",
    "mini": DATA_DIR / "food11_processed_mini",
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="mini")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def make_loaders(root: Path, batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    normalize = transforms.Normalize(MEAN, STD)
    train_tf = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])

    train_ds = datasets.ImageFolder(root / "training", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "validation", transform=eval_tf)
    test_ds = datasets.ImageFolder(root / "evaluation", transform=eval_tf)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size),
    )


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        total_loss += criterion(outputs, labels).item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = make_loaders(DATASETS[args.dataset], args.batch_size)
    model = build_model(len(train_loader.dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("food11")

    with mlflow.start_run():
        mlflow.log_params(
            {
                "dataset": args.dataset,
                "epochs": args.epochs,
                "lr": args.lr,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "model": "resnet18",
                "optimizer": "adam",
            }
        )

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
            print(
                f"epoch {epoch}/{args.epochs} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}"
            )

        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_accuracy)
        print(f"test_loss={test_loss:.4f} test_accuracy={test_accuracy:.4f}")

        example = next(iter(test_loader))[0][:2].numpy()
        mlflow.pytorch.log_model(model.cpu(), name="model", input_example=example)


if __name__ == "__main__":
    main()
