#!/usr/bin/env python3

import sys
import re
import platform
import shutil
import os
import subprocess
import traceback
from pathlib import Path

try:
    import venv
except ModuleNotFoundError:
    print("venv not found. Is python3-venv installed?")
    sys.exit(1)

VENV_PATH = Path(".torch_venv")

if platform.system() == "Windows":
    PYTHON_BIN = VENV_PATH / "Scripts" / "python.exe"
else:
    PYTHON_BIN = VENV_PATH / "bin" / "python"


def create_and_setup_venv():
    print(f"Creating virtual environment at {VENV_PATH}")
    venv.create(VENV_PATH, with_pip=True)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
        task = progress.add_task("Upgrading pip...", total=None)
        subprocess.check_call([str(PYTHON_BIN), "-m", "pip", "install", "--upgrade", "pip"])
        progress.update(task, completed=1)

        progress.update(task, description="Installing torch and rich...")
        subprocess.check_call([str(PYTHON_BIN), "-m", "pip", "install", "rich", "torch", "torchvision"])
        progress.update(task, completed=1)
    console.print("[green]Virtual environment setup complete.[/green]")


def restart_with_venv():
    console.print("[yellow]Restarting script inside virtual environment...[/yellow]")
    try:
        result = subprocess.run(
            [str(PYTHON_BIN)] + sys.argv,
            text=True,
            check=True,
            env=dict(**os.environ)
        )
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Subprocess Error with exit code {e.returncode}[/red]")
        sys.exit(e.returncode)
    except Exception as e:
        console.print(f"[red]Unexpected error while restarting python: {e}[/red]")
        sys.exit(1)


def ensure_venv_and_rich():
    try:
        import rich
        import torch
        import torchvision
    except ModuleNotFoundError:
        if not VENV_PATH.exists():
            create_and_setup_venv()
        else:
            try:
                subprocess.check_call([str(PYTHON_BIN), "-m", "pip", "install", "-q", "rich", "torch", "torchvision"])
            except subprocess.CalledProcessError:
                shutil.rmtree(VENV_PATH)
                create_and_setup_venv()
        restart_with_venv()

ensure_venv_and_rich()

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# -----------------------------------------------------------------------------------
# Benchmark Code: CIFAR-10 mit PyTorch, Early Stopping, NaN-Check, rich-Ausgabe, CLI Params
# -----------------------------------------------------------------------------------

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # (B,32,32,32)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3),            # (B,32,30,30)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # (B,32,15,15)
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1), # (B,64,15,15)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),             # (B,64,13,13)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                  # (B,64,6,6)
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR-10 Benchmark with PyTorch, EarlyStopping, NaN check")
    parser.add_argument('--learning_rate', type=float, required=True, help='Lernrate z.B. 0.001')
    parser.add_argument('--batch_size', type=int, required=True, help='Batchgröße z.B. 64')
    parser.add_argument('--epochs', type=int, required=True, help='Maximale Anzahl an Epochen')
    parser.add_argument('--seed', type=int, default=None, help='Optionaler Seed für Reproduzierbarkeit')
    parser.add_argument('--early_stopping_patience', type=int, default=5, help='Epochen ohne Verbesserung bis Stop')
    return parser.parse_args()


def train_epoch(model, device, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError("Loss is NaN or Inf. Aborting training.")

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, device, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total


def main_benchmark():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[green]Using device:[/green] {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_acc = 0.0
    epochs_no_improve = 0

    with Progress(
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[green]Training model...", total=args.epochs)

        for epoch in range(1, args.epochs + 1):
            try:
                train_loss = train_epoch(model, device, trainloader, optimizer, criterion)
            except ValueError as e:
                console.print(f"[red]{e}[/red]")
                console.print("[red]Aborting training due to NaN/Inf loss.[/red]")
                sys.exit(1)

            val_acc = evaluate(model, device, testloader)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            progress.update(task, advance=1, description=f"[green]Epoch {epoch}/{args.epochs} Train Loss: {train_loss:.4f} Val Acc: {val_acc:.4f}")

            if epochs_no_improve >= args.early_stopping_patience:
                console.print(f"[yellow]Early stopping triggered after {epoch} epochs.[/yellow]")
                break

    console.print(f"[bold green]RESULT: {best_val_acc:.6f}[/bold green]")


if __name__ == '__main__':
    try:
        main_benchmark()
    except KeyboardInterrupt:
        console.print("[red]Cancelled by CTRL-C.[/red]")
        sys.exit(130)
    except Exception:
        console.print("[red]Unexpected error:[/red]")
        console.print_exception()
        sys.exit(1)
