#!/usr/bin/env python3

import sys
import os
import re
import shutil
import platform
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
    from rich.console import Console
    from rich.spinner import Spinner

    console = Console()
    console.print("[yellow]Creating virtual environment...[/yellow]")

    with console.status("[bold green]Setting up virtual environment...", spinner="earth"):
        venv.create(VENV_PATH, with_pip=True)
        subprocess.check_call([str(PYTHON_BIN), "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([str(PYTHON_BIN), "-m", "pip", "install", "rich", "torch", "torchvision"])
    
    console.print("[green]Virtual environment setup complete.[/green]")

def restart_with_venv():
    try:
        result = subprocess.run(
            [str(PYTHON_BIN)] + sys.argv,
            text=True,
            check=True,
            env=dict(**os.environ)
        )
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"Subprocess Error with exit code {e.returncode}")
        sys.exit(e.returncode)
    except Exception as e:
        print(f"Unexpected error while restarting python: {e}")
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

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rich.console import Console
from rich.table import Table
from rich.traceback import install
from rich.progress import Progress, SpinnerColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn

install()
console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple neural network on MNIST with CLI hyperparameters.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and testing.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for optimizer.")
    parser.add_argument("--hidden_size", type=int, default=128, help="Number of neurons in the hidden layer.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to train on: 'cuda' or 'cpu'. Default is 'cuda' if available.")
    return parser.parse_args()

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train(model, device, train_loader, criterion, optimizer, epoch, total_epochs):
    model.train()
    correct = 0
    total = 0

    with Progress(
        SpinnerColumn(),
        "[bold green]Training...",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task(f"Epoch {epoch}/{total_epochs}", total=len(train_loader))
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            progress.advance(task)

    accuracy = 100.0 * correct / total
    console.log(f"[cyan]Epoch {epoch} Training Accuracy: {accuracy:.2f}%[/cyan]")

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        with Progress(
            SpinnerColumn(),
            "[bold blue]Evaluating...",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Validating", total=len(test_loader))
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                progress.advance(task)

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / total
    console.log(f"[bold blue]Validation Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%[/bold blue]")
    return test_loss, accuracy

def main():
    try:
        args = parse_args()
        console.print(f"[bold magenta]Using device:[/bold magenta] {args.device}")

        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = SimpleMLP(28 * 28, args.hidden_size, 10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

        val_loss = None
        val_acc = None

        for epoch in range(1, args.epochs + 1):
            train(model, device, train_loader, criterion, optimizer, epoch, args.epochs)
            val_loss, val_acc = test(model, device, test_loader, criterion)

        table = Table(title="Training Summary", show_header=True, header_style="bold green")
        table.add_column("Epochs")
        table.add_column("Batch Size")
        table.add_column("Learning Rate")
        table.add_column("Hidden Size")
        table.add_column("Validation Loss")
        table.add_column("Validation Accuracy")

        table.add_row(
            str(args.epochs),
            str(args.batch_size),
            f"{args.learning_rate:.4f}",
            str(args.hidden_size),
            f"{val_loss:.4f}" if val_loss is not None else "N/A",
            f"{val_acc:.2f}%" if val_acc is not None else "N/A"
        )
        console.print(table)

        if val_loss is not None:
            console.print(f"[bold red]RESULT: {val_loss:.3f}[/bold red]")

    except Exception as e:
        console.print_exception(show_locals=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
