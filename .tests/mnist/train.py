#!/usr/bin/env python3

import sys
import os
import platform
import shutil
import subprocess
import signal
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
    console = Console()
    with console.status("[bold green]Creating virtual environment...", spinner="dots"):
        venv.create(VENV_PATH, with_pip=True)
        subprocess.check_call([str(PYTHON_BIN), "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([str(PYTHON_BIN), "-m", "pip", "install", "rich", "torch", "torchvision"])

def restart_with_venv():
    try:
        result = subprocess.run([str(PYTHON_BIN)] + sys.argv, text=True, check=True, env=dict(**os.environ))
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"Subprocess Error with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        sys.exit(0)
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
    except KeyboardInterrupt:
        sys.exit(0)

ensure_venv_and_rich()

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box

console = Console()

def graceful_exit(signum, frame):
    console.print("\n[bold red]You pressed Ctrl+C. Exiting gracefully.[/bold red]")
    sys.exit(0)

signal.signal(signal.SIGINT, graceful_exit)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple neural network on MNIST with CLI hyperparameters.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and testing.")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for optimizer.")
    parser.add_argument("--hidden_size", type=int, default=128, help="Number of neurons in the hidden layer.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on: 'cuda' or 'cpu'. Default is 'cuda' if available.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability (0.0 = no dropout).")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam", "rmsprop"], default="adam", help="Optimizer to use.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 weight decay (regularization).")
    parser.add_argument("--activation", type=str, choices=["relu", "tanh", "leaky_relu", "sigmoid"], default="relu", help="Activation function to use.")
    parser.add_argument("--init", type=str, choices=["xavier", "kaiming", "normal", "none"], default="none", help="Weight initialization method.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    return parser.parse_args()

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout, activation, init_mode):
        super(SimpleMLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

        self.activation = self.get_activation(activation)
        self.fc2 = nn.Linear(hidden_size, num_classes)

        if init_mode != "none":
            self.init_weights(init_mode)

    def get_activation(self, name):
        if name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "leaky_relu":
            return nn.LeakyReLU()
        elif name == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation function: {name}")

    def init_weights(self, mode):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if mode == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif mode == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                elif mode == "normal":
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def show_hyperparams(args):
    table = Table(title="Hyperparameters", box=box.ROUNDED, title_style="bold magenta")
    table.add_column("Parameter", style="bold cyan")
    table.add_column("Value", style="bold green")

    table.add_row("Device", args.device)
    table.add_row("Epochs", str(args.epochs))
    table.add_row("Batch size", str(args.batch_size))
    table.add_row("Learning rate", str(args.learning_rate))
    table.add_row("Hidden size", str(args.hidden_size))
    table.add_row("Dropout", str(args.dropout))
    table.add_row("Optimizer", args.optimizer)
    table.add_row("Momentum", str(args.momentum))
    table.add_row("Weight Decay", str(args.weight_decay))
    table.add_row("Activation", args.activation)
    table.add_row("Init Method", args.init)
    table.add_row("Seed", str(args.seed) if args.seed is not None else "None")

    from rich.console import Console
    console = Console()
    console.print(table)

def train(model, device, train_loader, criterion, optimizer, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    console.rule(f"[bold blue]Epoch {epoch}/{total_epochs} - Training")

    with Progress(
        TextColumn("[bold blue]Batch {task.completed}/{task.total}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TextColumn("Loss: [green]{task.fields[loss]:.4f}"),
        TextColumn("Acc: [cyan]{task.fields[acc]:.2f}%"),
        transient=True,
    ) as progress:
        task_id = progress.add_task("Training", total=len(train_loader), loss=0.0, acc=0.0)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            progress.update(task_id, advance=1,
                            loss=running_loss / (batch_idx + 1),
                            acc=100.0 * correct / total)

def test(model, device, test_loader, criterion, epoch, total_epochs):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    console.rule(f"[bold magenta]Epoch {epoch}/{total_epochs} - Validation")

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / total

    console.print(Panel.fit(
        f"[bold green]Validation Loss:[/] {test_loss:.4f}\n[bold green]Accuracy:[/] {accuracy:.2f}%",
        title=f"Epoch {epoch}/{total_epochs} Summary",
        box=box.DOUBLE
    ))
    return test_loss, accuracy

def main():
    try:
        args = parse_args()
        show_hyperparams(args)
        console.print(f"[bold green]Using device:[/] {args.device}")

        device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        if args.seed is not None:
            torch.manual_seed(args.seed)

        model = SimpleMLP(
            input_size=28 * 28,
            hidden_size=args.hidden_size,
            num_classes=10,
            dropout=args.dropout,
            activation=args.activation,
            init_mode=args.init
        ).to(args.device)

        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.optimizer == "rmsprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer}")

        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, args.epochs + 1):
            train(model, device, train_loader, criterion, optimizer, epoch, args.epochs)
            val_loss, val_acc = test(model, device, test_loader, criterion, epoch, args.epochs)

        print(f"VAL_LOSS: {val_loss}")
        print(f"VAL_ACC: {val_acc}")

    except KeyboardInterrupt:
        graceful_exit(None, None)
    except Exception as e:
        console.print(f"[bold red]An error occurred: {e}[/bold red]", style="bold red")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
