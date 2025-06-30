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

    subprocess.check_call([str(PYTHON_BIN), "-m", "pip", "install", "--upgrade", "pip"])

    subprocess.check_call([str(PYTHON_BIN), "-m", "pip", "install", "rich", "torch", "torchvision"])

    print("Virtual environment setup complete.")

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

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

import argparse
import tarfile
import urllib.request
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

DATA_DIR = Path("places365_data")
VAL_TAR_URL = "http://data.csail.mit.edu/places/places365/val_256.tar"
VAL_TAR_PATH = DATA_DIR / "val_256.tar"
VAL_EXTRACTED_DIR = DATA_DIR / "val_256"

def download_and_extract_places365():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not VAL_TAR_PATH.exists():
        console.print("⏬ Downloading Places365 val_256.tar ...")
        with urllib.request.urlopen(VAL_TAR_URL) as response, open(VAL_TAR_PATH, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        console.print("✅ Download complete.")

    if not VAL_EXTRACTED_DIR.exists():
        console.print("📦 Extracting val_256.tar ...")
        with tarfile.open(VAL_TAR_PATH) as tar:
            tar.extractall(path=VAL_EXTRACTED_DIR)
        console.print("✅ Extraction complete.")

class Places365Dataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.image_paths = sorted(list(self.root.glob("*.jpg")))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image  # input == target for reconstruction

def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    dataset = Places365Dataset(VAL_EXTRACTED_DIR, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision.utils as vutils
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Autoencoder Training on Places365 val_256")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for output images and loss.txt")
    return parser.parse_args()

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # 112x112
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # 56x56
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 28x28
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),# 14x14
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),   # 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),    # 224x224
            nn.Sigmoid()  # map to [0,1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def denormalize(tensor):
    # Unnormalize from ImageNet mean/std
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return tensor * std + mean

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataloader = get_dataloader(args.batch_size)
    model = SimpleAutoencoder().to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    os.makedirs(args.output_dir, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn()
    ) as progress:
        task = progress.add_task("Training...", total=args.epochs)

        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for images, targets in dataloader:
                images = images.to(device)
                targets = targets.to(device)

                outputs = model(images)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            avg_loss = running_loss / len(dataloader.dataset)
            progress.update(task, advance=1, description=f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f}")

            if epoch == args.epochs - 1:
                model.eval()
                with torch.no_grad():
                    example_batch, _ = next(iter(dataloader))
                    example_batch = example_batch.to(device)
                    outputs = model(example_batch)

                    # Save original + reconstructed images
                    vutils.save_image(
                        denormalize(example_batch)[:8],
                        os.path.join(args.output_dir, "originals.png"),
                        normalize=True
                    )
                    vutils.save_image(
                        denormalize(outputs)[:8],
                        os.path.join(args.output_dir, "reconstructed.png"),
                        normalize=True
                    )

                with open(os.path.join(args.output_dir, "loss.txt"), "w") as f:
                    f.write(f"{avg_loss:.6f}\n")

                print(f"\nFinal Loss: {avg_loss:.6f}")
                print(f"Images saved to {args.output_dir}/")

if __name__ == "__main__":
    args = parse_args()
    download_and_extract_places365()
    train_model(args)

