#!/usr/bin/env python3

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
from colorama import Fore, Style

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, n_heads, n_classes, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_heads, dim_feedforward=hidden_size * 4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

def collate_fn(batch, tokenizer, max_len):
    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]
    encodings = tokenizer(texts, padding='max_length', truncation=True, max_length=max_len, return_tensors='pt')
    return encodings['input_ids'], torch.tensor(labels)

def get_datasets():
    return {
        "ag_news": {
            "hf_id": "ag_news",
            "size": "small (120k train)",
            "text_key": "text",
            "label_key": "label",
            "num_labels": 4
        },
        "yelp_review_full": {
            "hf_id": "yelp_review_full",
            "size": "medium (650k train)",
            "text_key": "text",
            "label_key": "label",
            "num_labels": 5
        },
        "dbpedia_14": {
            "hf_id": "dbpedia_14",
            "size": "large (560k train)",
            "text_key": "content",
            "label_key": "label",
            "num_labels": 14
        },
        "imdb": {
            "hf_id": "imdb",
            "size": "medium (25k train)",
            "text_key": "text",
            "label_key": "label",
            "num_labels": 2
        },
        "civil_comments": {
            "hf_id": "civil_comments",
            "size": "very large (1.8M train)",
            "text_key": "text",
            "label_key": "toxicity",
            "num_labels": 2
        },
        "bbc_news": {
            "hf_id": "bbc_news",
            "size": "small (2.2k train)",
            "text_key": "text",
            "label_key": "category",
            "num_labels": 5
        },
        "climate_fever": {
            "hf_id": "climate_fever",
            "size": "small (13k train)",
            "text_key": "claim",
            "label_key": "label",
            "num_labels": 2
        },
        "snli": {
            "hf_id": "snli",
            "size": "small (570k train)",
            "text_key": "sentence1",
            "label_key": "label",
            "num_labels": 3
        }
    }

def load_data(dataset_name, tokenizer, batch_size, max_len):
    config = get_datasets()[dataset_name]
    dataset = load_dataset(config["hf_id"])
    text_key = config["text_key"]
    label_key = config["label_key"]

    def format(example):
        return {"text": example[text_key], "label": int(example[label_key] >= 0.5) if config["num_labels"] == 2 and isinstance(example[label_key], float) else example[label_key]}

    train_data = dataset["train"].map(format).shuffle(seed=42)
    test_data = dataset["test"].map(format).shuffle(seed=42)

    if len(train_data) > 50000:
        train_data = train_data.select(range(50000))
    if len(test_data) > 5000:
        test_data = test_data.select(range(5000))

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer, max_len))
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, tokenizer, max_len))

    return train_loader, test_loader, config["num_labels"]

def train(model, loader, optimizer, criterion, device):
    model.train()
    progress_bar = tqdm(loader, desc=f'{Fore.CYAN}Training{Style.RESET_ALL}', dynamic_ncols=True)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        progress_bar.set_postfix(loss=loss.item())

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(loader, desc=f'{Fore.YELLOW}Evaluating{Style.RESET_ALL}', dynamic_ncols=True)
    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            loss = criterion(output, labels)
            total_loss += loss.item()
            preds = output.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            progress_bar.set_postfix(loss=total_loss / (total + 1e-8), accuracy=correct / total)
    return total_loss / len(loader), correct / total

def download_datasets():
    for key, cfg in get_datasets().items():
        print(f"{Fore.MAGENTA}Downloading {key} ({cfg['size']}) ...{Style.RESET_ALL}")
        try:
            load_dataset(cfg["hf_id"])
        except Exception as e:
            print(f"{Fore.RED}Failed to download {key}: {e}{Style.RESET_ALL}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=get_datasets().keys(), default='ag_news',
        help='Dataset to use. Sizes: ag_news (small), imdb (medium), yelp_review_full (medium), dbpedia_14 (large), civil_comments (very large), bbc_news (small), climate_fever (small), snli (small)'
    )

    parser.add_argument('--download', action='store_true', help='Only download datasets and exit')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='adam')
    args = parser.parse_args()

    if args.download:
        download_datasets()
        sys.exit(0)

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size

    train_loader, test_loader, num_labels = load_data(args.dataset, tokenizer, args.batch_size, args.max_len)

    model = TransformerClassifier(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_classes=num_labels,
        dropout=args.dropout
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    else:
        print(f"{Fore.RED}ERROR: Unsupported optimizer '{args.optimizer}'{Style.RESET_ALL}", file=sys.stderr)
        sys.exit(1)

    print(f"{Fore.GREEN}Starting training...{Style.RESET_ALL}")
    for epoch in range(args.epochs):
        print(f"{Fore.CYAN}Epoch {epoch + 1}/{args.epochs}{Style.RESET_ALL}")
        train(model, train_loader, optimizer, criterion, device)

    print(f"{Fore.GREEN}Training complete, evaluating...{Style.RESET_ALL}")
    loss, acc = evaluate(model, test_loader, criterion, device)
    duration = time.time() - start_time

    print(f"{Fore.YELLOW}LOSS: {loss:.4f}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}ACCURACY: {acc:.4f}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}TIME: {duration:.4f} seconds{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}RESULT_JSON: {{\"loss\": {loss:.6f}, \"accuracy\": {acc:.6f}, \"time\": {duration:.4f}}}{Style.RESET_ALL}")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"{Fore.RED}You pressed CTRL-c{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"{Fore.RED}EXCEPTION: {str(e)}{Style.RESET_ALL}", file=sys.stderr)
        sys.exit(2)
