import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from utils import *

def get_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=128, type=int, help='Size of each batch')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--subtask', default='1', type=str,
                    help='Choose subtask 1.2.<1/2/3>. For example for subtask 1.2.2 write 2.')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    return parser.parse_args()


class Classifier(nn.Module):
    def __init__(self, input_channels, flattened_size):
        super(Classifier, self).__init__()
        self.encoder = nn.Sequential( 
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.35),
            nn.Flatten(), 
            nn.Linear(flattened_size, 128)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
    
def train_classifier(model, train_loader, val_loader, device, num_epochs=15):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    best_acc = 0.0
    patience_counter = 0
    patience_limit = 10

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss, total, correct = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"CLS Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100 * correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_acc)
        
        print(f'CLS Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%'
              f', Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Scheduler step
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                print("Early stopping triggered.")
                break
    
    # plot_losses(train_losses, val_losses, 'Classifier Losses')
    # plot_losses(train_accuracies, val_accuracies, 'Classifier Accuracies', 'accuracy')
    
    return model




def evaluate_classifier(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def plot_losses(train_losses, val_losses, title, objective='loss'):
    plt.figure()
    plt.plot(train_losses, label=f'Train {objective}')
    plt.plot(val_losses, label=f'Validation {objective}')
    plt.xlabel('Epoch')
    plt.ylabel(f'{objective.capitalize()}')
    plt.title(title)
    plt.legend()
    plt.show()


def main():
    flattened = (64*7*7) if args.mnist else (64*8*8)
    model = Classifier(input_channels=cfg['input_channels'], flattened_size=flattened).to(device)
    model = train_classifier(model, train_loader, val_loader, device, num_epochs=cfg['cls_epochs'][subtask_id])
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

    # plot_tsne(model.encoder, test_loader, device)




if __name__ == "__main__":
    args = get_args()
    freeze_seeds(args.seed)
    device = args.device
    path = args.data_path
    latent_dim = 128
    batch_size = args.batch_size
    subtask_id = args.subtask
    learning_rate = 1e-3
    patience = 10
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_dir = 'checkpoints'
    patience_classifier = 5
    best_val_loss_classifier = float('inf')
    patience_counter_classifier = 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    dataset_name = 'MNIST' if args.mnist else 'CIFAR10'
    cfg = config[dataset_name]
    train_dataset, val_dataset, test_dataset = load_dataset(dataset_name, subtask_id, path)
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)

    main()
