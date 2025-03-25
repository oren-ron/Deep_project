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

class Autoencoder(nn.Module):
    def __init__(self, input_channels, input_size):
        super().__init__()
        self.input_dim = input_channels * input_size ** 2
        self.input_channels = input_channels
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, self.input_dim)
        )

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(-1, self.input_channels, self.input_size, self.input_size)   


class Classifier(nn.Module):
    def __init__(self, encoder, input_channels, input_size, num_classes):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.input_channels = input_channels
        self.input_size = input_size
        self.input_dim = input_channels * input_size ** 2
        
        latent_dim = 128
        
        fc_dim = 2048
        self.fc1 = nn.Linear(latent_dim, fc_dim)
        self.bn1 = nn.BatchNorm1d(fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.bn2 = nn.BatchNorm1d(fc_dim)
        self.fc3 = nn.Linear(fc_dim, num_classes)
        self.dropout = nn.Dropout(0.65)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        
        x = self.encoder(x)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    

def train_autoencoder(model, train_loader, val_loader, device, num_epochs, dataset_name):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    patience_counter = 0
    checkpoint_dir = f'checkpoints/{dataset_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, _ in tqdm(train_loader, desc=f"AE Epoch {epoch+1}"):
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_losses.append(train_loss/len(train_loader))
        val_loss = evaluate(model, val_loader, device, criterion)
        val_losses.append(val_loss)
        print(f'AE Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')

        # Early stopping and checkpoint logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{checkpoint_dir}/ae_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
        
        
        # if isinstance(model, Autoencoder):
        #     with torch.no_grad():
        #         dataiter = iter(val_loader)
        #         images, _ = next(dataiter)
        #         images = images.to(device)
        #         outputs = model(images)
        #         images = images.cpu()
        #         outputs = outputs.cpu()
                
        #         # Get image dimensions automatically
        #         batch_size, channels, height, width = images.shape
        #         random_indices = torch.randperm(batch_size)[:5]

        #         # Denormalize images using the config values
        #         def denormalize(x, dataset_name):
        #             mean = torch.tensor(config[dataset_name]['mean']).view(channels, 1, 1)
        #             std = torch.tensor(config[dataset_name]['std']).view(channels, 1, 1)
        #             return x * std + mean
                
        #         images_display = denormalize(images, dataset_name)
        #         outputs_display = denormalize(outputs, dataset_name)
                
        #         fig, axes = plt.subplots(2, 5, figsize=(12, 4))
        #         for i in range(5):
        #             idx = random_indices[i]
        #             if channels == 1:
        #                 axes[0, i].imshow(images_display[idx].squeeze(), cmap='gray')
        #                 axes[0, i].axis('off')
        #                 axes[1, i].imshow(outputs_display[idx].squeeze(), cmap='gray')
        #                 axes[1, i].axis('off')
        #             else:
        #                 axes[0, i].imshow(images_display[idx].permute(1, 2, 0).clamp(0, 1))
        #                 axes[0, i].axis('off')
        #                 axes[1, i].imshow(outputs_display[idx].permute(1, 2, 0).clamp(0, 1))
        #                 axes[1, i].axis('off')
                
        #         axes[0, 0].set_title('Original', size=14)
        #         axes[1, 0].set_title('Reconstructed', size=14)
        #         plt.tight_layout()
        #         plt.show()
        
    # plot_losses(train_losses, val_losses, 'Autoencoder Losses')

    return model

def train_classifier(model, train_loader, val_loader, device, num_epochs, dataset_name):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    best_val_acc = 0
    patience_counter = 0
    checkpoint_dir = f'checkpoints/{dataset_name}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        for images, labels in tqdm(train_loader, desc=f"CLS Epoch {epoch+1}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.*correct/total
        val_acc = evaluate_classifier(model, val_loader, device)
        train_losses.append(train_loss/len(train_loader))
        train_accuracies.append(train_acc)
        val_losses.append(evaluate_classifier_loss(model, val_loader, device, criterion))
        val_accuracies.append(val_acc)
        print(f'CLS Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%'
              f', Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%')

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), f'{checkpoint_dir}/cls_best.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # plot_losses(train_losses, val_losses, 'Classifier Losses')
    # plot_losses(train_accuracies, val_accuracies, 'Classifier Accuracies', 'accuracy')
    return model

def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, images).item()
    return total_loss / len(loader)

def evaluate_classifier(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def evaluate_classifier_loss(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(loader)

def plot_reconstructions_with_error(model, dataloader, device, num_images=5):
    model.eval()
    images, reconstructions = [], []
    reconstruction_errors = []
    
    with torch.no_grad():
        for data in dataloader:
            img, _ = data
            img = img.to(device)
            recon = model(img)
            images.append(img.cpu())
            reconstructions.append(recon.cpu())
            # Compute reconstruction error (MSE) for the batch
            error = F.mse_loss(recon, img, reduction='none').view(img.size(0), -1).mean(dim=1)
            reconstruction_errors.extend(error.cpu().numpy())
            if len(images) >= num_images:
                break
    
    images = torch.cat(images)[:num_images]
    reconstructions = torch.cat(reconstructions)[:num_images]
    
    # Compute mean reconstruction error
    mean_error = sum(reconstruction_errors) / len(reconstruction_errors)
    print(f"Mean Reconstruction Error: {mean_error:.6f}")
    
    # # Plot original and reconstructed images
    # fig, axes = plt.subplots(2, num_images, figsize=(15, 5))
    # for i in range(num_images):
    #     axes[0, i].imshow(images[i].permute(1, 2, 0).squeeze(), cmap='gray')
    #     axes[0, i].axis('off')
    #     axes[1, i].imshow(reconstructions[i].permute(1, 2, 0).squeeze(), cmap='gray')
    #     axes[1, i].axis('off')
    
    # axes[0, 0].set_title('Original Images', size=14)
    # axes[1, 0].set_title('Reconstructed Images', size=14)
    # plt.tight_layout()
    # plt.show()



def main():
    ae = Autoencoder(input_channels=cfg['input_channels'], input_size=cfg['input_size']).to(device)
    classifier = Classifier(ae.encoder, input_channels=cfg['input_channels'], input_size=cfg['input_size'], 
                            num_classes=cfg['num_classes']).to(device)
    print(f"Training Autoencoder on {dataset_name}")
    ae = train_autoencoder(ae, train_loader, val_loader, device, cfg['ae_epochs'][subtask_id], dataset_name)
    ae_test_loss = evaluate(ae, test_loader, device, criterion = nn.MSELoss())
    print(f'Test Loss: {ae_test_loss:.4f}')
    for param in classifier.encoder.parameters():
        param.requires_grad = False
    print(f"\nTraining Classifier on {dataset_name}")
    classifier = train_classifier(classifier, train_loader, val_loader, device, cfg['cls_epochs'][subtask_id], dataset_name)
    # Final test
    test_acc = evaluate_classifier(classifier, test_loader, device)
    print(f'\nFinal Test Accuracy on {dataset_name}: {test_acc:.2f}%')
    # Visualization
    # plot_tsne(ae.encoder, test_loader, device)
    plot_reconstructions_with_error(ae, test_loader, device)

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
