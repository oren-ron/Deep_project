import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from utils import *

def get_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=64, type=int, help='Size of each batch')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--subtask', default='1', type=str,
                    help='Choose subtask 1.2.<1/2/3>. For example for subtask 1.2.2 write 2.')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    return parser.parse_args()



class MNISTAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), 

            nn.Conv2d(32, 64, 3, 1, 1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.projection = nn.Sequential(
            nn.Linear(1152, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.decoder_fc = nn.Linear(1152, 128 * 3 * 3)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, 2, 0),  
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, 2, 0),  
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 2, 2, 0),  
            nn.Sigmoid()
            )

    def forward(self, x):
        encoded = self.encoder(x)
        flat = encoded.view(encoded.size(0), -1)
        projected = F.normalize(self.projection(flat), p=2, dim=1)
        x_recon = self.decoder_fc(flat)
        x_recon = x_recon.view(-1, 128, 3, 3)
        reconstruction = self.decoder_conv(x_recon)
        return reconstruction, projected





class CifarAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        

        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

        
        self.decoder_fc = nn.Linear(2048, 256 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        flat = encoded.view(encoded.size(0), -1)
        projected = F.normalize(self.projection(flat), p=2, dim=1)
        x_recon = self.decoder_fc(flat)
        x_recon = x_recon.view(-1, 256, 4, 4)
        reconstruction = self.decoder_conv(x_recon)
        return reconstruction, projected


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        self_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        mask = mask - self_mask
        
        max_sim, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - max_sim.detach()
        
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-10)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        mean_log_prob_pos = torch.nan_to_num(mean_log_prob_pos, nan=0.0)
        
        loss = -mean_log_prob_pos.mean()

        return loss
    

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()

        fc_dim = 512
        self.fc1 = nn.Linear(input_dim, fc_dim)
        self.bn1 = nn.BatchNorm1d(fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.bn2 = nn.BatchNorm1d(fc_dim)
        self.fc3 = nn.Linear(fc_dim, num_classes)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x



def plot_original_vs_reconstructed_with_error(model, data_loader, device):
    model.eval() 
    with torch.no_grad():

        for inputs, _ in data_loader:
            inputs = inputs.to(device)

            reconstructed, _ = model(inputs)
            break  


    reconstruction_errors = F.mse_loss(reconstructed, inputs, reduction='none')  
    reconstruction_errors = reconstruction_errors.view(reconstruction_errors.size(0), -1).mean(dim=1) 
    mean_error = reconstruction_errors.mean().item() 

    print(f"Mean Reconstruction Error: {mean_error:.6f}")


    inputs = inputs.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    inputs = np.transpose(inputs, (0, 2, 3, 1))
    reconstructed = np.transpose(reconstructed, (0, 2, 3, 1))

    # Plot original and reconstructed images
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for i in range(5):
        axes[0, i].imshow(np.clip(inputs[i], 0, 1))
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
        axes[1, i].imshow(np.clip(reconstructed[i], 0, 1))
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.show()


def extract_latent(loader, model, device):
    model.to(device)
    latent_vectors = []
    labels = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            features = model.encoder(data).flatten(1)
            latent = model.projection(features).cpu().numpy()
            latent_vectors.append(latent)
            labels.extend(target.numpy())
    latent_vectors = np.vstack(latent_vectors)
    labels = np.array(labels)
    return latent_vectors, labels

def train_ae(model, train_loader, num_epochs, contrastive_weight=0.5):
    print("starting encoder training")
    supcon_criterion = SupConLoss(temperature=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    recon_criterion = nn.MSELoss()
    best_val_loss = float('inf')
    for epoch in range(num_epochs):

        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            

            reconstructions, projections = model(inputs)
            recon_loss = recon_criterion(reconstructions, inputs)
            supcon_loss = supcon_criterion(projections, labels)
            total_loss = (1 - contrastive_weight) * recon_loss + contrastive_weight * supcon_loss
            

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()
        

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                reconstructions, projections = model(inputs)
                recon_loss = recon_criterion(reconstructions, inputs)
                supcon_loss = supcon_criterion(projections, labels)
                total_loss = recon_loss + contrastive_weight * supcon_loss
                
                val_loss += total_loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model_paper.pth')
            print("saved model")
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

def main(train_loader, val_loader, test_loader, mnist):

    model = MNISTAutoencoder().to(device) if mnist else CifarAutoencoder().to(device)
    supcon_criterion = SupConLoss(temperature=0.2)
    recon_criterion = nn.MSELoss()

    num_epochs = cfg['ae_epochs'][subtask_id]

    train_ae(model, train_loader, num_epochs, contrastive_weight=0.8)
    model.load_state_dict(torch.load('best_model_paper.pth'))
    train_dataset, val_dataset, test_dataset = load_dataset(dataset_name, '1', path)
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, 64)
    train_ae(model, train_loader, int(num_epochs / 2), contrastive_weight=1.0)
    torch.save(model.state_dict(), 'best_model_paper.pth')
    model.eval()
    # plot_original_vs_reconstructed_with_error(model, test_loader, device)
    # plot_tsne(model.encoder, test_loader, device)
    batch_size = 256

    X_train, y_train = extract_latent(train_loader, model, device)
    X_test, y_test = extract_latent(test_loader, model, device)
    X_val, y_val = extract_latent(val_loader, model, device)


    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)


    train_dataset_latent = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset_latent = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset_latent = TensorDataset(X_test_tensor, y_test_tensor)
    val_loader_latent = DataLoader(val_dataset_latent, batch_size=batch_size, shuffle=False)
    train_loader_latent = DataLoader(train_dataset_latent, batch_size=batch_size, shuffle=True)
    test_loader_latent = DataLoader(test_dataset_latent, batch_size=batch_size, shuffle=False)

    input_dim = 128
    num_classes = 10
    classifier = Classifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.0001)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_acc = 0.0
    num_epochs = cfg['cls_epochs'][subtask_id]
    print("starting classifier training")
    for epoch in range(num_epochs):
        classifier.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
    
        for inputs, labels in train_loader_latent:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()
    

        classifier.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
    
        with torch.no_grad():
            for inputs, labels in val_loader_latent:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = classifier(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
    

        train_acc = 100. * correct_train / total_train
        val_acc = 100. * correct_val / total_val
        avg_train_loss = train_loss / len(train_loader_latent)
        avg_val_loss = val_loss / len(val_loader_latent)
    

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(classifier.state_dict(), 'best_classifier_new.pth')
    
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f} | Acc: {train_acc:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f} | Acc: {val_acc:.2f}%')
        print('-' * 50)

    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    # plt.plot(range(1, num_epochs+1), val_losses, label='Val Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss')
    # plt.legend()
    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Acc')
    # plt.plot(range(1, num_epochs+1), val_accuracies, label='Val Acc')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy (%)')
    # plt.title('Training and Validation Accuracy')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    classifier.load_state_dict(torch.load('best_classifier_new.pth'))
    classifier.eval()
    
    test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for inputs, labels in test_loader_latent:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()
    
    avg_test_loss = test_loss / len(test_loader_latent)
    test_acc = 100. * correct_test / total_test
    
    print(f'Test Results:')
    print(f'Loss: {avg_test_loss:.4f} | Accuracy: {test_acc:.2f}%')

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
    main(train_loader, val_loader, test_loader, args.mnist)
