import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import random

def plot_tsne(model, dataloader, device):
    '''
    model - torch.nn.Module subclass. This is your encoder model
    dataloader - test dataloader to iterate over data for which you wish to compute projections
    device - cuda or cpu (as a string)
    '''
    model.eval()
    
    images_list = []
    labels_list = []
    latent_list = []
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # # Flatten the images before passing to the encoder
            # flattened_images = images.view(images.size(0), -1)
            # # Approximate the latent space from data
            # latent_vector = model(flattened_images)            
            latent_vector = model(images)
            images_list.append(images.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            latent_list.append(latent_vector.cpu().numpy())
    
    images = np.concatenate(images_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    latent_vectors = np.concatenate(latent_list, axis=0)
        
    # Flatten latent vectors if they have more than 2 dimensions
    if len(latent_vectors.shape) > 2:
        latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], -1)
    
    # Plot TSNE for latent space
    tsne_latent = TSNE(n_components=2, random_state=0)
    latent_tsne = tsne_latent.fit_transform(latent_vectors)
    
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab10', s=10)  # Smaller points
    plt.colorbar(scatter)
    plt.title('t-SNE of Latent Space')
    plt.savefig('latent_tsne.png')
    plt.close()
    
    # Plot TSNE for image space
    # tsne_image = TSNE(n_components=2, random_state=42)
    # images_flattened = images.reshape(images.shape[0], -1)
    # image_tsne = tsne_image.fit_transform(images_flattened)
    
    # plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(image_tsne[:, 0], image_tsne[:, 1], c=labels, cmap='tab10', s=10)  
    # plt.colorbar(scatter)
    # plt.title('t-SNE of Image Space')
    # plt.savefig('image_tsne.png')
    # plt.close()

def plot_tsne_simclr(model, dataloader, device, use_projection=True):
        model.eval()
        
        images_list = []
        labels_list = []
        latent_list = []
        
        with torch.no_grad():
            for data in dataloader:
                # Handle SimCLR data format (views, labels)
                if isinstance(data[0], tuple):
                    views, labels = data
                    images = views[0]  # Use only first view
                else:
                    images, labels = data
                    
                images, labels = images.to(device), labels.to(device)
                
                # For SimCLR model
                z = model.encode(images)
                if use_projection:
                    # Use projected features (what's used in contrastive loss)
                    latent_vector = model.projection(z)
                else:
                    # Use raw latent representation
                    latent_vector = z
                    
                images_list.append(images.cpu().numpy())
                labels_list.append(labels.cpu().numpy())
                latent_list.append(latent_vector.cpu().numpy())
        
        # Rest of the function remains the same
        images = np.concatenate(images_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        latent_vectors = np.concatenate(latent_list, axis=0)
        
        if len(latent_vectors.shape) > 2:
            latent_vectors = latent_vectors.reshape(latent_vectors.shape[0], -1)
        
        tsne_latent = TSNE(n_components=2, random_state=0)
        latent_tsne = tsne_latent.fit_transform(latent_vectors)
        
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=labels, cmap='tab10', s=10)
        plt.colorbar(scatter)
        plt.title('t-SNE of SimCLR Latent Space')
        plt.savefig('simclr_latent_tsne.png')
        plt.show()


mean_mnist, std_mnist = (0.1307,), (0.3081,)
mean_cifar, std_cifar = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
config = {
    'MNIST': {
        'input_channels': 1,
        'input_size': 28,
        'num_classes': 10,
        'train_transform': {
            '1': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean_mnist, std_mnist)
            ]),
            '2': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean_mnist, std_mnist)
            ]),
            '3': transforms.Compose([
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(0, shear=10),
                transforms.ToTensor(),
                transforms.Normalize(mean_mnist, std_mnist) 
            ])
        },
        'val_transform': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean_mnist, std_mnist)
            ]),
        'ae_epochs': {
            '1': 10,
            '2': 10,
            '3': 15
        },
        'cls_epochs': {
            '1': 20,
            '2': 20,
            '3': 15
        }
    },
    'CIFAR10': {
        'input_channels': 3,
        'input_size': 32,
        'num_classes': 10,
        'train_transform': {
            '1': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean_cifar, std_cifar)
            ]),
            '2': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean_cifar, std_cifar)
            ]),
            '3': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(0, shear=10),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
        },
        'val_transform': 
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean_cifar, std_cifar)
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ]),
        'ae_epochs': {
            '1': 30,
            '2': 30,
            '3': 12
        },
        'cls_epochs': {
            '1': 25,
            '2': 20,
            '3': 15
        }
    }
}



class MemoryDataset(Dataset):
    def __init__(self, dataset):
        self.data = [dataset[i] for i in range(len(dataset))]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def load_dataset(dataset_name, subtask_id, path):
    cfg = config[dataset_name]
    if dataset_name == 'MNIST':
        try:
            train_dataset = datasets.MNIST(root=path, train=True, download=False, transform=cfg['train_transform'][subtask_id])
            print("Loading datasets to memory, please wait... (up to 3 minutes for cifar)")
            train_dataset = MemoryDataset(train_dataset)
            test_dataset = datasets.MNIST(root=path, train=False, download=False, transform=cfg['val_transform'])
            test_dataset = MemoryDataset(test_dataset)
        except Exception as e:
            print("Didn't find MNIST dataset so downloading...")
            train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=cfg['train_transform'][subtask_id])
            print("Loading datasets to memory, please wait... (up to 3 minutes for cifar)")
            train_dataset = MemoryDataset(train_dataset)
            test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=cfg['val_transform'])
            test_dataset = MemoryDataset(test_dataset)
            path = './data'
    else:
        try:
            train_dataset = datasets.CIFAR10(root=path, train=True, download=False, transform=cfg['train_transform'][subtask_id])
            print("Loading datasets to memory, please wait... (up to 3 minutes for cifar)")

            train_dataset = MemoryDataset(train_dataset)
            test_dataset = datasets.CIFAR10(root=path, train=False, download=False, transform=cfg['val_transform'])
            test_dataset = MemoryDataset(test_dataset)
        except Exception as e:
            print("Didn't find CIFAR10 dataset so downloading...")
            train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=cfg['train_transform'][subtask_id])
            print("Loading datasets to memory, please wait... (up to 3 minutes for cifar)")
            train_dataset = MemoryDataset(train_dataset)
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=cfg['val_transform'])
            test_dataset = MemoryDataset(test_dataset)
            path = './data'
    

    train_size = len(train_dataset) - 10000
    train_dataset, val_dataset = random_split(train_dataset, [train_size, 10000])
    val_dataset.dataset.transform = cfg['val_transform']
    print("Finished loading to memory")
    return train_dataset, val_dataset, test_dataset

def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )

def plot_losses(train_losses, val_losses, title, objective='loss'):
    plt.figure()
    plt.plot(train_losses, label=f'Train {objective}')
    plt.plot(val_losses, label=f'Validation {objective}')
    plt.xlabel('Epoch')
    plt.ylabel(f'{objective.capitalize()}')
    plt.title(title)
    plt.legend()
    plt.show()

def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)