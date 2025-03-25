import torch
from torchvision import datasets, transforms
import numpy as np
from matplotlib import pyplot as plt
from utils import plot_tsne
import numpy as np
import random
import argparse
import subprocess
import sys


def freeze_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=128, type=int, help='Size of each batch')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    parser.add_argument('--subtask', default=1, type=int,
                        help='Choose subtask 1.2.<1/2/3>. For example for subtask 1.2.2 write 2.')
    return parser.parse_args()
    

if __name__ == "__main__":

    args = get_args()
    freeze_seeds(args.seed)

    if args.subtask not in [1, 2, 3]:
        print("Error: Invalid value for --subtask. Allowed values are 1, 2, or 3.")
        sys.exit(1)

    path = args.data_path
    if args.mnist:
        try:
            train_dataset = datasets.MNIST(root=path, train=True, download=False)
            test_dataset = datasets.MNIST(root=path, train=False, download=False)
        except Exception as e:
            print("Didn't find MNIST dataset so downloading...")
            train_dataset = datasets.MNIST(root='./data', train=True, download=True)
            test_dataset = datasets.MNIST(root='./data', train=False, download=True)
            path = './data'
    else:
        try:
            train_dataset = datasets.CIFAR10(root=path, train=True, download=False)
            test_dataset = datasets.CIFAR10(root=path, train=False, download=False)
        except Exception as e:
            print("Didn't find CIFAR10 dataset so downloading...")
            train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
            test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)
            path = './data'
    
    command = [
        'python', f'subtask{args.subtask}.py',
        '--seed', str(args.seed),
        '--data-path', path,
        '--batch-size', str(args.batch_size),
        '--device', args.device,
    ]

    if args.mnist:
        command.append('--mnist')

    # Run the command
    result = subprocess.run(command)

