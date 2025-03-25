import argparse
import torch

def get_args():   
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='Seed for random number generators')
    parser.add_argument('--data-path', default="/datasets/cv_datasets/data", type=str, help='Path to dataset')
    parser.add_argument('--batch-size', default=128, type=int, help='Size of each batch')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='Default device to use')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='Whether to use MNIST (True) or CIFAR10 (False) data')
    return parser.parse_args()

def main():
    args = get_args()
    print("hello")
    print(args.seed)
    print(args.mnist)

if __name__ == "__main__":
    main()