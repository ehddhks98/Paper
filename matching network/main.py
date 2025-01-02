from PIL import Image
import argparse
import torch
from torch import nn, optim
from dataset import get_dataset, get_dataloader
from model import create_model
from train import train_matching_net
from test import test_matching_net

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 로드
    train_set, test_set = get_dataset(args.image_size)

    # 데이터로더 설정
    train_dataloader = get_dataloader(train_set, args.n_class, args.n_support, args.n_query_train, args.n_batch_train)
    test_dataloader = get_dataloader(test_set, args.n_class, args.n_support, args.n_query_test, args.n_batch_test)

    model = create_model()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_matching_net(model, train_dataloader, optimizer, args.n_epoch)

    test_matching_net(model, test_dataloader)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Prototypical Networks on Omniglot dataset")
    parser.add_argument("--image_size", type=int, default=28, help="Image size (default: 28)")
    parser.add_argument("--n_class", type=int, default=5, help="Number of classes in a task (default: 60)")
    parser.add_argument("--n_support", type=int, default=5, help="Number of images per class in the support set (default: 1)")
    parser.add_argument("--n_query_train", type=int, default=5, help="Number of images per class in the query set (default: 5)")
    parser.add_argument("--n_query_test", type=int, default=5, help="Number of images per class in the query set (default: 5)")
    parser.add_argument("--n_batch_train", type=float, default=100, help="Learning rate (default: 0.001)")
    parser.add_argument("--n_batch_test", type=float, default=1000, help="Learning rate (default: 0.001)")
    parser.add_argument("--n_epoch", type=float, default=100, help="Learning rate (default: 0.001)")    
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate (default: 0.001)")    
    args = parser.parse_args()
    main(args)
