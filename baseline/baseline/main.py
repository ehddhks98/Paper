import torch
import torch.optim as optim
import argparse
from dataset import get_dataset, normal_dataloader, episode_dataloader
from model import create_model
from train import train
from test import fine_tuning
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):    
        root_dir = "/home/work/study/dongwan/data/mini_imagenet"        
        train_set, test_set = get_dataset(root_dir)   
        train_dataloader = normal_dataloader(train_set, args.n_batch_size)        
        test_dataloader = episode_dataloader(test_set, args.n_class, args.n_support, args.n_query, args.n_batch)

        model = create_model()
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        train(model, train_dataloader, optimizer, args.n_epoch)        
        fine_tuning(model, test_dataloader, optimizer, args.n_iter, args.n_experiment)


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Train and evaluate Baseline on mini Imagenet")
        parser.add_argument("--n_epoch", type=int, default=100)
        parser.add_argument("--n_batch_size", type=int, default=16)
        parser.add_argument("--n_class", type=int, default=5)
        parser.add_argument("--n_support", type=int, default=1)
        parser.add_argument("--n_query", type=int, default=16)
        parser.add_argument("--n_iter", type=int, default=100)
        parser.add_argument("--n_experiment", type=int, default=100)
        parser.add_argument("--n_batch", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=0.001)
        args = parser.parse_args()
        main(args)
