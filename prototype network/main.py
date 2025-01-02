from PIL import Image
import argparse
import torch
from torch import nn, optim
from dataset import get_datasets
from module import create_model
from episode import get_dataloaders
from train import train
from test import evaluate

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set, test_set = get_datasets(args.image_size)
    train_loader, test_loader = get_dataloaders(train_set, test_set, args.n_way, args.n_shot, args.n_query, args.n_training_episodes, args.n_evaluation_tasks)

    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train(model, train_loader, optimizer, criterion, args.log_update_frequency)

    correct_predictions, total_predictions = evaluate(model, test_loader)

    print(f"Model tested on {len(test_loader)} tasks. Accuracy: {(100 * correct_predictions / total_predictions):.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Prototypical Networks on Omniglot dataset")
    parser.add_argument("--image_size", type=int, default=28, help="Image size (default: 28)")
    parser.add_argument("--n_way", type=int, default=60, help="Number of classes in a task (default: 60)")
    parser.add_argument("--n_shot", type=int, default=1, help="Number of images per class in the support set (default: 1)")
    parser.add_argument("--n_query", type=int, default=5, help="Number of images per class in the query set (default: 5)")
    parser.add_argument("--n_training_episodes", type=int, default=10000, help="Number of training episodes (default: 1000)")
    parser.add_argument("--n_evaluation_tasks", type=int, default=100, help="Number of evaluation tasks (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--log_update_frequency", type=int, default=10, help="Log update frequency (default: 10)")

    args = parser.parse_args()
    main(args)
