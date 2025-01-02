from PIL import Image
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import get_dataset, get_dataloader
from sampler import BatchSampler, custom_collate_fn
from train import train_relationnet
from test import test_relationnet
from model import create_model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터셋 로드
    train_set, test_set = get_dataset(args.image_size)

    # 데이터로더 설정
    train_dataloader = get_dataloader(
        train_set, args.n_classes, args.n_support, args.n_query_train, args.n_batches_train
    )
    test_dataloader = get_dataloader(
        test_set, args.n_classes, args.n_support, args.n_query_test, args.n_batches_test
    )

    # 모델 생성
    model = create_model()
    model.to(device)

    # 손실 함수 및 옵티마이저 설정
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # 모델 학습
    train_relationnet(model, train_dataloader, optimizer, args.n_epochs)

    # 모델 테스트
    test_relationnet(model, test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate Prototypical Networks on Omniglot dataset"
    )
    parser.add_argument(
        "--image_size", type=int, default=28, help="Image size (default: 28)"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=5,
        help="Number of classes per task (default: 5)",
    )
    parser.add_argument(
        "--n_support",
        type=int,
        default=5,
        help="Number of support examples per class (default: 5)",
    )
    parser.add_argument(
        "--n_query_train",
        type=int,
        default=15,
        help="Number of query examples per class (default: 5)",
    )
    parser.add_argument(
        "--n_query_test",
        type=int,
        default=5,
        help="Number of query examples per class (default: 5)",
    )
    parser.add_argument(
        "--n_batches_train",
        type=int,
        default=100,
        help="Number of batches per epoch (default: 100)",
    )
    parser.add_argument(
        "--n_batches_test",
        type=int,
        default=1000,
        help="Number of batches per epoch (default: 100)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)",
    )

    args = parser.parse_args()
    main(args)
