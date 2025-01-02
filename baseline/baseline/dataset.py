import random
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from functools import partial
from sampler import BatchSampler, custom_collate_fn


def get_dataset(root_dir):
# 데이터셋의 루트 디렉터리 경로
    root_dir = '/home/work/study/dongwan/data/mini_imagenet'  # 원본 미니 이미지넷 데이터셋의 루트 디렉터리

    # 모든 클래스 목록을 가져오기
    all_class = os.listdir(root_dir) # 폴더명 리스트로 반환
    random.shuffle(all_class)  # 리스트를 무작위로 섞어주고

    # 학습 및 테스트로 나누기
    train_class = all_class[:64]  # 64개의 클래스는 학습용으로
    test_class = all_class[64:]   # 나머지 36개의 클래스는 테스트용으로

    # 이미지 변환 설정
    transform = transforms.Compose([
        transforms.Resize((84,84)),
        transforms.ToTensor()        
    ])

    # 전체 데이터셋 로드 (아직 클래스별로 분리되지 않은 상태)
    full_dataset = datasets.ImageFolder(root=root_dir, transform=transform) # image, label

    # 학습 및 테스트 클래스 인덱스 필터링
    # samples = 전체 리스트 w/ (path, class_index)
    # classes = class label list
    train_index = [i for i, (_, index) in enumerate(full_dataset.samples) if full_dataset.classes[index] in train_class]
    test_index = [i for i, (_, index) in enumerate(full_dataset.samples) if full_dataset.classes[index] in test_class]

    # Subset을 통해 학습용과 테스트용 데이터셋 생성
    train_dataset = Subset(full_dataset, train_index)
    test_dataset = Subset(full_dataset, test_index)

    return train_dataset, test_dataset
    
def normal_dataloader(dataset, n_batch_size):
    dataloader = DataLoader(dataset, n_batch_size, shuffle = True)
    return dataloader

def episode_dataloader(dataset, n_classes, n_support, n_query, n_batch, batch_sampler=BatchSampler, collate_fn=custom_collate_fn, num_workers=4):    
    batch_sampler = batch_sampler(dataset, n_classes=n_classes, n_support=n_support, n_query=n_query, n_batch=n_batch)
    collate_fn = partial(collate_fn, n_support=n_support, n_query=n_query)
    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)    
    
    return dataloader

