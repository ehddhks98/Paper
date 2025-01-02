import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model import extract_data
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter(log_dir="MatchingNet/runs/1shot")


def train_matching_net(model, dataloader, optimizer, n_epochs):
    model.train()  # 모델을 학습 모드로 설정

    for epoch in range(n_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        # 각 에포크에서 여러 배치에 대해 반복
        for batch_idx, (support_set, query_set) in enumerate(dataloader):
            # support_set과 query_set에서 데이터와 레이블을 추출
            support_data, support_label, query_data, query_label = extract_data(support_set, query_set)
            support_label, query_label = remap_labels(support_label, query_label)

            # 데이터를 GPU로 이동 (가능한 경우)
            support_data, query_data, support_label, query_label = (
                support_data.to(device),
                query_data.to(device),
                support_label.to(device),
                query_label.to(device),
            )

            # 모델을 통해 점수 계산 (쿼리 이미지에 대한 예측 점수)
            cos_similarity = model(support_data, query_data, support_label)

            # 손실 계산 (소프트맥스 함수 사용)
            loss = F.cross_entropy(cos_similarity, query_label)            

            # 정확도 계산
            label_prediction = cos_similarity.argmax(dim=1)
            correct += (label_prediction == query_label).sum().item()
            total += query_label.size(0)

            # 역전파 및 옵티마이저 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 배치당 손실 누적
            total_loss += loss.item()

            # 현재 배치 진행 상태 출력 (옵션)
            print(f"Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

            # 텐서보드에 배치당 손실 기록
            writer.add_scalar(
                "Loss/batch/train", loss.item(), epoch * len(dataloader) + batch_idx
            )

        # 에포크당 평균 손실 및 정확도 출력
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        print(
            f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        )

        # 텐서보드에 기록
        writer.add_scalar("Loss/epoch/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/epoch/train", accuracy, epoch)

    writer.close()


def remap_labels(support_label, query_label):
    unique_labels = torch.unique(torch.cat([support_label, query_label]))
    label_map = {label.item(): i for i, label in enumerate(unique_labels)}

    remapped_support_label = torch.tensor(
        [label_map[label.item()] for label in support_label],
        device=support_label.device,
    )
    remapped_query_label = torch.tensor(
        [label_map[label.item()] for label in query_label], device=query_label.device
    )

    return remapped_support_label, remapped_query_label
