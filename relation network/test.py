import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from model import extract_data
from train import remap_labels

writer = SummaryWriter(log_dir='./runs/5way 5shot')


def test_relationnet(model, dataloader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch_idx, (support_set, query_set) in enumerate(dataloader):
            # support_set과 query_set에서 데이터와 레이블을 추출
            support_data, support_label, query_data, query_label = extract_data(
                support_set, query_set
            )
            support_label, query_label = remap_labels(support_label, query_label)

            # 데이터를 GPU로 이동 (가능한 경우)
            support_data, query_data, support_label, query_label = (
                support_data.to(device),
                query_data.to(device),
                support_label.to(device),
                query_label.to(device),
            )

            # 모델을 통해 점수 계산 (쿼리 이미지에 대한 예측 점수)
            scores = model(support_data, query_data, support_label)
            scores = scores.squeeze(-1).squeeze(-1)

            # 실제 모델 레이블의 원 핫 벡터 생성
            num_classes = support_label.unique().size(0)
            one_hot_label = F.one_hot(query_label, num_classes=num_classes).float()
            one_hot_label = one_hot_label.squeeze(-1)

            # 손실 계산 (소프트맥스 함수 사용)
            loss = nn.MSELoss()(scores, one_hot_label)

            # 정확도 계산
            
            predicted_labels = scores.argmax(dim=1)
            correct += (predicted_labels == query_label).sum().item()
            total += query_label.size(0)

            # 손실 누적
            total_loss += loss.item()

            # 현재 배치 진행 상태 출력
            print(f"Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

            # 텐서보드에 배치당 손실 기록
            writer.add_scalar("Loss/Batch", loss.item(), batch_idx)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")

        writer.add_scalar("Test Loss", avg_loss)
        writer.add_scalar("Test Acurracy", accuracy)

    writer.close()
