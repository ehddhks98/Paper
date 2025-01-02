import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import remap_label
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir="/home/work/study/dongwan/baseline++/runs")


def train(model, dataloader, optimizer, n_epoch):
    model.train()
    
    for epoch in range(n_epoch):
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 각 에포크에서 여러 배치에 대해 반복
        for batch_idx, batch in enumerate(dataloader):
            data, label = batch

            # 데이터를 GPU로 이동 (가능한 경우)
            data, label = (data.to(device), label.to(device))
            label = remap_label(label)
            prob = model(data, label)
            loss = F.cross_entropy(prob, label)            

            # 정확도 계산
            label_prediction = prob.argmax(dim=1).to(device)
            correct += (label_prediction == label).sum().item()
            total += label.size(0)

            # 역전파 및 옵티마이저 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 배치당 손실 누적
            total_loss += loss.item()

            # 현재 배치 진행 상태 출력 (옵션)
            print(f"Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")            

        # 에포크당 평균 손실 및 정확도 출력
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        print(f"Epoch [{epoch+1}/{n_epoch}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # 텐서보드에 기록
        writer.add_scalar("Loss/epoch/train", avg_loss, epoch)
        writer.add_scalar("Accuracy/epoch/train", accuracy, epoch)

    writer.close()

        

