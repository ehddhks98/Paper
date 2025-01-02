import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import remap_label, extract_data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir="/home/work/study/dongwan/baseline++/runs")


def fine_tuning(model, dataloader, optimizer, n_iteration, n_experiment):
    experimenet_accuracy = []
    for experiment in range(n_experiment):
        for support_set, query_set in dataloader:
            
            model.train()
            # support, query data 획득
            support_data, support_label, query_data, query_label = extract_data(support_set, query_set)
            support_data, support_label = support_data.to(device), support_label.to(device)
            query_data, query_label = query_data.to(device), query_label.to(device)

            # classifier 초기화
            model.initialize_classifier(torch.unique(support_label).size(0))    

            for iteration in range(n_iteration):
                # support sample 중 학습에 사용할 sample 선택 (sequence, k)
                selected_indices = random.sample(range(support_data.size(0)), 4)
                selected_support_data = support_data[selected_indices]
                selected_support_label = support_label[selected_indices]
                selected_support_label = remap_label(selected_support_label)
                optimizer.zero_grad()             

                output = model(selected_support_data, selected_support_label)
                loss = F.cross_entropy(output, selected_support_label)
                loss.backward()                
                optimizer.step()
                print(f"Iteration {iteration + 1}/{n_iteration}, Loss: {loss.item():.4f}")
        
            # Query set 테스트
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                query_label = remap_label(query_label)
                output = model(query_data, query_label)
                predict_label = output.argmax(dim=1)
                total += query_label.size(0)
                correct += (predict_label == query_label).sum().item()

            accuracy = 100 * correct / total
            experimenet_accuracy.append(accuracy) 
            print(f"Query Set Accuracy: {accuracy:.2f}%")
            writer.add_scalar("Accuracy/Experiment", accuracy)
            
    avg_accuracy = sum(experimenet_accuracy) / len(experimenet_accuracy)
    writer.add_scalar("Total Accuracy/Experiment",  avg_accuracy)
    writer.close()