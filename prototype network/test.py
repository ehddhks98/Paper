import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_on_one_task(model, support_images, support_labels, query_images, query_labels):
    return (
        torch.max(
            model(support_images.to(device), support_labels.to(device), query_images.to(device)).detach().data, 1
        )[1] == query_labels.to(device)
    ).sum().item(), len(query_labels)

def evaluate(model, data_loader):
    total_predictions = 0
    correct_predictions = 0

    model.eval()
    with torch.no_grad():
        for episode_index, (support_images, support_labels, query_images, query_labels, class_ids) in tqdm(enumerate(data_loader), total=len(data_loader)):
            correct, total = evaluate_on_one_task(model, support_images, support_labels, query_images, query_labels)
            total_predictions += total
            correct_predictions += correct

    return correct_predictions, total_predictions