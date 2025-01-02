import torch

def remap_label_episode(support_label, query_label):
    unique_labels = torch.unique(torch.cat([support_label, query_label]))
    label_map = {label.item(): i for i, label in enumerate(unique_labels)}

    remapped_support_label = torch.tensor([label_map[label.item()] for label in support_label],device=support_label.device)
    remapped_query_label = torch.tensor([label_map[label.item()] for label in query_label], device=query_label.device)

    return remapped_support_label, remapped_query_label

def remap_label(support_label ):
    unique_labels = torch.unique(support_label)
    label_map = {label.item(): i for i, label in enumerate(unique_labels)}
    remapped_support_label = torch.tensor([label_map[label.item()] for label in support_label],device=support_label.device)
    
    return remapped_support_label

def extract_data(support_set, query_set):
    support_data = torch.stack([data for data, _ in support_set])
    query_data = torch.stack([data for data, _ in query_set])
    support_label = torch.tensor([label for _, label in support_set])
    query_label = torch.tensor([label for _, label in query_set])
    
    return support_data, support_label, query_data, query_label
