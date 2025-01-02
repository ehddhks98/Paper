import torch
import torch.nn as nn
from torchvision.models import resnet18
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrototypicalNetworks(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(PrototypicalNetworks, self).__init__()
        self.backbone = backbone

    def forward(self, support_images, support_labels, query_images):
        z_support = self.backbone.forward(support_images)
        z_query = self.backbone.forward(query_images)
        n_way = len(torch.unique(support_labels))
        
        z_proto = torch.cat([z_support[torch.nonzero(support_labels == label)].mean(0) for label in range(n_way)])
        
        dists = torch.cdist(z_query, z_proto)
        scores = -dists
        return scores

def create_model():
    convolutional_network = resnet18(pretrained=True)
    convolutional_network.fc = nn.Flatten()
    model = PrototypicalNetworks(convolutional_network).to(device)
    return model
