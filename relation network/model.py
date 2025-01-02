import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbeddingNetwork(nn.Module):
    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
        self.cnn_blocks = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.cnn_blocks(x)


class RelationNet(nn.Module):
    def __init__(self):
        super(RelationNet, self).__init__()
        self.cnn_blocks = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.cnn_blocks(x)


class RelationScore(nn.Module):
    def __init__(self, embedding_net, relation_net):
        super(RelationScore, self).__init__()
        self.embedding_net = embedding_net
        self.relation_net = relation_net

    def forward(self, support_data, query_data, support_label):
        # support set, query set을 CNN을 통해 임베딩
        support_embeddings = self.embedding_net(support_data)
        query_embeddings = self.embedding_net(query_data)

        support_mean = []
        for c in support_label.unique():
            class_mean = support_embeddings[support_label == c].mean(0)
            support_mean.append(class_mean)
        support_mean = torch.stack(support_mean)

        # 쿼리마다 클래스별 점수를 계산
        all_scores = []
        for query in query_embeddings:
            class_scores = []
            for class_mean in support_mean:
                # 쿼리와 클래스 평균 임베딩을 결합
                concat_vector = torch.cat((query, class_mean), dim=0)
                score = self.relation_net(concat_vector.unsqueeze(0))
                class_scores.append(score)
            class_scores = torch.stack(class_scores)
            all_scores.append(class_scores)

        # 최종 출력 텐서
        all_scores = torch.stack(all_scores)
        return all_scores


def extract_data(support_set, query_set):
    support_data = torch.stack([data for data, _ in support_set])
    query_data = torch.stack([data for data, _ in query_set])
    support_label = torch.tensor([label for _, label in support_set])
    query_label = torch.tensor([label for _, label in query_set])
    return support_data, support_label, query_data, query_label


def create_model():
    embedding_net = EmbeddingNetwork()
    relation_net = RelationNet()
    model = RelationScore(embedding_net, relation_net).to(device)
    return model
