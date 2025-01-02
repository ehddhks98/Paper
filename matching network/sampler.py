import torch
from torch.utils.data import Sampler


class BatchSampler(Sampler):
    def __init__(self, dataset, n_classes, n_support, n_query, n_batches):
        super().__init__(dataset)
        self.n_classes = n_classes  # 배치당 선택할 클래스 수
        self.n_support = n_support  # 각 클래스당 support 샘플 수
        self.n_query = n_query  # 각 클래스당 query 샘플 수
        self.n_batches = n_batches  # 총 배치 수
        self.n_samples = n_support + n_query  # 각 클래스당 선택할 전체 샘플 수

        # 데이터셋의 레이블을 PyTorch 텐서로 변환
        self.labels = torch.tensor([label for _, label in dataset])

        # 고유한 클래스 레이블 추출 (torch.unique 사용)
        self.unique_classes = torch.unique(self.labels)

        # 클래스별 인덱스를 미리 계산
        self.class_indices = [torch.where(self.labels == c)[0] for c in self.unique_classes]

    def __iter__(self):
        for _ in range(self.n_batches):
            indices = []
            # 1. 전체 클래스에서 무작위로 n_classes 개의 클래스 선택
            selected_class_indices = torch.randperm(len(self.unique_classes))[: self.n_classes]

            for class_idx in selected_class_indices:
                # 2. 각 클래스에서 무작위로 n_samples 개의 샘플 선택
                class_samples = self.class_indices[class_idx]
                selected_samples = class_samples[torch.randperm(len(class_samples))[: self.n_samples]]

                # 3. 선택된 샘플의 인덱스를 리스트에 추가
                indices.extend(selected_samples.tolist())

            yield indices

    def __len__(self):
        return self.n_batches


def custom_collate_fn(batch, n_support, n_query):
    n_samples_per_class = n_support + n_query
    support_set = []
    query_set = []

    for i in range(0, len(batch), n_samples_per_class):
        support_set.extend(batch[i : i + n_support])
        query_set.extend(batch[i + n_support : i + n_samples_per_class])

    return support_set, query_set
