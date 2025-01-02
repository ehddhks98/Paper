import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False

class Conv4(nn.Module):
    def __init__(self):
        super(Conv4, self).__init__()
        self.cnn_blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
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
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.cnn_blocks(x)    
    
# fine_tuing 학습 과정은 test.py에서
# conv4 적용하면 (64,84,84) -> (64,5,5)
class Baseline(nn.Module):  
    def __init__(self, backbone, in_feature, num_classes):
        super(Baseline, self).__init__()
        self.backbone = backbone
        self.in_feature = in_feature
        
        # 학습 가능한 파라미터 벡터 정의 (initial num_classes)
        self.parameter_vector = nn.Parameter(torch.randn(num_classes, in_feature))
        
    def forward(self, x, y):
        x = x.to(device)
        y = y.to(device)        
        n_class = torch.unique(y).size(0) # 배치별 클래스 개수
        sample_feature = self.backbone(x) # 데이터 임베딩 (batch_size, 64, 5, 5)
        sample_feature = torch.flatten(sample_feature, start_dim=1) # (batch_size, 64*5*5)

        if self.parameter_vector.size(0) != n_class:
            # 클래스 수에 따라 파라미터 조정
            self.parameter_vector = self.adjust_parameter_vector(n_class)
        
        # 코사인 유사도 계산
        cos_sim = torch.mm(F.normalize(sample_feature, dim=1), F.normalize(self.parameter_vector, dim=1).T)
        
        return cos_sim
    
    def initialize_classifier(self, num_class):
        self.parameter_vector = nn.Parameter(torch.randn(num_class, self.in_feature))

    def adjust_parameter_vector(self, new_out_feature):
        # 새로운 파라미터 벡터 생성
        new_parameter_vector = nn.Parameter(torch.randn(new_out_feature, self.in_feature).to(device))
        
        with torch.no_grad():
            # 기존 파라미터 복사
            min_feature = min(self.parameter_vector.size(0), new_out_feature)
            new_parameter_vector[:min_feature] = self.parameter_vector[:min_feature]

        return new_parameter_vector

def create_model():
    backbone = Conv4()
    in_feature = 64*5*5
    num_classes = 5
    model = Baseline(backbone, in_feature, num_classes).to(device)
    return model