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
    
# torch.nn.Linear(in_features, out_features)
# fine_tuing 학습 과정은 test.py에서
# conv4 적용하면 (64,84,84) -> (64,5,5)
class Baseline(nn.Module):  
    def __init__(self, backbone, in_feature):
        super(Baseline, self).__init__()
        self.backbone = backbone
        self.in_feature = in_feature
        self.linear = None # Linear layer 유동적으로 설정
        
    def forward(self, x, y):
        x = x.to(device)
        y = y.to(device)        
        n_class = torch.unique(y).size(0) # 배치별 클래스 개수     
        sample_feature = self.backbone(x) # 데이터 임베딩 (16, 64, 5, 5)
        sample_feature = torch.flatten(sample_feature, start_dim=1) # linear layer 통과 위한 2차원 변경/(16, 64*5*5)

        if self.linear is None:
            # 처음에는 그냥 초기화
            self.linear = nn.Linear(self.in_feature, n_class).to(device)
        elif self.linear.out_features != n_class:
            # 클래스 수 증가/감소에 따른 파라미터 조정
            self.linear = adjust_linear_layer(self.linear, n_class)    

        linear_output = self.linear(sample_feature)        
        
        return linear_output
    
    # fine tuning 과정 classifier 초기화 메서드
    def initialize_classifier(self, num_class):
        self.linear = nn.Linear(self.in_feature, num_class)

def adjust_linear_layer(linear_layer, new_out_feature):
    # 새로운 Linear layer 생성
    new_linear = nn.Linear(linear_layer.in_features, new_out_feature).to(device)
    
    # 기존 가중치 복사 (클래스 수 감소에 따라 가중치 일부만 복사)
    # weight.tensor (out_features, input_features) 
    # bias (out_features)
    with torch.no_grad():
        min_feature = min(linear_layer.out_features, new_out_feature)
        new_linear.weight[:min_feature] = linear_layer.weight[:min_feature]
        new_linear.bias[:min_feature] = linear_layer.bias[:min_feature]

    return new_linear

def create_model():
    backbone = Conv4()
    in_feature = 64*5*5
    model = Baseline(backbone, in_feature).to(device)
    return model