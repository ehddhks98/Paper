import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNBlock(nn.Module):
    def __init__(self):
        super(CNNBlock, self).__init__()
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
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.cnn_blocks(x)


class MatchingNet(nn.Module):
    def __init__(self,cnn_net, input_size, hidden_size):
        super(MatchingNet, self).__init__()
        self.cnn = cnn_net
        self.bi_lstm = nn.LSTM(input_size, hidden_size=hidden_size, num_layers = 1, batch_first=True, bidirectional=True)
        self.att_lstm = nn.LSTMCell(input_size=2*hidden_size, hidden_size=hidden_size)

    def forward(self, support_data, query_data, support_labels):
        init_sup_data = self.cnn(support_data) # (5, 64, 1,1)
        support_embedding = self.sup_LSTM(init_sup_data) # (5,64)

        init_query_data = self.cnn(query_data)
        query_embedding = self.query_LSTM(init_query_data, support_embedding) # (5,64)

        cos_similarity = torch.mm(F.normalize(query_embedding, dim=1), F.normalize(support_embedding, dim=1).T) #(5,5)
        
        return cos_similarity

    

    def sup_LSTM(self, init_sup_data):
        """
        LSTM input = [batch_size, sequence, input_size]
        LSTM output = output, (h_n, c_n)
        """
        hidden_size = self.bi_lstm.hidden_size 
        init_sup_data = init_sup_data.squeeze().unsqueeze(0) #  (5,64,1,1) -> (1, 5, 64)
        hidden_state, _ = self.bi_lstm(init_sup_data) # (1, 5, 128) bidrectional로 출력하면 뭐가 생기지?
        hidden_state = hidden_state.squeeze(0) # (5,128)
        support_embedding = init_sup_data.squeeze(0) + hidden_state[:,:hidden_size] + hidden_state[:,:hidden_size]
        return support_embedding # (5,64)

    def query_LSTM(self, init_query_data, support_embedding):
        """
        LSTMCell input = (batch, input_size)

        """
        n_support = support_embedding.size(0)
        init_query_data = init_query_data.squeeze()
        hidden_state = init_query_data # (5, 64)
        
        cell_state = torch.zeros_like(hidden_state)

        for i in range(n_support):
            dot_tensor = torch.mm(hidden_state, support_embedding.T) #(5,5) (query, support)
            attention = F.softmax(dot_tensor, dim=1) # support의 유사도 계산
            read_out = torch.mm(attention, support_embedding) #(5,5), (5,64) => (5, 64)
            lstm_input = torch.cat((init_query_data, read_out), dim=1) # (5,128)
            hidden_state, cell_state = self.att_lstm(lstm_input, (hidden_state, cell_state))
            hidden_state = hidden_state + init_query_data
        
        return hidden_state # (5,64)
            
    
def extract_data(support_set, query_set):
    support_data = torch.stack([data for data, _ in support_set])
    query_data = torch.stack([data for data, _ in query_set])
    support_label = torch.tensor([label for _, label in support_set])
    query_label = torch.tensor([label for _, label in query_set])
    return support_data, support_label, query_data, query_label

    
def create_model():
    cnn_net = CNNBlock()
    input_size = 64
    hidden_size = 64
    model = MatchingNet(cnn_net, input_size, hidden_size).to(device)
    return model
    
