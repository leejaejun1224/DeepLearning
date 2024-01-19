import torch
import torch.nn as nn
from torch.autograd import Variable


# class LSTM(nn.Module):
#     def __init__(self, output_class_size, input_size, hidden_size, num_layers, seq_length, use_bn, ):
#         super(LSTM, self).__init__()
#         #여기서 num class는 아래에도 나오지만 output의 크기이다. 
#         self.output_classes_size = output_class_size
#         self.hidden_size = hidden_size
#         self.input_size = input_size
#         self.num_layers = num_layers
#         self.seq_length = seq_length
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first = True)
#         # self.hidden = self.init_hidden()

#         layers = []
#         if use_bn:
#             #왜 batch size가 아니지?
#             layers.append(nn.BatchNorm1d(self.hidden_size))
#         layers.append(nn.Linear(self.hidden_size, self.hidden_size//2))
#         layers.append(nn.ReLU())
#         layers.append(nn.Linear(self.hidden_size//2, self.output_classes_size))
#         self.regressor = nn.Sequential(*layers)

#     def init_hidden(self, data_size):
#         return (torch.zeros(self.num_layers, data_size, self.hidden_size),
#                 torch.zeros(self.num_layers, data_size, self.hidden_size))


#     def forward(self, x):
#         (self.hidden, present_cell) = self.init_hidden(x.size(0))
#         output, (self.hidden, c0) = self.lstm(x, (present_cell, self.hidden))
#         self.hidden = self.hidden.view(-1, self.hidden_size)
#         y_pred = self.regressor(self.hidden)
#         return y_pred


class LSTM(nn.Module) :
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length) :
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        # 인풋 사이즈
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, 128)
        self.fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        print(seq_length)
        
    def forward(self, x) :
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        output, (hn, cn) = self.lstm(x, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc(out)
        return out

