import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import urllib.request
import time
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
train_data = pd.read_csv('ChatBotData.csv')
train_data.head()
print('챗봇 샘플의 개수 :', len(train_data))

questions = []
for sentence in train_data['Q']:
    # 구두점에 대해서 띄어쓰기
    # ex) 12시 땡! -> 12시 땡 !
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    questions.append(sentence)

answers = []
for sentence in train_data['A']:
    # 구두점에 대해서 띄어쓰기
    # ex) 12시 땡! -> 12시 땡 !
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    answers.append(sentence)

print(len(questions))
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# BPE 토크나이저 및 트레이너 초기화
tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=2**13)

# 토크나이저 훈련
tokenizer.train_from_iterator(questions + answers, trainer=trainer)

# 단어 집합의 크기 계산
VOCAB_SIZE = tokenizer.get_vocab_size()
START_TOKEN = VOCAB_SIZE
END_TOKEN = VOCAB_SIZE + 1
# 특수 토큰을 추가했으므로, 추가된 특수 토큰의 수를 고려
VOCAB_SIZE += 2
# print('시작 토큰 번호 :',START_TOKEN)
# print('종료 토큰 번호 :',END_TOKEN)
# print('단어 집합의 크기 :',VOCAB_SIZE)
# sample_string = questions[20]
# print('임의의 질문 샘플을 정수 인코딩 : {}'.format(tokenizer.encode(questions[20]).ids))
# tokenized_string = (tokenizer.encode(questions[20]).ids)
# print ('정수 인코딩 후의 문장 {}'.format(tokenized_string))

# original_string = tokenizer.decode(tokenized_string)
# print ('기존 문장: {}'.format(original_string))

# for ts in tokenized_string:
#     print ('{} ----> {}'.format(ts, tokenizer.decode([ts])))

MAX_LENGTH = 40

# 토큰화 / 정수 인코딩 / 시작 토큰과 종료 토큰 추가 / 패딩

def pad_to_max_length(tensor, max_length):
    if len(tensor) > max_length:
        return tensor[:max_length]  # 시퀀스 자르기
    elif len(tensor) < max_length:
        # 시퀀스에 패딩 추가
        return torch.cat([tensor, torch.zeros(max_length - len(tensor), dtype=torch.long)])
    else:
        return tensor

def tokenize_and_filter(inputs, outputs):
    tokenized_inputs, tokenized_outputs = [], []

    for (sentence1, sentence2) in zip(inputs, outputs):
        sentence1 = [START_TOKEN] + tokenizer.encode(sentence1).ids + [END_TOKEN]
        sentence2 = [START_TOKEN] + tokenizer.encode(sentence2).ids + [END_TOKEN]

        tokenized_inputs.append(torch.tensor(sentence1))
        tokenized_outputs.append(torch.tensor(sentence2))

    # 패딩
    tokenized_inputs = pad_sequence([pad_to_max_length(t, MAX_LENGTH) for t in tokenized_inputs], batch_first=True, padding_value=0)
    tokenized_outputs = pad_sequence([pad_to_max_length(t, MAX_LENGTH) for t in tokenized_outputs], batch_first=True, padding_value=0)

    return tokenized_inputs, tokenized_outputs

questions, answers = tokenize_and_filter(questions, answers)

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
BATCH_SIZE = 64
BUFFER_SIZE = 20000

class CustomDataset(Dataset):
    def __init__(self, questions, answers):
        self.questions = questions
        self.answers = answers

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return {
            'inputs': torch.tensor(self.questions[idx], dtype=torch.long),
            'dec_inputs': torch.tensor(self.answers[idx][:-1], dtype=torch.long),  # 마지막 토큰 제거
            'outputs': torch.tensor(self.answers[idx][1:], dtype=torch.long)  # 시작 토큰 제거
        }
    
dataset = CustomDataset(questions, answers)

# DataLoader를 사용하여 데이터셋을 셔플하고 배치 처리
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

for batch in data_loader:
    print(batch['inputs'].shape)
    break


from transformer import Transformer
num_epochs = 1
num_vocabs = 9000
num_layers = 2
dff = 512
d_model = 256
num_heads = 8
dropout=0.1
small_transformer = Transformer(d_model, num_heads, dff, num_vocabs, dropout, num_layers)

from torch.optim import Adam
from scheduler import CustomSchedule

learning_rate = CustomSchedule(d_model=d_model)

optimizer = Adam(params=small_transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)  # 적절한 옵티마이저를 정의하세요

def accuracy(y_true, y_pred):
    # y_true: (batch_size, MAX_LENGTH - 1)
    # y_pred: (batch_size, MAX_LENGTH - 1, num_classes)
    y_true = y_true.view(-1, MAX_LENGTH - 1)
    y_pred = y_pred.max(dim=2)[1]  # 가장 높은 확률을 가진 클래스 선택
    correct = (y_true == y_pred).float()  # 예측이 맞았는지 확인
    return correct.sum() / y_true.numel()

# 손실 함수
loss_function = nn.CrossEntropyLoss()

# 학습 루프 예시
for epoch in range(num_epochs):
    # 에폭 시작 시 lr_scheduler 업데이트
    for batch in data_loader:
        small_transformer.train()
        optimizer.zero_grad()
        # 모델의 예측
        y_pred = small_transformer(batch['inputs'], batch['dec_inputs'])
        # shape은 64*39*9000
        # 손실 계산을 위한 y_pred의 형태 확인 및 조정
        batch_size, sequence_length, _ = y_pred.size()
        loss = loss_function(y_pred.view(batch_size * sequence_length, num_vocabs), batch['outputs'].view(-1))

        loss.backward()
        optimizer.step()
        
    if (epoch+1)%2 == 1:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

import numpy as np
import re

# !나 ? 같은거 있으면 그 앞에 띄어쓰기 하나 해줌
def preprocess_sentence(sentence):
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    return sentence

def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    # 토크나이징, START_TOKEN과 END_TOKEN 추가
    sentence = torch.tensor([START_TOKEN] + tokenizer.encode(sentence).ids + [END_TOKEN]).unsqueeze(0)

    #처음부터 시작해서 하나하나붙여갈거야
    output = torch.tensor([START_TOKEN]).unsqueeze(0)

    small_transformer.eval()
    with torch.no_grad():
        for i in range(MAX_LENGTH):
            predictions = small_transformer(sentence, output)

            # 현재 시점의 예측 단어를 받아온다.
            predicted_id = torch.argmax(predictions[:, -1, :], dim=-1).unsqueeze(0)

            # 종료 토큰이라면 예측을 중단
            if predicted_id == [END_TOKEN][0]:
                break

            # 현재 시점의 예측 단어를 output에 연결한다.
            output = torch.cat((output, predicted_id), dim=-1)

    return output.squeeze(0)

def predict(sentence):
    prediction = evaluate(sentence)

    # 토크나이저를 통해 정수 시퀀스를 문자열로 디코딩
    predicted_sentence = tokenizer.decode([i for i in prediction if i < VOCAB_SIZE-2])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence

output = predict("영화 볼래?")
print(output)