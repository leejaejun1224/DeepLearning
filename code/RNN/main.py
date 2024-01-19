import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model import *
sentences = ["i like dog", "i love coffee", "i hate milk", "you like cat", "you love milk", "you hate coffee"]
dtype = torch.float


word_list = list(set(" ".join(sentences).split()))
word_dict = {w: i for i, w in enumerate(word_list)}
number_dict = {i: w for i, w in enumerate(word_list)}
n_class = len(word_dict)

batch_size = len(sentences) # 데이터 6개
n_step = 2  # 학습 하려고 하는 문장의 길이 - 1
hidden_size = 5  # 은닉층 사이즈
epoch = 500

def make_batch(sentences):
    input_batch = []
    target_batch = []
    for sentence in sentences:
        word = sentence.split()
        # input을 숫자로 만들어보자
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]
        input_batch.append(np.eye(n_class)[input]) #인코딩
        target_batch.append(target)

    return input_batch, target_batch

input_batch, target_batch = make_batch(sentences=sentences)

rnn = RNN()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), lr = 0.01)

loss_list = []
for i in range(epoch):
    # 각 hidden 마다 숫자가 하나가 나와야겠지?
    hidden = torch.zeros(1, 1, hidden_size)

    output, hidden = rnn(input_batch, hidden)
    loss = loss_function(output, target_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i%100 == 0:
        print("in epoch ", i, " loss is : ", loss)
        loss_list.append(loss)        

hidden = torch.zeros(1, batch_size, hidden_size, requires_grad=True)
predict = rnn(input_batch, hidden).data.max(1, keepdim=True)[1]
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])

