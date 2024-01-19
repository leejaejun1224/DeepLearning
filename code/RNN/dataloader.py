# -*- coding: utf-8 -*-
#! /usr/bin/env python3

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
from model import *

def findFiles(path):
    return glob.glob(path)



# 모든 영대소문자를 포함하는 문자열임
# abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 에다가 .,;'이 더해져서 나오게 됨.
# 물론 indexing도 가능함
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

#내가 원하는 글자만 남기기
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )


# 각 언어의 이름 목록인 category_lines dictionary 생성 -> 숫자 : 언어이름
category_lines = {}
all_categories = []


# 파일을 읽고 줄 단위로 분리
def readLines(filename):
    # '\n' 엔터를 기준으로 분리해서 리스트로
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('data/names/*.txt'):

    #파일의 이름을 짜개서 .txt 빼고 앞에거만 category에 저장
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename=filename)
    category_lines[category] = lines

n_categories = len(all_categories)


#문자의 주소를 찾아준다 a는0, b는 1이렇게 
def letterToIndex(letter):
    return all_letters.find(letter)

# 이걸 one-hot encoding이라고 한다.
# 한 줄 이름은 (글자수 x 1 x 전체 알파벳의 수)
def lineToTensor(line):
    tensor = torch.zeros(len(line),1, n_letters)
    # 하나의 line 예를 들어 jaejun이면 여기서 j,a,e,j,u,n 하나씩 가져와서
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    
    return tensor

n_hidden = 128

input = lineToTensor('Albert')
hidden = torch.zeros(1,n_hidden)

#글자의 총 갯수, hidden의 갯수, output 카테고리의 갯수
#초기화 한다고 생각 하면 됨

rnn = RNN(n_letters, n_hidden, n_categories)
output, next_hidden = rnn(input[1], hidden)

def categoryFromOutput(output):
    #1을 인자로 줬으니 tensor에서 상위 1개의 값의 index와 그 값을 반환 근데 그 값을 숫자만 땡그랑 하는게 아니라 (1,1) tensor로 반환 topk의 반환은 다 tensor임
    top_n, top_i =  output.topk(1)
    # 가장 softmax값이 큰 인덱스가 몇 번째 index인지 알아내고 그걸 category에서 찾음
    category_i = top_i[0].item()
    return all_categories[category_i], category_i


import random

# 입력으로 넣은 요소들 중 하나를 랜덤으로 추출
# jaejun -> a 막 이렇게나 랜덤으로
def randomChoice(l):
    return l[random.randint(0, len(l)-1)]

def randomTrainingExamples():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])

    # 그냥 카테코리의 넘버에 해당하는 숫자를 가진 tensor를 만듬
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)

    return category, line, category_tensor, line_tensor



# nn.logsoftmax로 softmax를 쓸거면 loss function은 NLLLoss가 적합하다.
criterion = nn.NLLLoss()


learning_rate = 0.005

def train(category_tensor, line_tensor):
    hidden = rnn.init_hidden()
    rnn.zero_grad()

    #띄어쓰기 포함 단어의 알파벳 수
    for i in range(line_tensor.size()[0]):
        # hidden은 계속 바뀌고
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


import time
import math
n_iters = 100000
print_every = 5000
plot_every = 1000


current_loss = 0
all_losses = []


def timeSince(since):
    now = time.time()
    s = now - since
    # floor는 내림
    m = math.floor(s/60)
    s -= m*60

    return '%dm %ds' % (m,s)

start = time.time()


for iter in range(1, n_iters-1):
    category, line, category_tensor, line_tensor = randomTrainingExamples()
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss
    
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)


confusion = torch.zeros(n_categories, n_categories)
n_confusion = 10000

def evaluate(line_tensor):
    hidden = rnn.init_hidden()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output


for i in range(n_confusion):
    category, line, category_tensor, line_tensor = randomTrainingExamples()
    output = evaluate(line_tensor)
    quess, quess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][category_i] += 1

for i in range(n_categories):
    confusion[i] = confusion[i]/confusion[i].sum()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()

def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)

    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

predict('Dovesky')
predict('Jackson')
predict('Satoshi')
predict('jaejun')

