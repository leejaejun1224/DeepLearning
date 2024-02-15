import torch
import torch.nn as nn
from transformer import Transformer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import re
import urllib.request
import pandas as pd


# BPE 토크나이저 및 트레이너 초기화

urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")
train_data = pd.read_csv('ChatBotData.csv')
train_data.head()
# print('챗봇 샘플의 개수 :', len(train_data))

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

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=2**13)
tokenizer.train_from_iterator(questions + answers, trainer=trainer)

num_epochs = 50
num_vocabs = 9000
num_layers = 2
dff = 512
d_model = 256
num_heads = 8
dropout=0.1
model = Transformer(d_model, num_heads, dff, num_vocabs, dropout, num_layers)
model.load_state_dict(torch.load('./weights/model.pth'))
model.eval()

VOCAB_SIZE = tokenizer.get_vocab_size()
START_TOKEN = VOCAB_SIZE
END_TOKEN = VOCAB_SIZE + 1
# 특수 토큰을 추가했으므로, 추가된 특수 토큰의 수를 고려
VOCAB_SIZE += 2
MAX_LENGTH = 40

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
    with torch.no_grad():
        for i in range(MAX_LENGTH):
            predictions = model(sentence, output)

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
    predicted_sentence = tokenizer.decode([i for i in prediction if i < VOCAB_SIZE])

    print('Input: {}'.format(sentence))
    print('Output: {}'.format(predicted_sentence))

    return predicted_sentence

predict("살을 빼야해")
