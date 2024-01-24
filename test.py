import torch
import torch.nn as nn


def get_angles(position, i, d_model):
    # 이 함수는 각 position과 i에 대해 계산된 각도를 반환합니다.
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / torch.tensor(d_model, dtype=torch.float32))
    return position * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(
        position=torch.arange(position, dtype=torch.float32).unsqueeze(1),
        i=torch.arange(d_model, dtype=torch.float32).unsqueeze(0),
        d_model=d_model)

    return angle_rads

pos, d_model = 50, 128
position = torch.arange(pos, dtype=torch.float32).unsqueeze(1)
i=torch.arange(d_model, dtype=torch.float32).unsqueeze(0)


# print(positional_encoding(pos, d_model))
array = torch.arange(9)
print(2*array)