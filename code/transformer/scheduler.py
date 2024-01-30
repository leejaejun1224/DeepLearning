import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Adam
import matplotlib.pyplot as plt

# class CustomSchedule(_LRScheduler):
#     def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch=-1):
#         self.d_model = d_model
#         self.warmup_steps = warmup_steps
#         super(CustomSchedule, self).__init__(optimizer, last_epoch)

#     def get_lr(self):
#         step = max(self._step_count, 1)
#         arg1 = step ** -0.5
#         arg2 = step * (self.warmup_steps ** -1.5)

#         return [base_lr * (self.d_model ** -0.5) * min(arg1, arg2) for base_lr in self.base_lrs]

class CustomSchedule():
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = torch.tensor(d_model, dtype=torch.float32).rsqrt()  # 미리 계산
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        if isinstance(step, torch.Tensor):
            step = step.clone().detach().float()  # 이미 텐서라면 clone().detach() 사용
        else:
            step = torch.tensor(step, dtype=torch.float32)
        arg1 = torch.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return self.d_model * torch.min(arg1, arg2)

# sample_learning_rate = CustomSchedule(d_model=128)

# plt.plot(sample_learning_rate(torch.arange(200000, dtype=torch.float32)))
# plt.ylabel("Learning Rate")
# plt.xlabel("Train Step")
# plt.savefig('/container/home/DeepLearning/code/transformer/custom_learning_rate.png')