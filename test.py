import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(10,10),
    nn.ReLU(),
    nn.Linear(10,1),
)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)


scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

num_epochs = 10
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)
print(inputs.shape)
print(targets.shape)
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)
    print(outputs.shape)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch+1)%10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


