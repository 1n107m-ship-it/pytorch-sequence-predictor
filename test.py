import torch
import random
import torch.nn as nn
import torch.optim as optim

data = []
label = []

training_size = 1000

for i in range(training_size):
    d = []
    rando_batch = random.randint(1, 50)
    rando_loap = random.randint(1, 15)
    for v in range(4):
        g = v + 1
        d.append(rando_loap + g * rando_batch)
    data.append(d)
    label.append(rando_loap + 5 * rando_batch)

data = torch.tensor(data, dtype=torch.float32)
label = torch.tensor(label, dtype=torch.float32).unsqueeze(1)

class SeqPredictAI(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        x = self.sequential_layers(x)
        return x

model = SeqPredictAI()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5000):
    output = model(data)
    loss = criterion(output, label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

ccss = list(map(int, input().split()))
ffcc = torch.tensor(ccss, dtype=torch.float32).unsqueeze(0)
print(f"Accurate answer :: {round(model(ffcc).item(), 0)}")
