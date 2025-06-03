import torch
from torch.autograd import Variable

# 1) Datele noastre
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))

# 2) Modelul
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = Model()

# 3) Loss & Optimizer
criterion = torch.nn.MSELoss(reduction='sum')      # înlocuiește size_average=False
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 4) Training loop
for epoch in range(500):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    # afișăm scalari cu .item()
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 5) după antrenament
hour_var = Variable(torch.Tensor([[4.0]]))
pred = model(hour_var)
print("predict (after training)", 4, pred.item())
