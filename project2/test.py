import torch
from torch.autograd import Variable

# datele noastre
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# parametrul w
w = Variable(torch.Tensor([1.0]), requires_grad=True)

def forward(x):
    return x * w

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

print("predict (before training)", 4, forward(4).item())

# hyperparameters
lr = 0.01
epochs = 100

for epoch in range(epochs):
    for x_val, y_val in zip(x_data, y_data):
        # 1) loss și backprop
        l = loss(x_val, y_val)
        l.backward()

        # afișăm gradientul
        print("\tgrad:", x_val, y_val, w.grad.data[0])

        # 2) actualizare manuală folosind exact linia cerută,
        #    dar în interiorul torch.no_grad()
        with torch.no_grad():
            w.data = w.data - lr * w.grad.data

        # 3) resetăm gradientul pentru următoare iterație
        w.grad.data.zero_()

    # afișăm progresul după fiecare epocă
    print("progress:", epoch, "w =", w.data[0].item(), "loss =", l.item())

print("predict (after training)", 4, forward(4).item())
