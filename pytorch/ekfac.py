"""
Fast Approximate Natural Gradient Descent in a Kronecker-factored Eigenbasis
George, Thomas, et al. 2018
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import input_data
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('error')


np.random.seed(9999)
torch.manual_seed(9999)

mnist = input_data.read_data_sets('../MNIST_data', one_hot=False)

X_test = mnist.test.images
t_test = mnist.test.labels


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(784, 200, bias=False)
        self.fc2 = nn.Linear(200, 100, bias=False)
        self.fc3 = nn.Linear(100, 10, bias=False)

        self.W = [self.fc1.weight, self.fc2.weight, self.fc3.weight]

    def forward(self, x):
        a1 = self.fc1(x)
        h1 = F.relu(a1)
        a2 = self.fc2(h1)
        h2 = F.relu(a2)
        z = self.fc3(h2)

        cache = (a1, h1, a2, h2)

        z.retain_grad()
        for c in cache:
            c.retain_grad()

        return z, cache


# Model
model = Model()

m = 128  # mb size
alpha = 0.001

A = []  # KFAC A
G = []  # KFAC G

for Wi in model.W:
    A.append(torch.zeros(Wi.size(1)))
    G.append(torch.zeros(Wi.size(0)))

eps = 1e-3
eigbasis_comp_freq = 20

# Visualization stuffs
losses = []

# Training
for i in range(1, 5000):
    X_mb, t_mb = mnist.train.next_batch(m)
    X_mb, t_mb = torch.from_numpy(X_mb), torch.from_numpy(t_mb).long()

    # Forward
    z, cache = model.forward(X_mb)
    a1, h1, a2, h2 = cache

    # Loss
    loss = F.cross_entropy(z, t_mb)
    loss.backward()

    if (i-1) % 100 == 0:
        print(f'Iter-{i}; Loss: {loss:.3f}')

    losses.append(loss if i == 1 else 0.99*losses[-1] + 0.01*loss)

    # Compute eigenbasis
    if (i-1) % eigbasis_comp_freq == 0:
        G1_ = 1/m * a1.grad.t() @ a1.grad
        A1_ = 1/m * X_mb.t() @ X_mb
        G2_ = 1/m * a2.grad.t() @ a2.grad
        A2_ = 1/m * h1.t() @ h1
        G3_ = 1/m * z.grad.t() @ z.grad
        A3_ = 1/m * h2.t() @ h2

        A_ = [A1_, A2_, A3_]
        G_ = [G1_, G2_, G3_]

         # Update running estimates of KFAC
        rho = min(1-1/i, 0.95)

        for k in range(3):
            A[k] = rho*A[k] + (1-rho)*A_[k]
            G[k] = rho*G[k] + (1-rho)*G_[k]

        # Compute eigenbases
        U_A = []
        U_G = []

        for k in range(3):
            _, U_Ak = torch.symeig(A[k], eigenvectors=True)
            _, U_Gk = torch.symeig(G[k], eigenvectors=True)

            U_A.append(U_Ak)
            U_G.append(U_Gk)

    # Step
    for k in range(3):
        # Compute diagonal scaling
        s = (U_G[k] @ model.W[k].grad.data @ U_A[k].t())**2

        # Apply diagonal scaling in eigenbasis
        delta = U_G[k] @ model.W[k].grad.data @ U_A[k].t()
        delta /= s + eps

        # Project back to original basis
        delta = U_G[k].t() @ delta @ U_A[k]

        # Update parameters
        model.W[k].data -= alpha * delta

    # PyTorch stuffs
    model.zero_grad()


z, _ = model.forward(torch.from_numpy(X_test))
y = z.argmax(dim=1)
acc = np.mean(y.numpy() == t_test)

print(f'Accuracy: {acc:.3f}')
# np.save('temp/ekfac_losses.npy', losses)
np.save('/content/logs/temp/ekfac_losses.npy', losses)

