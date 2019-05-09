import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import numpy as np
import input_data
from sklearn.utils import shuffle
import warnings
warnings.filterwarnings('error')

from utils import *
import argparse
import sys




np.random.seed(9999)
torch.manual_seed(9999)

mnist = input_data.read_data_sets('../MNIST_data', one_hot=False)

X_test = mnist.test.images
t_test = mnist.test.labels


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        
        layersizes = [784, 200, 100, 10]
        self.numlayers = len(layersizes) - 1
        
        self.fc = []
        for l in range(0, self.numlayers):
            self.fc.append(nn.Linear(layersizes[l], layersizes[l+1], bias=False))
        
        

        self.fc1 = nn.Linear(784, 200, bias=False)
        self.fc2 = nn.Linear(200, 100, bias=False)
        self.fc3 = nn.Linear(100, 10, bias=False)

#         self.W = [self.fc1.weight, self.fc2.weight, self.fc3.weight]       
        self.W = [fci.weight for fci in self.fc]

    def forward(self, x):
        a1 = self.fc1(x)
        h1 = F.relu(a1)
        a2 = self.fc2(h1)
        h2 = F.relu(a2)
        z = self.fc3(h2)
        
        a = []
        h = []
        for l in range(0, self.numlayers - 1):
            if l == 0:
                a.append(self.fc[l](x))
            else:
                a.append(self.fc[l](a[-1]))
            h.append(F.relu(a[-1]))
        z = self.fc[-1](h[-1])
            

        cache = (a1, h1, a2, h2)
            
        cache = []
        for l in range(0, self.numlayers - 1):
            cache.append((a[l],h[l]))
        cache = tuple(acche) 

        z.retain_grad()
        for c in cache:
            c.retain_grad()

        return z, cache

    
params = {}

m = 128  # mb size
alpha = 0.001
eps = 1e-2
inverse_update_freq = 20


params['m'] = m
params['inverse_update_freq'] = inverse_update_freq
params['eps'] = eps
params['alpha'] = alpha

parser = argparse.ArgumentParser()
parser.add_argument('algorithm', type=str, )
args = parser.parse_args()
# print args.accumulate(args.algorithm)
params['algorithm'] = args.algorithm




data_ = {}

# Model
model = Model()

data_['model'] = model



if params['algorithm'] == 'kfac':

    A = []  # KFAC A
    G = []  # KFAC G

    for Wi in model.W:
        A.append(torch.zeros(Wi.size(1)))
        G.append(torch.zeros(Wi.size(0)))
    
    A_inv, G_inv = 3*[0], 3*[0]
    
    
    data_['A'] = A
    data_['G'] = G
    data_['A_inv'] = A_inv
    data_['G_inv'] = G_inv
    



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
        print(f'Iter-{i-1}; Loss: {loss:.3f}')

    losses.append(loss if i == 1 else 0.99*losses[-1] + 0.01*loss)

    """
    # KFAC matrices
    G1_ = 1/m * a1.grad.t() @ a1.grad
    A1_ = 1/m * X_mb.t() @ X_mb
    G2_ = 1/m * a2.grad.t() @ a2.grad
    A2_ = 1/m * h1.t() @ h1
    G3_ = 1/m * z.grad.t() @ z.grad
    A3_ = 1/m * h2.t() @ h2

    G_ = [G1_, G2_, G3_]
    A_ = [A1_, A2_, A3_]

    # Update running estimates of KFAC
    rho = min(1-1/i, 0.95)

    for k in range(3):
        A[k] = rho*A[k] + (1-rho)*A_[k]
        G[k] = rho*G[k] + (1-rho)*G_[k]

    # Step
    for k in range(3):
        # Amortize the inverse. Only update inverses every now and then
        if (i-1) % inverse_update_freq == 0:
            A_inv[k] = (A[k] + eps*torch.eye(A[k].shape[0])).inverse()
            G_inv[k] = (G[k] + eps*torch.eye(G[k].shape[0])).inverse()

        delta = G_inv[k] @ model.W[k].grad.data @ A_inv[k]
        model.W[k].data -= alpha * delta
    """
    
    data_['X_mb'] = X_mb
    
#     data_['a1'] = a1
#     data_['a2'] = a2
#     data_['h1'] = h1
#     data_['h2'] = h2
    data_['cache'] = cache
    
    data_['z'] = z
    
    if params['algorithm'] == 'kfac':    
        params['i'] = i
    
    
    
        data_ = kfac_update(data_, params)
    
        model = data_['model']
    elif params['algorithm'] == 'SMW-Fisher':
        data_ = SMW_Fisher_update(data_, params)
    else:
        print('Error!')
        sys.exit()
        
    

    # PyTorch stuffs
    model.zero_grad()


z, _ = model.forward(torch.from_numpy(X_test))
y = z.argmax(dim=1)
acc = np.mean(y.numpy() == t_test)

print(f'Accuracy: {acc:.3f}')
# np.save('temp/kfac_losses.npy', losses)
np.save('/content/logs/temp/kfac_losses.npy', losses)

