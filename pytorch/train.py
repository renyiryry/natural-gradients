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
        
#         self.fc = self.numlayers * [0]
        self.fc = list(range(self.numlayers))
#         for l in range(self.numlayers):
#             self.fc[l] = nn.Linear(layersizes[l], layersizes[l+1], bias=False)
            
        self.fc[0] = nn.Linear(layersizes[0], layersizes[0+1], bias=False)
        self.fc[1] = nn.Linear(layersizes[1], layersizes[1+1], bias=False)
        self.fc[2] = nn.Linear(layersizes[2], layersizes[2+1], bias=False)
        
        self.fc = tuple(self.fc)
        
        

#         self.fc1 = nn.Linear(784, 200, bias=False)
#         self.fc2 = nn.Linear(200, 100, bias=False)
#         self.fc3 = nn.Linear(100, 10, bias=False)

#         self.W = [self.fc1.weight, self.fc2.weight, self.fc3.weight]  
#         self.W = self.numlayers * [0]
#         for l in range(self.numlayers):
#             self.W[l] = self.fc[l].weight
#         self.W = [fci.weight for fci in self.fc]

#         self.W = []
#         for l in range(self.numlayers):
#             self.W.append(self.fc[l].weight)
            
        self.W = list(range(3))
        self.W[0] = self.fc[0].weight
        self.W[1] = self.fc[1].weight
        self.W[2] = self.fc[2].weight
        
        self.W = tuple(self.W)
    
#         print('size(self.W[0]): ', self.W[0].numel())
#         print('size(self.W[1]): ', self.W[1].numel())
#         print('size(self.W[2]): ', self.W[2].numel())

    def forward(self, x):
#         a1 = self.fc1(x)
#         h1 = F.relu(a1)
#         a2 = self.fc2(h1)
#         h2 = F.relu(a2)
#         z = self.fc3(h2)
        
#         a = (self.numlayers - 1) * [0]
#         h = (self.numlayers - 1) * [0]
#         for l in range(self.numlayers - 1):
#             if l == 0:
#                 a[l] = self.fc[l](x)
#             else:
#                 a[l] = self.fc[l](h[l-1])
#             h[l] = F.relu(a[l])
            
            
        
        
#         a = []
#         h = []
#         for l in range(self.numlayers - 1):
#             if l == 0:
#                 a.append(self.fc[l](x))
#             else:
#                 a.append(self.fc[l](h[l-1]))
#             h.append(F.relu(a[l]))
            
        a = list(range(2))
        h = list(range(2))
        a[0] = self.fc[0](x)
        h[0] = F.relu(a[0])
        a[1] = self.fc[1](h[0])
        h[1] = F.relu(a[1])
        
        a = tuple(a)
        h = tuple(h)
            
        z = self.fc[2](h[1])
            

#         cache = (a1, h1, a2, h2)
            
        cache = ((self.numlayers - 1)) * 2 * [0]
        for l in range(0, self.numlayers - 1):
#             cache = cache + [a[l],h[l]]
#             cache.append([a[l],h[l]])
            cache[2*l] = a[l]
            cache[2*l+1] = h[l]
        cache = tuple(cache) 
    
#         print('len(cache): ', len(cache))

        z.retain_grad()
        for c in cache:
            c.retain_grad()

        return z, cache

# Model
model = Model()

# print('model: ', model)
    
params = {}

parser = argparse.ArgumentParser()
parser.add_argument('algorithm', type=str)
parser.add_argument('max_iter', type=int)
args = parser.parse_args()
# print args.accumulate(args.algorithm)
params['algorithm'] = args.algorithm
max_iter = args.max_iter

N1 = 128  # mini-batch size (for gradient)

if params['algorithm'] == 'kfac':
    N2 = 128
elif params['algorithm'] == 'SMW-Fisher':
    N2 = 64
else:
    print('Error!')
    sys.exit()

alpha = 0.001
eps = 1e-2
inverse_update_freq = 20


params['N1'] = N1
params['N2'] = N2
params['inverse_update_freq'] = inverse_update_freq
params['eps'] = eps
params['alpha'] = alpha
params['numlayers'] = model.numlayers






data_ = {}

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

# max_iter = 5000
# max_iter = 5

# Training
for i in range(1, max_iter):
    X_mb, t_mb = mnist.train.next_batch(m)
    X_mb, t_mb = torch.from_numpy(X_mb), torch.from_numpy(t_mb).long()

    # Forward
    
#     print('i: ', i)
#     print('X_mb: ', X_mb)
    
#     print('model.W[0]: ', model.W[0])
    
#     print('model.W: ', model.W)
    
    z, cache = model.forward(X_mb)
    
#     a1, h1, a2, h2 = cache

    # Loss
    
    
    
    
#     print('a1: ', cache[0])
#     print('a2: ', cache[2])
#     print('z: ', z)
    
    loss = F.cross_entropy(z, t_mb)
    
#     print('loss: ', loss)
    
#     model.zero_grad()

#     print('model.W[1]: ', model.W[1])
#     print('model.W[1].grad.data: ', model.W[1].grad.data)
    
    loss.backward()
    
#     print('model.W[1]: ', model.W[1])
    
#     print('z.grad: ', z.grad)
    
#     print('h1.grad: ', cache[1].grad)
#     print('h2.grad: ', cache[3].grad)
    
#     print('h1: ', cache[1])
#     print('h2: ', cache[3])
    
#     print('a1.grad: ', cache[0].grad)
#     print('a2.grad: ', cache[2].grad)
    
#     print('X_mb.grad: ', X_mb.grad)
    
#     print('X_mb: ', X_mb)
    
#     print('a2.grad.size: ', cache[2].grad.size())
#     print('h1.size: ',cache[1].size())
    
#     print('1/m * cache[1].t() @ cache[2].grad: ', 1/m * cache[1].t() @ cache[2].grad)
    
#     print('cache[2].grad.t() @ cache[1]: ', cache[2].grad.t() @ cache[1])
    
#     print('model.W[1].grad.data: ', model.W[1].grad.data)
    
#     print('model.W[1].grad: ', model.W[1].grad)
    
#     print('model.fc[1].weight.grad: ', model.fc[1].weight.grad)

    if (i-1) % 100 == 0:
        
#         print(z)
        
#         print(t_mb)
        
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
#     print('gradeint zerod')
    
#     print('model.fc[1].weight.grad: ', model.fc[1].weight.grad)
    
    for fci in model.fc:
        fci.zero_grad()
        
#     print('model.fc[1].weight.grad: ', model.fc[1].weight.grad)


z, _ = model.forward(torch.from_numpy(X_test))
y = z.argmax(dim=1)
acc = np.mean(y.numpy() == t_test)

print(f'Accuracy: {acc:.3f}')
# np.save('temp/kfac_losses.npy', losses)
np.save('/content/logs/temp/kfac_losses.npy', losses)

