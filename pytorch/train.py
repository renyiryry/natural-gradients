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
import time
import copy




np.random.seed(9999)
torch.manual_seed(9999)

mnist = input_data.read_data_sets('../MNIST_data', one_hot=False)

X_test = mnist.test.images
t_test = mnist.test.labels


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        
        self.layersizes = [784, 200, 100, 10]
        self.numlayers = len(self.layersizes) - 1
        
#         self.fc = self.numlayers * [0]
        self.fc = list(range(self.numlayers))
#         for l in range(self.numlayers):
#             self.fc[l] = nn.Linear(layersizes[l], layersizes[l+1], bias=False)
            
        self.fc[0] = nn.Linear(self.layersizes[0], self.layersizes[0+1], bias=False)
        self.fc[1] = nn.Linear(self.layersizes[1], self.layersizes[1+1], bias=False)
        self.fc[2] = nn.Linear(self.layersizes[2], self.layersizes[2+1], bias=False)
        
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
parser.add_argument('--max_iter', type=int)
parser.add_argument('N1', type=int)
parser.add_argument('N2', type=int)
parser.add_argument('--alpha', type=float)
parser.add_argument('--lambda_', type=float)
args = parser.parse_args()
# print args.accumulate(args.algorithm)
algorithm = args.algorithm
params['algorithm'] = algorithm
max_iter = args.max_iter

if algorithm == 'kfac' or algorithm == 'SMW-Fisher' or algorithm == 'SMW-Fisher-momentum':
    lambda_ = args.lambda_
    params['lambda_'] = lambda_
    boost = 1.01
    drop = 1 / 1.01
    params['boost'] = boost
    params['drop'] = drop
elif algorithm == 'SGD':
    1
else:
    print('Error!')
    sys.exit()

# N1 = 128  # mini-batch size (for gradient)
N1 = args.N1
N2 = args.N2

if N2 > N1:
    print('Error!')
    sys.exit()



# alpha = 0.001
alpha = args.alpha
# eps = 1e-2
inverse_update_freq = 20


params['N1'] = N1
params['N2'] = N2
params['inverse_update_freq'] = inverse_update_freq

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
    
#     print('A[0].size(): ', A[0].size())
    
    A_inv, G_inv = 3*[0], 3*[0]
    
    
    data_['A'] = A
    data_['G'] = G
    data_['A_inv'] = A_inv
    data_['G_inv'] = G_inv

elif params['algorithm'] == 'SMW-Fisher-momentum':
    a_grad_momentum = []
    h_momentum = []
    
    layersizes = model.layersizes
    
    for l in range(model.numlayers):
        a_grad_momentum.append(torch.zeros(N2, layersizes[l+1]))
        h_momentum.append(torch.zeros(N2, layersizes[l]))
        
    data_['a_grad_momentum'] = a_grad_momentum
    data_['h_momentum'] = h_momentum
elif params['algorithm'] == 'SMW-Fisher' or algorithm == 'SGD':
    1;
else:
    print('Error!')
    sys.exit()

    



# Visualization stuffs
losses = []
times = []

# max_iter = 5000
# max_iter = 5

# Training
for i in range(1, max_iter):
    X_mb, t_mb = mnist.train.next_batch(N1)
    X_mb, t_mb = torch.from_numpy(X_mb), torch.from_numpy(t_mb).long()

    # Forward
    
#     print('i: ', i)
#     print('X_mb: ', X_mb)
    
#     print('model.W[1]: ', model.W[1])
    
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

    get_model_grad_zerod(model)
    
    loss.backward(retain_graph=True)
    
    
    
    ###########
    # test
    
#     loss = nn.MSELoss(z, t_mb)
#     loss.backward()
    
#     print('test')
    
    ###########
    
#     print('model.W[1]: ', model.W[1])
    
#     print('z.grad: ', z.grad)
    
#     print('h1.grad: ', cache[1].grad)
#     print('h2.grad: ', cache[3].grad)
    
#     print('h1 in train: ', cache[1])
#     print('h2: ', cache[3])
    
#     print('a1.grad: ', cache[0].grad)
#     print('a2.grad in train: ', cache[2].grad)
    
#     print('X_mb.grad: ', X_mb.grad)
    
#     print('X_mb: ', X_mb)
    
#     print('a2.grad.size: ', cache[2].grad.size())
#     print('h1.size: ',cache[1].size())
    
#     print('1/m * cache[1].t() @ cache[2].grad: ', 1/m * cache[1].t() @ cache[2].grad)
    
#     print('cache[2].grad.t() @ cache[1] in train: ', cache[2].grad.t() @ cache[1])
    
#     print('model.W[1].grad.data: ', model.W[1].grad.data)
    
#     print('model.W[1].grad in train: ', model.W[1].grad)
    
#     print('model.fc[1].weight.grad: ', model.fc[1].weight.grad)

    if (i-1) % 100 == 0:
        
#         print(z)
        
#         print(t_mb)
        
        print(f'Iter-{i-1}; Loss: {loss:.3f}')
        if algorithm == 'SMW-Fisher' or algorithm == 'SMW-Fisher-momentum' or algorithm == 'kfac':
            lambda_ = params['lambda_']
            print('lambda = ', lambda_)
            print('\n')

    losses.append(loss if i == 1 else 0.99*losses[-1] + 0.01*loss)
    

    
    
    data_['X_mb'] = X_mb
    data_['loss'] = loss
    data_['t_mb'] = t_mb
    
#     data_['a1'] = a1
#     data_['a2'] = a2
#     data_['h1'] = h1
#     data_['h2'] = h2
    data_['cache'] = cache
    
    data_['z'] = z
    
    params['i'] = i
    
#     print(data_['model'])

    model = data_['model']

    start_time = time.time()
    
    
    p = []
    for l in range(numlayers):
        p.append(copy.deepcopy(model.W[l].grad))   
    lambda_ = update_lambda(p, data_, params)
    print('test')

    
    if params['algorithm'] == 'kfac':    
        
    
    
    
        data_, params = kfac_update(data_, params)
    
        
    elif params['algorithm'] == 'SMW-Fisher' or params['algorithm'] == 'SMW-Fisher-momentum':
        
        
        
        data_, params = SMW_Fisher_update(data_, params)
    elif algorithm == 'SGD':
        data_ = SGD_update(data_, params)
    else:
        print('Error!')
        sys.exit()
        
    p = data_['p']
    
    
    if algorithm == 'kfac' or algorithm == 'SMW-Fisher' or algorithm == 'SMW-Fisher-momentum':
        
        lambda_ = update_lambda(p, data_, params)
        params['lambda_'] = lambda_
        
    model = update_parameter(p, model, params)
        

    times.append(time.time() - start_time)
        
    
        
    get_model_grad_zerod(model)

    


z, _ = model.forward(torch.from_numpy(X_test))
y = z.argmax(dim=1)
acc = np.mean(y.numpy() == t_test)

print(f'Accuracy: {acc:.3f}')
# np.save('temp/kfac_losses.npy', losses)
# np.save('/content/logs/temp/kfac_losses.npy', losses)
np.save('/content/logs/temp/' + algorithm + '_losses.npy', losses)
np.save('/content/logs/temp/' + algorithm + '_times.npy', times)

np.save('/content/gdrive/My Drive/Gauss_Newton/result/' + algorithm + '_losses.npy', losses)
np.save('/content/gdrive/My Drive/Gauss_Newton/result/' + algorithm + '_times.npy', times)


