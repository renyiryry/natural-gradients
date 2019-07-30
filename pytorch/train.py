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




class Model_3(nn.Module):

    def __init__(self, activation, name_dataset):
        super(Model_3, self).__init__()
        
#         self.layersizes = [784, 200, 100, 10]

        self.activation = activation

#         self.layersizes = [784, 400, 400, 10]
        if name_dataset == 'MNIST':
            self.layersizes = [784, 500, 10]
        elif name_dataset == 'CIFAR':
            self.layersizes = [3072, 400, 400, 10]
        elif name_dataset == 'MNIST-autoencoder':
            # reference: https://arxiv.org/pdf/1301.3641.pdf
            self.layersizes = [784, 1000, 500, 250, 30, 250, 500, 1000, 784]
        else:
            print('Dateset not supported!')
            sys.exit()
            
        self.numlayers = len(self.layersizes) - 1
        
        self.fc = list(range(self.numlayers))
        for l in range(self.numlayers):
            self.fc[l] = nn.Linear(self.layersizes[l], self.layersizes[l+1], bias=False)
        
        self.fc = tuple(self.fc)

 



            
        self.W = list(range(self.numlayers))
        for l in range(self.numlayers):
            self.W[l] = self.fc[l].weight
#         self.W[1] = self.fc[1].weight
#         self.W[2] = self.fc[2].weight
        
        self.W = tuple(self.W)


    def forward(self, x):
        
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
            
        a = list(range(self.numlayers - 1))
        h = list(range(self.numlayers - 1))
        
        for l in range(self.numlayers - 1):
            if l == 0:
                a[l] = self.fc[l](x)
            else:
                a[l] = self.fc[l](h[l-1])
            if self.activation == 'relu':
                h[l] = F.relu(a[l])
            elif self.activation == 'sigmoid':
                h[l] = torch.sigmoid(a[l])
            
        z = self.fc[-1](h[-1])
        
#         loss = F.cross_entropy(z, t, reduction = 'none')
#         weighted_loss = torch.dot(loss, v)

        z.retain_grad()
#         for c in cache:
#             c.retain_grad()
        for c in a:
            c.retain_grad()
        for c in h:
            c.retain_grad()
        
        h = [x] + h
        a = a + [z]
        

        return z, a, h










    
params = {}

parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str)
parser.add_argument('--matrix_name', type=str)
parser.add_argument('--momentum_gradient', type=int)
parser.add_argument('--max_epoch', type=float)
parser.add_argument('--record_epoch', type=float)
parser.add_argument('--N1', type=int)
parser.add_argument('--N2', type=int)
parser.add_argument('--alpha', type=float)
parser.add_argument('--lambda_', type=float)
parser.add_argument('--inverse_update_freq', type=int)
parser.add_argument('--inverse_update_freq_D_t', type=int)
parser.add_argument('--rho_kfac', type=float)
parser.add_argument('--activation', type=str)
parser.add_argument('--dataset', type=str)
args = parser.parse_args()
# print args.accumulate(args.algorithm)
algorithm = args.algorithm
matrix_name = args.matrix_name
params['algorithm'] = algorithm
max_epoch = args.max_epoch
record_epoch = args.record_epoch
inverse_update_freq = args.inverse_update_freq
inverse_update_freq_D_t = args.inverse_update_freq_D_t
rho_kfac = args.rho_kfac
activation = args.activation
name_dataset = args.dataset

if_momentum_gradient = args.momentum_gradient
if if_momentum_gradient != 0 and if_momentum_gradient != 1:
    print('if_momentum_gradient')
    print(if_momentum_gradient)
    print('Error!')
    sys.exit()
if_momentum_gradient = bool(if_momentum_gradient)




dataset = input_data.read_data_sets(name_dataset, one_hot=False)



X_test = dataset.test.images
t_test = dataset.test.labels
X_train = dataset.train.images
t_train = dataset.train.labels
X_train, t_train = torch.from_numpy(X_train), torch.from_numpy(t_train).long()

print('X_train.shape')
print(X_train.shape)
print('t_train.shape')
print(t_train.shape)







# Model
model = Model_3(activation, name_dataset)

print('Model created.')

# print('model.W[1] when initialize: ', model.W[1])

params['layersizes'] = model.layersizes

if algorithm == 'kfac' or algorithm == 'SMW-Fisher' or algorithm == 'SMW-Fisher-momentum' or algorithm == 'SMW-GN'\
    or algorithm == 'Fisher-block' or algorithm == 'SMW-Fisher-D_t-momentum'\
    or algorithm == 'SMW-Fisher-momentum-D_t-momentum':
    init_lambda_ = args.lambda_
    params['lambda_'] = init_lambda_
    boost = 1.01
    drop = 1 / 1.01
    params['boost'] = boost
    params['drop'] = drop
elif algorithm == 'SGD':
    init_lambda_ = args.lambda_
    params['lambda_'] = init_lambda_
else:
    print('Error: algorithm not defined.')
    sys.exit()

# N1 = 128  # mini-batch size (for gradient)
N1 = args.N1
N2 = args.N2

if N2 > N1:
    print('Error! 1432')
    sys.exit()

alpha = args.alpha
# eps = 1e-2



params['N1'] = N1
params['N2'] = N2
params['inverse_update_freq'] = inverse_update_freq
params['inverse_update_freq_D_t'] = inverse_update_freq_D_t
params['rho_kfac'] = rho_kfac

params['alpha'] = alpha
params['numlayers'] = model.numlayers






data_ = {}

data_['model'] = model



if params['algorithm'] == 'kfac' or algorithm == 'Fisher-block':

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
    
    D_t_inv = np.zeros((N2, N2))
    data_['D_t_inv'] = D_t_inv

elif params['algorithm'] == 'SMW-Fisher-D_t-momentum':
    data_['J_J_transpose'] = np.float32(np.zeros((N2, N2)))
    
elif params['algorithm'] == 'SMW-Fisher-momentum-D_t-momentum':
    a_grad_momentum = []
    h_momentum = []
    
    layersizes = model.layersizes
    
    for l in range(model.numlayers):
        a_grad_momentum.append(torch.zeros(N2, layersizes[l+1]))
        h_momentum.append(torch.zeros(N2, layersizes[l]))
        
    data_['a_grad_momentum'] = a_grad_momentum
    data_['h_momentum'] = h_momentum
    
    data_['J_J_transpose'] = np.float32(np.zeros((N2, N2)))
    
    
elif algorithm == 'SMW-Fisher' or algorithm == 'SGD' or algorithm == 'SMW-GN':
    1;
else:
    print('Error: algorithm not defined.')
    sys.exit()
    
if if_momentum_gradient:
    
#     print('model.W[1].size')
#     print(model.W[1].size())
    
#     print(params['layersizes'])
    
    data_['model_grad'] = get_zero(params)
    

    



# Visualization stuffs
len_record = int(max_epoch / record_epoch)
losses = np.zeros(len_record + 1)
acces = np.zeros(len_record + 1)
times = np.zeros(len_record + 1)
epochs = np.zeros(len_record + 1)
lambdas = np.zeros(len_record + 1)

acces[0] = get_acc(model, X_test, t_test)
losses[0] = get_loss(model, X_train, t_train)
times[0] = 0
epochs[0] = 0
lambdas[0] = params['lambda_']


# times[0] = 0

iter_per_epoch = int(len(t_train) / N1)

iter_per_record = int(len(t_train) * record_epoch / N1)

# Training
print('Begin training...')
epoch = -1
for i in range(int(max_epoch * iter_per_epoch)):
    
    if i % iter_per_record == 0:
        start_time = time.time()
        epoch += 1
    
    # get minibatch
    X_mb, t_mb = dataset.train.next_batch(N1)
    X_mb, t_mb = torch.from_numpy(X_mb), torch.from_numpy(t_mb).long()
    
#     print('t_mb.size()')
#     print(t_mb.size())

    # Forward
    z, a, h = model.forward(X_mb)
    
#     print('z.size()')
#     print(z.size())
    
    
    loss = F.cross_entropy(z, t_mb, reduction = 'mean')    
#     loss = F.cross_entropy(z, t_mb)


#     print('torch.sum(a[-1], dim=0).size():', torch.sum(a[-1], dim=0).size())
#     print('torch.sum(a[-1], dim=0):', torch.sum(a[-1], dim=0))
#     print('torch.sum(a[-1], dim=1).size():', torch.sum(a[-1], dim=1).size())
#     print('torch.sum(a[-1], dim=1):', torch.sum(a[-1], dim=1))
#     print('loss: ', loss)


    # backward and gradient
    model = get_model_grad_zerod(model)
    
#         test_start_time = time.time()
    

    

    loss.backward()
    
    
    

#         print('time of loss:', time.time() - test_start_time)

#     print('a[0].grad')
#     print(a[0].grad)



    

    model_grad = []
    for l in range(model.numlayers):
        model_grad.append(copy.deepcopy(model.W[l].grad))
    
    



    if if_momentum_gradient:
        rho = min(1-1/(i+1), 0.9)
        data_['model_grad'] = get_plus(\
                                       get_multiply(rho, data_['model_grad'], params),
                                       get_multiply(1 - rho, model_grad, params),
                                       params)
    else:
        data_['model_grad'] = model_grad
        
    
    
    
    
    

    
    
    

    

    

    
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

    
    # get second order caches

    
    
    data_['X_mb'] = X_mb
    data_['loss'] = loss
    data_['t_mb'] = t_mb
    
    if matrix_name == 'EF':
        data_['t_mb_pred'] = t_mb
    
        data_['a'] = a
        data_['h'] = h
    elif matrix_name == 'Fisher':
        from torch.utils.data import WeightedRandomSampler
        
#         print('F.softmax(z, dim=0)')
#         print(F.softmax(z, dim=0))
        
#         print('torch.sum(F.softmax(z, dim=0), dim=0)')
#         print(torch.sum(F.softmax(z, dim=0), dim=0))
        
#         print('torch.sum(F.softmax(z, dim=0), dim=1)')
#         print(torch.sum(F.softmax(z, dim=0), dim=1))
        
        pred_dist = F.softmax(z, dim=1)
        
#         print('WeightedRandomSampler(pred_dist, 1)')
#         print(len(list(WeightedRandomSampler(pred_dist, 1))))
        
        t_mb_pred = list(WeightedRandomSampler(pred_dist, 1))
        t_mb_pred = np.asarray(t_mb_pred)
        t_mb_pred = np.squeeze(t_mb_pred)
        
#         print(t_mb_pred)
        
        t_mb_pred = torch.from_numpy(t_mb_pred).long()
        
        
#         print('np.sum(t_mb_pred == t_mb) / len(t_mb)')
#         print(np.sum(t_mb_pred.data.numpy() == t_mb.data.numpy()) / len(t_mb))
        
        
        data_['t_mb_pred'] = t_mb_pred
        
#         print('z.size()')
#         print(z.size())
        
#         print('t_mb_pred.size()')
#         print(t_mb_pred.size())
        
#         print('t_mb.size()')
#         print(t_mb.size())

#         print('t_mb_pred')
#         print(t_mb_pred)
        
        z, a, h = model.forward(X_mb)
        loss = F.cross_entropy(z, t_mb_pred, reduction = 'mean')    
        model = get_model_grad_zerod(model)
        loss.backward()
        
#         print('a[0].grad')
#         print(a[0].grad)
        
        
    
        data_['a'] = a
        data_['h'] = h
        
    else:
        print('Error.')
        sys.exit()
    
#         i = epoch * iter_per_epoch + iter_
    params['i'] = i

    model = data_['model']

    
    
    
#     p = []
#     for l in range(model.numlayers):
#         p.append(copy.deepcopy(model.W[l].grad))   
#     lambda_ = update_lambda(p, data_, params)
#     print('test')

#     test_start_time = time.time()
    
#     print('time first half: ', time.time() - start_time)
    
    
    

    
    if algorithm == 'kfac' or algorithm == 'Fisher-block':    
        data_, params = kfac_update(data_, params)
    
        
    elif params['algorithm'] == 'SMW-Fisher' or params['algorithm'] == 'SMW-Fisher-momentum'\
        or params['algorithm'] == 'SMW-Fisher-D_t-momentum'\
        or params['algorithm'] == 'SMW-Fisher-momentum-D_t-momentum':
        
        
        
        data_, params = SMW_Fisher_update(data_, params)
    elif algorithm == 'SGD':
        data_ = SGD_update(data_, params)
    elif algorithm == 'SMW-GN':
        data_ = SMW_GN_update(data_, params)
    else:
        print('Error: algorithm not defined.')
        sys.exit()
        
     
        
        
#     print('time of second order:', time.time() - test_start_time)
    
#     print('time 3/4: ', time.time() - start_time)
        
    p = data_['p']
    
#     print('p[0]: ', p[0])
#     print('p[1]: ', p[1])
#     print('p[2]: ', p[2])
    
    
    if algorithm == 'kfac' or algorithm == 'SMW-Fisher' or algorithm == 'SMW-Fisher-momentum' or algorithm == 'SMW-GN'\
        or algorithm == 'Fisher-block' or algorithm == 'SMW-Fisher-D_t-momentum'\
        or algorithm == 'SMW-Fisher-momentum-D_t-momentum':
        
        lambda_ = update_lambda(p, data_, params)
        
#         lambda_ = init_lambda_
#         print('test')
        
        params['lambda_'] = lambda_
    elif algorithm == 'SGD':
        1
    else:
        print('Error! 1435')
        sys.exit()
#     print('no update lambda')
        
    model = update_parameter(p, model, params)
    
#     print('time 7/8: ', time.time() - start_time)
        
    
    
    if (i+1) % iter_per_record == 0:
        times[epoch+1] = time.time() - start_time
    
#         print('time this iter: ', times[i-1])
    
        if epoch > 0:
            times[epoch+1] = times[epoch+1] + times[epoch]
            
        losses[epoch+1] = get_loss(model, X_train, t_train)
        
        acces[epoch+1] = get_acc(model, X_test, t_test)
        
        epochs[epoch+1] = (epoch + 1) * record_epoch
        
        lambdas[epoch+1] = params['lambda_']
    
        
#         print(z)
        
#         print(t_mb)
            
        
        
        

            
        
    
        
        
        
    

    
#         loss = F.cross_entropy(z, t_train)    
            
        
        
        print(f'Iter-{(epoch+1) * record_epoch:.3f}; Loss: {losses[epoch+1]:.3f}')
        print(f'Accuracy: {acces[epoch+1]:.3f}')
        
        if epoch > 0:
            print('elapsed time: ', times[epoch+1] - times[epoch])
        else:
            print('elapsed time: ', times[epoch+1])
            
        
            
        if algorithm == 'SMW-Fisher' or algorithm == 'SMW-Fisher-momentum' or algorithm == 'kfac'\
            or algorithm == 'SMW-GN'\
            or algorithm == 'SMW-Fisher-momentum-D_t-momentum':
#             lambda_ = params['lambda_']
            print('lambda = ', params['lambda_'])
        elif algorithm == 'SGD':
            1
        else:
            print('Error: algorithm not defined.')
            sys.exit
        
        print('\n')
        
    
        
    model = get_model_grad_zerod(model)

    





# times = np.asarray([0] + [times])
# times = np.insert(times, 0, 0)
# losses = np.insert(losses, init_loss.data, 0)

# np.save('temp/kfac_losses.npy', losses)
# np.save('/content/logs/temp/kfac_losses.npy', losses)

print('Begin saving results...')

name_result = name_dataset + '_' + algorithm +\
'_matrix_name_' + matrix_name +\
'_momentum_gradient_' + str(int(if_momentum_gradient)) +\
'_alpha_' + str(alpha)

np.save('/content/logs/temp/' + name_result + '_losses.npy', losses)
np.save('/content/logs/temp/' + name_result + '_acces.npy', acces)
np.save('/content/logs/temp/' + name_result + '_lambdas.npy', lambdas)
np.save('/content/logs/temp/' + name_result + '_times.npy', times)
np.save('/content/logs/temp/' + name_result + '_epochs.npy', epochs)

np.save('/content/gdrive/My Drive/Gauss_Newton/result/' + name_result + '_losses.npy', losses)
np.save('/content/gdrive/My Drive/Gauss_Newton/result/' + name_result + '_acces.npy', acces)
np.save('/content/gdrive/My Drive/Gauss_Newton/result/' + name_result + '_lambdas.npy', lambdas)
np.save('/content/gdrive/My Drive/Gauss_Newton/result/' + name_result + '_times.npy', times)
np.save('/content/gdrive/My Drive/Gauss_Newton/result/' + name_result + '_epochs.npy', epochs)

print('Saved at /content/gdrive/My Drive/Gauss_Newton/result/' + name_result + '.')










"""
class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        
        self.layersizes = [784, 200, 100, 10]
        self.numlayers = len(self.layersizes) - 1
        
#         self.fc = self.numlayers * [0]
        self.fc = list(range(self.numlayers))
        for l in range(self.numlayers):
            self.fc[l] = nn.Linear(self.layersizes[l], self.layersizes[l+1], bias=False)
            
#         self.fc[0] = nn.Linear(self.layersizes[0], self.layersizes[0+1], bias=False)
#         self.fc[1] = nn.Linear(self.layersizes[1], self.layersizes[1+1], bias=False)
#         self.fc[2] = nn.Linear(self.layersizes[2], self.layersizes[2+1], bias=False)
        
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
        for l in range(self.numlayers):
            self.W[l] = self.fc[l].weight
#         self.W[1] = self.fc[1].weight
#         self.W[2] = self.fc[2].weight
        
        self.W = tuple(self.W)
    
#         print('size(self.W[0]): ', self.W[0].numel())
#         print('size(self.W[1]): ', self.W[1].numel())
#         print('size(self.W[2]): ', self.W[2].numel())

    def forward(self, x):
        
#         a = (self.numlayers - 1) * [0]
#         h = (self.numlayers - 1) * [0]
#         for l in range(self.numlayers - 1):
#             if l == 0:
#                 a[l] = self.fc[l](x)
#             else:
#                 a[l] = self.fc[l](h[l-1])
#             h[l] = F.relu(a[l])
            
            
        
        

            
        a = list(range(self.numlayers - 1))
        h = list(range(self.numlayers - 1))
        
        for l in range(self.numlayers - 1):
            if l == 0:
                a[l] = self.fc[l](x)
            else:
                a[l] = self.fc[l](h[l-1])
            h[l] = F.relu(a[l])
        
#         a[0] = self.fc[0](x)
#         h[0] = F.relu(a[0])
#         a[1] = self.fc[1](h[0])
#         h[1] = F.relu(a[1])
            
        z = self.fc[-1](h[-1])
            

#         cache = (a1, h1, a2, h2)
            
#         cache = ((self.numlayers - 1)) * 2 * [0]
#         for l in range(0, self.numlayers - 1):            
#             cache[2*l] = a[l]
#             cache[2*l+1] = h[l]
#         cache = tuple(cache) 
        

    
#         print('len(cache): ', len(cache))

        z.retain_grad()
#         for c in cache:
#             c.retain_grad()
        for c in a:
            c.retain_grad()
        for c in h:
            c.retain_grad()
        
        h = [x] + h
        a = a + [z]
        

        return z, a, h
"""

"""
class Model_2(nn.Module):

    def __init__(self):
        super(Model_2, self).__init__()
        
        self.layersizes = [784, 200, 100, 10]
        self.numlayers = len(self.layersizes) - 1
        
        self.fc = list(range(self.numlayers))
        for l in range(self.numlayers):
            self.fc[l] = nn.Linear(self.layersizes[l], self.layersizes[l+1], bias=False)
        
        self.fc = tuple(self.fc)

 



            
        self.W = list(range(3))
        for l in range(self.numlayers):
            self.W[l] = self.fc[l].weight
#         self.W[1] = self.fc[1].weight
#         self.W[2] = self.fc[2].weight
        
        self.W = tuple(self.W)


    def forward(self, x, t, v):
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
            
        a = list(range(self.numlayers - 1))
        h = list(range(self.numlayers - 1))
        
        for l in range(self.numlayers - 1):
            if l == 0:
                a[l] = self.fc[l](x)
            else:
                a[l] = self.fc[l](h[l-1])
            h[l] = F.relu(a[l])
        
#         a = tuple(a)
#         h = tuple(h)
            
        z = self.fc[-1](h[-1])
        
        loss = F.cross_entropy(z, t, reduction = 'none')
        weighted_loss = torch.dot(loss, v)
            
#         cache = ((self.numlayers - 1)) * 2 * [0]
#         for l in range(0, self.numlayers - 1):            
#             cache[2*l] = a[l]
#             cache[2*l+1] = h[l]
#         cache = tuple(cache) 
        

    
#         print('len(cache): ', len(cache))

        z.retain_grad()
#         for c in cache:
#             c.retain_grad()
        for c in a:
            c.retain_grad()
        for c in h:
            c.retain_grad()
        
        h = [x] + h
        a = a + [z]
        

        return weighted_loss, a, h
"""


