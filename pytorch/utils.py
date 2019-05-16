import torch
import sys
import numpy as np
import scipy

"""
import torch.nn as nn
import torch.nn.functional as F


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
        
#         a[0] = self.fc[0](x)
#         h[0] = F.relu(a[0])
#         a[1] = self.fc[1](h[0])
#         h[1] = F.relu(a[1])
        
#         a = tuple(a)
#         h = tuple(h)
            
        z = self.fc[-1](h[-1])
        
        loss = F.cross_entropy(z, t, reduction = 'none')
        weighted_loss = torch.dot(loss, v)
            

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
        

        return weighted_loss, a, h
"""


def get_loss(model, x, t):
    v = 1 / len(x) * torch.ones(len(x))
    loss, _, _ = model.forward(x, t, v)
    return loss.data.numpy()

def get_acc(model, x, t):
    v = 1 / len(x) * torch.ones(len(x))
    _, a , _= model.forward(torch.from_numpy(x), torch.from_numpy(t).long(), v)
    z = a[-1]
    y = z.argmax(dim=1)
    acc = np.mean(y.numpy() == t)
    return acc
    
def compute_sum_J_transpose_V_backp(v, data_, params):
    # use backpropagation
    import copy
    import torch.nn.functional as F
#     import torch
    
    numlayers = params['numlayers']
    
    model = data_['model']
    
    X_mb = data_['X_mb']
    t_mb = data_['t_mb']
    
    N2_index = params['N2_index']
    
#     model.detach()
    
    
#     model_1 = copy.deepcopy(data_['model'])
#     model = data_['model']
#     X_mb = data_['X_mb']
#     t_mb = data_['t_mb']
    
#     N2_index = params['N2_index']
    
#     model_new = Model()
    
#     model_new.W = model.W
    
#     print(model_new.fc[1].weight)
#     print(model.fc[1].weight)

#     print('X_mb[N2_index].size(): ', X_mb[N2_index].size())



#     model = get_model_grad_zerod(model)  


    
    
    
    
    
    
   
    
    
    
    
    
    
    
    

    



    


#     model_new = Model_2()
    
#     print('model_new.W[1]): ', model_new.W[1])
#     print('model.W[1]): ', model.W[1])
    
#     model_new.load_state_dict(model.state_dict())

#     for l in range(numlayers):
#         model_new.W[l].data = model.W[l].data
#     print('test')
    
#     z, _, _ = model_new.forward(X_mb[N2_index])
    
    
    
#     loss = F.cross_entropy(z, t_mb[N2_index], reduction = 'none')
#     loss = F.cross_entropy(z, t_mb[N2_index])
    
#     weighted_loss = torch.dot(loss, v)
    
    model = get_model_grad_zerod(model)
    loss, _, _ = model.forward(X_mb[N2_index], t_mb[N2_index], v)
    
#     weighted_loss.backward(retain_graph = True)
    loss.backward()
    
#     print('test 10:28')
    
#     print('model_1.W[1].size():', model_1.W[1].size())
    
#     a_grad_momentum = data_['a_grad_momentum']
#     h_momentum = data_['h_momentum']
    
    
    
    delta = list(range(numlayers))
    for l in range(numlayers):
#         delta[l] = a_grad_momentum[l][:, :, None] @ h_momentum[l][:, None, :] # [N2, m[l+1], m[l]]
#         delta[l] = v[:, None, None] * delta[l] # [N2, m[l+1], m[l]]
        
#         delta = torch.sum(delta, dim = 0) # [m[l+1], m[l]]
        delta[l] = copy.deepcopy(model.W[l].grad)
    
#     cache.detach_()
    
    model = get_model_grad_zerod(model)
    
#     del model_1
    
    
    return delta

    
def get_D_t(data_, params):
#     import sys
#     import torch
    algorithm = params['algorithm']
    
    N2 = params['N2']
    numlayers = params['numlayers']
    
    if algorithm == 'SMW-Fisher' or algorithm == 'SMW-Fisher-momentum':
    
#         from torch import eye
    
        a_grad_momentum = data_['a_grad_momentum']
        h_momentum = data_['h_momentum']
    
        lambda_ = params['lambda_']
        
        

        # compute D_t 
        D_t = lambda_ * torch.eye(N2)
    
#     print('D_t aftre lambda: ', D_t)
#     print('lambda: ', lambda_)
    
        for l in range(numlayers):
        
#         print('h[l][N2_index] @ h[l][N2_index].t().size(): ', (h[l][N2_index] @ h[l][N2_index].t()).size())
#         print('(a[l].grad)[N2_index] @ (a[l].grad)[N2_index].t().size(): ',
#               ((a[l].grad)[N2_index] @ (a[l].grad)[N2_index].t()).size())
        
            D_t += 1 / N2 * (a_grad_momentum[l] @ a_grad_momentum[l].t()) * (h_momentum[l] @ h_momentum[l].t())
    elif algorithm == 'SMW-GN':
        
#         from numpy import kron
#         import numpy as np
    
        GN_cache = data_['GN_cache']
        h = GN_cache['h']
        a_grad = GN_cache['a_grad']
        
#         print('h.size(): ', h.size())
        
        m_L = params['m_L']
        
        
        D_t = torch.zeros(m_L * N2, m_L * N2)
        
        # a_grad[l]: N2, m_L, m_l
        
        
        for l in range(numlayers):
            
            a_grad_l = a_grad[l]
#             h_l = h[l]
            
            # h[l]: N2 * m[l]
            
            # a_grad[l]
            
            print('len(a_grad_l)', len(a_grad_l))
            
            print('a_grad_l[0].size(): ', a_grad_l[0].size())
            
            permuted_a_grad_l = a_grad_l.permute(1, 0, 2).contiguous().view(m_L * N2, model.layersizes[l+1])
            
            print('permuted_a_grad_l.size(): ', permuted_a_grad_l.size())
            
            D_t += torch.from_numpy(np.kron(h[l] @ h[l].t(), np.ones(m_L, m_L))) * (permuted_a_grad_l.t() @ permuted_a_grad_l)
    else:
        print('Error!')
        sys.exit()
    return D_t

def compute_JV(V, data_, params):
#     import sys
    algorithm = params['algorithm']
    
    numlayers = params['numlayers']
    N2 = params['N2']
    
    if algorithm == 'SMW-Fisher' or algorithm == 'SMW-Fisher-momentum' or algorithm == 'kfac':
#         import torch
    
        a_grad_momentum = data_['a_grad_momentum']
        h_momentum = data_['h_momentum']
    
        
        
    
        v = torch.zeros(N2)
    
#     print('model.W[0].size(): ', model.W[0].size())
#     print('model.W[1].size(): ', model.W[1].size())
#     print('model.W[2].size(): ', model.W[2].size())        
    
        for l in range(numlayers):
        
#         model.W[l] @ h[l] # m[l+1] * N2
    
        
            v += torch.sum((a_grad_momentum[l] @ V[l]) * h_momentum[l], dim = 1)
            
            
#         print('v:', v)
        v = v.data
        
        
#     print('V[1]: ', V[1])
#     print('1/N2 * h_momentum[1].t() @ a_grad_momentum[1]:', 1/N2 * h_momentum[1].t() @ a_grad_momentum[1])
    elif algorithm == 'SMW-GN':
        
#         import torch
        
        GN_cache = data_['GN_cache']
        
        m_L = params['m_L']
        
        a_grad = GN_cache['a_grad']
        h = GN_cache['h']
        
        
        
        
        
        
        
        v = torch.zeros(m_L, N2)
        
        print('len(a_grad): ', len(a_grad))
        
        print('len(a_grad[1]): ', len(a_grad[1]))
        
        print('a_grad[1][0].size(): ', a_grad[1][0].size())
        
        print('len(h)', len(h))
        
        print('h[1].size(): ', h[1].size())
        
        # a_grad[l]: N2, m_L, m_l
        
        for l in range(numlayers):
        
            a_grad_l = a_grad[l]
#             h_i = h[i]
            for i in range(m_L):
#                 print('len(a_grad_momentum): ', len(a_grad_momentum))

    # a[l][N2_index] @ model.W[l] # N2 * m[l]
    # (a[l][N2_index] @ model.W[l]) * h[l][N2_index] # N2 * m[l]
    #  torch.sum((a[l][N2_index] @ model.W[l]) * h[l][N2_index], dim = 1)
    
    # a[l].grad: size N1 * m[l+1], it has a coefficient 1 / N1, which should be first compensate
    # h[l]: size N1 * m[l]
    # model.W[l]: size m[l+1] * m[l]
                
                v[i] += torch.sum((a_grad_l[:, i, :] @ V[l]) * h[l], dim = 1)
        
        v = v.view(m_L * N2)
        
    else:
        print('Error!')
        sys.exit()
    
    return v

def get_cache_momentum(data_, params):
    algorithm = params['algorithm']
    
    N2 = params['N2']
    
    
    
    if algorithm == 'SMW-GN':
#         import torch
        import copy
    
        X_mb = data_['X_mb']
        t_mb = data_['t_mb']
        
        model = data_['model']
        
        numlayers = params['numlayers']
#         layersizes = params['layersizes']
        
#         layersizes = model.layersizes
        m_L = params['m_L']
        
        N2_index = params['N2_index']
        
        
        
#         import torch.nn.functional as F
#         t_mb = data_['t_mb']
#         loss = F.cross_entropy(z, t_mb[N2_index],reduction = 'none')
#         print('loss.size(): ', loss.size())
#         loss.backward()
        
        
        
        
        
#         z.backward(torch.Tensor(z.size()))
        
#         h_momentum = []
        
        
        
#         print('h[0].size(): ', h[0].size())
#         print('h[1].size(): ', h[1].size())
#         print('h[2].size(): ', h[2].size())
#         print('h: ', h)
#         print('h[0]: ', h[0])
#         print('h[1]: ', h[1])
#         print('h[2]: ', h[2])
        
        
        
#         a_grad_momentum = list(range(numlayers))
#         a_grad_momentum = numlayers * [[]]
        a_grad_momentum = []
        for l in range(numlayers):
            a_grad_momentum.append(torch.ones(N2, m_L, model.layersizes[l+1]))
        
#         print('a_grad_momentum', a_grad_momentum)
        
        v_tmp = 1 / len(X_mb[N2_index]) * torch.ones(len(X_mb[N2_index]))
        
        for i in range(m_L):
            
            
            _, a, h = model.forward(X_mb[N2_index], t_mb[N2_index], v_tmp)
        
            z = a[-1]
            
#         Jacobian_z = []

            
    
#             print('z.size(): ', z.size())

            model = get_model_grad_zerod(model)
    
#             torch.sum(z[:, i]).backward(retain_graph = True)
            torch.sum(z[:, i]).backward()
        
            
        
#             a_grad_momentum.append(copy.deepcopy())

#             a_grad_momentum.append([copy.deepcopy(ai.grad) for ai in a])
#             for l in range(numlayers - 1):
#                 a_grad_momentum_i.append(copy.deepcopy(a[l].grad))

            for l in range(numlayers):
        
#                 print('a_grad_momentum[l]: ', a_grad_momentum[l])
#                 print('a[l]: ', a[l])


                a_grad_momentum[l][:, i, :] = copy.deepcopy(a[l].grad)
                
#                 if i == 0:
#                     a_grad_momentum[l] = [copy.deepcopy(a[l].grad)]
#                 else:
                    
#                     print('a_grad_momentum[i]', a_grad_momentum[i])
                    
                    
                    
#                     a_grad_momentum[l].append(copy.deepcopy(a[l].grad))    
        
                
             
        
#             print('a:', a)
#             print('h: ', h)
    
#         print('model.W[1].grad.size(): ', model.W[1].grad.size())

        h_momentum = [copy.deepcopy(hi.data) for hi in h]
        
        GN_cache = {}
        GN_cache['a_grad'] = a_grad_momentum
        GN_cache['h'] = h_momentum
        
        
        data_['GN_cache'] = GN_cache
        
        
        
        
    else:
    
        
        a = data_['a']
        h = data_['h']
#         z = data_['z']
    
    
    
    
        N1 = params['N1']
        
        i = params['i']
        
        numlayers = params['numlayers']
    
        N2_index = params['N2_index']
    
        if algorithm == 'SMW-Fisher-momentum':
            a_grad_momentum = data_['a_grad_momentum']
            h_momentum = data_['h_momentum']
    
    
#         a = []
#         h = [X_mb] + h
#         for ii in range(len(cache)):
#             if ii % 2 == 0:
#                 a.append(cache[ii])
#             else:
#                 h.append(cache[ii])        
#         a.append(z)
    
    
    
#     print('a[0].size(): ', a[0].size())
#     print('a[1].size(): ', a[1].size())
#     print('a[2].size(): ', a[2].size())
    
#     print('h[0].size(): ', h[0].size())
#     print('h[1].size(): ', h[1].size())
#     print('h[2].size(): ', h[2].size())
    
#     print('a[0].grad: ', a[0].grad)
#     print('a[1].grad: ', a[1].grad)
#     print('a[2].grad: ', a[2].grad)
    
#     print('h[0]: ', h[0])
#     print('h[1]: ', h[1])
#     print('h[2]: ', h[2])

    
    
    
    
    
    # Update running estimates
        if algorithm == 'SMW-Fisher-momentum':
            rho = min(1 - 1/(i+1), 0.95)
        
            for l in range(numlayers):
                a_grad_momentum[l] = rho * a_grad_momentum[l] + (1-rho) * N1 * (a[l].grad)[N2_index]
                h_momentum[l] = rho * h_momentum[l] + (1-rho) * h[l][N2_index]
        
        elif algorithm == 'SMW-Fisher' or algorithm =='kfac':
            a_grad_momentum = []
            h_momentum = []
            for l in range(numlayers):
                a_grad_momentum.append(N1 * (a[l].grad)[N2_index])
                h_momentum.append(h[l][N2_index])
            
    
        
        
        else:
            print('Error!')
            sys.exit()
        
        data_['a_grad_momentum'] = a_grad_momentum
        data_['h_momentum'] = h_momentum


    return data_



def get_subtract(model_grad, delta, params):
    numlayers = params['numlayers']
    for l in range(numlayers):
        delta[l] = model_grad[l] - delta[l]
    return delta

def get_plus(model_grad, delta, params):
    numlayers = params['numlayers']
    for l in range(numlayers):
        delta[l] = model_grad[l] + delta[l]
    return delta

def get_multiply(alpha, delta, params):
    numlayers = params['numlayers']
    for l in range(numlayers):
        delta[l] = alpha * delta[l]
    return delta

def get_minus(delta, params):
    numlayers = params['numlayers']
    
    p = []
    for l in range(numlayers):
        p.append(-delta[l])
        
    return p


def SMW_GN_update(data_, params):
    # a[l].grad: size N1 * m[l+1], it has a coefficient 1 / N1, which should be first compensate
    # h[l]: size N1 * m[l]
    # model.W[l]: size m[l+1] * m[l]
    
    

    
#     algorithm = params['algorithm']
#     model = data_['model']
    
    model_grad = data_['model_grad']
    
#     X_mb = data_['X_mb']
#     t_mb = data_['t_mb']    
#     cache = data_['cache']
#     z = data_['z']

    
    
#     if algorithm == 'SMW-Fisher-momentum':
#         a_grad_momentum = data_['a_grad_momentum']
#         h_momentum = data_['h_momentum']
        
#     loss = data_['loss']
    
    
    
    N1 = params['N1']
    N2 = params['N2']
#     i = params['i']
#     alpha = params['alpha']
#     lambda_ = params['lambda_']
#     numlayers = params['numlayers']
#     boost = params['boost']
#     drop = params['drop']
    
    N2_index = np.random.permutation(N1)[:N2]
    params['N2_index'] = N2_index
    
    m_L = data_['model'].layersizes[-1]
    params['m_L'] = m_L
    
    
#     start_time = time.time()
    
    data_ = get_cache_momentum(data_, params)

#     print('time for get cache momentum: ', start_time - time.time())


    
    
    
    
    
        
    # compute the vector after D_t    
    

    
#     start_time = time.time()
    
    v = compute_JV(model_grad, data_, params)
    
#     print('time for compute JV: ', start_time - time.time())
    

        
    
    
    # compute hat_v
    

#     start_time = time.time()
        
    D_t = get_D_t(data_, params)
    
#     print('D_t:', D_t)
    
#     print('v:', v)
#     print('torch.mean(v): ', torch.mean(v))
    
#     print('time for get D_t: ', start_time - time.time())
    
#     start_time = time.time()
    
    D_t_cho_fac = scipy.linalg.cho_factor(D_t.data.numpy())
    hat_v = scipy.linalg.cho_solve(D_t_cho_fac, v.data.numpy())
    
    hat_v = torch.from_numpy(hat_v)
    
    hat_v = compute_HV(hat_v)
    
#     print('time for solve linear system: ', start_time - time.time())
    
#     print('hat_v: ', hat_v)
    
#     print('torch.mean(hat_v): ', torch.mean(hat_v))
    
    
#     print('get_dot_product(model_grad, model_grad, params): ', get_dot_product(model_grad, model_grad, params))
#     print('1 - hat_v: ', 1 - hat_v)
#     print('torch.max(hat_v): ', torch.max(hat_v))
#     print('torch.min(hat_v): ', torch.min(hat_v))

#     hat_v = torch.ones(N2)
    
#     print('hat_v: ', hat_v)
#     print('1 - hat_v: ', 1 - hat_v)

    # compute natural gradient
    

    
#     start_time = time.time()
    
    
    
    delta = compute_sum_J_transpose_V_1_backp(hat_v, model, X_mb[N2_index], t_mb[N2_index], params)

#     print('test delta')
#     delta = model_grad
    
#     print('time for compute J transpose V: ', start_time - time.time())
    
#     print('\n')
    

    
    
        

        
        
#     print('delta[1]: ', delta[1])
#     print('model_grad[1]: ', model_grad[1]
        

    delta = get_multiply(1 / N2, delta, params)
    
    
    delta = get_subtract(model_grad, delta, params)
    
    delta = get_multiply(1 / lambda_, delta, params)
        
    p = get_minus(delta, params)
    

            

            
            

        

#     data_['model'] = model
    data_['p'] = p
    
#     if algorithm == 'SMW-Fisher-momentum':
#         data_['a_grad_momentum'] = a_grad_momentum
#         data_['h_momentum'] = h_momentum
    
#     print('model.W[1] in utils: ', model.W[1])
#     print('model.W[1].data in utils: ', model.W[1].data)
        
    return data_

def get_model_grad_zerod(model):
    model.zero_grad()
#     print('gradeint zerod')
    
#     print('model.fc[1].weight.grad: ', model.fc[1].weight.grad)
    
    for fci in model.fc:
        fci.zero_grad()
        
#     print('model.fc[1].weight.grad: ', model.fc[1].weight.grad)
    return model



def compute_sum_J_transpose_V(v, data_, params):
    
    a_grad_momentum = data_['a_grad_momentum']
    h_momentum = data_['h_momentum']
    
    numlayers = params['numlayers']
    
    delta = list(range(numlayers))
    for l in range(numlayers):
        delta[l] = a_grad_momentum[l][:, :, None] @ h_momentum[l][:, None, :] # [N2, m[l+1], m[l]]
        delta[l] = v[:, None, None] * delta[l] # [N2, m[l+1], m[l]]
        
#         delta = torch.sum(delta, dim = 0) # [m[l+1], m[l]]

    for l in range(numlayers):
        delta[l] = torch.sum(delta[l], dim = 0) # [m[l+1], m[l]]
        
    print('check if correct')
    
    sys.exit()
    
    
    return delta





def update_lambda(p, data_, params):
#     import sys
    
    model = data_['model']
    X_mb = data_['X_mb']
    t_mb = data_['t_mb']
    loss = data_['loss']
    
    model_grad = data_['model_grad']
    
    numlayers = params['numlayers']
    lambda_ = params['lambda_']
    boost = params['boost']
    drop = params['drop']
    
    algorithm = params['algorithm']
    
    
    # compute rho
      

#     [ll_chunk, ~] =...
#             computeLL(paramsp + test_p, indata, outdata, numchunks, targetchunk)

#     print('model.W[1].grad: ', model.W[1].grad)


#     import time
#     start_time = time.time()
    
    
    
    
    
#     print('time for update lambda 1/2: ', time.time() - start_time)


        
    ll_chunk = get_new_loss(model, p, X_mb, t_mb)
    
    
        
#     [oldll_chunk, ~] =...
#             computeLL(paramsp, indata, outdata, numchunks, targetchunk)
    oldll_chunk = loss
        
    
    



   
    
        
        
    if oldll_chunk - ll_chunk < 0:
        rho = float("-inf")
    else:
        if algorithm == 'SMW-Fisher' or algorithm == 'SMW-GN':
            denom = - 0.5 * get_dot_product(model_grad, p, params)
        elif algorithm == 'kfac' or algorithm == 'SMW-Fisher-momentum':
            denom = computeFV(p, data_, params)
                
            denom = get_dot_product(p, denom, params)
            denom = -0.5 * denom
            denom = denom - get_dot_product(model_grad, p, params) 
                
        else:
            print('Error!')
            sys.exit()
    
#     print('time for update lambda 1/4: ', time.time() - start_time)

            
        
        rho = (oldll_chunk - ll_chunk) / denom
        
#         print('ll_chunk: ', ll_chunk)
#         print('old ll_chunk: ', oldll_chunk)
#         print('oldll_chunk - ll_chunk: ', oldll_chunk - ll_chunk)
#         print('denom: ', denom)
#         print('rho: ', rho)
        
    
    
    
    # update lambda   
    if rho < 0.25:
        lambda_ = lambda_ * boost
    elif rho > 0.75:
        lambda_ = lambda_ * drop
        
    return lambda_

def SMW_Fisher_update(data_, params):
#     import torch
#     import numpy as np

    import time
    import copy
    
    algorithm = params['algorithm']
    model = data_['model']
    
    model_grad = data_['model_grad']
    
    X_mb = data_['X_mb']
    t_mb = data_['t_mb']    
    a = data_['a']
    h = data_['h']
#     z = data_['z']


        

    
    
    if algorithm == 'SMW-Fisher-momentum':
        a_grad_momentum = data_['a_grad_momentum']
        h_momentum = data_['h_momentum']
        
    loss = data_['loss']
    
    
    
    N1 = params['N1']
    N2 = params['N2']
    i = params['i']
#     inverse_update_freq = params['inverse_update_freq']
    alpha = params['alpha']
    lambda_ = params['lambda_']
    numlayers = params['numlayers']
    
    
    
    
    N2_index = np.random.permutation(N1)[:N2]
    params['N2_index'] = N2_index
    
    
#     start_time = time.time()
    
    data_ = get_cache_momentum(data_, params)

#     print('time for get cache momentum: ', start_time - time.time())

#     model_grad = []
#     for l in range(numlayers):
#         model_grad.append(copy.deepcopy(model.W[l].grad))
    
    
    
    
    
        
    # compute the vector after D_t    
    
#     data_compute_JV = {}
#     data_compute_JV['a_grad_momentum'] = a_grad_momentum
#     data_compute_JV['h_momentum'] = h_momentum
    
#     start_time = time.time()
    
    v = compute_JV(model_grad, data_, params)
    
#     print('time for compute JV: ', start_time - time.time())
    
#     data_compute_JV = {}
        
    
    
    # compute hat_v
    

#     start_time = time.time()

    if algorithm == 'SMW-Fisher':
        
        D_t = get_D_t(data_, params)
    
#     print('D_t:', D_t)
    
#     print('v:', v)
#     print('torch.mean(v): ', torch.mean(v))
    
#     print('time for get D_t: ', start_time - time.time())
    
#     start_time = time.time()
    
        D_t_cho_fac = scipy.linalg.cho_factor(D_t.data.numpy())
        hat_v = scipy.linalg.cho_solve(D_t_cho_fac, v.data.numpy())
    
        hat_v = torch.from_numpy(hat_v)
        
    elif algorithm == 'SMW-Fisher-momentum':
            
        
        inverse_update_freq = params['inverse_update_freq']
        
        if i % inverse_update_freq == 0 or i < 100:
            D_t = get_D_t(data_, params)
            D_t_inv = D_t.inverse()
            data_['D_t_inv'] = D_t_inv
        else:
            D_t_inv = data_['D_t_inv']
        
#         print('D_t_inv ', D_t_inv)
#         print('D_t_inv', D_t_inv.size())
#         print('v: ', v)
        
        hat_v = D_t_inv @ v
        
    else:
        print('Error!')
        sys.exit()
        
    
    
#     print('time for solve linear system: ', start_time - time.time())
    
#     print('hat_v: ', hat_v)
    
#     print('torch.mean(hat_v): ', torch.mean(hat_v))
    
    
#     print('get_dot_product(model_grad, model_grad, params): ', get_dot_product(model_grad, model_grad, params))
#     print('1 - hat_v: ', 1 - hat_v)
#     print('torch.max(hat_v): ', torch.max(hat_v))
#     print('torch.min(hat_v): ', torch.min(hat_v))

#     hat_v = torch.ones(N2)
#     hat_v = torch.zeros(N2)
    
#     print('test hat_v: ', hat_v)
#     print('1 - hat_v: ', 1 - hat_v)

    # compute natural gradient
    

    
#     start_time = time.time()
    
    
    

    if algorithm == 'SMW-Fisher':
        delta = compute_sum_J_transpose_V_backp(hat_v, data_, params)
    elif algorithm == 'SMW-Fisher-momentum':
        delta = compute_sum_J_transpose_V(hat_v, data_, params)
    else:
        print('Error!')
        sys.exit()

#     print('test delta')
#     delta = model_grad
    
#     print('time for compute J transpose V: ', start_time - time.time())
    
#     print('\n')
    

        
#     print('delta[1] 1: ', delta[1])    
   
    
    delta = get_multiply(1 / N2, delta, params)
    
#     print('delta[1]: ', delta[1])
#     print('model_grad[1]: ', model_grad[1])
    
#     print('delta[1] 2: ', delta[1])    
    
    delta = get_subtract(model_grad, delta, params)
    
#     print('delta[1] 3: ', delta[1])
        
    delta = get_multiply(1 / lambda_, delta, params)
    
#     print('delta[1] 4: ', delta[1])

    ########################
#     print('model_grad[1]: ', model_grad[1])
    
#     should_be_grad = computeFV(delta, data_, params)
#     should_be_grad = get_plus(should_be_grad, get_multiply(lambda_, delta, params), params) 
    
#     print('should_be_grad[1]: ', should_be_grad[1])
    
    
#     print('test')

        
    p = get_minus(delta, params)
    

    
    # update parameters
#     for l in range(numlayers):
        
#         For two 2D tensors a and b (of size [b,n] and [b,m] respectively),
# a[:, :, None] @ b[:, None, :] (of size [b,n,m]) gives the outer product operated on each item in the batch.
        
#         print('a[l][N2_index][:, :, None] @ h[l][N2_index][:, None, :].size(): ', 
#               (a[l][N2_index][:, :, None] @ h[l][N2_index][:, None, :]).size()) 
        
#         print('(torch.mean((1 - hat_v)[:, None, None] * (a[l][N2_index][:, :, None] @ h[l][N2_index][:, None, :]). dim=0)).size(): ', 
#               (torch.mean((1 - hat_v)[:, None, None] * (a[l][N2_index][:, :, None] @ h[l][N2_index][:, None, :]), dim=0)).size())
        
#         print('a[l][N2_index][:, :, None]: ', a[l][N2_index][:, :, None])
#         print('h[l][N2_index][:, None, :]: ', h[l][N2_index][:, None, :])
    
        
        
#         print('delta.size():', delta.size())
#         print('delta: ', delta)
    
#         print('delta * N2: ', delta * N2)
        
#         print('model.W[l].grad.size(): ', model.W[l].grad.size())
#         print('model.W[l].grad: ', model.W[l].grad)

#         print('model.W[1].data in utils: ', model.W[1].data)
    
#         model.W[l].data -= alpha * delta[l]
        
#         print('model.W[1].data in utils: ', model.W[1].data)
        
    
    

    
    
        
    
    

            

            
            

        

#     data_['model'] = model
    data_['p'] = p
    
    if algorithm == 'SMW-Fisher-momentum':
        data_['a_grad_momentum'] = a_grad_momentum
        data_['h_momentum'] = h_momentum
    
#     print('model.W[1] in utils: ', model.W[1])
#     print('model.W[1].data in utils: ', model.W[1].data)
        
    return data_, params

def get_new_loss(model, p, x, t):
    import torch.nn.functional as F
    import copy
    
#     print('p[1]: ', p[1])
    
    model_new = copy.deepcopy(model)
    
#     print('model_new.W[1]: ', model_new.W[1])
    
    for l in range(model_new.numlayers):
        model_new.W[l].data += p[l]
        
#     print('model_new.W[1]: ', model_new.W[1])
        
#     z, _, _ = model_new.forward(x)
    
#     loss = F.cross_entropy(z, t)
    
    v = 1 / len(x) * torch.ones(len(x))
    loss, _, _ = model_new.forward(x, t, v)
    
    return loss

def get_dot_product(delta_1, delta_2, params):
    import torch
    numlayers = params['numlayers']
    
    dot_product = 0
    for l in range(numlayers):
        dot_product += torch.sum(delta_1[l] * delta_2[l])
    
    return dot_product

"""
def get_mean(delta, params):
#     import torch
    numlayers = params['numlayers']
    for l in range(numlayers):
        delta[l] = torch.mean(delta[l], dim=0)
    return delta
"""


def computeFV(delta, data_, params):
    
    X_mb = data_['X_mb']
    t_mb = data_['t_mb']
    model = data_['model']
    
    N1 = params['N1']
    N2 = params['N2']
#     N2_index = np.random.permutation(N1)[:N2]

    N2_index = params['N2_index']
    
    algorithm = params['algorithm']
    
#     a_grad_momentum = data_['a_grad_momentum']
#     h_momentum = data_['h_momentum']
    
    
#     import time
#     start_time = time.time()

    
    v = compute_JV(delta, data_, params)
    
#     print('time for FV 1/2: ', time.time() - start_time)

    
    
    

    delta = compute_sum_J_transpose_V_backp(v, data_, params)

    
#     print('delta[1].size(): ', delta[1].size())
    
    
    
#     delta = get_mean(delta, params)
    
    delta = get_multiply(1 / N2, delta, params)
    
#     print('delta[1].size(): ', delta[1].size())
    
    return delta




    
    
    



def kfac_update(data_, params):
#     import numpy as np
    
    X_mb = data_['X_mb']
    
#     a1 = data_['a1']
#     a2 = data_['a2']
#     h1 = data_['h1']
#     h2 = data_['h2']
    
#     a = data_['a']
#     h = data_['h']
    
#     z = data_['z']
    A = data_['A']
    G = data_['G']
    A_inv = data_['A_inv']
    G_inv = data_['G_inv']
    model = data_['model']
    
    model_grad = data_['model_grad']
    
    N1 = params['N1']
    N2 = params['N2']
    i = params['i']
    inverse_update_freq = params['inverse_update_freq']
    lambda_ = params['lambda_']
    alpha = params['alpha']
    numlayers = params['numlayers']
    rho_kfac = params['rho_kfac']
    
    N2_index = np.random.permutation(N1)[:N2]
    params['N2_index'] = N2_index
    
#     a = []
#     h = []
#     for ii in range(0, len(cache)):
#         if ii % 2 == 0:
#             a.append(cache[ii])
#         else:
#             h.append(cache[ii])

    data_ = get_cache_momentum(data_, params)
    
    
    
    a_grad_momentum = data_['a_grad_momentum']
    h_momentum = data_['h_momentum']
    
    G_ = []
    for l in range(0, numlayers):
        G_.append(1/N2 * a_grad_momentum[l].t() @ a_grad_momentum[l])
        
            
    A_ = []
    for l in range(0, numlayers):
#         if l == 0:
#             A_.append(1/N1 * X_mb.t() @ X_mb)
#         else:
        A_.append(1/N2 * h_momentum[l].t() @ h_momentum[l])
        

#     G_ = [G1_, G2_, G3_]
#     A_ = [A1_, A2_, A3_]

    # Update running estimates of KFAC
    rho = min(1-1/(i+1), rho_kfac)
    
#     print()

    for l in range(numlayers):
        
#         print('in utils')
#         print('A[l].size(): ', A[l].size())
#         print('A_[l].size(): ', A_[l].size())
        
        A[l] = rho*A[l] + (1-rho)*A_[l]
        G[l] = rho*G[l] + (1-rho)*G_[l]
        
#         print('A[l].size(): ', A[l].size())
        
#      print('G[1]: ', G[1])

    # Step
    delta = []
    for l in range(numlayers):
        
#         print(type(G[l]))
        
#         print('i = ', i)
        
        # Amortize the inverse. Only update inverses every now and then
        if i % inverse_update_freq == 0:
            
            
            
            A_inv[l] = (A[l] + np.sqrt(lambda_) * torch.eye(A[l].shape[0])).inverse()
            
#             print('G[l] + eps*torch.eye(G[l].shape[0]): ', G[l] + eps*torch.eye(G[l].shape[0]))
            
            G_inv[l] = (G[l] + np.sqrt(lambda_) * torch.eye(G[l].shape[0])).inverse()

#         print(type(G_inv[l]))
#         print(type(model.W[l].grad.data))
#         print(type(A_inv[l]))

#         print('G_inv[l]: ', G_inv[l])
#         print('model.W[l]: ', model.W[l])
#         print('model.W[l].grad.data: ', model.W[l].grad.data)
#         print('A_inv[l]: ', A_inv[l])

#         print('A_inv[l].size(): ', A_inv[l].size())
    
#         print('G_inv[l].size(): ', G_inv[l].size())
        
            
        delta.append(G_inv[l] @ model_grad[l] @ A_inv[l])
        
#         print('delta: ', delta)

    
    
    p = []
    for l in range(numlayers):
        p.append(-delta[l])
        

        
    
    
    

    
        
    data_['A'] = A
    data_['G'] = G
    data_['A_inv'] = A_inv
    data_['G_inv'] = G_inv
#     data_['model'] = model
    
    data_['p'] = p
    
    
        
    return data_, params



def SGD_update(data_, params):
    
#     print(data_['model'])
    
    model = data_['model']
    
#     print(data_['model'])
    

#     alpha = params['alpha']
    numlayers = params['numlayers']
    

    p = []
    for l in range(numlayers):
        p.append(-model.W[l].grad)
    
    
        
#     print(data_['model'])
        
    
#     data_['model'] = model
    
#     print(data_['model'])

    data_['p'] = p
    
    
        
    return data_

def update_parameter(p, model, params):
    numlayers = params['numlayers']
    alpha = params['alpha']
    
    for l in range(numlayers):
        model.W[l].data += alpha * p[l]
        
    return model
    
