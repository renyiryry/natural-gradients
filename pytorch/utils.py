import torch
import torch.nn.functional as F
import sys
import numpy as np
import scipy
import copy

import time

def get_zero(params):
    layersizes = params['layersizes']
    W = []
    for l in range(len(layersizes) - 1):
        W.append(torch.zeros(layersizes[l+1], layersizes[l]))
    return W

def get_loss(model, x, t):
    z, _, _ = model.forward(x)
    loss = F.cross_entropy(z, t, reduction = 'mean')
    
#     print('loss', loss)
    
    return loss

def get_acc(model, x, t):
    z, _ , _= model.forward(torch.from_numpy(x))
    y = z.argmax(dim=1)
    acc = np.mean(y.numpy() == t)
    return acc
    
def compute_sum_J_transpose_V_backp(v, data_, params):
    # use backpropagation
    algorithm = params['algorithm']
    N2_index = params['N2_index']
    N2 = params['N2']
    numlayers = params['numlayers']
    
    model = data_['model']
    X_mb = data_['X_mb']
    
    model = get_model_grad_zerod(model)
    
    z, _, _ = model.forward(X_mb[N2_index])
    
    if algorithm == 'kfac' or algorithm == 'SMW-Fisher' or algorithm == 'SMW-Fisher-momentum':
        t_mb = data_['t_mb']
    
        
    
    
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
    
        
        
        
        loss = F.cross_entropy(z, t_mb[N2_index], reduction = 'none')
        
        loss = torch.dot(loss, v)
    
#     weighted_loss.backward(retain_graph = True)
        
    elif algorithm == 'SMW-GN':
        
        m_L = params['m_L']
        
        v = v.view(N2, m_L)
        
#         print('print(z.dtype): ', z.dtype)
        
#         print('print(v.dtype): ', v.dtype)
        
        loss = torch.sum(z * v)
        
    else:
        print('Error! 1500')
        sys.exit()
    
#     del model_1

    loss.backward()
    
#     print('test 10:28')
    
#     print('model_1.W[1].size():', model_1.W[1].size())
    

    
    
    
    delta = list(range(numlayers))
    for l in range(numlayers):
#         delta[l] = a_grad_momentum[l][:, :, None] @ h_momentum[l][:, None, :] # [N2, m[l+1], m[l]]
#         delta[l] = v[:, None, None] * delta[l] # [N2, m[l+1], m[l]]
        
#         delta = torch.sum(delta, dim = 0) # [m[l+1], m[l]]
        delta[l] = copy.deepcopy(model.W[l].grad)
    
#     cache.detach_()
    
    model = get_model_grad_zerod(model)
    
    
    return delta

    
def get_D_t(data_, params):
    algorithm = params['algorithm'] 
    N2 = params['N2']
    numlayers = params['numlayers']
    
    if algorithm == 'SMW-Fisher' or algorithm == 'SMW-Fisher-momentum':   
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
        
        D_t = D_t.data.numpy()
    elif algorithm == 'SMW-GN':
    
        GN_cache = data_['GN_cache']
        h = GN_cache['h']
        a_grad = GN_cache['a_grad']
        
#         print('h.size(): ', h.size())
        
        m_L = params['m_L']
        lambda_ = params['lambda_']
        
        
        D_t = np.zeros((m_L * N2, m_L * N2))
        
        # a_grad[l]: N2, m_L, m_l
        
        model = data_['model']
        
#         start_time = time.time()
        
        
        for l in range(numlayers):
            
            a_grad_l = a_grad[l]
#             h_l = h[l]
            
            # h[l]: N2 * m[l]
            
            # a_grad[l]
            
#             print('len(a_grad_l)', len(a_grad_l))
            
#             print('a_grad_l[0].size(): ', a_grad_l[0].size())
            
    
#             print('a_grad_l.permute(1, 0, 2).contiguous().size():', a_grad_l.permute(1, 0, 2).contiguous().size())
            
            
            permuted_a_grad_l = a_grad_l.permute(1, 0, 2).contiguous().view(m_L * N2, model.layersizes[l+1])
            
#             print('h[l]', h[l])
            
#             print('torch.max(h[l])', torch.max(h[l]))
            
#             print('(h[l].t())', (h[l].t()))
            
#             print('h[l] @ (h[l].t())', h[l] @ (h[l].t()))
            
#             print('print(torch.mm(h[l], h[l].t()))', torch.mm(h[l], h[l].t()))
            
            h_l_h_l_t = h[l] @ (h[l].t())
            
#             print('h_l_h_l_t', h_l_h_l_t)
            
            h_l_h_l_t = h_l_h_l_t.data.numpy()
            
#             print('h_l_h_l_t', h_l_h_l_t)
            
            h_kron = np.kron(h_l_h_l_t, np.ones((m_L, m_L)))
            
#             h_kron = torch.from_numpy(h_kron)
            
#             h_kron = h_kron.type(torch.DoubleTensor)
            
#             
            
#             print('(permuted_a_grad_l @ permuted_a_grad_l.t())', (permuted_a_grad_l @ permuted_a_grad_l.t()).dtype).type(torch.DoubleTensor)
            
#             print('permuted_a_grad_l.size(): ', permuted_a_grad_l.size())
            
#             print('(torch.from_numpy(np.kron(h[l] @ h[l].t(), np.ones(m_L, m_L)))).size()' (torch.from_numpy(np.kron(h[l] @ h[l].t(), np.ones(m_L, m_L)))).size())
            
#             print()
    
#             
    
#             print('np.kron(h_l_h_l_t, np.ones(m_L, m_L))', np.kron(h_l_h_l_t, np.ones(m_L, m_L)))
            
#             print('torch.from_numpy(np.kron(h_l_h_l_t, np.ones(m_L, m_L))) ', torch.from_numpy(np.kron(h_l_h_l_t, np.ones(m_L, m_L))))
    
#             print('(permuted_a_grad_l.t() @ permuted_a_grad_l).size()', (permuted_a_grad_l.t() @ permuted_a_grad_l).size())
            
#             print('(permuted_a_grad_l.t() @ permuted_a_grad_l)', (permuted_a_grad_l.t() @ permuted_a_grad_l))
    
#             print('print((torch.from_numpy(np.kron(h_l_h_l_t, np.ones((m_L, m_L))))).size())', (torch.from_numpy(np.kron(h_l_h_l_t, np.ones((m_L, m_L))))).size())
            
#             print(' print(permuted_a_grad_l.szie())', permuted_a_grad_l.size())
            
#             print('print((permuted_a_grad_l.t() @ permuted_a_grad_l).size())', (permuted_a_grad_l.t() @ permuted_a_grad_l).size())
            
            
    
            
    
#             print('h_kron', h_kron)
        
#             print('print((permuted_a_grad_l @ permuted_a_grad_l.t()).data.numpy())', (permuted_a_grad_l @ permuted_a_grad_l.t()).data.numpy())
    
            D_t += np.multiply(h_kron, (permuted_a_grad_l @ permuted_a_grad_l.t()).data.numpy())
        
#         print('time for compute J J transpose: ', time.time() - start_time)
        
        # add the H term
        
#         start_time = time.time()
        
        D_t = get_JH(D_t, data_, params)
        
        
        
#         D_t = np.transpose(D_t)
        
#         for i in range(N2 * m_L):
#             D_t[:, i] = get_HV(D_t[:, i], data_, params)
        
#         D_t = np.transpose(D_t)
        
#         print('time for compute H: ', time.time() - start_time)
        
        
        
        D_t = 1 / N2 * D_t
        
#         H = get_H(data_, params)
#         for i in range(N2):
#             D_t[i * m_L: (i+1) * m_L, i * m_L: (i+1) * m_L] += lambda_ * H[i]
        D_t = D_t + lambda_ * np.eye(m_L * N2)
        
    else:
        print('Error! 1501')
        sys.exit()
    return D_t

def get_JH(D_t, data_, params):
    
    y = data_['y']
    
    N2 = params['N2']
    m_L = params['m_L']
    
#     print('y.size', y.size())
    
    # D_t_1
    
    diag_y = y.view(m_L * N2)
    
    diag_y = diag_y.repeat(N2 * m_L, 1)
    
    D_t_1 = D_t * diag_y.data.numpy()
    
    # D_t_2
    
#     D_t_2 = torch.zeros(m_L * N2, N2)
    D_t_3 = np.zeros((m_L * N2, m_L * N2))
    for i in range(N2):
#         D_t_2[:, i] = D_t[:, i * m_L : (i+1) * m_L] @ y[i]

        y_i = y[i].data.numpy()[:, np.newaxis]
    
    

        D_t_3[:, i * m_L : (i+1) * m_L] = np.matmul(np.matmul(D_t[:, i * m_L : (i+1) * m_L], y_i), np.transpose(y_i))
    
    D_t = D_t_1 - D_t_3
    return D_t
    

def get_H(data_, params):
    
    print('wrong')
    sys.exit()
    
    N2_index = params['N2_index']
    m_L = params['m_L']
    N2 = params['N2']
    
    a = data_['a']
    z = a[-1]
    z_data = z[N2_index].data.numpy()
    
#     print('z_data', z_data)
    
    H = np.zeros((N2, m_L, m_L))
    for i in range(N2):
        H[i] -= np.outer(z_data[i], z_data[i])
        H[np.diag_indices(m_L)] += z_data[i]
    
    
    return H

def get_HV(V, data_, params):
    
    N2 = params['N2']
    N2_index = params['N2_index']
    m_L = params['m_L']
    
    V = np.reshape(V, (N2, m_L))
    
#     a = data_['a']
#     z = a[-1]
#     z_data = z[N2_index]
    
#     y = F.softmax(z_data, dim = 1)
    
    y = data_['y']
    
    y = y.data.numpy()
    
    HV = np.multiply(y, V)
    
    sum_HV = np.sum(HV, 1) # length N2
    
    for i in range(N2):
        HV[i] -= sum_HV[i] * y[i]
        
    
    HV = np.reshape(HV, m_L * N2)
    
    return HV

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
        
#         print('len(a_grad): ', len(a_grad))
        
#         print('len(a_grad[1]): ', len(a_grad[1]))
        
#         print('a_grad[1][0].size(): ', a_grad[1][0].size())
        
#         print('len(h)', len(h))
        
#         print('h[1].size(): ', h[1].size())
        
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
        print('Error! 1502')
        sys.exit()
    
    return v

def get_cache_momentum(data_, params):
    algorithm = params['algorithm']
    
    N2 = params['N2']
    
    
    
    if algorithm == 'SMW-GN':
    
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
        
        
        

        
        
        
#         a_grad_momentum = list(range(numlayers))
#         a_grad_momentum = numlayers * [[]]
        a_grad_momentum = []
        for l in range(numlayers):
            a_grad_momentum.append(torch.ones(N2, m_L, model.layersizes[l+1]))
        
#         print('a_grad_momentum', a_grad_momentum)
        
        v_tmp = 1 / len(X_mb[N2_index]) * torch.ones(len(X_mb[N2_index]))
        
        for i in range(m_L):
            
            
            z, a, h = model.forward(X_mb[N2_index])
            
            
        
#             z = a[-1]
            
            
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
        
        elif algorithm == 'SMW-Fisher' or algorithm == 'kfac' or algorithm == 'Fisher-block':
            a_grad_momentum = []
            h_momentum = []
            for l in range(numlayers):
                a_grad_momentum.append(N1 * (a[l].grad)[N2_index])
                h_momentum.append(h[l][N2_index])
            
    
        
        
        else:
            print('Error! 1503')
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
    lambda_ = params['lambda_']
#     numlayers = params['numlayers']
#     boost = params['boost']
#     drop = params['drop']
    
    N2_index = np.random.permutation(N1)[:N2]
    params['N2_index'] = N2_index
    
    m_L = data_['model'].layersizes[-1]
    params['m_L'] = m_L
    
    
    
    
    
    
    
#     start_time = time.time()
    
    data_ = get_cache_momentum(data_, params)

#     print('time for get cache momentum: ', time.time() - start_time)
    
#     start_time = time.time()
    
    a = data_['a']
    z = a[-1]
    z_data = z[N2_index]
    
    y = F.softmax(z_data, dim = 1)
    
    data_['y'] = y
    
#     print('time for compute y: ', time.time() - start_time)


    
    
    
    
    
        
    # compute the vector after D_t    
    

    
#     start_time = time.time()
    
    v = compute_JV(model_grad, data_, params)
    
    
#     print('v of compute JV: ', v)
    
#     print('time for compute JV: ', time.time() - start_time)
    

        
    
    
    # compute hat_v
    

#     start_time = time.time()
        
    D_t = get_D_t(data_, params)
    
#     print('D_t:', D_t)
    
#     print('v:', v)
#     print('torch.mean(v): ', torch.mean(v))
    
#     print('time for get D_t: ', time.time() - start_time)
    
#     start_time = time.time()
    
#     D_t_cho_fac = scipy.linalg.cho_factor(D_t)
#     hat_v = scipy.linalg.cho_solve(D_t_cho_fac, v.data.numpy())
    
    hat_v = np.linalg.solve(D_t, v.data.numpy())
    
    
    
    hat_v = get_HV(hat_v, data_, params)
    
    hat_v = np.float32(hat_v)
    
    hat_v = torch.from_numpy(hat_v)
    
#     hat_v = hat_v.long()
    
#     print('time for solve linear system: ', time.time() - start_time)
    
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
    
    
    
    delta = compute_sum_J_transpose_V_backp(hat_v, data_, params)

#     print('test delta')
#     delta = model_grad
    
#     print('time for compute J transpose V: ', time.time() - start_time)
    
#     print('\n')
    

    
    
        

        
        
#     print('delta[1]: ', delta[1])
#     print('model_grad[1]: ', model_grad[1])
        

    delta = get_multiply(1 / N2, delta, params)
    
    
    delta = get_subtract(model_grad, delta, params)
    
    delta = get_multiply(1 / lambda_, delta, params)
    
    
#     print('delta[1]: ', delta[1])
    
    
    
    ########################
#     print('model_grad[1]: ', model_grad[1])
#     
#     should_be_grad = computeFV(delta, data_, params)
    
    
#     print(get_dot_product(should_be_grad, delta))
#     should_be_grad = get_plus(should_be_grad, get_multiply(lambda_, delta, params), params) 
    
#     print('should_be_grad[1]: ', should_be_grad[1])
    
#     F_grad = computeFV(model_grad, data_, params)
    
#     print('print(get_dot_product(F_grad, model_grad, params))', get_dot_product(F_grad, model_grad, params))
    
#     print()
    
    
#     print('test')
    
        
    p = get_minus(delta, params)
    
#     p = get_minus(model_grad, params)
#     print('test sgd')

#     print('print(get_dot_product(p, model_grad, params))', get_dot_product(p, model_grad, params))
            

            
            

        

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
        
#     print('check if correct')
    
#     sys.exit()
    
    
    return delta





def update_lambda(p, data_, params):
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
    oldll_chunk = loss
    
    
        
#     print('ll_chunk: ', ll_chunk)
#     print('old ll_chunk: ', oldll_chunk)



   
    
        
        
    if oldll_chunk - ll_chunk < 0:
        rho = float("-inf")
    else:
        if algorithm == 'SMW-Fisher' or algorithm == 'SMW-GN':
            denom = - 0.5 * get_dot_product(model_grad, p, params)
        elif algorithm == 'kfac' or algorithm == 'SMW-Fisher-momentum':
            denom = computeFV(p, data_, params)
                
            denom = get_dot_product(p, denom, params)
            
#             print('p F p')
#             print(denom)
            
            
            denom = -0.5 * denom
            denom = denom - get_dot_product(model_grad, p, params) 
            
#             print('get_dot_product(model_grad, p, params) ')
#             print(get_dot_product(model_grad, p, params) )
                
        else:
            print('Error! 1504')
            sys.exit()
    
#     print('time for update lambda 1/4: ', time.time() - start_time)

            
        
        rho = (oldll_chunk - ll_chunk) / denom
        

#         print('oldll_chunk - ll_chunk: ', oldll_chunk - ll_chunk)
#         print('denom: ', denom)
#         print('rho: ', rho)

#     print('lambda', lambda_)
        
    
    
    
    # update lambda   
    if rho < 0.25:
        lambda_ = lambda_ * boost
    elif rho > 0.75:
        lambda_ = lambda_ * drop
        
    return lambda_

def SMW_Fisher_update(data_, params):   
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
    elif algorithm == 'SMW-Fisher-D_t-momentum':
        D_t_momentum = data_['D_t']
        
    loss = data_['loss']
    
    
    
    N1 = params['N1']
    N2 = params['N2']
    i = params['i']
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
    
        D_t_cho_fac = scipy.linalg.cho_factor(D_t)
        hat_v = scipy.linalg.cho_solve(D_t_cho_fac, v.data.numpy())
    
        hat_v = torch.from_numpy(hat_v)
    elif algorithm == 'SMW-Fisher-D_t-momentum':
        
        
    elif algorithm == 'SMW-Fisher-momentum':
            
        
        inverse_update_freq_D_t = params['inverse_update_freq_D_t']
        
        if i % inverse_update_freq_D_t == 0 or i < 100:
            D_t = get_D_t(data_, params)
            D_t_inv = torch.from_numpy(D_t).inverse()
            data_['D_t_inv'] = D_t_inv
        else:
            D_t_inv = data_['D_t_inv']
        
#         print('D_t_inv ', D_t_inv)
#         print('D_t_inv', D_t_inv.size())
#         print('v: ', v)
        
        hat_v = D_t_inv @ v
        
    else:
        print('Error! 1505')
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
        print('Error! 1506')
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
    
  

#     data_['model'] = model
    data_['p'] = p
    
#     if algorithm == 'SMW-Fisher-momentum':
#         data_['a_grad_momentum'] = a_grad_momentum
#         data_['h_momentum'] = h_momentum
    

        
    return data_, params

def get_new_loss(model, p, x, t):
    
#     print('p[1]: ', p[1])
    
    model_new = copy.deepcopy(model)
    
#     print('model_new.W[1]: ', model_new.W[1])
    
    for l in range(model_new.numlayers):
        model_new.W[l].data += p[l]
        
#     print('model_new.W[1]: ', model_new.W[1])
        
#     z, _, _ = model_new.forward(x)
    
#     print('z:', z)
    
#     print('t:', t)
    
#     loss = F.cross_entropy(z, t, reduction = 'mean')
#     print('test')
    
    loss= get_loss(model_new, x, t)
    
#     v = 1 / len(x) * torch.ones(len(x))
    
#     z, _, _ = model_new.forward(x)
#     loss = F.cross_entropy(z, t, reduction = 'mean')  

#     print('print(get_loss(model, x, t))', get_loss(model, x, t))

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

    if algorithm == 'SMW-GN':
        v = v.data.numpy()
        v = get_HV(v, data_, params)
        v = torch.from_numpy(v)
    
    

    delta = compute_sum_J_transpose_V_backp(v, data_, params)
    
    
    #############
#     N2 = params['N2']
#     m_L = params['m_L']
#     test_v = torch.zeros(m_L * N2)
#     test_v[0] = 1
    
#     print('print(compute_sum_J_transpose_V_backp(test_v, data_, params)): ', compute_sum_J_transpose_V_backp(test_v, data_, params))
    
#     print('test')
    
    
    ###############
    
    
#     test_v = torch.ones(m_L * N2)
    
#     aver_J = compute_sum_J_transpose_V_backp(test_v, data_, params)
    
    
    
#     aver_J = get_multiply(1 / N2, aver_J, params)
    
#     aver_J = get_multiply(1 / (N2 * m_L), aver_J, params)
    
#     print('norm 1', torch.sum(compute_JV(aver_J, data_, params)) / (m_L * N2))
    
#     print('norm 2', get_dot_product(aver_J, aver_J, params))
    
#     print('test 2')
    
    ###############
    
    
#     test_v = torch.ones(m_L * N2)
    
#     sum_J = compute_sum_J_transpose_V_backp(test_v, data_, params)
    
    
    
#     aver_J = get_multiply(1 / N2, aver_J, params)
    
#     aver_J = get_multiply(1 / (N2 * m_L), aver_J, params)
    
#     print('norm 1', torch.sum(compute_JV(sum_J, data_, params)))
    
#     print('norm 2', get_dot_product(sum_J, sum_J, params))
    
#     print('test 3')

    
#     print('delta[1].size(): ', delta[1].size())
    
    
    
#     delta = get_mean(delta, params)
    
    delta = get_multiply(1 / N2, delta, params)
    
#     print('delta[1].size(): ', delta[1].size())
    
    return delta




    
    
    



def kfac_update(data_, params):
    
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
            
            phi_ = np.sqrt( ( np.trace(A[l].data.numpy()) / A[l].shape[0] ) / ( np.trace(G[l].data.numpy()) / G[l].shape[0] ) )
            
            A_inv[l] = (A[l] + (phi_ * np.sqrt(lambda_)) * torch.eye(A[l].shape[0])).inverse()
            
#             print('G[l] + eps*torch.eye(G[l].shape[0]): ', G[l] + eps*torch.eye(G[l].shape[0]))
            
            G_inv[l] = (G[l] + (1 / phi_ * np.sqrt(lambda_)) * torch.eye(G[l].shape[0])).inverse()

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


    ##############
    algorithm = params['algorithm']
    layersizes = model.layersizes
    if algorithm == 'Fisher-block':
        delta = []
        for l in range(numlayers):
#             F = lambda_ * torch.eye(layersizes[l] * layersizes[l+1]) + np.kron(A_block, G_block)
            
#             print('print(model_grad[l])', model_grad[l])

            params['algorithm'] = 'SMW-Fisher'
    
            data_test, _ = SMW_Fisher_update(data_, params)
    
#             print('SMW_Fisher_update(data_, params)', SMW_Fisher_update(data_, params))
            

            p = data_test['p']
        
            delta.append(-p)
        
            params['algorithm'] = 'Fisher-block'
            
#             delta.append()
        

    
    
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
    
