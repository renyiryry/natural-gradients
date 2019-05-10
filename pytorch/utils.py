def SMW_Fisher_update(data_, params):
    # a[l]: size N1 * m[l+1]
    # h[l]: size N1 * m[l]
    # model.W[l]: size m[l+1] * m[l]
    
    
    import torch
    import numpy as np
    import scipy
    
    X_mb = data_['X_mb']
    cache = data_['cache']
    z = data_['z']
#     A = data_['A']
#     G = data_['G']
#     A_inv = data_['A_inv']
#     G_inv = data_['G_inv']
    model = data_['model']
    
    a_grad_momentum = data_['a_grad_momentum']
    h_momnetum = data_['h_momentum']
    
    
    
    N1 = params['N1']
    N2 = params['N2']
#     i = params['i']
#     inverse_update_freq = params['inverse_update_freq']
    eps = params['eps']
    alpha = params['alpha']
    lambda_ = params['lambda_']
    numlayers = params['numlayers']
    algorithm = params['algorithm']
    
    a = []
    h = [X_mb]
    for ii in range(len(cache)):
        if ii % 2 == 0:
            a.append(cache[ii])
        else:
            h.append(cache[ii])        
    a.append(z)
    
    
    
#     print('a[0].size(): ', a[0].size())
#     print('a[1].size(): ', a[1].size())
#     print('a[2].size(): ', a[2].size())
    
#     print('h[0].size(): ', h[0].size())
#     print('h[1].size(): ', h[1].size())
#     print('h[2].size(): ', h[2].size())

    
    
    
    N2_index = np.random.permutation(N1)[:N2]
    
    # Update running estimates of KFAC
    if algorithm == 'SMW-Fisher-momentum':
        rho = min(1-1/i, 0.95)
    elif algorithm == 'SMW-Fisher':
        rho = 0
    else:
        print('Error!')
        sys.exit()

    for l in range(3):
        a_grad_momentum[l] = rho * a_grad_momentum[l] + (1-rho) * (a[l].grad)[N2_index]
        h_momentum[l] = rho * h_momentum[l] + (1-rho) * h[l][N2_index]
    
    
    
    
    # compute D_t
    D_t = lambda_ * torch.eye(N2)
    for l in range(numlayers):
        
#         print('h[l][N2_index] @ h[l][N2_index].t().size(): ', (h[l][N2_index] @ h[l][N2_index].t()).size())
#         print('(a[l].grad)[N2_index] @ (a[l].grad)[N2_index].t().size(): ',
#               ((a[l].grad)[N2_index] @ (a[l].grad)[N2_index].t()).size())
        
        D_t += 1 / N2 * (a_grad_momentum[l] @ a_grad_momentum[l].t()) * (h_momentum[l] @ h_momentum[l].t())
        
    # compute the vector after D_t
    v = torch.zeros(N2)
    
#     print('model.W[0].size(): ', model.W[0].size())
#     print('model.W[1].size(): ', model.W[1].size())
#     print('model.W[2].size(): ', model.W[2].size())
    
    
    
    for l in range(numlayers):
        
#         model.W[l] @ h[l] # m[l+1] * N2
    # a[l][N2_index] @ model.W[l] # N2 * m[l]
    # (a[l][N2_index] @ model.W[l]) * h[l][N2_index] # N2 * m[l]
    #  torch.sum((a[l][N2_index] @ model.W[l]) * h[l][N2_index], dim = 1)


        
        v += torch.sum((a_grad_momentum[l] @ model.W[l]) * h_momentum[l], dim = 1)
        
    
    
    # compute hat_v
#     hat_v, _ = torch.solve(v, D_t)

#     print('D_t:', D_t)

    hat_v = scipy.linalg.cho_solve(scipy.linalg.cho_factor(D_t.data.numpy()), v.data.numpy())
    
    hat_v = torch.from_numpy(hat_v)
    
#     print('hat_v: ', hat_v)
#     print('1 - hat_v: ', 1 - hat_v)
#     print('torch.max(hat_v): ', torch.max(hat_v))
#     print('torch.min(hat_v): ', torch.min(hat_v))

#     hat_v = torch.zeros(N2)
    
#     print('hat_v: ', hat_v)
#     print('1 - hat_v: ', 1 - hat_v)
    

    
    # update parameters
    for l in range(numlayers):
        
#         For two 2D tensors a and b (of size [b,n] and [b,m] respectively),
# a[:, :, None] @ b[:, None, :] (of size [b,n,m]) gives the outer product operated on each item in the batch.
        
#         print('a[l][N2_index][:, :, None] @ h[l][N2_index][:, None, :].size(): ', 
#               (a[l][N2_index][:, :, None] @ h[l][N2_index][:, None, :]).size()) 
        
#         print('(torch.mean((1 - hat_v)[:, None, None] * (a[l][N2_index][:, :, None] @ h[l][N2_index][:, None, :]). dim=0)).size(): ', 
#               (torch.mean((1 - hat_v)[:, None, None] * (a[l][N2_index][:, :, None] @ h[l][N2_index][:, None, :]), dim=0)).size())
        
#         print('a[l][N2_index][:, :, None]: ', a[l][N2_index][:, :, None])
#         print('h[l][N2_index][:, None, :]: ', h[l][N2_index][:, None, :])
    
        delta = a_grad_momentum[l][:, :, None] @ h_momentum[l][:, None, :] # [N2, m[l+1], m[l]]
        delta = (1 - hat_v)[:, None, None] * delta # [N2, m[l+1], m[l]]
#         delta = torch.mean(delta, dim = 0) # [m[l+1], m[l]]
        delta = torch.sum(delta, dim = 0) # [m[l+1], m[l]]
        delta = 1 / lambda_ * delta
        
#         print('delta.size():', delta.size())
#         print('delta: ', delta)
    
#         print('delta * N2: ', delta * N2)
        
#         print('model.W[l].grad.size(): ', model.W[l].grad.size())
#         print('model.W[l].grad: ', model.W[l].grad)

#         print('model.W[1].data in utils: ', model.W[1].data)
    
        model.W[l].data -= alpha * delta
        
#         print('model.W[1].data in utils: ', model.W[1].data)
        
    
    

    

    # Step
#     for k in range(3):
#         Amortize the inverse. Only update inverses every now and then
#         if (i-1) % inverse_update_freq == 0:
#             A_inv[k] = (A[k] + eps*torch.eye(A[k].shape[0])).inverse()
#             G_inv[k] = (G[k] + eps*torch.eye(G[k].shape[0])).inverse()

#         delta = G_inv[k] @ model.W[k].grad.data @ A_inv[k]
#         model.W[k].data -= alpha * delta
        
#     data_['A'] = A
#     data_['G'] = G
#     data_['A_inv'] = A_inv
#     data_['G_inv'] = G_inv
    data_['model'] = model
    
    data_['a_grad_momentum'] = a_grad_momentum
    data_['h_momentum'] = h_momnetum
    
#     print('model.W[1] in utils: ', model.W[1])
#     print('model.W[1].data in utils: ', model.W[1].data)
        
    return data_

def kfac_update(data_, params):
    import torch
    import sys
    
    X_mb = data_['X_mb']
    
#     a1 = data_['a1']
#     a2 = data_['a2']
#     h1 = data_['h1']
#     h2 = data_['h2']
    
    cache = data_['cache']
    
    z = data_['z']
    A = data_['A']
    G = data_['G']
    A_inv = data_['A_inv']
    G_inv = data_['G_inv']
    model = data_['model']
    
    N1 = params['N1']
    i = params['i']
    inverse_update_freq = params['inverse_update_freq']
    eps = params['eps']
    alpha = params['alpha']
    numlayers = params['numlayers']
    
    a = []
    h = []
    for ii in range(0, len(cache)):
        if ii % 2 == 0:
            a.append(cache[ii])
        else:
            h.append(cache[ii])
    
    
    
    # KFAC matrices
#     G1_ = 1/m * a1.grad.t() @ a1.grad
#     G2_ = 1/m * a2.grad.t() @ a2.grad
#     G3_ = 1/m * z.grad.t() @ z.grad
    
    
#     A1_ = 1/m * X_mb.t() @ X_mb
#     A2_ = 1/m * h1.t() @ h1
#     A3_ = 1/m * h2.t() @ h2
    
    G_ = []
    for l in range(0, numlayers):
        if l < numlayers - 1:
            G_.append(1/N1 * a[l].grad.t() @ a[l].grad)
        elif l == numlayers - 1:
            G_.append(1/N1 * z.grad.t() @ z.grad)
        else:
            print('Error!')
            sys.exit()
            
    A_ = []
    for l in range(0, numlayers):
        if l == 0:
            A_.append(1/N1 * X_mb.t() @ X_mb)
        else:
            A_.append(1/N1 * h[l-1].t() @ h[l-1])
        

#     G_ = [G1_, G2_, G3_]
#     A_ = [A1_, A2_, A3_]

    # Update running estimates of KFAC
    rho = min(1-1/i, 0.95)

    for l in range(numlayers):
        
        print('in utils')
        print('A[l].size(): ', A[l].size())
        print('A_[l].size(): ', A_[l].size())
        
        A[l] = rho*A[l] + (1-rho)*A_[l]
        G[l] = rho*G[l] + (1-rho)*G_[l]
        
        print('A[l].size(): ', A[l].size())
        
#         print('G[l]: ', G[l])

    # Step
    for l in range(numlayers):
        
#         print(type(G[l]))
        
#         print('i = ', i)
        
        # Amortize the inverse. Only update inverses every now and then
        if (i-1) % inverse_update_freq == 0:
            
            
            
            A_inv[l] = (A[l] + eps*torch.eye(A[l].shape[0])).inverse()
            
#             print('G[l] + eps*torch.eye(G[l].shape[0]): ', G[l] + eps*torch.eye(G[l].shape[0]))
            
            G_inv[l] = (G[l] + eps*torch.eye(G[l].shape[0])).inverse()

#         print(type(G_inv[l]))
#         print(type(model.W[l].grad.data))
#         print(type(A_inv[l]))

#         print('G_inv[l]: ', G_inv[l])
#         print('model.W[l]: ', model.W[l])
#         print('model.W[l].grad.data: ', model.W[l].grad.data)
#         print('A_inv[l]: ', A_inv[l])
        
            
        delta = G_inv[l] @ model.W[l].grad.data @ A_inv[l]
        
#         print('delta: ', delta)
        
        model.W[l].data -= alpha * delta
        
    data_['A'] = A
    data_['G'] = G
    data_['A_inv'] = A_inv
    data_['G_inv'] = G_inv
    data_['model'] = model
        
    return data_
