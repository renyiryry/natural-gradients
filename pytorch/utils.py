def SMW_Fisher_update(data_, params):
    import torch
    
    X_mb = data_['X_mb']
    
    a1 = data_['a1']
    a2 = data_['a2']
    h1 = data_['h1']
    h2 = data_['h2']
    
    cache = data_['cache']
    
    z = data_['z']
#     A = data_['A']
#     G = data_['G']
#     A_inv = data_['A_inv']
#     G_inv = data_['G_inv']
    model = data_['model']
    
    m = params['m']
#     i = params['i']
#     inverse_update_freq = params['inverse_update_freq']
    eps = params['eps']
    alpha = params['alpha']
    
    a = []
    h = []
    for ii in range(0, len(cache)):
        if ii % 2 == 0:
            a.append(cache[ii])
        else:
            h.append(cache[ii])
    
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
#     rho = min(1-1/i, 0.95)

#     for k in range(3):
#         A[k] = rho*A[k] + (1-rho)*A_[k]
#         G[k] = rho*G[k] + (1-rho)*G_[k]

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
    
    m = params['m']
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
            G_.append(1/m * a[l].grad.t() @ a[l].grad)
        elif l == numlayers - 1:
            G_.append(1/m * z.grad.t() @ z.grad)
        else:
            print('Error!')
            sys.exit()
            
    A_ = []
    for l in range(0, numlayers):
        if l == 0:
            A_.append(1/m * X_mb.t() @ X_mb)
        else:
            A_.append(1/m * h[l-1].t() @ h[l-1])
        

#     G_ = [G1_, G2_, G3_]
#     A_ = [A1_, A2_, A3_]

    # Update running estimates of KFAC
    rho = min(1-1/i, 0.95)

    for l in range(numlayers):
        A[l] = rho*A[l] + (1-rho)*A_[l]
        G[l] = rho*G[l] + (1-rho)*G_[l]

    # Step
    for l in range(numlayers):
        
#         print(type(G[l]))
        
#         print('i = ', i)
        
        # Amortize the inverse. Only update inverses every now and then
        if (i-1) % inverse_update_freq == 0:
            
            
            
            A_inv[l] = (A[l] + eps*torch.eye(A[l].shape[0])).inverse()
            G_inv[l] = (G[l] + eps*torch.eye(G[l].shape[0])).inverse()

#         print(type(G_inv[l]))
#         print(type(model.W[l].grad.data))
#         print(type(A_inv[l]))
        
            
        delta = G_inv[l] @ model.W[l].grad.data @ A_inv[l]
        
        print('delta: ', delta)
        
        model.W[l].data -= alpha * delta
        
    data_['A'] = A
    data_['G'] = G
    data_['A_inv'] = A_inv
    data_['G_inv'] = G_inv
    data_['model'] = model
        
    return data_
