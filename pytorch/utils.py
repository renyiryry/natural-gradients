def kfac_update(data_, params):
    import torch
    
    X_mb = data_['X_mb']
    a1 = data_['a1']
    a2 = data_['a2']
    h1 = data_['h1']
    h2 = data_['h2']
    z = data_['z']
    A = data_['A']
    G = data_['G']
    A_inv = data_['A_inv']
    G_inv = data_['G_inv']
    
    m = params['m']
    i = params['i']
    inverse_update_freq = params['inverse_update_freq']
    eps = params['eps']
    
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
        
    data_['A'] = A
    data_['G'] = G
    data_['A_inv'] = A_inv
    data_['G_inv'] = G_inv
        
    return data_
