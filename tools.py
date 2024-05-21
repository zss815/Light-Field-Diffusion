import numpy as np
import torch
import torch.nn as nn


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def beta_schedule(schedule, beta_start, beta_end, num_diffusion_steps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_steps, dtype=np.float64) ** 2)
    elif schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_steps, dtype=np.float64)
    elif schedule == "const":
        betas = beta_end * np.ones(num_diffusion_steps, dtype=np.float64)
    elif schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_steps, 1, num_diffusion_steps, dtype=np.float64)
    elif schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_steps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(schedule)
    assert betas.shape == (num_diffusion_steps,)
    return betas


def Sample_LD(inp, seq, model, betas, eta=0.):
    with torch.no_grad():
        B,_,H,W = inp.shape
        zt = torch.randn(B,4,H//8,W//8).to(inp.device)   #[B,4,h,w]
        seq_pre = [-1] + list(seq[:-1])
        betas = torch.cat([torch.zeros(1).to(inp.device), betas], dim=0)  #(0,betas)
        alphas = (1 - betas).cumprod(dim=0)  #(1,alphas)

        for i, j in zip(reversed(seq), reversed(seq_pre)):
            t = (torch.ones(B) * i).to(inp.device)  #[B]
            t_pre = (torch.ones(B) * j).to(inp.device)  #[B]
            a = alphas.index_select(0, (t + 1).long()).reshape(-1,1,1,1)  #[B,1,1,1]
            a_pre = alphas.index_select(0, (t_pre + 1).long()).reshape(-1,1,1,1)  #[B,1,1,1]
            
            e = model(zt, inp, t)  #[B,4,h,w]

            z0 = (zt - e * (1 - a).sqrt()) / a.sqrt()  #[B,4,h,w]
            c1 = eta * ((1 - a / a_pre) * (1 - a_pre) / (1 - a)).sqrt()
            c2 = ((1 - a_pre) - c1 ** 2).sqrt()
            zt = a_pre.sqrt() * z0 + c1 * torch.randn_like(z0) + c2 * e  #[B,4,h,w]

    return zt


def Sample_LDLF(inp, seq, model, betas, eta=0.):
    with torch.no_grad():
        B,N,_,H,W=inp.shape   #[B,A*A,6,H,W]
        inp=inp.reshape(B*N,-1,H,W)  #[B*A*A,6,H,W]

        #Noise Prior (shared central noise)
        p_sq=0.8
        ec = torch.randn(B,4,H//8,W//8).to(inp.device)  #[B,4,h,w]
        ei = torch.randn(B,N,4,H//8,W//8).to(inp.device)*((1-p_sq)**0.5)    #[B,A*A,4,h,w]
        e = ec[:,None,:,:,:].repeat(1,N,1,1,1)*(p_sq**0.5)+ei   #[B,A*A,4,h,w]
        e[:,N//2,:,:,:]= ec
        zt=e.reshape(B*N,4,H//8,W//8)   #[B*A*A,4,h,w]

        seq_pre = [-1] + list(seq[:-1])
        betas = torch.cat([torch.zeros(1).to(inp.device), betas], dim=0)  #(0,betas)
        alphas = (1 - betas).cumprod(dim=0)  #(1,alphas)

        for i, j in zip(reversed(seq), reversed(seq_pre)):
            t = (torch.ones(B) * i).to(inp.device)  #[B]
            t_pre = (torch.ones(B) * j).to(inp.device)  #[B]
            a = alphas.index_select(0, (t + 1).long()).reshape(-1,1,1,1,1).repeat(1,N,1,1,1).reshape(B*N,1,1,1)   #[B*A*A,1,1,1]
            a_pre = alphas.index_select(0, (t_pre + 1).long()).reshape(-1,1,1,1,1).repeat(1,N,1,1,1).reshape(B*N,1,1,1)  #[B*A*A,1,1,1]
            
            e = model(zt, inp, t)  #[B*A*A,4,h,w]

            z0 = (zt - e * (1 - a).sqrt()) / a.sqrt()  #[B*A*A,4,h,w]
            c1 = eta * ((1 - a / a_pre) * (1 - a_pre) / (1 - a)).sqrt()
            c2 = ((1 - a_pre) - c1 ** 2).sqrt()
            zt = a_pre.sqrt() * z0 + c1 * torch.randn_like(z0) + c2 * e  #[B*A*A,4,h,w]

    return zt
        

class EMA(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict