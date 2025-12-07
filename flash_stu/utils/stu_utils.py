import numpy as np
import torch
import torch.nn.functional as F

from flashfftconv import FlashFFTConv

from flash_stu.utils.numerics import nearest_power_of_two

'''
def get_hankel(seq_len: int, use_hankel_L: bool = False) -> np.ndarray:
    entries = np.arange(1, seq_len + 1, dtype=np.float64)
    i_plus_j = entries[:, None] + entries[None, :]

    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    elif not use_hankel_L:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    else:
        raise ValueError("use_hankel_L must be a boolean")

    return Z

def get_spectral_filters(
    seq_len: int, 
    K: int, 
    use_hankel_L: bool = False, 
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    assert torch.cuda.is_available(), "CUDA is required."
    Z = get_hankel(seq_len, use_hankel_L)
    sigma, phi = np.linalg.eigh(Z)
    sigma, phi = sigma[-K:], phi[:, -K:]
    phi *= sigma ** 0.25
    return torch.tensor(phi, device=device, dtype=dtype)
'''

def get_hankel_vals(
  seq_len: int, 
  use_hankel_L: bool = False,
  device: torch.device = None      
) -> torch.Tensor:
    
    k = torch.arange(2, 2 * seq_len + 1, dtype=torch.float64, device=device)

    if use_hankel_L:
        sgn = (-1.0) ** (k - 2.0) + 1.0
        denom = (k + 3.0) * (k - 1.0) * (k + 1.0)
        vals = sgn * (8.0 / denom)
    elif not use_hankel_L:
        vals = 2.0 / (k**3 - k)
    else:
        raise ValueError("use_hankel_L must be a boolean")
     
    return vals

def implicitHankelRPCholesky(
    seq_len: int,
    k: int,
    use_hankel_L: bool = False, 
    tol: float = 1e-8,
    device: torch.device = None,
    greedy: bool = True
) -> torch.Tensor:
    n = seq_len
    hankel_vals = get_hankel_vals(n, use_hankel_L, device=device)
    diag_init = hankel_vals[::2]
    diag = diag_init
    energy = torch.zeros_like(diag)
    pivots = []
    trace_init = torch.sum(diag)
    max_init = torch.max(diag)
    F = torch.zeros((n, k), device=device, dtype=diag.dtype)
    for i in range(k):
        if greedy:
            pivot_id = torch.argmax(diag)       
        else:
            trace = torch.sum(diag)
            if trace < tol * trace_init: break
            pivot_id = torch.multinomial(diag/trace, 1).item()
        pivot = diag[pivot_id]
        if pivot < tol * max_init:
            break
        else: 
            pivots.append(pivot_id)
        g = hankel_vals[pivot_id:pivot_id + n]
        g = g - F[:,:i] @ torch.conj(F[pivot_id,0:i]).T
        F[:, i] = g / torch.sqrt(g[pivot_id])
        energy +=  torch.abs(F[:, i]) ** 2
        diag = diag_init - energy
        diag = torch.clamp(diag, 0)
    rank = len(pivots)
    F = F[:, :rank]
    return F, pivots

def get_spectral_filters(
    seq_len: int, 
    k: int, 
    use_hankel_L: bool = False, 
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
    greedy: bool = True
) -> torch.Tensor:
    assert torch.cuda.is_available(), "CUDA is required."
    F, p = implicitHankelRPCholesky(seq_len, k, 1e-8, use_hankel_L, device, greedy)
    U, S, _ = torch.linalg.svd(F, full_matrices=False)
    phi = U * (S ** 0.5).unsqueeze(0)
    return phi.to(device=device, dtype=dtype)   

def convolve(u: torch.Tensor, v: torch.Tensor, n: int, use_approx: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, d_in = u.shape

    sgn = torch.full((1, seq_len, 1), 1, device=u.device)
    sgn[:, 1::2] *= -1
    if use_approx:
        _, d_out = v.shape
        v = v.view(1, -1, d_out, 1).to(torch.float32)
    else:
        _, K = v.shape
        sgn = sgn.unsqueeze(-1)
        v = v.view(1, -1, K, 1, 1).to(torch.float32) # (bsz, seq_len, K, d_in, stack)
        u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)

    v = torch.fft.rfft(v, n=n, dim=1)
    U = torch.stack([u, u * sgn], dim=-1).to(torch.float32)
    U = torch.fft.rfft(U, n=n, dim=1)
    U_conv = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)
    U_minus = U_minus * sgn

    return U_plus, U_minus

def flash_convolve(
    u: torch.Tensor, v: torch.Tensor, flash_fft: FlashFFTConv, use_approx: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, d_in = u.shape
    _, K = v.shape

    padded_len = nearest_power_of_two(seq_len, round_up=True)
    pad_len = padded_len - seq_len

    sgn = torch.full((1, 1, padded_len), 1, device=u.device)
    sgn[:, :, 1::2] = -1

    if use_approx:
        u_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).contiguous()
        v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).contiguous()
        u_conv = torch.stack([u_padded, u_padded * sgn], dim=0).reshape(2 * bsz, d_in, padded_len)
    else:
        u_k_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).repeat_interleave(K, dim=1).contiguous()
        v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).repeat(d_in, 1).contiguous()
        u_conv = torch.stack([u_k_padded, u_k_padded * sgn], dim=0).reshape(2 * bsz, K * d_in, padded_len)

    U_conv = flash_fft(u_conv, v_padded)

    # Trim the output back to the original sequence length
    U_conv = U_conv[..., :seq_len]

    u_plus, u_minus = torch.chunk(U_conv, 2, dim=0)

    if use_approx:
        u_minus = u_minus * sgn[:, :, :seq_len]
        U_plus, U_minus = u_plus.transpose(1, 2), u_minus.transpose(1, 2)
    else:
        sgn = sgn[:, :, :seq_len].unsqueeze(-1).transpose(1, 2)
        U_plus = u_plus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous()
        U_minus = u_minus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous() * sgn

    return U_plus, U_minus
