"""
kernel.py — Generalized batched thin SVD + Procrustes alignment.

Part of the GEOLIP ecosystem.
Repository: AbstractEyes/geolip-core
Package: geolip

Provides:
  batched_svd(A)                    — Auto-dispatched thin SVD for (B, M, N)
  batched_svd2(A)                   — Fused Triton kernel for N=2
  batched_svd3(A)                   — Fused Triton kernel for N=3
  gram_eigh_svd(A)                  — Gram + eigh hybrid for any N
  newton_schulz_invsqrt(G)          — Batched G^{-1/2} via pure bmm
  batched_procrustes(src, tgt)      — Subspace-preserving Procrustes alignment

Performance (NVIDIA RTX PRO 6000 Blackwell, B=512, M=1024):
  N=2:   0.021ms  (3,850× vs torch)
  N=3:   0.022ms  (5,488× vs torch)
  N=8:   0.290ms  (584× vs torch)
  N=32:  0.781ms  (388× vs torch)

Mathematical lineage:
  Eckart-Young (1936), Jacobi (1846), Golub-Reinsch (1970), Batcher (1968)

Author: AbstractPhil + Claude Opus 4.6
License: Apache 2.0
"""

import math
import torch
import torch.nn.functional as F

__all__ = [
    'batched_svd',
    'batched_svd2',
    'batched_svd3',
    'gram_eigh_svd',
    'newton_schulz_invsqrt',
    'batched_procrustes',
    'HAS_TRITON',
]


# ═══════════════════════════════════════════════════════════════════════════════
# TRITON FUSED KERNELS (N=2, N=3)
# ═══════════════════════════════════════════════════════════════════════════════

HAS_TRITON = False

try:
    import triton
    import triton.language as tl

    # ── N=2: Closed-form Jacobi rotation ─────────────────────────────────

    @triton.jit
    def _svd2_kernel(
        A_ptr, U_ptr, S_ptr, Vh_ptr,
        M: tl.constexpr, BLOCK_M: tl.constexpr, EPS: tl.constexpr,
    ):
        bid = tl.program_id(0)
        base = bid * M * 2
        g00 = tl.zeros([], dtype=tl.float32)
        g01 = tl.zeros([], dtype=tl.float32)
        g11 = tl.zeros([], dtype=tl.float32)
        for block_start in range(0, M, BLOCK_M):
            offs = tl.arange(0, BLOCK_M)
            row_idx = block_start + offs
            mask = row_idx < M
            a0 = tl.load(A_ptr + base + row_idx * 2 + 0, mask=mask, other=0.0).to(tl.float32)
            a1 = tl.load(A_ptr + base + row_idx * 2 + 1, mask=mask, other=0.0).to(tl.float32)
            g00 += tl.sum(a0 * a0)
            g01 += tl.sum(a0 * a1)
            g11 += tl.sum(a1 * a1)
        # Jacobi rotation (single step, no iteration needed for 2×2)
        off_diag = g01
        diag_diff = g11 - g00
        abs_off = tl.abs(off_diag)
        tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
        t = tl.where(abs_off > EPS,
            tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)),
            0.0)
        c = 1.0 / tl.sqrt(1.0 + t * t)
        s = t * c
        eig0 = c * c * g00 - 2.0 * s * c * g01 + s * s * g11
        eig1 = s * s * g00 + 2.0 * s * c * g01 + c * c * g11
        s0 = tl.sqrt(tl.maximum(eig0, EPS))
        s1 = tl.sqrt(tl.maximum(eig1, EPS))
        v00 = c
        v01 = s
        v10 = -s
        v11 = c
        # Sort descending
        do_swap = s0 < s1
        s0, s1 = tl.where(do_swap, s1, s0), tl.where(do_swap, s0, s1)
        tv = v00
        v00 = tl.where(do_swap, v01, v00)
        v01 = tl.where(do_swap, tv, v01)
        tv = v10
        v10 = tl.where(do_swap, v11, v10)
        v11 = tl.where(do_swap, tv, v11)
        # Write S, Vh
        tl.store(S_ptr + bid * 2 + 0, s0)
        tl.store(S_ptr + bid * 2 + 1, s1)
        vh_base = bid * 4
        tl.store(Vh_ptr + vh_base + 0, v00)
        tl.store(Vh_ptr + vh_base + 1, v10)
        tl.store(Vh_ptr + vh_base + 2, v01)
        tl.store(Vh_ptr + vh_base + 3, v11)
        # U recovery
        inv_s0 = 1.0 / (s0 + EPS)
        inv_s1 = 1.0 / (s1 + EPS)
        for block_start in range(0, M, BLOCK_M):
            offs = tl.arange(0, BLOCK_M)
            row_idx = block_start + offs
            mask = row_idx < M
            a0 = tl.load(A_ptr + base + row_idx * 2 + 0, mask=mask, other=0.0).to(tl.float32)
            a1 = tl.load(A_ptr + base + row_idx * 2 + 1, mask=mask, other=0.0).to(tl.float32)
            u0 = (a0 * v00 + a1 * v10) * inv_s0
            u1 = (a0 * v01 + a1 * v11) * inv_s1
            u_base = bid * M * 2
            tl.store(U_ptr + u_base + row_idx * 2 + 0, u0, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 2 + 1, u1, mask=mask)

    # ── N=3: Cyclic Jacobi in scalar registers ───────────────────────────

    @triton.jit
    def _svd3_kernel(
        A_ptr, U_ptr, S_ptr, Vh_ptr,
        M: tl.constexpr, BLOCK_M: tl.constexpr,
        JACOBI_ITERS: tl.constexpr, EPS: tl.constexpr,
    ):
        bid = tl.program_id(0)
        g00 = tl.zeros([], dtype=tl.float32)
        g01 = tl.zeros([], dtype=tl.float32)
        g02 = tl.zeros([], dtype=tl.float32)
        g11 = tl.zeros([], dtype=tl.float32)
        g12 = tl.zeros([], dtype=tl.float32)
        g22 = tl.zeros([], dtype=tl.float32)
        base = bid * M * 3
        for block_start in range(0, M, BLOCK_M):
            offs = tl.arange(0, BLOCK_M)
            row_idx = block_start + offs
            mask = row_idx < M
            a0 = tl.load(A_ptr + base + row_idx * 3 + 0, mask=mask, other=0.0).to(tl.float32)
            a1 = tl.load(A_ptr + base + row_idx * 3 + 1, mask=mask, other=0.0).to(tl.float32)
            a2 = tl.load(A_ptr + base + row_idx * 3 + 2, mask=mask, other=0.0).to(tl.float32)
            g00 += tl.sum(a0 * a0)
            g01 += tl.sum(a0 * a1)
            g02 += tl.sum(a0 * a2)
            g11 += tl.sum(a1 * a1)
            g12 += tl.sum(a1 * a2)
            g22 += tl.sum(a2 * a2)
        v00 = 1.0
        v01 = 0.0
        v02 = 0.0
        v10 = 0.0
        v11 = 1.0
        v12 = 0.0
        v20 = 0.0
        v21 = 0.0
        v22 = 1.0
        for _ in range(JACOBI_ITERS):
            # pair (0,1)
            off_diag = g01
            diag_diff = g11 - g00
            abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t)
            s = t * c
            ng00 = c*c*g00 - 2.0*s*c*g01 + s*s*g11
            ng11 = s*s*g00 + 2.0*s*c*g01 + c*c*g11
            ng02 = c*g02 - s*g12
            ng12 = s*g02 + c*g12
            g00 = ng00
            g11 = ng11
            g01 = 0.0
            g02 = ng02
            g12 = ng12
            nv00 = c*v00 - s*v01
            nv01 = s*v00 + c*v01
            nv10 = c*v10 - s*v11
            nv11 = s*v10 + c*v11
            nv20 = c*v20 - s*v21
            nv21 = s*v20 + c*v21
            v00 = nv00
            v01 = nv01
            v10 = nv10
            v11 = nv11
            v20 = nv20
            v21 = nv21
            # pair (0,2)
            off_diag = g02
            diag_diff = g22 - g00
            abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t)
            s = t * c
            ng00 = c*c*g00 - 2.0*s*c*g02 + s*s*g22
            ng22 = s*s*g00 + 2.0*s*c*g02 + c*c*g22
            ng01 = c*g01 - s*g12
            ng12b = s*g01 + c*g12
            g00 = ng00
            g22 = ng22
            g02 = 0.0
            g01 = ng01
            g12 = ng12b
            nv00 = c*v00 - s*v02
            nv02 = s*v00 + c*v02
            nv10 = c*v10 - s*v12
            nv12 = s*v10 + c*v12
            nv20 = c*v20 - s*v22
            nv22 = s*v20 + c*v22
            v00 = nv00
            v02 = nv02
            v10 = nv10
            v12 = nv12
            v20 = nv20
            v22 = nv22
            # pair (1,2)
            off_diag = g12
            diag_diff = g22 - g11
            abs_off = tl.abs(off_diag)
            tau = tl.where(abs_off > EPS, diag_diff / (2.0 * off_diag), 0.0)
            t = tl.where(abs_off > EPS, tl.where(tau >= 0, 1.0, -1.0) / (tl.abs(tau) + tl.sqrt(1.0 + tau * tau)), 0.0)
            c = 1.0 / tl.sqrt(1.0 + t * t)
            s = t * c
            ng11 = c*c*g11 - 2.0*s*c*g12 + s*s*g22
            ng22 = s*s*g11 + 2.0*s*c*g12 + c*c*g22
            ng01 = c*g01 - s*g02
            ng02b = s*g01 + c*g02
            g11 = ng11
            g22 = ng22
            g12 = 0.0
            g01 = ng01
            g02 = ng02b
            nv01 = c*v01 - s*v02
            nv02 = s*v01 + c*v02
            nv11 = c*v11 - s*v12
            nv12 = s*v11 + c*v12
            nv21 = c*v21 - s*v22
            nv22 = s*v21 + c*v22
            v01 = nv01
            v02 = nv02
            v11 = nv11
            v12 = nv12
            v21 = nv21
            v22 = nv22
        # Sort descending
        s0 = tl.sqrt(tl.maximum(g00, EPS))
        s1 = tl.sqrt(tl.maximum(g11, EPS))
        s2 = tl.sqrt(tl.maximum(g22, EPS))
        do_swap = s0 < s1
        s0, s1 = tl.where(do_swap, s1, s0), tl.where(do_swap, s0, s1)
        tv = v00
        v00 = tl.where(do_swap, v01, v00)
        v01 = tl.where(do_swap, tv, v01)
        tv = v10
        v10 = tl.where(do_swap, v11, v10)
        v11 = tl.where(do_swap, tv, v11)
        tv = v20
        v20 = tl.where(do_swap, v21, v20)
        v21 = tl.where(do_swap, tv, v21)
        do_swap = s0 < s2
        s0, s2 = tl.where(do_swap, s2, s0), tl.where(do_swap, s0, s2)
        tv = v00
        v00 = tl.where(do_swap, v02, v00)
        v02 = tl.where(do_swap, tv, v02)
        tv = v10
        v10 = tl.where(do_swap, v12, v10)
        v12 = tl.where(do_swap, tv, v12)
        tv = v20
        v20 = tl.where(do_swap, v22, v20)
        v22 = tl.where(do_swap, tv, v22)
        do_swap = s1 < s2
        s1, s2 = tl.where(do_swap, s2, s1), tl.where(do_swap, s1, s2)
        tv = v01
        v01 = tl.where(do_swap, v02, v01)
        v02 = tl.where(do_swap, tv, v02)
        tv = v11
        v11 = tl.where(do_swap, v12, v11)
        v12 = tl.where(do_swap, tv, v12)
        tv = v21
        v21 = tl.where(do_swap, v22, v21)
        v22 = tl.where(do_swap, tv, v22)
        # Write S
        s_base = bid * 3
        tl.store(S_ptr + s_base + 0, s0)
        tl.store(S_ptr + s_base + 1, s1)
        tl.store(S_ptr + s_base + 2, s2)
        # Write Vh = V^T
        vh_base = bid * 9
        tl.store(Vh_ptr + vh_base + 0, v00)
        tl.store(Vh_ptr + vh_base + 1, v10)
        tl.store(Vh_ptr + vh_base + 2, v20)
        tl.store(Vh_ptr + vh_base + 3, v01)
        tl.store(Vh_ptr + vh_base + 4, v11)
        tl.store(Vh_ptr + vh_base + 5, v21)
        tl.store(Vh_ptr + vh_base + 6, v02)
        tl.store(Vh_ptr + vh_base + 7, v12)
        tl.store(Vh_ptr + vh_base + 8, v22)
        # U recovery
        inv_s0 = 1.0 / (s0 + EPS)
        inv_s1 = 1.0 / (s1 + EPS)
        inv_s2 = 1.0 / (s2 + EPS)
        for block_start in range(0, M, BLOCK_M):
            offs = tl.arange(0, BLOCK_M)
            row_idx = block_start + offs
            mask = row_idx < M
            a0 = tl.load(A_ptr + base + row_idx * 3 + 0, mask=mask, other=0.0).to(tl.float32)
            a1 = tl.load(A_ptr + base + row_idx * 3 + 1, mask=mask, other=0.0).to(tl.float32)
            a2 = tl.load(A_ptr + base + row_idx * 3 + 2, mask=mask, other=0.0).to(tl.float32)
            u0 = (a0 * v00 + a1 * v10 + a2 * v20) * inv_s0
            u1 = (a0 * v01 + a1 * v11 + a2 * v21) * inv_s1
            u2 = (a0 * v02 + a1 * v12 + a2 * v22) * inv_s2
            u_base = bid * M * 3
            tl.store(U_ptr + u_base + row_idx * 3 + 0, u0, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 3 + 1, u1, mask=mask)
            tl.store(U_ptr + u_base + row_idx * 3 + 2, u2, mask=mask)

    HAS_TRITON = True

except ImportError:
    import logging
    logging.info("[SVD Kernel] Triton not available — fused N=2/N=3 kernels disabled, using PyTorch fallback")


# ═══════════════════════════════════════════════════════════════════════════════
# PYTHON WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════════

def batched_svd2(A, block_m=128):
    """Fused Triton SVD for (B, M, 2) tensors. Falls back to torch if no Triton.

    Returns: U (B,M,2), S (B,2), Vh (B,2,2)
    """
    if not HAS_TRITON or not A.is_cuda:
        return torch.linalg.svd(A.float(), full_matrices=False)
    assert A.ndim == 3 and A.shape[2] == 2
    B, M, _ = A.shape
    A_f32 = A.contiguous().float()
    U = torch.empty((B, M, 2), dtype=torch.float32, device=A.device)
    S = torch.empty((B, 2), dtype=torch.float32, device=A.device)
    Vh = torch.empty((B, 2, 2), dtype=torch.float32, device=A.device)
    _svd2_kernel[(B,)](A_f32, U, S, Vh, M=M, BLOCK_M=block_m, EPS=1e-12)
    return U, S, Vh


def batched_svd3(A, block_m=128, jacobi_iters=6):
    """Fused Triton SVD for (B, M, 3) tensors. Falls back to torch if no Triton.

    Returns: U (B,M,3), S (B,3), Vh (B,3,3)
    """
    if not HAS_TRITON or not A.is_cuda:
        return torch.linalg.svd(A.float(), full_matrices=False)
    assert A.ndim == 3 and A.shape[2] == 3
    B, M, _ = A.shape
    A_f32 = A.contiguous().float()
    U = torch.empty((B, M, 3), dtype=torch.float32, device=A.device)
    S = torch.empty((B, 3), dtype=torch.float32, device=A.device)
    Vh = torch.empty((B, 3, 3), dtype=torch.float32, device=A.device)
    _svd3_kernel[(B,)](A_f32, U, S, Vh, M=M, BLOCK_M=block_m,
                       JACOBI_ITERS=jacobi_iters, EPS=1e-12)
    return U, S, Vh


# ═══════════════════════════════════════════════════════════════════════════════
# GRAM-EIGH HYBRID (N ≥ 4)
# ═══════════════════════════════════════════════════════════════════════════════

def gram_eigh_svd(A):
    """Thin SVD via Gram matrix eigendecomposition. Works for any N.

    G = A^T A → eigh(G) → S = sqrt(eigenvalues), V = eigenvectors, U = AV/S

    AMP-safe: disables autocast internally to prevent bf16 eigh failure.

    Args:
        A: (B, M, N) tensor, M >= N

    Returns: U (B,M,N), S (B,N), Vh (B,N,N) — singular values descending.
    """
    B, M, N = A.shape
    with torch.amp.autocast('cuda', enabled=False):
        A_f = A.float()
        G = torch.bmm(A_f.transpose(1, 2), A_f)
        eigenvalues, V = torch.linalg.eigh(G)
        eigenvalues = eigenvalues.flip(-1)
        V = V.flip(-1)
        S = torch.sqrt(eigenvalues.clamp(min=1e-12))
        U = torch.bmm(A_f, V) / S.unsqueeze(1)
        Vh = V.transpose(-2, -1).contiguous()
    return U, S, Vh


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════

def batched_svd(A, method='auto', block_m=128):
    """Batched thin SVD for (B, M, N) tensors. M >= N.

    Auto-dispatches by N:
      N=2:  Fused Triton         ~0.02ms
      N=3:  Fused Triton         ~0.02ms
      N≥4:  Gram + eigh          ~0.25ms (N=4) to ~0.78ms (N=32)

    Note: N≥48 hits eigh serialization cliff (~344ms). For Procrustes
    alignment at large N, use batched_procrustes() which bypasses this.

    Args:
        A:        (B, M, N) tensor, CUDA for Triton kernels
        method:   'auto', 'triton', 'gram_eigh', 'torch'
        block_m:  Tile size for Triton kernels

    Returns: U (B,M,N), S (B,N), Vh (B,N,N) — singular values descending.
    """
    assert A.ndim == 3, f"Expected (B, M, N), got {A.shape}"
    B, M, N = A.shape
    assert M >= N, f"Thin SVD requires M >= N, got M={M}, N={N}"

    if method == 'auto':
        if N == 2 and HAS_TRITON and A.is_cuda:
            return batched_svd2(A, block_m)
        elif N == 3 and HAS_TRITON and A.is_cuda:
            return batched_svd3(A, block_m)
        else:
            return gram_eigh_svd(A)
    elif method == 'triton':
        if N == 2:
            return batched_svd2(A, block_m)
        elif N == 3:
            return batched_svd3(A, block_m)
        raise ValueError(f"Triton kernel only for N=2,3, got N={N}")
    elif method == 'gram_eigh':
        return gram_eigh_svd(A)
    elif method == 'torch':
        return torch.linalg.svd(A.float(), full_matrices=False)
    raise ValueError(f"Unknown method '{method}'. Use: auto, triton, gram_eigh, torch")


# ═══════════════════════════════════════════════════════════════════════════════
# NEWTON-SCHULZ INVERSE SQUARE ROOT
# ═══════════════════════════════════════════════════════════════════════════════

def newton_schulz_invsqrt(G, iters=10):
    """Batched G^{-1/2} via Newton-Schulz iteration.

    Pure bmm — zero eigensolvers. Quadratic convergence.
    Use for Procrustes whitening: W = X @ newton_schulz_invsqrt(X^T X)

    AMP-safe: disables autocast internally.

    Args:
        G:     (B, N, N) symmetric PSD matrices
        iters: Iteration count (10 conservative, 7 usually sufficient)

    Returns: (B, N, N) inverse square root matrices
    """
    B, N, _ = G.shape
    device = G.device
    with torch.amp.autocast('cuda', enabled=False):
        G = G.float()
        trace = G.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1).clamp(min=1e-8)
        G_norm = G / trace
        I = torch.eye(N, device=device, dtype=torch.float32).unsqueeze(0).expand(B, -1, -1)
        Y = G_norm.clone()
        Z = I.clone()
        for _ in range(iters):
            ZY = torch.bmm(Z, Y)
            factor = 1.5 * I - 0.5 * ZY
            Y = torch.bmm(Y, factor)
            Z = torch.bmm(factor, Z)
        return Z / trace.sqrt()


# ═══════════════════════════════════════════════════════════════════════════════
# SUBSPACE-PRESERVING PROCRUSTES ALIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def batched_procrustes(source, target, rank=24, whiten=True, schulz_iters=10):
    """Batched Procrustes alignment with subspace-preserving rotation.

    N ≤ 32: full N-d Procrustes via SVD (sub-ms).
    N > 32: project to rank-d, align there, lift back preserving
            orthogonal complement exactly.

    Validated: 1.000 nearest-neighbor agreement with full Procrustes
    across N=32-128, k=8-64.

    AMP-safe: disables autocast internally.

    Args:
        source:       (B, n_samples, N) or (n_samples, N) — source embeddings
        target:       (B, n_samples, N) or (n_samples, N) — target embeddings
        rank:         Projection rank for N > 32 (default 24)
        whiten:       Apply Newton-Schulz whitening (default True)
        schulz_iters: Iterations for whitening (default 10)

    Returns:
        aligned: same shape as source — source aligned to target
        info:    dict with method, rotation matrix, diagnostics
    """
    unbatched = source.ndim == 2
    if unbatched:
        source = source.unsqueeze(0)
        target = target.unsqueeze(0)

    B, n_samples, N = source.shape
    device = source.device

    with torch.amp.autocast('cuda', enabled=False):
        source_f = source.float()
        target_f = target.float()

        # Center
        src_mean = source_f.mean(1, keepdim=True)
        tgt_mean = target_f.mean(1, keepdim=True)
        src_c = source_f - src_mean
        tgt_c = target_f - tgt_mean

        # Whiten
        if whiten:
            src_cov = torch.bmm(src_c.transpose(1, 2), src_c) / max(n_samples - 1, 1)
            tgt_cov = torch.bmm(tgt_c.transpose(1, 2), tgt_c) / max(n_samples - 1, 1)
            src_W = newton_schulz_invsqrt(src_cov, iters=schulz_iters)
            tgt_W = newton_schulz_invsqrt(tgt_cov, iters=schulz_iters)
            src_w = F.normalize(torch.bmm(src_c, src_W), dim=-1)
            tgt_w = F.normalize(torch.bmm(tgt_c, tgt_W), dim=-1)
        else:
            src_w = src_c
            tgt_w = tgt_c

        use_projection = N > 32 and rank < N

        if not use_projection:
            # Full N-d Procrustes
            C = torch.bmm(src_w.transpose(1, 2), tgt_w)
            U, _, Vh = torch.linalg.svd(C)
            R = torch.bmm(U, Vh)
            aligned_w = torch.bmm(src_w, R)
            if whiten:
                aligned = torch.bmm(aligned_w, torch.linalg.pinv(tgt_W)) + tgt_mean
            else:
                aligned = aligned_w + tgt_mean
            cos_after = F.cosine_similarity(
                aligned_w[:, :min(1000, n_samples)],
                tgt_w[:, :min(1000, n_samples)], dim=-1).mean().item()
            info = {'method': 'full', 'N': N, 'rank': N,
                    'rotation': R, 'cos_after': cos_after}
        else:
            # Subspace-preserving rank-k Procrustes
            k = min(rank, N - 1)
            P = torch.linalg.qr(
                torch.randn(B, N, k, device=device, dtype=torch.float32)).Q
            src_proj = torch.bmm(src_w, P)
            tgt_proj = torch.bmm(tgt_w, P)
            C_k = torch.bmm(src_proj.transpose(1, 2), tgt_proj)
            U_k, _, Vh_k = torch.linalg.svd(C_k)
            R_k = torch.bmm(U_k, Vh_k)
            # Decompose and rotate only in-subspace
            src_in = torch.bmm(src_w, P)
            P_T = P.transpose(1, 2)
            src_perp = src_w - torch.bmm(src_in, P_T)
            src_rotated = torch.bmm(torch.bmm(src_in, R_k), P_T)
            aligned_w = src_rotated + src_perp
            if whiten:
                aligned = torch.bmm(aligned_w, torch.linalg.pinv(tgt_W)) + tgt_mean
            else:
                aligned = aligned_w + tgt_mean
            cos_after = F.cosine_similarity(
                aligned_w[:, :min(1000, n_samples)],
                tgt_w[:, :min(1000, n_samples)], dim=-1).mean().item()
            info = {'method': 'subspace', 'N': N, 'rank': k,
                    'rotation_k': R_k, 'projection': P, 'cos_after': cos_after}

    if unbatched:
        aligned = aligned.squeeze(0)

    return aligned, info


# ═══════════════════════════════════════════════════════════════════════════════
# INLINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"kernel.py — validation on {device}")
    print(f"  HAS_TRITON: {HAS_TRITON}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name()}")
    print()

    B, M = 32, 256
    passed = 0
    failed = 0

    def _check(name, condition, detail=""):
        global passed, failed
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name}  {detail}")

    def _validate_svd(A, U, S, Vh, label):
        """Check reconstruction, orthogonality, descending S."""
        B, M, N = A.shape
        recon = torch.bmm(U * S.unsqueeze(1), Vh)
        recon_err = (A.float() - recon).pow(2).mean().sqrt().item()
        UtU = torch.bmm(U.transpose(1, 2), U)
        I_N = torch.eye(N, device=A.device).unsqueeze(0)
        orth_err = (UtU - I_N).pow(2).mean().sqrt().item()
        desc = (S[:, :-1] >= S[:, 1:] - 1e-5).all().item()
        _check(f"{label} recon",  recon_err < 1e-3, f"err={recon_err:.2e}")
        _check(f"{label} orth",   orth_err < 1e-3,  f"err={orth_err:.2e}")
        _check(f"{label} desc",   desc)

    # ── batched_svd auto-dispatch ──
    print("batched_svd (auto-dispatch):")
    for N in [2, 3, 8, 16, 32]:
        A = torch.randn(B, M, N, device=device)
        U, S, Vh = batched_svd(A)
        _check(f"  N={N:>2} shapes", U.shape == (B, M, N) and S.shape == (B, N) and Vh.shape == (B, N, N))
        _validate_svd(A, U, S, Vh, f"  N={N:>2}")

    # ── Triton kernels explicitly ──
    if HAS_TRITON and device == 'cuda':
        print("\nbatched_svd2 (Triton):")
        A2 = torch.randn(B, M, 2, device=device)
        U2, S2, Vh2 = batched_svd2(A2)
        _validate_svd(A2, U2, S2, Vh2, "  N=2 triton")

        print("\nbatched_svd3 (Triton):")
        A3 = torch.randn(B, M, 3, device=device)
        U3, S3, Vh3 = batched_svd3(A3)
        _validate_svd(A3, U3, S3, Vh3, "  N=3 triton")

    # ── gram_eigh_svd directly ──
    print("\ngram_eigh_svd:")
    for N in [4, 24, 48]:
        A = torch.randn(B, M, N, device=device)
        U, S, Vh = gram_eigh_svd(A)
        _validate_svd(A, U, S, Vh, f"  N={N}")

    # ── newton_schulz_invsqrt ──
    print("\nnewton_schulz_invsqrt:")
    N = 16
    X = torch.randn(B, 100, N, device=device)
    G = torch.bmm(X.transpose(1, 2), X) / 99
    G_inv_sqrt = newton_schulz_invsqrt(G)
    # G_inv_sqrt @ G @ G_inv_sqrt should ≈ I
    product = torch.bmm(torch.bmm(G_inv_sqrt, G), G_inv_sqrt)
    I_N = torch.eye(N, device=device).unsqueeze(0)
    ns_err = (product - I_N).pow(2).mean().sqrt().item()
    _check("  invsqrt identity", ns_err < 1e-2, f"err={ns_err:.2e}")
    _check("  invsqrt shape", G_inv_sqrt.shape == (B, N, N))

    # ── batched_procrustes (full, N ≤ 32) ──
    print("\nbatched_procrustes (full):")
    N = 24
    shared = torch.randn(500, N, device=device)
    src = shared + 0.3 * torch.randn(500, N, device=device)
    tgt = shared + 0.3 * torch.randn(500, N, device=device)
    cos_before = F.cosine_similarity(src, tgt, dim=-1).mean().item()
    aligned, info = batched_procrustes(src, tgt, rank=24)
    cos_after = F.cosine_similarity(aligned, tgt, dim=-1).mean().item()
    _check("  full method",  info['method'] == 'full')
    _check("  full shape",   aligned.shape == src.shape)
    _check("  full improved", cos_after > cos_before, f"{cos_before:.4f} → {cos_after:.4f}")

    # ── batched_procrustes (subspace, N > 32) ──
    print("\nbatched_procrustes (subspace):")
    N = 64
    shared = torch.randn(500, N, device=device)
    src = shared + 0.3 * torch.randn(500, N, device=device)
    tgt = shared + 0.3 * torch.randn(500, N, device=device)
    cos_before = F.cosine_similarity(src, tgt, dim=-1).mean().item()
    aligned, info = batched_procrustes(src, tgt, rank=24)
    cos_after = F.cosine_similarity(aligned, tgt, dim=-1).mean().item()
    _check("  subspace method",  info['method'] == 'subspace')
    _check("  subspace rank",    info['rank'] == 24)
    _check("  subspace shape",   aligned.shape == src.shape)
    _check("  subspace improved", cos_after > cos_before, f"{cos_before:.4f} → {cos_after:.4f}")

    # ── batched interface ──
    print("\nbatched_procrustes (batched):")
    src_b = torch.randn(4, 200, 32, device=device)
    tgt_b = src_b * 0.5 + torch.randn_like(src_b) * 0.3
    aligned_b, info_b = batched_procrustes(src_b, tgt_b)
    _check("  batched shape", aligned_b.shape == src_b.shape)
    _check("  batched method", info_b['method'] == 'full')

    # ── Summary ──
    total = passed + failed
    print(f"\n{'='*50}")
    print(f"  {passed}/{total} passed" + (f"  ({failed} FAILED)" if failed else "  — all clear"))
    print(f"{'='*50}")
