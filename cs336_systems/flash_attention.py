from multiprocessing import Value
from sympy import Q
import torch
from torch import K, nn
from einops import einsum
import triton
import triton.language as tl
import math

class FlashAttention2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool = False):
        # Q, K, V: [batch, n_q/n_k, d]
        # We ignore is_causal for this PyTorch reference implementation
        # assert Q.dim() == 3 and K.dim() == 3 and V.dim() == 3, "Q, K, V must be [B, N, D]"
        B, Nq, d = Q.shape
        _, Nk, d_k = K.shape
        # assert d == d_k == V.shape[-1], "Last dim of Q, K, V must match"

        # Tile sizes (at least 16x16 as requested)
        Bq = 16
        Bk = 16
        scale = (1.0 / (d ** 0.5))

        O = torch.zeros(B, Nq, d, device=Q.device, dtype=Q.dtype)
        L = torch.empty(B, Nq, device=Q.device, dtype=Q.dtype)

        # Iterate over query tiles
        for qi in range(0, Nq, Bq):
            q_end = qi + Bq
            Qi = Q[:, qi:q_end, :]  # [B, Bq, d]

            # Initialize running values per Algorithm 1
            O_running = torch.zeros(B, Qi.shape[1], d, device=Q.device, dtype=Q.dtype)
            l_running = torch.zeros(B, Qi.shape[1], device=Q.device, dtype=Q.dtype)
            m_running = torch.full((B, Qi.shape[1]), -float('inf'), device=Q.device, dtype=Q.dtype)

            # Iterate over key/value tiles
            for kj in range(0, Nk, Bk):
                k_end = kj + Bk
                Kj = K[:, kj:k_end, :]  # [B, Bk, d]
                Vj = V[:, kj:k_end, :]  # [B, Bk, d]

                # S_ij = Qi @ Kj^T scaled, shape [B, Bq, Bk]
                Sij = einsum(Qi, Kj, 'b q d, b k d -> b q k') * scale

                # New row-wise max across keys for stability
                row_max = Sij.max(dim=-1).values  # [B, Bq]
                m_new = torch.maximum(m_running, row_max)

                # Exponentiated and rescaled probabilities
                P_tilde = torch.exp(Sij - m_new[..., None])  # [B, Bq, Bk]

                # Update running normalizer and output per FA-2
                exp_factor = torch.exp(m_running - m_new)  # [B, Bq]
                l_new = exp_factor * l_running + P_tilde.sum(dim=-1)  # [B, Bq]
                O_new = exp_factor[..., None] * O_running + einsum(P_tilde, Vj, 'b q k, b k d -> b q d')  # [B, Bq, d]

                m_running, l_running, O_running = m_new, l_new, O_new

            # Final normalization and log-sum-exp
            O[:, qi:q_end, :] = O_running / l_running[..., None]
            L[:, qi:q_end] = m_running + torch.log(l_running)

        # Save tensors for backward
        ctx.save_for_backward(L, Q, K, V, O)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented yet.")

@triton.jit
def flash_attention_fwd(
    Q_ptr, K_ptr, V_ptr,  
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS, scale, 
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    ):
    # no causal mask yet
    # program indices
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # offset each program with the corresponding batch index, mutlitply with the batch stride for each tensor
    Q_block_ptr = tl.make_block_ptr (
        Q_ptr + batch_index * stride_qb,
        shape = (N_QUERIES, D),
        strides = (stride_qq, stride_qd),
        offsets = (query_tile_index * Q_TILE_SIZE, 0),
        block_shape =(Q_TILE_SIZE, D),
        order =(1, 0),
    )

    K_block_ptr = tl.make_block_ptr (
        K_ptr + batch_index * stride_kb,
        shape = (N_KEYS, D),
        strides = (stride_kk, stride_kd),
        offsets = (0, 0),
        block_shape =(K_TILE_SIZE, D),
        order =(1, 0),
    )

    V_block_ptr = tl.make_block_ptr (
        V_ptr + batch_index * stride_vb,
        shape = (N_KEYS, D),
        strides = (stride_vk, stride_vd),
        offsets = (0, 0),
        block_shape =(K_TILE_SIZE, D),
        order =(1, 0),
    )

    O_block_ptr = tl.make_block_ptr (
    O_ptr + batch_index * stride_ob,
    shape = (N_QUERIES, D),
    strides = (stride_oq, stride_od),
    offsets =  (query_tile_index * Q_TILE_SIZE, 0),
    block_shape =(Q_TILE_SIZE, D),
    order =(1, 0),
    )

    L_block_ptr = tl.make_block_ptr (
    L_ptr + batch_index * stride_lb,
    shape = (N_QUERIES, ),
    strides = (stride_lq, ),
    offsets =  (query_tile_index * Q_TILE_SIZE, ),
    block_shape =(Q_TILE_SIZE,),
    order =(1, 0),
    )
    # load Qi
    Q_i = tl.load (Q_block_ptr, boundary_check = (0, ), padding_option = "zero")

    # Initialize running buffer
    acc = tl.float32
    O_running = tl.zeros((Q_TILE_SIZE,D), dtype = acc)
    l_running = tl.zeros((Q_TILE_SIZE,), dtype = acc)
    m_running = tl.full((Q_TILE_SIZE,), -float('inf'),  dtype = acc)


    # Computation in a for loop
    for j in range(tl.cdiv(N_KEYS, K_TILE_SIZE)):
        # load K_j and V_J
        K_j = tl.load (K_block_ptr, boundary_check = (0,), padding_option = "zero") # is this boundary check correct? 
        V_j = tl.load (V_block_ptr, boundary_check = (0,), padding_option = "zero")
        # compute flash attention parameters
        S_ij = tl.dot (Q_i, tl.trans(K_j), acc = acc) * scale
        m_new = tl.max(m_running, tl.max(S_ij, axis = 1))
        alpha = tl.exp (m_running - m_new)
        P_ij = tl.exp(S_ij - m_new[:, None])
        P_ij_casted = P_ij.to(V_j.dtype)
        l_new = alpha * l_running + tl.sum(P_ij, axis = 1)
        O_new = alpha[:, None] * O_running + tl.dot(P_ij_casted, V_j, acc = acc)
        m_running, l_running, O_running = m_new, l_new, O_new
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    # final computation and store back
    O_final = (O_running / l_running[:, None]).to(O_block_ptr.type.element_ty)
    L_final = m_running  + tl.log(l_running)
    tl.store (O_block_ptr, O_final, boundary_check = (0,))
    tl.store(L_block_ptr, L_final, boundary_check = (0,))

class FlashAttention_triton(torch.autograd.Function):
    @ staticmethod
    def forward(ctx, K, Q, V):
        B, Nq, D = Q.shape
        _, Nk, _ = K.shape
        Q_TILE_SIZE = 16
        K_TILE_SIZE = 16
        scale = (1.0 / (D ** 0.5))

        # initialize empty result tensors
        O = torch.empty(B, Nq, D, device=Q.device, dtype=Q.dtype)
        L = torch.empty(B, Nq, device=Q.device, dtype=Q.dtype)
        # lauch our kernel
        Q_num = triton.cdiv(Nq, Q_TILE_SIZE)
        flash_attention_fwd[(Q_num, B)](
            Q, K, V, 
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            scale, 
            D = D, Q_TILE_SIZE = Q_TILE_SIZE
        )
        # Save tensors for backward
        ctx.save_for_backward(L, Q, K, V, O)
        return O
    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError("Backward pass not implemented yet.")




