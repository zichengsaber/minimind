import torch


def precompute_freqs_cis_minimind(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # （dim // 2,） 是偶数
    t = torch.arange(end, device=freqs.device) # 生成一个从0到end的序列 (end,)
    freqs = torch.outer(t, freqs).float() # 将t和freqs进行外积，得到一个(end, dim // 2)的矩阵
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) # (end, dim)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) # (end, dim)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb_minimind(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    # q,k 的shape 是 (bsz, seq_len, n_heads, dim)
    # cos,sin 的shape 是 (seq_len, dim)
    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed



def precompute_freqs_cis_nanogpt(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # shape (seq_len, dim//2)
    freqs_cis_real = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # shape (seq_len, dim//2, 2) [cos, sin]
    return freqs_cis_real

def apply_rotary_emb_nanogpt(x, freqs_cis):
    # shape gymnastics let's go
    # x is (bs, seqlen, n_heads, head_dim), e.g. (4, 8, 32, 128)
    # freqs_cis is (seq_len, head_dim/2, 2), e.g. (8, 64, 2)
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    # xshaped is (bs, seqlen, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    # freqs_cis becomes (1, seqlen, 1, head_dim/2, 2), e.g. (1, 8, 1, 64, 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1], # x_i*cos(theta_i) - x_{i+1}*sin(theta_i)
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1], # x_{i+1}*cos(theta_i) + x_i*sin(theta_i)
        ],
        -1,
    )
    # x_out2 at this point is (bs, seqlen, n_heads, head_dim/2, 2), e.g. (4, 8, 32, 64, 2)
    x_out2 = x_out2.flatten(3)
    # x_out2 is now (bs, seqlen, n_heads, head_dim), e.g. (4, 8, 32, 128)
    return x_out2.type_as(x)


def test_rope_implementations():
    # 设置小规模的测试参数
    dim = 4  # 头维度
    seq_len = 4  # 序列长度
    bsz = 2  # 批大小
    n_heads = 1  # 头数
    
    # 生成随机输入
    q = torch.randn(bsz, seq_len, n_heads, dim)
    k = torch.randn(bsz, seq_len, n_heads, dim)
    
    # 使用minimind实现
    cos_mini, sin_mini = precompute_freqs_cis_minimind(dim, seq_len)
    q_mini, k_mini = apply_rotary_pos_emb_minimind(q, k, cos_mini, sin_mini)
    
    # 使用nanogpt实现
    freqs_nano = precompute_freqs_cis_nanogpt(dim, seq_len)
    q_nano = apply_rotary_emb_nanogpt(q, freqs_nano)
    k_nano = apply_rotary_emb_nanogpt(k, freqs_nano)
    
    # 比较结果
    q_close = torch.allclose(q_mini, q_nano, atol=1e-6)
    k_close = torch.allclose(k_mini, k_nano, atol=1e-6)
    
    print(f"Q embeddings match: {q_close}")
    print(f"K embeddings match: {k_close}")
    
    if not q_close or not k_close:
        print("\nDifferences:")
        if not q_close:
            print("Q max difference:", torch.max(torch.abs(q_mini - q_nano)).item())
        if not k_close:
            print("K max difference:", torch.max(torch.abs(k_mini - k_nano)).item())
        
        # 打印部分值进行比较
        print("\nSample comparison (first element):")
        print("minimind Q:", q_mini)
        print("nanogpt Q:", q_nano)
        print("minimind K:", k_mini)
        print("nanogpt K:", k_nano)
    
    return q_close and k_close

# 运行测试
if __name__ == "__main__":
    test_passed = test_rope_implementations()
    print("\nTest passed:", test_passed)

