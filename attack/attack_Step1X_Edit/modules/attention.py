import math

import torch
import torch.nn.functional as F
from xfuser.model_executor.layers.usp import USP

try:
    import flash_attn
    from flash_attn.flash_attn_interface import (
        _flash_attn_forward,
        flash_attn_func,
        flash_attn_varlen_func,
    )
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None
    flash_attn_func = None

MEMORY_LAYOUT = {
    # flash模式:
    # 预处理: 输入 [batch_size, seq_len, num_heads, head_dim]
    # 后处理: 保持形状不变
    "flash": (
        lambda x: x,  # 保持形状
        lambda x: x,  # 保持形状
    ),
    # torch/vanilla模式:
    # 预处理: 交换序列和注意力头的维度 [B,S,A,D] -> [B,A,S,D]
    # 后处理: 交换回原始维度 [B,A,S,D] -> [B,S,A,D]
    "torch": (
        lambda x: x.transpose(1, 2),  # (B,S,A,D) -> (B,A,S,D)
        lambda x: x.transpose(1, 2),  # (B,A,S,D) -> (B,S,A,D)
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "xdit": (
        lambda x: x.transpose(1, 2),  # (B,S,A,D) -> (B,A,S,D)
        lambda x: x.transpose(1, 2),  # (B,A,S,D) -> (B,S,A,D)
    )
}

import matplotlib.pyplot as plt
import numpy as np

BLOCK_MASK_CONFIG = {
    'enabled': True,  # 是否启用屏蔽
    'timestep_range': (0, 5),  # 前10个timestep (索引0-9)
    'double_blocks': [], # double block的前2个
    'single_blocks': list(range(25)),  # single block 0-25
}


# 在 attention.py 顶部添加全局变量
STATS_COLLECTOR = {
    'double': {},  # {timestep: [block0_weight, block1_weight, ...]}
    'single': {},  # {timestep: [block0_weight, block1_weight, ...]}
    'double_counter': 0,
    'single_counter': 0,
    'current_timestep': None,
    'timestep_index': None  # 新增：当前是第几个timestep
}

def set_current_timestep(t, timestep_idx):
    """在推理循环中调用，设置当前timestep"""
    STATS_COLLECTOR['current_timestep'] = float(t)
    STATS_COLLECTOR['timestep_index'] = timestep_idx  # 设置索引
    STATS_COLLECTOR['double_counter'] = 0
    STATS_COLLECTOR['single_counter'] = 0


def plot_context_proportion():
    """绘制折线图：viridis配色表示去噪前期→后期（指定timestep列表）"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # 指定要绘制的timestep索引
    target_timesteps = [0, 1, 2, 3, 4, 6, 8, 12, 20, 22, 26]
    
    for ax, block_type, num_blocks, title in [
        (axes[0], 'double', 19, 'Double Blocks'),
        (axes[1], 'single', 38, 'Single Blocks')
    ]:
        # 按timestep从大到小排序（去噪前期→后期）
        all_timesteps = sorted(STATS_COLLECTOR[block_type].keys(), reverse=True)
        
        # 只选择列表中索引对应的timestep
        timesteps = [all_timesteps[i] for i in target_timesteps if i < len(all_timesteps)]
        
        # viridis配色（与之前的plot_multi_step_proportions保持一致）
        colors = plt.cm.viridis(np.linspace(0, 1, len(timesteps)))
        
        for timestep, color in zip(timesteps, colors):
            weights = STATS_COLLECTOR[block_type][timestep]
            ax.plot(range(len(weights)), weights, marker='o', 
                   color=color, label=f't={timestep:.3f}', 
                   linewidth=2, markersize=4, alpha=0.8)
        
        ax.set_xlabel('Block Number', fontsize=12)
        ax.set_ylabel('Context Proportion', fontsize=12)
        ax.set_title(f'{title} ({num_blocks} blocks)', fontsize=14, fontweight='bold')
        ax.legend(title='Timestep\n(purple→yellow = early→late)', 
                 bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(-0.5, num_blocks-0.5)
    
    plt.tight_layout()
    plt.savefig('context_proportion.png', dpi=150, bbox_inches='tight')
    print("✓ 图表已保存: context_proportion.png")
    plt.close()




def attention(
    q,
    k,
    v,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    causal=False,
):
    """
    执行QKV自注意力计算

    Args:
        q (torch.Tensor): 查询张量，形状 [batch_size, seq_len, num_heads, head_dim]
        k (torch.Tensor): 键张量，形状 [batch_size, seq_len_kv, num_heads, head_dim]
        v (torch.Tensor): 值张量，形状 [batch_size, seq_len_kv, num_heads, head_dim]
        mode (str): 注意力模式，可选 'flash', 'torch', 'vanilla'
        drop_rate (float): 注意力矩阵的dropout概率
        attn_mask (torch.Tensor): 注意力掩码，形状根据模式不同而变化
        causal (bool): 是否使用因果注意力（仅关注前面位置）

    Returns:
        torch.Tensor: 注意力输出，形状 [batch_size, seq_len, num_heads * head_dim]
    """
    # 获取预处理和后处理函数
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
    # print(q.shape)
    # 应用预处理变换
    q = pre_attn_layout(q)  # 形状根据模式变化
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)
    # print(q.shape)

    if mode == "torch":
        # 使用PyTorch原生的scaled_dot_product_attention
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
    elif mode == "flash":

        if flash_attn_func is None:  #改了
            mode = "torch"  # 自动降级到 SDPA
        # assert flash_attn_func is not None, "flash_attn_func未定义"
        # assert attn_mask is None, "不支持的注意力掩码"
        # x: torch.Tensor = flash_attn_func(
        #     q, k, v, dropout_p=drop_rate, causal=causal, softmax_scale=None
        # )  # type: ignore

    elif mode == "vanilla":
        scale_factor = 1 / math.sqrt(q.size(-1))
        # print(q.shape)
        # # print(k.shape)
        b, a, s, _ = q.shape
        s1 = k.size(2)
        
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)
        
        attn = (q @ k.transpose(-2, -1)) * scale_factor
        attn += attn_bias
        #print(attn)

        # # === 有针对性的屏蔽 ===
        # if s == 2688 and BLOCK_MASK_CONFIG['enabled']:
        #     timestep_idx = STATS_COLLECTOR['timestep_index']
            
            
        #     if timestep_idx is not None:
        #         # 判断当前block类型和编号
        #         if STATS_COLLECTOR['double_counter'] < 19:
        #             block_type = 'double'
        #             block_num = STATS_COLLECTOR['double_counter']
        #         else:
        #             block_type = 'single'
        #             block_num = STATS_COLLECTOR['single_counter']
                
        #         # 检查是否需要屏蔽
        #         should_mask = False
                
        #         # 检查timestep范围
        #         t_start, t_end = BLOCK_MASK_CONFIG['timestep_range']
        #         if t_start <= timestep_idx < t_end:
        #             # 检查block编号
        #             if block_type == 'double' and block_num in BLOCK_MASK_CONFIG['double_blocks']:
        #                 should_mask = True
        #             elif block_type == 'single' and block_num in BLOCK_MASK_CONFIG['single_blocks']:
        #                 should_mask = True
                
        #         # 执行屏蔽
        #         if should_mask:
        #             attn[0, :, 640:1664, 1664:2688] = float('-inf')
        #             # print(f"[Masked] timestep_idx={timestep_idx}, {block_type} block {block_num}")


        attn = attn.softmax(dim=-1)
        
        
        # === 统计 Context Proportion ===
        if s == 2688 and STATS_COLLECTOR['current_timestep'] is not None:
            # 只取第一个batch
            target_to_condition = attn[0, :, 640:1664, 1664:2688]

            # shape: [num_heads, target_len, condition_len]
            #        [24, 1024, 1024]
            
            # 对每个target query在condition上求和
            sum_per_query = target_to_condition.sum(dim=-1)  # [24, 1024]
            
            # 平均所有head和所有query
            avg_weight = sum_per_query.mean().item()
            
            timestep = STATS_COLLECTOR['current_timestep']
            
            # 判断block类型
            if STATS_COLLECTOR['double_counter'] < 19:
                block_type = 'double'
                counter_key = 'double_counter'
                block_num = STATS_COLLECTOR['double_counter']
            else:
                block_type = 'single'
                counter_key = 'single_counter'
                block_num = STATS_COLLECTOR['single_counter']  # single block的编号
            
            # 记录
            if timestep not in STATS_COLLECTOR[block_type]:
                STATS_COLLECTOR[block_type][timestep] = []

            if block_type == 'single' and 0 <= block_num <= 25:
                print(f"single_block_{block_num}, t={timestep:.3f}, ctx_prop={avg_weight:.4f}")
                
                # 当收集完第26个block (block_num=25) 时，计算并打印平均值
                if block_num == 25:
                    single_blocks_0_25 = STATS_COLLECTOR['single'][timestep][:26]  # 取前26个
                    avg_of_26_blocks = sum(single_blocks_0_25) / len(single_blocks_0_25)
                    print(f">>> Average of single blocks 0-25 at t={timestep:.3f}: {avg_of_26_blocks:.4f}\n")

            STATS_COLLECTOR[block_type][timestep].append(avg_weight)
            STATS_COLLECTOR[counter_key] += 1
        

        attn = torch.dropout(attn, p=drop_rate, train=True)

        # 计算输出
        x = attn @ v  # [B,A,S,D]
    elif mode == "xdit":
        x: torch.Tensor = USP(q, k, v, dropout_p=drop_rate, is_causal=causal)
    else:
        raise NotImplementedError(f"不支持的注意力模式: {mode}")

    # 应用后处理变换
    x = post_attn_layout(x)  # 恢复原始维度顺序

    # 合并注意力头维度
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)  # [B,S,A*D]
    return out
