from cs336_basics.model import BasicsTransformerLM, TransformerBlock, CausalMultiHeadSelfAttention, Linear, Embedding, RMSNorm, RotaryEmbedding, SwiGLU, scaled_dot_product_attention
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_basics.nn_utils import softmax, log_softmax, cross_entropy, clip_gradient
import torch
import timeit
import numpy as np
# 导入 PyTorch 的 checkpoint 模块
from torch.utils.checkpoint import checkpoint

def benchmark(d_model, d_ff, num_layers, num_heads, warmup_w, n, max_lr=6e-4, min_lr=6e-5, vocab=10000, batch_size=4, context_length=1024, type:int=3, warmup=True, use_ckpt=True):
    '''
    type==1: only forward, 
    type==2: forward and backward,
    type==3: forward and backward with optimizer step
    use_ckpt: 是否启用重计算 (Activation Checkpointing)
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BasicsTransformerLM(vocab, context_length, d_model, num_layers, num_heads, d_ff).to(device)
    
    # ==============================================================
    # 🎯 核心精准改造：完美适配你的 BasicsTransformerLM 结构
    # ==============================================================
    if use_ckpt:
        print("💡 已激活重计算 (Activation Checkpointing) 模式...")
        
        # 定义一个全新的 forward 函数，强行塞入 checkpoint
        def checkpointed_forward(x):
            # 1. 对应你源码中的: embedded_tokens = self.token_embeddings(x)
            x = model.token_embeddings(x)
            
            # 2. 核心：遍历每一层 TransformerBlock，用 checkpoint 包裹它
            for layer in model.layers:
                # use_reentrant=False 是 PyTorch 官方强烈推荐的新版实现
                x = checkpoint(layer, x, use_reentrant=False)
                
            # 3. 对应你源码中的收尾工作
            x = model.ln_final(x)
            logits = model.lm_head(x)
            return logits
            
        # 偷梁换柱：用我们的定制函数替换掉模型原有的 forward
        model.forward = checkpointed_forward

    # ==============================================================
    # 后续的数据准备、训练循环完全保持你原本的逻辑
    # ==============================================================
    data_X = torch.randint(0, vocab, (batch_size, context_length), device=device)
    data_Y = torch.randint(0, vocab, (batch_size, context_length), device=device)
    optimizer = AdamW(model.parameters())
    
    if warmup:
        with torch.cuda.nvtx.range("Warmup_Phase"):
            for i in range(warmup_w):
                lrt = get_cosine_lr(i, max_lr, min_lr, warmup_w, n)
                for group in optimizer.param_groups:
                    group['lr'] = lrt
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    result = model(data_X)
                    loss = cross_entropy(result, data_Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if device == "cuda":
                torch.cuda.synchronize() 
    
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    if type == 1:
        time_list = []
        with torch.no_grad():
            for it in range(n):
                with torch.cuda.nvtx.range(f"Step_{it}"):
                    start_time = timeit.default_timer()
                    with torch.cuda.nvtx.range("Forward_Pass"):
                        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                            result = model(data_X)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    end_time = timeit.default_timer()
                    time_list.append(end_time - start_time)
            timings = np.array(time_list)
            print(f"每一步的耗时明细: {timings}")
            print(f"仅forward单步平均时间: {np.mean(timings):.6f} 秒")
        torch.cuda.memory._dump_snapshot(f"context_1024_type1_ckpt_{use_ckpt}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        return

    if type == 2:
        timings = []
        for it in range(n):
            with torch.cuda.nvtx.range(f"Step_{it}"):
                start_time = timeit.default_timer()
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    with torch.cuda.nvtx.range("Forward_Pass"):
                        result = model(data_X)
                    with torch.cuda.nvtx.range("Loss_Computation"):
                        loss = cross_entropy(result, data_Y)
                optimizer.zero_grad()
                with torch.cuda.nvtx.range("Backward_Pass"):
                    loss.backward()
                if device == "cuda":
                    torch.cuda.synchronize()
                end_time = timeit.default_timer()
                timings.append(end_time - start_time)
        timings = np.array(timings)
        print(f"每一步的耗时明细: {timings}")
        print(f'forward+backward 单步平均时间: {np.mean(timings):.6f} 秒')
        torch.cuda.memory._dump_snapshot(f"context_1024_type2_ckpt_{use_ckpt}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        return

    if type == 3:
        timings = []
        for it in range(n):
            with torch.cuda.nvtx.range(f"Step_{it}"):
                start_time = timeit.default_timer()          
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    with torch.cuda.nvtx.range("Forward_Pass"):
                        result = model(data_X)
                    with torch.cuda.nvtx.range("Loss_Computation"):
                        loss = cross_entropy(result, data_Y)

                optimizer.zero_grad()
                with torch.cuda.nvtx.range("Backward_Pass"):
                    loss.backward()
                    
                with torch.cuda.nvtx.range("Optimizer_Step"):
                    optimizer.step()            
                    
                if device == "cuda":
                    torch.cuda.synchronize()                
                end_time = timeit.default_timer()
                timings.append(end_time - start_time)
            
        timings = np.array(timings)
        print(f"每一步的耗时明细: {timings}")
        print(f'forward+backward+optimizer 单步平均时间: {np.mean(timings):.6f} 秒')
        
        filename = f"context_1024_type3_ckpt_{use_ckpt}.pickle"
        torch.cuda.memory._dump_snapshot(filename)
        print(f"📸 显存快照已保存至: {filename}")
        torch.cuda.memory._record_memory_history(enabled=None)
        
        if device == "cuda":
            peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
            print(f"📊 当前模式显存巅峰 (Peak Memory): {peak_mem:.2f} MB")
        return

# ==============================================================
# 运行对比测试
# ==============================================================
if __name__ == "__main__":
    # 1. 运行传统模式 (type=3)
    '''print("=== 开始传统训练测试 ===")
    benchmark(768, 3072, 12, 12, 5, 10, type=3, context_length=1024, use_ckpt=False)
    
    print("\n" + "="*50 + "\n")'''
    
    # 2. 运行重计算模式 (type=3)
    print("=== 开始重计算 (Checkpoint) 测试 ===")
    benchmark(768, 3072, 12, 12, 5, 10, type=3, context_length=1024, use_ckpt=True)