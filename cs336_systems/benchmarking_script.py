from cs336_basics.model import BasicsTransformerLM, TransformerBlock, CausalMultiHeadSelfAttention, Linear, Embedding, RMSNorm, RotaryEmbedding, SwiGLU, scaled_dot_product_attention
from cs336_basics.data import get_batch
from cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_basics.nn_utils import softmax, log_softmax, cross_entropy, clip_gradient
import torch
import timeit
import numpy as np

def benchmark(d_model,d_ff,num_layers,num_heads,warmup_w,n,max_lr=6e-4,min_lr=6e-5,vocab=10000,batch_size=4,context_length=512,type:int=3,warmup=True):
    '''
    type==1:only forward, 
    type==2:forward and backward,
    type==3:forward and backward with optimizer step
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BasicsTransformerLM(vocab, context_length, d_model, num_layers, num_heads, d_ff).to(device)
    data_X = torch.randint(0, vocab, (batch_size, context_length), device=device)
    data_Y = torch.randint(0, vocab, (batch_size, context_length), device=device)
    optimizer=AdamW(model.parameters())
    
    if warmup:
        # 顺便给 Warmup 整体也打一个标签，方便在 Nsys 里一眼把它和后续的测试分离开
        with torch.cuda.nvtx.range("Warmup_Phase"):
            for i in range(warmup_w):
                lrt=get_cosine_lr(i,max_lr,min_lr,warmup_w,n)
                for group in optimizer.param_groups:
                    group['lr']=lrt
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    result=model(data_X)
                    loss=cross_entropy(result,data_Y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if device == "cuda":
                torch.cuda.synchronize()  #强制同步一次，将已有异步算子都算好
    
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    if type==1:
        time_list=[]
        with torch.no_grad():
            for it in range(n):
                # 为每一轮迭代（Step）打个范围标签
                with torch.cuda.nvtx.range(f"Step_{it}"):
                    start_time = timeit.default_timer()
                    
                    with torch.cuda.nvtx.range("Forward_Pass"):
                        with torch.autocast(device_type=device, dtype=torch.bfloat16):
                            result=model(data_X)
                        
                    if device == "cuda":
                        torch.cuda.synchronize()
                    end_time=timeit.default_timer()
                    timing=(-start_time+end_time)
                    time_list.append(timing)
                    
            timings = np.array(time_list)
            print(f"每一步的耗时明细: {timings}")
            print(f"仅forward单步平均时间: {np.mean(timings):.6f} 秒，标准差: {np.std(timings):.6f} 秒")
        torch.cuda.memory._dump_snapshot("context_128_all.pickle")
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
        print(f'forward+backward 单步平均时间: {np.mean(timings):.6f} 秒, 标准差: {np.std(timings):.6f} 秒')
        torch.cuda.memory._dump_snapshot("context_128_all.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        return

    if type == 3:
        timings = []
        for it in range(n):
            # 将每一轮迭代括起来
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
        print(f'forward+backward+optimizer 单步平均时间: {np.mean(timings):.6f} 秒, 标准差: {np.std(timings):.6f} 秒')
        torch.cuda.memory._dump_snapshot("context_1024_all.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        return


    print('type无意义')
    return
    

benchmark(768,3072,12,12,5,10,type=3,context_length=1024)