---
title: "Vllm Openvino后端环境安装和跑基测试"
description: 
date: 2024-07-12T21:50:16+08:00
image: 
math: 
license: 
hidden: false
comments: true
draft: false


tags: [
    "VLLM",
    "vllm-openvino",
]
categories: [
    "LLM",
    "CPU端侧部署",
]
---

# vllm-openvino环境安装
参考链接：  
1、[Installation with OpenVINO](https://docs.vllm.ai/en/latest/getting_started/openvino-installation.html#)

```bash
# step1: install virtual env
conda create -n vllm-openvino python=3.10 -y
conda activate vllm-openvion

# step2: 安装编译工具依赖
pip install --upgrade pip
pip install -r requirements-build.txt --extra-index-url https://download.pytorch.org/whl/cpu


# step3: 安装OpenVINO后端
PIP_PRE=1 PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cpu https://storage.openvinotoolkit.org/simple/wheels/nightly/" VLLM_TARGET_DEVICE=openvino python -m pip install -v .
```


# 基准性能测试
## 默认基准性能测试
基准测试命令：
```bash
    VLLM_OPENVINO_KVCACHE_SPACE=30  \
    VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8 \
    VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
    python3 vllm/benchmarks/benchmark_throughput.py \
            --input-len 128 \
            --output-len 128 \
            --num_prompts 500
```
基准测试运行结果：
可以发现，使用OpenVINO推理的时候，当模型不是OpenVION IR的时候，会先把模型转成OpenVION IR
```bash
(vllm-openvino) feng@feng-X99M-D3:~/llm/vllm$ VLLM_OPENVINO_KVCACHE_SPACE=30  VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8 VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON     python3 vllm/benchmarks/benchmark_throughput.py --input-len 128 --output-len 128 --num_prompts 500
WARNING 07-12 22:02:16 _custom_ops.py:14] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
Namespace(backend='vllm', dataset=None, input_len=128, output_len=128, model='facebook/opt-125m', tokenizer='facebook/opt-125m', quantization=None, tensor_parallel_size=1, n=1, use_beam_search=False, num_prompts=500, seed=0, hf_max_batch_size=None, trust_remote_code=False, max_model_len=None, dtype='auto', gpu_memory_utilization=0.9, enforce_eager=False, kv_cache_dtype='auto', quantization_param_path=None, device='auto', enable_prefix_caching=False, enable_chunked_prefill=False, max_num_batched_tokens=None, download_dir=None, output_json=None, distributed_executor_backend=None, load_format='auto')
INFO 07-12 22:02:18 llm_engine.py:174] Initializing an LLM engine (v0.5.1) with config: model='facebook/opt-125m', speculative_config=None, tokenizer='facebook/opt-125m', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=2048, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cpu, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None), seed=0, served_model_name=facebook/opt-125m, use_v2_block_manager=False, enable_prefix_caching=False)
WARNING 07-12 22:02:18 openvino_executor.py:132] Only float32 dtype is supported on OpenVINO, casting from torch.float16.
WARNING 07-12 22:02:18 openvino_executor.py:137] CUDA graph is not supported on OpenVINO backend, fallback to the eager mode.
INFO 07-12 22:02:18 openvino_executor.py:146] KV cache type is overried to u8 via VLLM_OPENVINO_CPU_KV_CACHE_PRECISION env var.
INFO 07-12 22:02:18 openvino_executor.py:159] OpenVINO optimal block size is 32, overriding currently set 16
INFO 07-12 22:02:21 selector.py:121] Cannot use _Backend.FLASH_ATTN backend on OpenVINO.
INFO 07-12 22:02:21 selector.py:69] Using OpenVINO Attention backend.
WARNING 07-12 22:02:22 openvino.py:123] Provided model id facebook/opt-125m does not contain OpenVINO IR, the model will be converted to IR with default options. If you need to use specific options for model conversion, use optimum-cli export openvino with desired options.
Framework not specified. Using pt to export the model.
Using framework PyTorch: 2.3.0+cpu
The BetterTransformer implementation does not support padding during training, as the fused kernels do not support attention masks. Beware that passing padded batched data during training may result in unexpected outputs. Please refer to https://huggingface.co/docs/optimum/bettertransformer/overview for more details.
Overriding 1 configuration item(s)
	- use_cache -> True
/home/feng/miniconda3/envs/vllm-openvino/lib/python3.10/site-packages/transformers/models/opt/modeling_opt.py:824: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  elif attention_mask.shape[1] != mask_seq_length:
/home/feng/miniconda3/envs/vllm-openvino/lib/python3.10/site-packages/transformers/modeling_attn_mask_utils.py:114: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if (input_shape[-1] > 1 or self.sliding_window is not None) and self.is_causal:
/home/feng/miniconda3/envs/vllm-openvino/lib/python3.10/site-packages/optimum/exporters/onnx/model_patcher.py:303: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if past_key_values_length > 0:
/home/feng/miniconda3/envs/vllm-openvino/lib/python3.10/site-packages/optimum/bettertransformer/models/attention.py:285: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if batch_size == 1 or self.training:
/home/feng/miniconda3/envs/vllm-openvino/lib/python3.10/site-packages/optimum/bettertransformer/models/attention.py:299: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if attn_output.size() != (batch_size, self.num_heads, tgt_len, self.head_dim):
['input_ids', 'attention_mask', 'past_key_values']
INFO:nncf:Statistics of the bitwidth distribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│              8 │ 100% (74 / 74)              │ 100% (74 / 74)                         │
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 74/74 • 0:00:01 • 0:00:00
INFO 07-12 22:02:38 openvino_executor.py:72] # CPU blocks: 48545
INFO 07-12 22:02:38 selector.py:121] Cannot use _Backend.FLASH_ATTN backend on OpenVINO.
INFO 07-12 22:02:38 selector.py:69] Using OpenVINO Attention backend.
Processed prompts: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [01:43<00:00,  4.84it/s, est. speed input: 624.89 toks/s, output: 620.05 toks/s]
Throughput: 4.84 requests/s, 1238.34 tokens/s
(vllm-openvino) feng@feng-X99M-D3:~/llm/vllm$ 
```


## 直接使用本地无量化模型进行基准性能测试
基准测试命令：
```bash
    VLLM_OPENVINO_KVCACHE_SPACE=30  \
    VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8 \
    VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
         python3 vllm/benchmarks/benchmark_throughput.py \
            --model /home/feng/disk1/models-llm/hf-models/Qwen1.5-1.8B-Chat \
            --input-len 32 \
            --output-len 32 \
            --num_prompts 100
```
基准测试运行结果：

```bash
(vllm-openvino) feng@feng-X99M-D3:~/llm/vllm$ VLLM_OPENVINO_KVCACHE_SPACE=30  VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8 VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON     python3 vllm/benchmarks/benchmark_throughput.py --input-len 32 --output-len 32  --num_prompts 100 --model /home/feng/disk1/models-llm/hf-models/Qwen1.5-1.8B-Chat
WARNING 07-12 22:30:59 _custom_ops.py:14] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
Namespace(backend='vllm', dataset=None, input_len=32, output_len=32, model='/home/feng/disk1/models-llm/hf-models/Qwen1.5-1.8B-Chat', tokenizer='/home/feng/disk1/models-llm/hf-models/Qwen1.5-1.8B-Chat', quantization=None, tensor_parallel_size=1, n=1, use_beam_search=False, num_prompts=100, seed=0, hf_max_batch_size=None, trust_remote_code=False, max_model_len=None, dtype='auto', gpu_memory_utilization=0.9, enforce_eager=False, kv_cache_dtype='auto', quantization_param_path=None, device='auto', enable_prefix_caching=False, enable_chunked_prefill=False, max_num_batched_tokens=None, download_dir=None, output_json=None, distributed_executor_backend=None, load_format='auto')
INFO 07-12 22:31:00 llm_engine.py:174] Initializing an LLM engine (v0.5.1) with config: model='/home/feng/disk1/models-llm/hf-models/Qwen1.5-1.8B-Chat', speculative_config=None, tokenizer='/home/feng/disk1/models-llm/hf-models/Qwen1.5-1.8B-Chat', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cpu, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None), seed=0, served_model_name=/home/feng/disk1/models-llm/hf-models/Qwen1.5-1.8B-Chat, use_v2_block_manager=False, enable_prefix_caching=False)
WARNING 07-12 22:31:00 openvino_executor.py:132] Only float32 dtype is supported on OpenVINO, casting from torch.bfloat16.
WARNING 07-12 22:31:00 openvino_executor.py:137] CUDA graph is not supported on OpenVINO backend, fallback to the eager mode.
INFO 07-12 22:31:00 openvino_executor.py:146] KV cache type is overried to u8 via VLLM_OPENVINO_CPU_KV_CACHE_PRECISION env var.
INFO 07-12 22:31:00 openvino_executor.py:159] OpenVINO optimal block size is 32, overriding currently set 16
INFO 07-12 22:31:03 selector.py:121] Cannot use _Backend.FLASH_ATTN backend on OpenVINO.
INFO 07-12 22:31:03 selector.py:69] Using OpenVINO Attention backend.
WARNING 07-12 22:31:03 openvino.py:123] Provided model id /home/feng/disk1/models-llm/hf-models/Qwen1.5-1.8B-Chat does not contain OpenVINO IR, the model will be converted to IR with default options. If you need to use specific options for model conversion, use optimum-cli export openvino with desired options.
Framework not specified. Using pt to export the model.
Using framework PyTorch: 2.3.0+cpu
Overriding 1 configuration item(s)
	- use_cache -> True
We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)
/home/feng/miniconda3/envs/vllm-openvino/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py:1116: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if sequence_length != 1:
/home/feng/miniconda3/envs/vllm-openvino/lib/python3.10/site-packages/transformers/models/qwen2/modeling_qwen2.py:128: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
  if seq_len > self.max_seq_len_cached:
['input_ids', 'attention_mask', 'position_ids', 'past_key_values']
INFO:nncf:Statistics of the bitwidth distribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
│              8 │ 100% (170 / 170)            │ 100% (170 / 170)                       │
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
Applying Weight Compression ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 170/170 • 0:00:13 • 0:00:00
INFO 07-12 22:31:43 openvino_executor.py:72] # CPU blocks: 9637
INFO 07-12 22:31:43 selector.py:121] Cannot use _Backend.FLASH_ATTN backend on OpenVINO.
INFO 07-12 22:31:43 selector.py:69] Using OpenVINO Attention backend.
Processed prompts: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:40<00:00,  2.47it/s, est. speed input: 76.44 toks/s, output: 78.91 toks/s]
Throughput: 2.46 requests/s, 157.72 tokens/s

```

## 直接使用本地OpenVINO IR格式的模型进行基准性能测试
基准测试命令：
```bash
    VLLM_OPENVINO_KVCACHE_SPACE=30  \
    VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8 \
    VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
         python3 vllm/benchmarks/benchmark_throughput.py \
            --model /home/feng/disk1/models-llm/openvino-models/Qwen1.5-4B-Chat-OpenVINO-int4 \
            --input-len 32 \
            --output-len 32 \
            --num_prompts 100
```

基准测试运行结果：
```bash
(vllm-openvino) feng@feng-X99M-D3:~/llm/vllm$     VLLM_OPENVINO_KVCACHE_SPACE=30  \
    VLLM_OPENVINO_CPU_KV_CACHE_PRECISION=u8 \
    VLLM_OPENVINO_ENABLE_QUANTIZED_WEIGHTS=ON \
         python3 vllm/benchmarks/benchmark_throughput.py \
            --model /home/feng/disk1/models-llm/openvino-models/Qwen1.5-4B-Chat-OpenVINO-int4 \
            --input-len 32 \
            --output-len 32 \
            --num_prompts 100
WARNING 07-12 22:36:38 _custom_ops.py:14] Failed to import from vllm._C with ModuleNotFoundError("No module named 'vllm._C'")
Namespace(backend='vllm', dataset=None, input_len=32, output_len=32, model='/home/feng/disk1/models-llm/openvino-models/Qwen1.5-4B-Chat-OpenVINO-int4', tokenizer='/home/feng/disk1/models-llm/openvino-models/Qwen1.5-4B-Chat-OpenVINO-int4', quantization=None, tensor_parallel_size=1, n=1, use_beam_search=False, num_prompts=100, seed=0, hf_max_batch_size=None, trust_remote_code=False, max_model_len=None, dtype='auto', gpu_memory_utilization=0.9, enforce_eager=False, kv_cache_dtype='auto', quantization_param_path=None, device='auto', enable_prefix_caching=False, enable_chunked_prefill=False, max_num_batched_tokens=None, download_dir=None, output_json=None, distributed_executor_backend=None, load_format='auto')
INFO 07-12 22:36:38 llm_engine.py:174] Initializing an LLM engine (v0.5.1) with config: model='/home/feng/disk1/models-llm/openvino-models/Qwen1.5-4B-Chat-OpenVINO-int4', speculative_config=None, tokenizer='/home/feng/disk1/models-llm/openvino-models/Qwen1.5-4B-Chat-OpenVINO-int4', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, rope_scaling=None, rope_theta=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.bfloat16, max_seq_len=32768, download_dir=None, load_format=LoadFormat.AUTO, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=False, kv_cache_dtype=auto, quantization_param_path=None, device_config=cpu, decoding_config=DecodingConfig(guided_decoding_backend='outlines'), observability_config=ObservabilityConfig(otlp_traces_endpoint=None), seed=0, served_model_name=/home/feng/disk1/models-llm/openvino-models/Qwen1.5-4B-Chat-OpenVINO-int4, use_v2_block_manager=False, enable_prefix_caching=False)
WARNING 07-12 22:36:39 openvino_executor.py:132] Only float32 dtype is supported on OpenVINO, casting from torch.bfloat16.
WARNING 07-12 22:36:39 openvino_executor.py:137] CUDA graph is not supported on OpenVINO backend, fallback to the eager mode.
INFO 07-12 22:36:39 openvino_executor.py:146] KV cache type is overried to u8 via VLLM_OPENVINO_CPU_KV_CACHE_PRECISION env var.
INFO 07-12 22:36:39 openvino_executor.py:159] OpenVINO optimal block size is 32, overriding currently set 16
INFO 07-12 22:36:42 selector.py:121] Cannot use _Backend.FLASH_ATTN backend on OpenVINO.
INFO 07-12 22:36:42 selector.py:69] Using OpenVINO Attention backend.
WARNING 07-12 22:36:42 openvino.py:130] OpenVINO IR is available for provided model id /home/feng/disk1/models-llm/openvino-models/Qwen1.5-4B-Chat-OpenVINO-int4. This IR will be used for inference as-is, all possible options that may affect model conversion are ignored.
INFO:nncf:Statistics of the bitwidth distribution:
┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
│ Num bits (N)   │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
INFO 07-12 22:36:54 openvino_executor.py:72] # CPU blocks: 4626
INFO 07-12 22:36:54 selector.py:121] Cannot use _Backend.FLASH_ATTN backend on OpenVINO.
INFO 07-12 22:36:54 selector.py:69] Using OpenVINO Attention backend.
Processed prompts: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [02:15<00:00,  1.35s/it, est. speed input: 22.96 toks/s, output: 23.70 toks/s]
Throughput: 0.74 requests/s, 47.39 tokens/s
```



# 问题记录

## 曾经在VLLM上提出过的issue

1、Installation with OpenVINO get dependency conflict Error !!! ——   https://github.com/vllm-project/vllm/issues/6243   
2、[Bug]: get that Exception in thread Thread-3 —— https://github.com/vllm-project/vllm/issues/6340