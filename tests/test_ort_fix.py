"""
ORT 修复验证测试脚本

用于在远程服务器上测试优化后的 ORT 配置：
1. 测试输出重复率是否降低
2. 验证数据类型正确性
3. 检查性能改进

使用方法：
python test_ort_fix.py --audio test_data/audio_16000_10s.wav
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path

# 添加项目路径（tests/ 的上一级为项目根目录）
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import soundfile as sf
import onnxruntime as ort

from src.feature_extractor import extract_features, load_cmvn
from src.tokenizer import Tokenizer

def test_ort_performance_with_fix():
    """测试 ORT 性能和质量改进"""
    print("=" * 70)
    print("ORT 修复验证测试")
    print("=" * 70)
    
    # 加载测试音频
    audio_file = "test_data/audio_16000_10s.wav"
    audio_path = os.path.join(project_root, audio_file)
    
    if not os.path.exists(audio_path):
        print(f"测试音频不存在: {audio_path}")
        return
    
    print(f"测试音频: {audio_file}")
    
    # 配置目录（am.mvn, tokens.json 所在）
    config_dir = os.path.join(project_root, "models/asr/pt")
    vocab_path = os.path.join(config_dir, "tokens.json")
    cmvn_path = os.path.join(config_dir, "am.mvn")
    if not os.path.exists(vocab_path):
        print(f"tokenizer 文件不存在: {vocab_path}")
        return
    
    # 测试模型路径
    model_path = os.path.join(project_root, "models/asr/fp32/model.onnx")
    if not os.path.exists(model_path):
        print(f"ORT 模型不存在: {model_path}")
        return
    
    print(f"ORT 模型: {model_path}")
    
    # 测试优化的 ORT 配置
    print("\n[1/4] 加载音频和提取特征...")
    pcm, sr = sf.read(audio_path, dtype="float32")
    if pcm.ndim > 1:
        pcm = pcm[:, 0]
    
    cmvn_mean, cmvn_istd = (None, None)
    if os.path.exists(cmvn_path):
        cmvn_mean, cmvn_istd = load_cmvn(cmvn_path)
    
    feats = extract_features(pcm, sample_rate=sr, cmvn_mean=cmvn_mean, cmvn_istd=cmvn_istd)
    
    print(f"音频时长: {len(pcm)/sr:.2f}s")
    print(f"特征维度: {feats.shape}")
    
    # 转换为适合推理的格式
    batch_size = 1
    seq_len = feats.shape[0]
    padded_feats = feats[np.newaxis, :, :].astype(np.float32)
    lengths = np.array([seq_len], dtype=np.int64)
    
    # 预加载 tokenizer（供解码用）
    tokenizer = Tokenizer()
    tokenizer.load(vocab_path)
    
    # 准备 bias_embed（全零）
    bias_embed = np.zeros((1, 1, 512), dtype=np.float32)
    
    # 测试两种配置
    test_configs = [
        {
            "name": "原始配置（禁用内存模式）",
            "enable_mem_pattern": False,
            "enable_cpu_mem_arena": False,
        },
        {
            "name": "优化配置（启用内存模式）", 
            "enable_mem_pattern": True,
            "enable_cpu_mem_arena": True,
        }
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n[2/4] 测试配置: {config['name']}")
        
        # 创建 ORT session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_mem_pattern = config["enable_mem_pattern"]
        sess_options.enable_cpu_mem_arena = config["enable_cpu_mem_arena"]
        sess_options.log_severity_level = 2
        
        providers = [
            ("CUDAExecutionProvider", {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
            }),
            "CPUExecutionProvider",
        ]
        
        try:
            session = ort.InferenceSession(model_path, sess_options, providers=providers)
            input_names = [i.name for i in session.get_inputs()]
            output_names = [o.name for o in session.get_outputs()]
            
            print(f"  输入: {input_names}")
            print(f"  输出: {output_names}")
            
            # 预热
            print("  预热中...")
            for i in range(5):
                dummy_feats = np.random.randn(1, seq_len, 560).astype(np.float32)
                dummy_lengths = np.array([seq_len], dtype=np.int64)
                dummy_bias = np.zeros((1, 1, 512), dtype=np.float32)
                
                feed = {}
                for name in input_names:
                    if name == "speech":
                        feed[name] = dummy_feats
                    elif name == "speech_lengths":
                        feed[name] = dummy_lengths
                    elif "bias_embed" in name:
                        feed[name] = dummy_bias
                
                session.run(output_names, feed)
            
            # 实际推理
            print("  推理中...")
            inference_times = []
            
            for i in range(10):
                feed = {}
                for name in input_names:
                    if name == "speech":
                        feed[name] = padded_feats
                    elif name == "speech_lengths":
                        feed[name] = lengths
                    elif "bias_embed" in name:
                        feed[name] = bias_embed
                
                start_time = time.time()
                outputs = session.run(output_names, feed)
                inference_times.append((time.time() - start_time) * 1000)  # 毫秒
            
            avg_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            rtf = avg_time / (len(pcm) / 16000 * 1000)  # 音频时长毫秒
            
            # 分析输出
            logits = outputs[0]  # (1, token_len, 8404)
            # token_num 输出 shape=(batch,)，取第 0 个样本的有效 token 数
            if len(outputs) > 1:
                token_num = int(round(float(np.asarray(outputs[1]).flatten()[0])))
            else:
                token_num = logits.shape[1]
            token_num = max(1, min(token_num, logits.shape[1]))
            
            # 解码测试
            try:
                # 取 top-1 token
                token_ids = np.argmax(logits[0, :token_num], axis=1)
                decoded = tokenizer.decode(token_ids)
                
                # 计算重复率（按字符级连续重复统计）
                if len(decoded) > 1:
                    repeated = sum(1 for i in range(1, len(decoded)) if decoded[i] == decoded[i-1])
                    repetition_rate = repeated / (len(decoded) - 1)
                else:
                    repetition_rate = 0
                
                result = {
                    "config_name": config["name"],
                    "avg_inference_ms": avg_time,
                    "std_inference_ms": std_time,
                    "rtf": rtf,
                    "token_num": int(token_num),
                    "repetition_rate": repetition_rate,
                    "decoded_preview": decoded[:100] + "..." if len(decoded) > 100 else decoded,
                    "status": "成功"
                }
                
                print(f"  平均推理时间: {avg_time:.2f}ms (±{std_time:.2f}ms)")
                print(f"  RTF: {rtf:.4f}")
                print(f"  输出 token 数: {int(token_num)}")
                print(f"  输出重复率: {repetition_rate:.2%}")
                print(f"  解码预览: {result['decoded_preview']}")
                
            except Exception as e:
                print(f"  解码失败: {e}")
                result = {
                    "config_name": config["name"],
                    "error": str(e),
                    "status": "失败"
                }
            
            results.append(result)
            
        except Exception as e:
            print(f"  ORT 配置测试失败: {e}")
            results.append({
                "config_name": config["name"],
                "error": str(e),
                "status": "失败"
            })
    
    print("\n" + "=" * 70)
    print("测试结果总结")
    print("=" * 70)
    
    for result in results:
        print(f"\n{result['config_name']}:")
        if result["status"] == "成功":
            print(f"  状态: ✓ {result['status']}")
            print(f"  推理时间: {result['avg_inference_ms']:.2f}ms (±{result['std_inference_ms']:.2f}ms)")
            print(f"  RTF: {result['rtf']:.4f}")
            print(f"  输出重复率: {result['repetition_rate']:.2%}")
            print(f"  解码预览: {result['decoded_preview']}")
        else:
            print(f"  状态: ✗ {result['status']}")
            print(f"  错误: {result.get('error', '未知错误')}")
    
    # 保存结果
    output_file = os.path.join(project_root, "ort_fix_test_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_audio": audio_file,
            "results": results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n详细结果已保存到: {output_file}")
    
    # 检查是否有所改进
    success_results = [r for r in results if r.get("status") == "成功"]
    if len(success_results) >= 2:
        original = next(r for r in success_results if "原始配置" in r["config_name"])
        optimized = next(r for r in success_results if "优化配置" in r["config_name"])
        
        print("\n" + "=" * 70)
        print("改进对比")
        print("=" * 70)
        
        repetition_improvement = original.get("repetition_rate", 1) - optimized.get("repetition_rate", 1)
        if repetition_improvement > 0:
            print(f"✓ 输出重复率改进: {repetition_improvement:.2%} 降低")
        elif repetition_improvement < 0:
            print(f"✗ 输出重复率恶化: {abs(repetition_improvement):.2%} 增加")
        else:
            print("○ 输出重复率无变化")
        
        rtf_improvement = original.get("rtf", 1) - optimized.get("rtf", 1)
        if rtf_improvement > 0:
            print(f"✓ RTF 改进: {rtf_improvement:.4f} 降低")
        elif rtf_improvement < 0:
            print(f"✗ RTF 恶化: {abs(rtf_improvement):.4f} 增加")
        else:
            print("○ RTF 无变化")

def main():
    parser = argparse.ArgumentParser(description="测试 ORT 修复效果")
    parser.add_argument("--audio", default="test_data/audio_16000_10s.wav",
                       help="测试音频文件路径")
    args = parser.parse_args()
    
    test_ort_performance_with_fix()

if __name__ == "__main__":
    main()