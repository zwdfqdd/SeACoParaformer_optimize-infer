"""
数据类型验证测试

专门测试 ORT 推理中的数据类型问题：
1. 验证输入数据类型是否与模型期望匹配
2. 测试不同数据类型的兼容性
3. 检测数据类型相关的问题

使用方法：
python tests/test_dtype_validation.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

# 添加项目路径（tests/ 的上一级为项目根目录）
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_ort_input_data_types():
    """测试 ORT 输入数据类型验证"""
    print("=" * 70)
    print("ORT 输入数据类型验证测试")
    print("=" * 70)
    
    # 加载模型
    model_path = os.path.join(project_root, "models/asr/fp32/model.onnx")
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    print(f"加载模型: {model_path}")
    
    # 创建 ORT session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_mem_pattern = True
    sess_options.enable_cpu_mem_arena = True
    
    providers = [
        ("CUDAExecutionProvider", {"device_id": 0}),
        "CPUExecutionProvider",
    ]
    
    try:
        session = ort.InferenceSession(model_path, sess_options, providers=providers)
        
        # 获取模型输入输出信息
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        print("\n模型输入信息:")
        for inp in inputs:
            print(f"  名称: {inp.name}")
            print(f"  类型: {inp.type}")
            print(f"  形状: {inp.shape}")
            print(f"  ")
        
        print("\n模型输出信息:")
        for out in outputs:
            print(f"  名称: {out.name}")
            print(f"  类型: {out.type}")
            print(f"  形状: {out.shape}")
            print(f"  ")
        
        # 测试不同的数据类型组合
        test_cases = [
            {
                "name": "正确数据类型（float32 + int64）",
                "speech_dtype": np.float32,
                "lengths_dtype": np.int64,
                "expected": "应该成功"
            },
            {
                "name": "错误数据类型1（float64 + int64）",
                "speech_dtype": np.float64,
                "lengths_dtype": np.int64,
                "expected": "可能失败或需要类型转换"
            },
            {
                "name": "错误数据类型2（float32 + int32）",
                "speech_dtype": np.float32,
                "lengths_dtype": np.int32,
                "expected": "已知问题：应该失败"
            },
            {
                "name": "错误数据类型3（float16 + int64）",
                "speech_dtype": np.float16,
                "lengths_dtype": np.int64,
                "expected": "可能失败或精度问题"
            }
        ]
        
        print("\n数据类型测试:")
        print("-" * 50)
        
        batch_size = 1
        seq_len = 100
        feat_dim = 560
        
        for i, test_case in enumerate(test_cases):
            print(f"\n[{i+1}] 测试: {test_case['name']}")
            print(f"   预期: {test_case['expected']}")
            
            try:
                # 创建测试输入
                speech = np.random.randn(batch_size, seq_len, feat_dim).astype(test_case["speech_dtype"])
                lengths = np.array([seq_len], dtype=test_case["lengths_dtype"])
                bias_embed = np.zeros((batch_size, 1, 512), dtype=np.float32)
                
                # 准备 feed（注意：speech_lengths 也含 "speech"，必须先判断 lengths）
                feed = {}
                for inp in inputs:
                    iname = inp.name.lower()
                    if "length" in iname:
                        feed[inp.name] = lengths
                    elif "bias" in iname:
                        feed[inp.name] = bias_embed
                    elif "speech" in iname:
                        feed[inp.name] = speech
                
                # 运行推理
                output_names = [out.name for out in outputs]
                outputs_result = session.run(output_names, feed)
                
                # 检查输出
                logits = outputs_result[0]
                print(f"   状态: ✓ 成功")
                print(f"   logits shape: {logits.shape}")
                print(f"   logits dtype: {logits.dtype}")
                
                # 检查是否有 NaN 或 Inf
                if np.any(np.isnan(logits)):
                    print(f"   警告: logits 包含 NaN")
                if np.any(np.isinf(logits)):
                    print(f"   警告: logits 包含 Inf")
                
                # 检查输出重复率
                if len(logits.shape) >= 3:
                    batch, token_num, vocab = logits.shape
                    if token_num > 1:
                        # 取 top-1 token
                        top_tokens = np.argmax(logits[0], axis=1)
                        repeated = np.sum(top_tokens[1:] == top_tokens[:-1])
                        repetition_rate = repeated / (token_num - 1)
                        print(f"   输出重复率: {repetition_rate:.2%}")
                
            except Exception as e:
                print(f"   状态: ✗ 失败: {e}")
                
                # 如果是 int32 问题，提供修复建议
                if test_case["lengths_dtype"] == np.int32:
                    print(f"   建议: 将 int32 转换为 int64: lengths.astype(np.int64)")
        
        # 测试类型转换
        print("\n\n类型转换测试:")
        print("-" * 50)
        
        # 测试 int32 转 int64
        print("\n[1] int32 到 int64 转换测试:")
        int32_lengths = np.array([seq_len], dtype=np.int32)
        int64_lengths = int32_lengths.astype(np.int64)
        print(f"   原始 int32: {int32_lengths.dtype}")
        print(f"   转换后 int64: {int64_lengths.dtype}")
        
        # 测试是否可以通过转换解决问题
        print("\n[2] 通过类型转换的推理测试:")
        speech = np.random.randn(batch_size, seq_len, feat_dim).astype(np.float32)
        wrong_lengths = np.array([seq_len], dtype=np.int32)
        correct_lengths = wrong_lengths.astype(np.int64)
        bias_embed = np.zeros((batch_size, 1, 512), dtype=np.float32)
        
        try:
            feed = {}
            for inp in inputs:
                iname = inp.name.lower()
                if "length" in iname:
                    feed[inp.name] = correct_lengths  # 使用转换后的 int64
                elif "bias" in iname:
                    feed[inp.name] = bias_embed
                elif "speech" in iname:
                    feed[inp.name] = speech
            
            outputs_result = session.run(output_names, feed)
            print(f"   状态: ✓ 类型转换后推理成功")
            
        except Exception as e:
            print(f"   状态: ✗ 类型转换后仍然失败: {e}")
        
        # 输出数据类型兼容性建议
        print("\n" + "=" * 70)
        print("数据类型兼容性建议")
        print("=" * 70)
        print("\n基于测试结果，建议：")
        print("1. 确保 speech_lengths 使用 np.int64（不是 np.int32）")
        print("2. 确保 speech 使用 np.float32")
        print("3. 确保 bias_embed 使用 np.float32")
        print("4. 在推理前进行类型检查：")
        print("   if lengths.dtype != np.int64:")
        print("       lengths = lengths.astype(np.int64)")
        print("   if speech.dtype != np.float32:")
        print("       speech = speech.astype(np.float32)")
        
    except Exception as e:
        print(f"ORT session 创建失败: {e}")

def check_current_code_type_handling():
    """检查当前代码中的类型处理"""
    print("\n" + "=" * 70)
    print("当前代码类型处理检查")
    print("=" * 70)
    
    print("\n1. _infer_batch_raw_ort 函数类型处理分析:")
    print("   - lengths.astype(np.int64) ✓ 正确转换")
    print("   - bias_embeddings.astype(np.float32) ✓ 正确转换")
    print("   - speech 数据类型没有显式转换，依赖调用方")
    
    print("\n2. 潜在问题:")
    print("   - 如果输入 speech 不是 float32，可能存在问题")
    print("   - 如果没有显式检查，可能传递错误类型")
    
    print("\n3. 建议改进:")
    print("   - 添加输入数据类型验证")
    print("   - 自动类型转换（如 float64 → float32）")
    print("   - 添加详细的错误信息")

def main():
    test_ort_input_data_types()
    check_current_code_type_handling()

if __name__ == "__main__":
    main()