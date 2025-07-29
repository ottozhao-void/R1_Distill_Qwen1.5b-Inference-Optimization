#!/usr/bin/env python3
"""
配置验证脚本

验证整个LLM推理工作流是否正确配置为使用DeepSeek-R1-Distill-Qwen-1.5B模型
"""

import os
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, ConfigPresets


def validate_model_path():
    """验证模型路径是否存在"""
    target_model = "/data1/zhaofanghan/.cache/huggingface/hub/DeepSeek-R1-Distill-Qwen-1.5B"
    
    print("🔍 验证模型路径...")
    
    if not os.path.exists(target_model):
        print(f"❌ 模型路径不存在: {target_model}")
        print("请确保模型已正确下载到指定路径")
        return False
    
    # 检查是否包含必要的模型文件
    required_files = ["config.json"]
    optional_files = ["pytorch_model.bin", "model.safetensors", "tokenizer.json"]
    
    missing_required = []
    for file in required_files:
        if not os.path.exists(os.path.join(target_model, file)):
            missing_required.append(file)
    
    if missing_required:
        print(f"❌ 缺少必要的模型文件: {missing_required}")
        return False
    
    # 检查可选文件
    found_optional = []
    for file in optional_files:
        if os.path.exists(os.path.join(target_model, file)):
            found_optional.append(file)
    
    print(f"✅ 模型路径验证通过: {target_model}")
    print(f"   必要文件: {required_files}")
    print(f"   可选文件: {found_optional}")
    
    return True


def validate_all_configs():
    """验证所有配置是否使用正确的模型"""
    target_model = "/data1/zhaofanghan/.cache/huggingface/hub/DeepSeek-R1-Distill-Qwen-1.5B"
    
    print("\n🔧 验证配置一致性...")
    
    configs_to_check = {
        "默认配置": Config(),
        "小模型配置": ConfigPresets.small_model_config(),
        "大模型配置": ConfigPresets.large_model_config()
    }
    
    all_consistent = True
    
    for name, config in configs_to_check.items():
        if config.model == target_model:
            print(f"✅ {name}: {config.model}")
        else:
            print(f"❌ {name}: {config.model} (应该是 {target_model})")
            all_consistent = False
    
    return all_consistent


def validate_device_config():
    """验证设备配置"""
    print("\n🖥️ 验证设备配置...")
    
    config = Config()
    
    try:
        import torch
        
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"可用GPU数量: {device_count}")
            
            if config.device:
                for device_id in config.device:
                    if device_id >= device_count:
                        print(f"⚠️ 配置的设备ID {device_id} 超出可用范围 (0-{device_count-1})")
                        return False
                    else:
                        device_name = torch.cuda.get_device_name(device_id)
                        print(f"✅ 设备 {device_id}: {device_name}")
            
            print(f"✅ 设备配置: {config.device}")
        else:
            print("⚠️ CUDA不可用，将使用CPU")
    
    except ImportError:
        print("⚠️ PyTorch未安装，无法验证设备配置")
        return False
    
    return True


def validate_dependencies():
    """验证依赖包"""
    print("\n📦 验证依赖包...")
    
    required_packages = {
        "torch": "PyTorch",
        "transformers": "Hugging Face Transformers", 
        "vllm": "vLLM",
        "numpy": "NumPy",
        "psutil": "psutil"
    }
    
    missing_packages = []
    installed_packages = []
    
    for package, name in required_packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            installed_packages.append(f"{name} ({version})")
        except ImportError:
            missing_packages.append(name)
    
    for package in installed_packages:
        print(f"✅ {package}")
    
    if missing_packages:
        print(f"❌ 缺少依赖包: {missing_packages}")
        return False
    
    return True


def generate_config_summary():
    """生成配置摘要"""
    print("\n📋 配置摘要")
    print("=" * 60)
    
    config = Config()
    
    print(f"模型路径: {config.model}")
    print(f"设备配置: {config.device}")
    print(f"最大tokens: {config.max_tokens}")
    print(f"温度: {config.temperature}")
    print(f"Top-p: {config.top_p}")
    print(f"批次大小: {config.batch_size}")
    print(f"测试迭代: {config.test_iterations}")
    print(f"预热迭代: {config.warmup_iterations}")
    print(f"输出目录: {config.output_dir}")
    
    print("\n设备信息:")
    print(f"设备字符串: {config.get_device_str()}")
    print(f"张量并行大小: {config.get_tensor_parallel_size()}")
    print(f"多GPU模式: {config.is_multi_gpu()}")


def main():
    """主验证函数"""
    print("🚀 LLM推理工作流配置验证")
    print("="*60)
    
    # 验证各个方面
    validations = [
        ("模型路径", validate_model_path),
        ("配置一致性", validate_all_configs),
        ("设备配置", validate_device_config),
        ("依赖包", validate_dependencies)
    ]
    
    all_passed = True
    
    for name, validation_func in validations:
        try:
            result = validation_func()
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {name}验证失败: {e}")
            all_passed = False
    
    # 生成摘要
    generate_config_summary()
    
    # 最终结果
    print("\n" + "="*60)
    if all_passed:
        print("🎉 所有验证通过！LLM推理工作流已正确配置为使用DeepSeek-R1-Distill-Qwen-1.5B模型")
        print("\n✅ 可以开始运行推理测试:")
        print("   python main.py --preset small")
        print("   python example.py")
        print("   ./run_benchmark.sh")
    else:
        print("❌ 部分验证失败，请检查并修复相关问题")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
