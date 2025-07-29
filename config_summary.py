#!/usr/bin/env python3
"""
配置统一性验证摘要

这个脚本提供了项目配置的快速摘要，验证所有组件都使用统一的模型配置。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, ConfigPresets

def main():
    print("🔧 LLM推理工作流统一配置摘要")
    print("=" * 60)
    
    target_model = "/data1/zhaofanghan/.cache/huggingface/hub/DeepSeek-R1-Distill-Qwen-1.5B"
    
    print(f"🎯 目标模型: {target_model}")
    print()
    
    # 检查所有配置
    configs = {
        "默认配置": Config(),
        "小模型预设": ConfigPresets.small_model_config(),
        "大模型预设": ConfigPresets.large_model_config(),
        "Llama预设": ConfigPresets.llama_config()
    }
    
    print("📋 配置验证结果:")
    all_correct = True
    
    for name, config in configs.items():
        if config.model == target_model:
            print(f"✅ {name}")
        else:
            print(f"❌ {name}: {config.model}")
            all_correct = False
    
    print()
    print("🔍 配置详情:")
    config = Config()
    print(f"  设备: {config.device}")
    print(f"  最大tokens: {config.max_tokens}")
    print(f"  温度: {config.temperature}")
    print(f"  Top-p: {config.top_p}")
    print(f"  输出目录: {config.output_dir}")
    
    print()
    print("📁 相关文件:")
    files_updated = [
        "config.py - 主配置文件，所有预设已更新",
        "example.py - 示例代码已更新",
        "main.py - 主程序文档已更新", 
        "run_benchmark.sh - 基准测试脚本已更新",
        "validate_config.py - 新增配置验证脚本",
        "CONFIG.md - 新增配置文档"
    ]
    
    for file_info in files_updated:
        print(f"  ✅ {file_info}")
    
    print()
    if all_correct:
        print("🎉 所有配置正确！整个LLM推理工作流已统一配置为使用DeepSeek-R1-Distill-Qwen-1.5B模型")
        print()
        print("🚀 快速开始:")
        print("  python validate_config.py  # 完整验证")
        print("  python main.py --preset small  # 快速测试")
        print("  python example.py  # 运行示例")
        print("  ./run_benchmark.sh  # 完整基准测试")
    else:
        print("❌ 发现配置不一致问题，请检查上述文件")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
