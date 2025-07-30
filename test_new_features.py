#!/usr/bin/env python3
"""
简单测试脚本 - 验证新的配置系统和量化推理模块
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import get_config_for_technique, save_config_to_yaml, load_config_from_yaml

def test_config_system():
    """测试配置系统"""
    print("🧪 测试配置系统...")
    
    # 测试PagedAttention配置
    print("\n1. 测试PagedAttention配置")
    config = get_config_for_technique("PagedAttention", "small", "test")
    print(f"  ✓ 配置加载成功: {config.technique}, {config.batch_level}")
    
    # 测试量化配置
    print("\n2. 测试量化配置")
    config = get_config_for_technique("量化", "medium", "test")
    print(f"  ✓ 配置加载成功: {config.technique}, {config.batch_level}")
    print(f"  ✓ 量化配置: {config.quantization_config}")
    
    print("\n✅ 配置系统测试通过")

def test_yaml_operations():
    """测试YAML操作"""
    print("\n🧪 测试YAML操作...")
    
    # 创建测试配置
    from config.config import Config
    test_config = Config()
    test_config.experiment_name = "yaml_test"
    test_config.technique = "量化"
    test_config.batch_level = "small"
    
    # 保存配置
    test_path = "test_config.yaml"
    save_config_to_yaml(test_config, test_path)
    
    # 加载配置
    loaded_config = load_config_from_yaml(test_path)
    print(f"  ✓ YAML保存和加载成功: {loaded_config.experiment_name}")
    
    # 清理
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("✅ YAML操作测试通过")

def main():
    """主函数"""
    print("🚀 新功能测试")
    print("="*50)
    
    try:
        test_config_system()
        test_yaml_operations()
        
        print("\n🎉 所有测试通过！")
        print("\n📝 使用方法:")
        print("  python main.py --technique PagedAttention --batch-level small")
        print("  python main.py --technique 量化 --batch-level medium")
        print("  python main.py --technique PagedAttention --benchmark-memory")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
