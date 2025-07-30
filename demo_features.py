#!/usr/bin/env python3
"""
完整功能演示脚本 - 展示所有新增功能
"""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def demo_config_system():
    """演示配置系统"""
    print("📋 配置系统演示")
    print("="*50)
    
    from config.config import get_config_for_technique, load_config_from_yaml
    
    # 展示不同技术和批次级别的配置
    techniques = ["PagedAttention", "量化"]
    batch_levels = ["small", "medium", "large"]
    
    for technique in techniques:
        print(f"\n🔧 {technique} 配置:")
        for level in batch_levels:
            config = get_config_for_technique(technique, level, "demo")
            print(f"  {level}: batch_size={config.batch_size}, num_prompts={config.num_prompts}, max_tokens={config.max_tokens}")

def demo_performance_metrics():
    """演示新的性能指标"""
    print("\n📊 新性能指标演示")
    print("="*50)
    
    from utils.performance_monitor import PerformanceMonitor, PerformanceMetrics
    import numpy as np
    
    monitor = PerformanceMonitor()
    
    # 模拟一些性能数据
    monitor.latencies = [1.5, 1.6, 1.4, 1.7, 1.5]
    monitor.token_counts = [50, 48, 52, 45, 49]
    monitor.first_token_times = [0.8, 0.9, 0.7, 0.85, 0.82]
    
    # 模拟监测时间
    monitor.start_time = time.time() - 10
    monitor.end_time = time.time()
    
    metrics = monitor.get_metrics("demo-model", 4)
    
    print(f"✓ Time to First Token: {metrics.time_to_first_token*1000:.2f}ms")
    print(f"✓ Time Per Output Token: {metrics.time_per_output_token*1000:.2f}ms")
    print(f"✓ 传统延迟指标: {metrics.avg_latency*1000:.2f}ms")
    print(f"✓ 吞吐量: {metrics.tokens_per_second:.2f} tokens/s")

def demo_report_formats():
    """演示报告格式"""
    print("\n📄 报告格式演示")
    print("="*50)
    
    print("✓ JSON格式报告 - 详细的机器可读数据")
    print("✓ Markdown格式报告 - 人类可读的格式化报告")
    print("✓ 对比实验报告 - 包含改进分析的完整报告")
    print("✓ 自动保存到技术特定文件夹")

def demo_yaml_configs():
    """演示YAML配置"""
    print("\n⚙️ YAML配置示例")
    print("="*50)
    
    sample_config = """# PagedAttention实验配置
experiment_name: production_test
technique: PagedAttention
batch_level: large
model: /path/to/model
device: [0, 1]
max_tokens: 200
temperature: 0.8
batch_size: 8
test_iterations: 10
quantization_config: null"""
    
    print("PagedAttention配置示例:")
    print(sample_config)
    
    print("\n量化配置会包含额外的量化参数:")
    print("quantization_config:")
    print("  load_in_8bit: true")
    print("  bnb_4bit_compute_dtype: float16")

def demo_usage_examples():
    """演示使用示例"""
    print("\n🚀 使用示例")
    print("="*50)
    
    examples = [
        ("快速PagedAttention测试", "python main.py --technique PagedAttention --batch-level small"),
        ("量化性能对比", "python main.py --technique 量化 --batch-level medium --method both"),
        ("大批次内存基准测试", "python main.py --technique PagedAttention --batch-level large --benchmark-memory"),
        ("自定义配置测试", "python main.py --config config/量化/量化_large_custom.yaml"),
        ("只运行优化推理", "python main.py --technique 量化 --method optimized --test-iterations 3")
    ]
    
    for desc, cmd in examples:
        print(f"📌 {desc}:")
        print(f"   {cmd}")
        print()

def main():
    """主演示函数"""
    print("🎉 R1_Distill_Qwen1.5b 推理优化项目 - 新功能演示")
    print("="*80)
    
    try:
        demo_config_system()
        demo_performance_metrics()
        demo_report_formats()
        demo_yaml_configs()
        demo_usage_examples()
        
        print("\n✅ 所有新功能演示完成！")
        print("\n📚 查看 NEW_FEATURES.md 获取详细使用说明")
        print("🧪 运行 test_new_features.py 验证系统功能")
        print("🚀 开始使用: python main.py --technique PagedAttention --batch-level small")
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
