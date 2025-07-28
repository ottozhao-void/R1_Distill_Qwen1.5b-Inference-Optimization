"""
示例脚本 - 演示如何使用各个模块

这个脚本展示了如何在代码中直接使用项目的各个模块，
而不是通过命令行参数。适合集成到其他项目中。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, ConfigPresets
from inference_paged_attention import InferenceOnPagedAttention
from performance_monitor import compare_metrics


def example_basic_usage():
    """基础使用示例"""
    print("🔧 基础使用示例")
    print("=" * 50)
    
    # 创建配置
    config = ConfigPresets.small_model_config()
    config.test_iterations = 2  # 减少迭代次数以节省时间
    config.warmup_iterations = 1
    
    print(f"使用模型: {config.model}")
    
    # 创建推理模块
    inference_module = InferenceOnPagedAttention(config)
    
    try:
        # 准备测试提示
        prompts = [
            "The future of AI is",
            "Machine learning will help us",
            "Climate change affects"
        ]
        
        print(f"测试提示数量: {len(prompts)}")
        
        # 运行推理测试
        results = inference_module.inference(prompts, method="both")
        
        # 打印一些生成的示例
        if "basic" in results and "optimized" in results:
            print("\n📝 生成示例:")
            basic_outputs = results["basic"]["outputs"]
            optimized_outputs = results["optimized"]["outputs"]
            
            for i, prompt in enumerate(prompts[:2]):  # 只显示前两个
                if i < len(basic_outputs) and i < len(optimized_outputs):
                    print(f"\n提示: {prompt}")
                    print(f"基础推理: {basic_outputs[i][:100]}...")
                    print(f"优化推理: {optimized_outputs[i][:100]}...")
        
        return results
        
    except Exception as e:
        print(f"❌ 基础使用示例失败: {e}")
        return None
    finally:
        inference_module.cleanup()


def example_custom_config():
    """自定义配置示例"""
    print("\n🔧 自定义配置示例")
    print("=" * 50)
    
    # 创建自定义配置
    config = Config(
        model="facebook/opt-125m",  # 使用小模型
        max_tokens=30,
        temperature=0.9,
        num_prompts=3,
        test_iterations=2,
        warmup_iterations=1,
        output_dir="custom_results"
    )
    
    print(f"自定义配置:")
    print(f"  模型: {config.model}")
    print(f"  最大tokens: {config.max_tokens}")
    print(f"  温度: {config.temperature}")
    
    inference_module = InferenceOnPagedAttention(config)
    
    try:
        # 使用自定义提示
        custom_prompts = [
            "Tell me about",
            "The best way to",
            "In my opinion"
        ]
        
        results = inference_module.inference(custom_prompts, method="optimized")
        
        if "optimized" in results:
            print("\n✅ 自定义配置测试完成")
            metrics = results["optimized"]["metrics"]
            print(f"平均延迟: {metrics.avg_latency*1000:.2f}ms")
            print(f"吞吐量: {metrics.tokens_per_second:.2f} tokens/s")
        
        return results
        
    except Exception as e:
        print(f"❌ 自定义配置示例失败: {e}")
        return None
    finally:
        inference_module.cleanup()


def example_performance_comparison():
    """性能对比示例"""
    print("\n📊 性能对比示例")
    print("=" * 50)
    
    config = ConfigPresets.small_model_config()
    config.test_iterations = 2
    
    inference_module = InferenceOnPagedAttention(config)
    
    try:
        prompts = ["Artificial intelligence is", "The future holds"]
        
        # 分别运行两种方法
        print("运行基础推理...")
        basic_results = inference_module.inference(prompts, method="basic")
        
        print("运行优化推理...")
        optimized_results = inference_module.inference(prompts, method="optimized")
        
        # 手动对比结果
        if "basic" in basic_results and "optimized" in optimized_results:
            basic_metrics = basic_results["basic"]["metrics"]
            optimized_metrics = optimized_results["optimized"]["metrics"]
            
            print("\n📈 性能对比结果:")
            print(f"基础推理延迟: {basic_metrics.avg_latency*1000:.2f}ms")
            print(f"优化推理延迟: {optimized_metrics.avg_latency*1000:.2f}ms")
            
            improvement = (basic_metrics.avg_latency - optimized_metrics.avg_latency) / basic_metrics.avg_latency * 100
            print(f"延迟改善: {improvement:.1f}%")
            
            # 使用内置的对比函数
            compare_metrics(basic_metrics, optimized_metrics, "传统方法", "PagedAttention")
        
        return basic_results, optimized_results
        
    except Exception as e:
        print(f"❌ 性能对比示例失败: {e}")
        return None, None
    finally:
        inference_module.cleanup()


def example_memory_benchmark():
    """内存基准测试示例"""
    print("\n💾 内存基准测试示例")
    print("=" * 50)
    
    config = ConfigPresets.small_model_config()
    config.test_iterations = 1  # 减少时间
    
    inference_module = InferenceOnPagedAttention(config)
    
    try:
        # 测试不同批次大小
        batch_sizes = [1, 2]
        results = inference_module.benchmark_memory_efficiency(batch_sizes)
        
        print("\n📊 内存使用对比:")
        for batch_size, result in results.items():
            if "error" not in result:
                basic_memory = result["basic"]["metrics"].gpu_memory_peak
                optimized_memory = result["optimized"]["metrics"].gpu_memory_peak
                print(f"批次大小 {batch_size}:")
                print(f"  基础方法: {basic_memory:.1f}MB")
                print(f"  优化方法: {optimized_memory:.1f}MB")
                if basic_memory > 0:
                    savings = (basic_memory - optimized_memory) / basic_memory * 100
                    print(f"  内存节省: {savings:.1f}%")
        
        return results
        
    except Exception as e:
        print(f"❌ 内存基准测试示例失败: {e}")
        return None
    finally:
        inference_module.cleanup()


def main():
    """运行所有示例"""
    print("🎯 LLM推理优化项目 - 示例脚本")
    print("=" * 60)
    
    # 检查基本依赖
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        print(f"✅ CUDA可用: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch未安装，某些示例可能无法运行")
    
    try:
        import transformers
        print(f"✅ Transformers版本: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers未安装")
        return
    
    try:
        import vllm
        print(f"✅ vLLM可用")
    except ImportError:
        print("❌ vLLM未安装")
        return
    
    # 运行示例
    try:
        # 示例1: 基础使用
        result1 = example_basic_usage()
        
        # 示例2: 自定义配置
        result2 = example_custom_config()
        
        # 示例3: 性能对比
        result3a, result3b = example_performance_comparison()
        
        # 示例4: 内存基准测试
        result4 = example_memory_benchmark()
        
        print("\n🎉 所有示例运行完成！")
        print("\n💡 更多用法请参考:")
        print("  - main.py: 命令行接口")
        print("  - README.md: 详细文档")
        print("  - run_benchmark.sh: 自动化测试脚本")
        
    except KeyboardInterrupt:
        print("\n⏹️ 示例被用户中断")
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
