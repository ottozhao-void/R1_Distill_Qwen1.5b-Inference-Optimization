"""
主程序 - 大语言模型推理和优化

这是项目的主入口文件，演示如何使用各个模块进行LLM推理性能对比测试。

注意：本项目已统一配置为使用 DeepSeek-R1-Distill-Qwen-1.5B 模型，
确保 PagedAttention 与传统 Key-Value Cache 对比实验的公平性。
"""

import argparse
import sys
import os
from typing import List, Optional
import vllm
import torch
import transformers

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config, load_config_from_yaml, get_config_for_technique, print_available_devices
from inference.inference_paged_attention import InferenceOnPagedAttention
from inference.inference_quantization import InferenceOnQuantization


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="LLM推理优化性能测试",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 优化技术选择
    parser.add_argument("--technique", type=str, 
                       choices=["PagedAttention", "量化"], 
                       default="PagedAttention",
                       help="选择优化技术")
    
    # 批次级别
    parser.add_argument("--batch-level", type=str,
                       choices=["small", "medium", "large"],
                       default="medium",
                       help="批次级别")
    
    # 实验名称
    parser.add_argument("--experiment-name", type=str, default="default",
                       help="实验名称")
    
    # 配置文件路径
    parser.add_argument("--config", type=str, default=None,
                       help="YAML配置文件路径")
    
    # 设备配置
    parser.add_argument("--device", type=str, default=None,
                       help="指定CUDA设备，例如: '0' 表示cuda:0, '0,1' 表示使用GPU 0和1, 'cpu' 表示使用CPU")
    parser.add_argument("--list-devices", action="store_true",
                       help="显示可用设备信息并退出")
    
    # 推理配置
    parser.add_argument("--max-tokens", type=int, default=None,
                       help="最大生成tokens数量")
    parser.add_argument("--temperature", type=float, default=None,
                       help="采样温度")
    parser.add_argument("--top-p", type=float, default=None,
                       help="核采样参数")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="批次大小")
    
    # 测试配置
    parser.add_argument("--num-prompts", type=int, default=None,
                       help="测试提示数量")
    parser.add_argument("--test-iterations", type=int, default=None,
                       help="测试迭代次数")
    parser.add_argument("--warmup-iterations", type=int, default=None,
                       help="预热迭代次数")
    
    # 运行模式
    parser.add_argument("--method", type=str, 
                       choices=["basic", "optimized", "both"], 
                       default="both",
                       help="推理方法")
    
    # 基准测试
    parser.add_argument("--benchmark-memory", action="store_true",
                       help="运行内存效率基准测试")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8",
                       help="内存基准测试的批次大小，用逗号分隔")
    
    # 输出配置
    parser.add_argument("--output-dir", type=str, default=None,
                       help="输出目录")
    parser.add_argument("--no-save", action="store_true",
                       help="不保存结果文件")
    
    # 自定义提示
    parser.add_argument("--prompts", nargs="+", default=None,
                       help="自定义测试提示")
    parser.add_argument("--prompts-file", type=str, default=None,
                       help="从文件读取测试提示")
    
    return parser.parse_args()


def load_prompts_from_file(filepath: str) -> List[str]:
    """从文件加载提示"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"从文件 {filepath} 加载了 {len(prompts)} 个提示")
        return prompts
    except Exception as e:
        print(f"无法从文件加载提示: {e}")
        return []


def create_config_from_args(args) -> Config:
    """根据命令行参数创建配置"""
    
    # 从指定配置文件加载
    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        # 根据技术和批次级别获取配置
        config = get_config_for_technique(
            technique=args.technique,
            batch_level=args.batch_level,
            experiment_name=args.experiment_name
        )
    
    # 用命令行参数覆盖配置
    if args.device:
        # 解析设备参数
        if args.device.lower() == 'cpu':
            config.device = []
        else:
            try:
                # 支持 "0" 或 "0,1,2" 格式
                device_ids = [int(x.strip()) for x in args.device.split(',')]
                config.device = device_ids
                print(f"设置设备为: {device_ids}")
            except ValueError:
                print(f"警告: 无效的设备参数 '{args.device}'，使用默认设备")
    
    if args.max_tokens:
        config.max_tokens = args.max_tokens
    
    if args.temperature:
        config.temperature = args.temperature
    
    if args.top_p:
        config.top_p = args.top_p
    
    if args.batch_size:
        config.batch_size = args.batch_size
    
    if args.num_prompts:
        config.num_prompts = args.num_prompts
    
    if args.test_iterations:
        config.test_iterations = args.test_iterations
    
    if args.warmup_iterations:
        config.warmup_iterations = args.warmup_iterations
    
    if args.output_dir:
        config.output_dir = args.output_dir
    
    if args.no_save:
        config.save_results = False
    
    return config



def main():

    # # 设置环境变量以确保CUDA库的正确加载
    # os.environ["LD_LIBRARY_PATH"] = "/home/zhaofanghan/tmp/lib:/home/zhaofanghan/tmp/cuda_stubs:" + os.environ.get("LD_LIBRARY_PATH", "")
    # os.environ["CUDA_HOME"] = "/usr/local/cuda"
    # os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ.get("PATH", "")
    
    
    """主函数"""
    print("🚀 大语言模型推理和优化性能测试")
    print("="*60)
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 如果用户请求显示设备信息，显示后退出
    if args.list_devices:
        print_available_devices()
        return
    
    # 创建配置
    config = create_config_from_args(args)
    
    print(f"\n📋 测试配置:")
    print(f"  优化技术: {config.technique}")
    print(f"  批次级别: {config.batch_level}")
    print(f"  实验名称: {config.experiment_name}")
    print(f"  模型: {config.model}")
    print(f"  设备: {config.device}")
    print(f"  最大tokens: {config.max_tokens}")
    print(f"  温度: {config.temperature}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  测试迭代: {config.test_iterations}")
    print(f"  输出目录: {config.output_dir}")
    
    # 准备测试提示
    custom_prompts = None
    if args.prompts:
        custom_prompts = args.prompts
        print(f"  使用自定义提示: {len(custom_prompts)}个")
    elif args.prompts_file:
        custom_prompts = load_prompts_from_file(args.prompts_file)
        if not custom_prompts:
            print("⚠️ 无法从文件加载提示，使用默认提示")
    
    
    
    try:
        # 根据优化技术创建推理模块
        if config.technique == "PagedAttention":
            inference_module = InferenceOnPagedAttention(config)
        elif config.technique == "量化":
            inference_module = InferenceOnQuantization(config)
        else:
            raise ValueError(f"不支持的优化技术: {config.technique}")
        
        if args.benchmark_memory:
            # 内存效率基准测试
            if hasattr(inference_module, 'benchmark_memory_efficiency'):
                batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
                print(f"\n🧪 开始内存效率基准测试...")
                results = inference_module.benchmark_memory_efficiency(batch_sizes)
                print(f"✅ 内存基准测试完成")
            else:
                print(f"⚠️ {config.technique} 不支持内存基准测试")
                return
            
        else:
            # 常规对比测试
            print(f"\n🧪 开始性能对比测试...")
            if args.method == "both":
                results = inference_module.run_comparison_test(custom_prompts)
            else:
                # 生成或使用自定义提示
                if custom_prompts is None:
                    test_prompts = inference_module.generate_test_prompts()
                else:
                    test_prompts = custom_prompts
                
                results = inference_module.inference(test_prompts, method=args.method)
            
            print(f"✅ 测试完成")
        
        print(f"\n📊 结果已保存到: {config.output_dir}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 测试被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 确保清理资源
        try:
            if 'inference_module' in locals():
                inference_module.cleanup()
        except:
            pass


if __name__ == "__main__":
    main()
