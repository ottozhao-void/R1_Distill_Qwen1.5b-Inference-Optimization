"""
主程序 - 大语言模型推理和优化

这是项目的主入口文件，演示如何使用各个模块进行LLM推理性能对比测试。
"""

import argparse
import sys
import os
from typing import List, Optional

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, ConfigPresets, load_config_from_env
from inference_paged_attention import InferenceOnPagedAttention


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="LLM推理优化性能测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 使用默认小模型进行快速测试
  python main.py --preset small
  
  # 使用自定义模型
  python main.py --model facebook/opt-1.3b --max-tokens 200
  
  # 只运行基础推理
  python main.py --method basic
  
  # 只运行优化推理
  python main.py --method optimized
  
  # 使用自定义GPU设备
  python main.py --devices 0,1 --model meta-llama/Llama-2-7b-chat-hf
  
  # 内存效率基准测试
  python main.py --benchmark-memory --batch-sizes 1,2,4,8
        """
    )
    
    # 模型配置
    parser.add_argument("--model", type=str, default=None,
                       help="模型名称或路径")
    parser.add_argument("--devices", type=str, default=None,
                       help="CUDA设备ID，用逗号分隔 (例如: 0,1)")
    
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
    
    # 预设配置
    parser.add_argument("--preset", type=str, choices=["small", "large", "llama"],
                       help="使用预设配置")
    
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
    
    # 使用预设配置
    if args.preset:
        if args.preset == "small":
            config = ConfigPresets.small_model_config()
        elif args.preset == "large":
            config = ConfigPresets.large_model_config()
        elif args.preset == "llama":
            config = ConfigPresets.llama_config()
    else:
        # 从环境变量加载配置或使用默认配置
        config = load_config_from_env()
    
    # 用命令行参数覆盖配置
    if args.model:
        config.model = args.model
    
    if args.devices:
        device_ids = [int(x.strip()) for x in args.devices.split(",")]
        config.device = device_ids
    
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


def check_dependencies():
    """检查依赖包是否安装"""
    required_packages = {
        "torch": "PyTorch",
        "transformers": "Hugging Face Transformers",
        "vllm": "vLLM",
        "numpy": "NumPy",
        "psutil": "psutil"
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(name)
    
    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ 所有依赖包已安装")
    return True


def print_system_info():
    """打印系统信息"""
    print("\n" + "="*60)
    print("系统信息")
    print("="*60)
    
    try:
        import torch
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"    内存: {memory_total:.1f}GB")
    except ImportError:
        print("PyTorch未安装")
    
    try:
        import transformers
        print(f"Transformers版本: {transformers.__version__}")
    except ImportError:
        print("Transformers未安装")
    
    try:
        import vllm
        print(f"vLLM版本: {vllm.__version__}")
    except ImportError:
        print("vLLM未安装")
    
    print("="*60)


def main():
    """主函数"""
    print("🚀 大语言模型推理和优化性能测试")
    print("="*60)
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 打印系统信息
    print_system_info()
    
    # 创建配置
    config = create_config_from_args(args)
    
    print(f"\n📋 测试配置:")
    print(f"  模型: {config.model}")
    print(f"  设备: {config.device}")
    print(f"  最大tokens: {config.max_tokens}")
    print(f"  温度: {config.temperature}")
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
        # 创建推理模块
        inference_module = InferenceOnPagedAttention(config)
        
        if args.benchmark_memory:
            # 内存效率基准测试
            batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
            print(f"\n🧪 开始内存效率基准测试...")
            results = inference_module.benchmark_memory_efficiency(batch_sizes)
            print(f"✅ 内存基准测试完成")
            
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
